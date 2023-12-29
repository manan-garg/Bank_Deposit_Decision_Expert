# Import relevant Libraries
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from scipy.sparse import hstack
import numpy as np

# Load saved TFIDF vectorizer and random forest model from local
loaded_tfidf_vect_train_dict = pickle.load(open(r'tfidf_vect_train_dict.pickle', 'rb'))
loaded_model = pickle.load(open(r'model.sav', 'rb'))

# Initiate Flask app
app = Flask(__name__)

# Create the API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the API call
        json_data = request.get_json()

        # Convert the JSON to a DataFrame
        df = pd.DataFrame([json_data])

        # Replacing '999' with '-1' in pdays
        df.pdays = df.pdays.replace(999,-1)

        # Creating extra feature
        df['campaign_inference'] = df['campaign'].apply(lambda x: 0 if x > 25 else 1)

        # Feature engineering - categorical
        X_test_cat_processed = pd.DataFrame()
        for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']:
          X_test_col = loaded_tfidf_vect_train_dict[col].transform(df[col])
          X_test_cat_processed = hstack((X_test_cat_processed, X_test_col))

        # Feature engineering - numerical
        cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign_inference']

        # Convert selected columns to NumPy array
        X_test_col = df[cols].values

        # Concatenate the arrays along the columns (axis=1)
        X_test_num_processed = np.concatenate((X_test_col,), axis=1)

        # Ensure both arrays are 2D
        if X_test_cat_processed.ndim == 1:
            X_test_cat_processed = X_test_cat_processed.reshape(-1, 1)

        if X_test_num_processed.ndim == 1:
            X_test_num_processed = X_test_num_processed.reshape(-1, 1)

        # Horizontally stacking categorical and numerical columns to prepare dataset
        X_test = hstack((X_test_cat_processed,X_test_num_processed))

        # Predicting on the test row
        y_pred = loaded_model.predict(X_test)

        # Final variable to return
        final_prediction = 'Not Converted'

        # Checking value of y_pred
        if y_pred==1:
            final_prediction = 'Converted'

        # Print final prediction
        print(final_prediction)

        # Return the prediction as JSON
        return jsonify({'Prediction': final_prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

# Driver function
if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True, port = 9000)