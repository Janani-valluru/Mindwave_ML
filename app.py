from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
from pymongo import MongoClient
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the model using joblib
try:
    model = load('C:/Users/JANANI.V.A/Desktop/mentalhealth/depression_prediction_model_v1.joblib')
except Exception as e:
    print("Error loading the model:", str(e))
    model = None

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MongoDB configuration
mongo_client = MongoClient('mongodb+srv://jananishetty24:junnu2404@cluster01.obkrmam.mongodb.net/')
db = mongo_client['test']
collection = db['testresults']

# Initialize LabelEncoder for encoding usernames
username_encoder = LabelEncoder()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if model is None:
        return jsonify(error="Model not loaded"), 500

    try:
        # Fetch real-time data from MongoDB
        data = list(collection.find({}, {'_id': 0, 'username': 1, 'score': 1}))
        print("Data from MongoDB:", data)

        # Preprocess and validate the data
        df = pd.DataFrame(data)
        df['username_encoded'] = username_encoder.fit_transform(df['username'])
        if not all(col in df.columns for col in ['username_encoded', 'score']):
            return jsonify(error="Missing required columns in input data"), 400

        # Make predictions
        predictions = model.predict(df[['username_encoded', 'score']])
        print("Predictions:", predictions)

        # Combine predictions with original data
        df['depression_level'] = predictions

        # Return predictions as JSON response
        return df[['username', 'depression_level']].to_json(orient='records')

    except Exception as e:
        print("Error predicting:", str(e))
        return jsonify(error="Prediction failed"), 500

if __name__ == '__main__':
    app.run(debug=True)
