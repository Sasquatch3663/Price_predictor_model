from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Use environment variable for security

# Function to fetch live crop price from SerpAPI
def fetch_crop_price(crop_name):
    params = {
        "engine": "google",
        "q": f"{crop_name} price per kg in India",
        "api_key": SERPAPI_KEY,
    }
    
    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()
    
    for result in data.get("organic_results", []):
        snippet = result.get("snippet", "")

        # Extract ₹ price using regex
        match = re.search(r"₹\s?(\d+)", snippet)
        if match:
            return int(match.group(1))  # Convert extracted price to integer

    return None  # No price found

# List of crops to predict prices for
crops = ["Rice", "Wheat", "Corn", "Moong Dal", "Arahar Dal", "Mustard", "Sugar Cane", "Mango", "Dragon Fruit", "Tea"]

# Fetch updated daily prices (Simulated historical data)
data = {"Date": [(datetime.today() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(29, -1, -1)]}
for crop in crops:
    fetched_price = fetch_crop_price(crop) or np.random.randint(40, 80)  # Default if price not found
    data[crop] = [fetched_price - 12, fetched_price - 10, fetched_price - 8, fetched_price - 5, 
                  fetched_price - 3, fetched_price, fetched_price + 3, fetched_price + 5, 
                  fetched_price + 7, fetched_price + 10] * 3
    data[crop] = data[crop][:30]  # Ensure exactly 30 values

df = pd.DataFrame(data)

# Function to predict crop price
def predict_crop_price(crop_name):
    if crop_name not in df.columns:
        return {"error": f"Data for {crop_name} not available"}

    df["Date"] = pd.to_datetime(df["Date"])
    df["Timestamp"] = df["Date"].map(pd.Timestamp.toordinal)  # Convert dates to numerical values
    
    X = df[["Timestamp"]]  # Features (Date as ordinal values)
    y = df[crop_name]  # Target (Crop Prices)

    # Normalize timestamps for better accuracy
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting dataset into training & testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

    # Train the Random Forest Regression Model
    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)

    # Predict price for the next day
    next_day = datetime.today() + timedelta(days=1)
    next_day_timestamp = scaler.transform([[next_day.toordinal()]])
    future_price = model.predict(next_day_timestamp)
    future_price_int = int(round(future_price[0]))  # Convert to integer

    return {
        "predicted_price": round(future_price[0], 2),
        "predicted_price_integer": future_price_int
    }

# API Endpoint for crop price prediction
@app.route('/predict_price', methods=['GET'])
def get_predicted_price():
    crop_name = request.args.get('crop_name')

    if not crop_name:
        return jsonify({"error": "Please provide a crop name"}), 400

    prediction = predict_crop_price(crop_name)
    
    # Fetch live price from SerpAPI
    live_price = fetch_crop_price(crop_name)

    return jsonify({
        "crop_name": crop_name,
        "predicted_price": prediction.get("predicted_price"),
        "predicted_price_integer": prediction.get("predicted_price_integer"),
        "live_price": live_price if live_price else "Not Available"
    })

# Home route
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Crop Price Prediction API is running!"})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
