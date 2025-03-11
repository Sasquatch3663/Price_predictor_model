# 🌾 Crop Price Prediction API
🚀 A Flask-based API that provides real-time crop price analysis and predicts next-day crop prices using Machine Learning (Random Forest Regressor). It fetches live market prices using SerpAPI and enables data-driven decision-making for farmers.

# 📌 Features
✅ Predicts next-day crop prices using 30-day historical data.
✅ Fetches real-time crop prices from Google Search (SerpAPI).
✅ Supports multiple crops (Rice, Wheat, Corn, etc.).
✅ Built with Flask API & Machine Learning.
✅ CORS-enabled for frontend integration.

# Installation & Setup
1. Install Dependencies
Make sure you have Python installed (3.7+ recommended). Then install required modules:
pip install flask flask-cors numpy pandas requests scikit-learn waitress

2. Set Up Environment Variable
Create a .env file or set the SerpAPI Key manually:
export SERPAPI_KEY="your_actual_serpapi_key"

Run the API
python app.py
Your API will be running at:
http://127.0.0.1:5000

# How It Works?
1️⃣ Fetches past 30 days of crop prices (simulated if unavailable).
2️⃣ Trains a Random Forest Regressor to learn price trends.
3️⃣ Predicts the next-day price based on historical data.
4️⃣ Fetches real-time price from SerpAPI for comparison.
5️⃣ Returns predicted & live prices in JSON format.

# Supported Crops
🌾 Rice, 🌾 Wheat, 🌽 Corn, 🥭 Mango, ☕ Tea, 🌿 Mustard, 🍌 Banana, 🥔 Moong Dal, 🫘 Arahar Dal, 🍬 Sugarcane.

# Contributions
🔹 Fork the repo, submit pull requests & enhance the project!
🔹 If you found this helpful, ⭐ Star this project on GitHub!

