import calendar
from datetime import datetime
import joblib
import traceback
import requests
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load the Gemini API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load model and encoders
model = joblib.load("price_model.pkl")
le_crop = joblib.load("crop_encoder.pkl")
le_state = joblib.load("state_encoder.pkl")
demand_model = joblib.load("demand_model.pkl")
scaler = joblib.load("scalers.pkl")
label_encoders = joblib.load("label_encoder.pkl")

# Initialize Firebase Admin SDK (use your actual path)
cred = credentials.Certificate("firebase-adminsdk.json")
firebase_admin.initialize_app(cred)

# Get Firestore database reference
db = firestore.client()

# Function to get weather data (to be replaced with real API calls)
def get_weather_data(state, month, year):
    # for future
    # get the latitude and longitude data from database for the state and pass to API to get weather data
    return {
        "temperature": np.random.uniform(20, 45),
        "rainfall": np.random.uniform(0, 300),
        "soil_moisture": np.random.uniform(0.1, 0.5),
        "ndvi": np.random.uniform(0.2, 0.7)
    }

def get_additional_features(state, crop, month, year):
    return {
        "seasonality": np.random.uniform(0.5, 1.5),
        "marketing_spend": np.random.uniform(10000, 50000),
        "competitor_price": np.random.uniform(10, 50),
        "special_event": np.random.choice([0, 1], p=[0.8, 0.2]),
        "supply": np.random.uniform(500, 2000)
    }

@app.route('/')
def home():
    return "Crop Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    print("Received Data:", request.json)  # Print received request
    data = request.json
    state = data.get("state")
    crop = data.get("crop")
    start_year = data.get("startYear")
    start_month = data.get("startMonth")
    end_year = data.get("endYear")
    end_month = data.get("endMonth")

    if not state or not crop or not start_year or not start_month or not end_year or not end_month:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Encode state and crop
        state_encoded = le_state.transform([state])[0]
        crop_encoded = le_crop.transform([crop])[0]

        predictions = []
        current_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)

        while current_date <= end_date:
            year = current_date.year
            month = current_date.month

            # Fetch weather data
            weather = get_weather_data(state, month, year)

            # Create input for model
            input_data = np.array([
                state_encoded,
                crop_encoded,
                year,
                month,
                weather["temperature"],
                weather["rainfall"],
                weather["soil_moisture"],
                weather["ndvi"]
            ]).reshape(1, -1)

            # Predict price
            predicted_price = model.predict(input_data)
            predictions.append({
                "month": f"{calendar.month_name[month]} {year}",
                "price": round(predicted_price[0], 2)
            })

            # Move to the next month
            next_month = month + 1 if month < 12 else 1
            next_year = year if month < 12 else year + 1
            current_date = datetime(next_year, next_month, 1)

        return jsonify(predictions)

    except Exception as e:
        print("Error:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/pastPrices', methods=['POST'])
def get_past_prices():
    data = request.json
    state = data.get("state")
    crop = data.get("crop")
    year = int(data.get("year"))
    month = int(data.get("month"))

    if not state or not crop or not year or not month:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        past_prices = []
        current_date = datetime(year, month, 1)

        # Get past 6 months
        for i in range(6):
            past_year = current_date.year
            past_month = current_date.month

            # Convert to database format "MM-YYYY"
            db_month = f"{past_month:02d}-{past_year}"

            # Fetch price from database if not present then predict
            record = fetch_price_from_db(state, crop, db_month)

            if not record.empty:
                price = record.iloc[0]["price"]
            else:
                # Predict price if not found in database
                price = predict_price(state, crop, db_month)

            past_prices.append({
                "month": f"{calendar.month_name[past_month]} {past_year}",
                "price": price
            })
            
            # Move to the previous month
            previous_month = past_month - 1 if past_month > 1 else 12
            previous_year = past_year if past_month > 1 else past_year - 1
            current_date = datetime(previous_year, previous_month, 1)

        return jsonify(past_prices)
    
    except Exception as e:
            return jsonify({"error": str(e)}), 500

def fetch_price_from_db(state, crop, db_month):
    try:
        # Fetch the document for the specific state and crop
        state_doc_ref = db.collection('crop_data').document(state)
        crop_collection_ref = state_doc_ref.collection(crop)

        # Fetch the document for the specific month
        month_doc_ref = crop_collection_ref.document(db_month)
        month_doc_snapshot = month_doc_ref.get()

        if month_doc_snapshot.exists:
            month_data = month_doc_snapshot.to_dict()
            if 'price' in month_data:
                # Create a DataFrame with the fetched price
                data = {
                    "price": [month_data['price']],
                    "month": [db_month]
                }
                return pd.DataFrame(data)
        
        # Return an empty DataFrame if no price is found
        return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching price from database: {e}")
        return pd.DataFrame()
    
@app.route('/cropsCollection', methods=['POST'])
def crops_collection():
    data = request.json
    selected_state = data.get("selectedState")
    previous_month = data.get("previousMonth")
    next_month = data.get("nextMonth")

    if not selected_state or not previous_month or not next_month:
        return jsonify({"error": "Missing required fields: selectedState, previousMonth, nextMonth"}), 400

    try:
        # 🔹 Get all collections inside the selected state (list of crops)
        state_doc_ref = db.collection('crop_data').document(selected_state)
        crops_collections = state_doc_ref.collections()

        fetched_crops = []

        for crop_collection in crops_collections:
            crop_name = crop_collection.id  # Crop name

            # 🔹 Get previous and next month documents
            previous_month_doc_ref = crop_collection.document(previous_month)
            next_month_doc_ref = crop_collection.document(next_month)

            # 🔹 Fetch previous month price & demand
            previous_month_snapshot = previous_month_doc_ref.get()
            previous_month_price = 0
            previous_month_demand = 0

            if previous_month_snapshot.exists:
                doc_data = previous_month_snapshot.to_dict()
                previous_month_price = doc_data.get('price', 0)
                previous_month_demand = doc_data.get('demand', 0)

            # Predict if missing
            if previous_month_price == 0:
                previous_month_price = predict_price(selected_state, crop_name, previous_month)
            if previous_month_demand == 0:
                previous_month_demand = predict_demand_value(selected_state, crop_name, previous_month)

            # 🔹 Fetch next month price & demand
            next_month_snapshot = next_month_doc_ref.get()
            next_month_price = 0
            next_month_demand = 0

            if next_month_snapshot.exists:
                doc_data = next_month_snapshot.to_dict()
                next_month_price = doc_data.get('price', 0)
                next_month_demand = doc_data.get('demand', 0)

            # Predict if missing
            if next_month_price == 0:
                next_month_price = predict_price(selected_state, crop_name, next_month)
            if next_month_demand == 0:
                next_month_demand = predict_demand_value(selected_state, crop_name, next_month)

            # Store result
            fetched_crops.append({
                "name": crop_name,
                "previousMonthPrice": previous_month_price,
                "nextMonthPrice": next_month_price,
                "previousMonthDemand": previous_month_demand,
                "nextMonthDemand": next_month_demand,
            })

        return jsonify({"crops": fetched_crops})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def predict_price(state, crop, month):
    try:
        # Encode state and crop
        state_encoded = le_state.transform([state])[0]
        crop_encoded = le_crop.transform([crop])[0]

        # Extract year and month from the month string (e.g., "01-2023")
        year = int(month.split('-')[1])
        month = int(month.split('-')[0])

        # Fetch weather data
        weather = get_weather_data(state, month, year)

        # Create input for model
        input_data = np.array([
            state_encoded,
            crop_encoded,
            year,
            month,
            weather["temperature"],
            weather["rainfall"],
            weather["soil_moisture"],
            weather["ndvi"]
        ]).reshape(1, -1)

        # Predict price
        predicted_price = model.predict(input_data)
        return round(predicted_price[0], 2)

    except Exception as e:
        print(f"Error predicting price: {e}")
        return 100  # Default price if prediction fails
    
@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    data = request.json
    state = data.get("state")
    crop = data.get("crop")
    start_year = data.get("startYear")
    start_month = data.get("startMonth")
    end_year = data.get("endYear")
    end_month = data.get("endMonth")

    if not state or not crop or not start_year or not start_month or not end_year or not end_month:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        state_encoded = le_state.transform([state])[0]
        crop_encoded = le_crop.transform([crop])[0]
        predictions = []
        current_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)

        while current_date <= end_date:
            year, month = current_date.year, current_date.month
            additional_features = get_additional_features(state, crop, month, year)
            price = get_price_or_predict(state, crop, year, month)

            # Function to safely encode categorical values
            def safe_encode(label_encoder, value):
                """Encodes a categorical value using LabelEncoder, handling unseen labels."""
                if value in label_encoder.classes_:
                    return label_encoder.transform([value])[0]  # Encode if it exists
                else:
                    print(f"⚠ Warning: Unseen label '{value}' detected. Using default encoding (0).")
                    return 0  # Assign a default value (e.g., 0) for unseen labels
                
            # Define numerical columns (same as in training)
            numerical_cols = ["price", "marketing_spend", "competitor_price", "supply"]

            # Encode categorical variables safely
            state_encoded = safe_encode(label_encoders["state"], state)
            crop_encoded = safe_encode(label_encoders["crop"], crop)

            # Prepare input data as a dictionary
            input_data = {
                "state": state_encoded, 
                "crop": crop_encoded, 
                "year": year, 
                "month": month,
                "seasonality": additional_features["seasonality"],
                "special_event": additional_features["special_event"],
                "price": price,
                "marketing_spend": additional_features["marketing_spend"],
                "competitor_price": additional_features["competitor_price"],
                "supply": additional_features["supply"]
            }

            # Convert to DataFrame with correct column names
            input_df = pd.DataFrame([input_data])

            # Apply scaling to numerical features
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

            # Ensure input order matches what the model expects
            input_df = input_df.reindex(columns=demand_model.feature_names_in_)

            # Predict demand
            predicted_demand = float(demand_model.predict(input_df)[0])  # Convert to float

            predictions.append({
                "month": f"{calendar.month_name[month]} {year}",
                "demand": round(predicted_demand, 2)
            })
            
            current_date = datetime(year + (month // 12), (month % 12) + 1, 1)
        
        return jsonify(predictions)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/pastDemand', methods=['POST'])
def get_past_demand():
    data = request.json
    state, crop = data.get("state"), data.get("crop")
    year, month = int(data.get("year")), int(data.get("month"))

    if not state or not crop or not year or not month:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        past_demand = []
        current_date = datetime(year, month, 1)

        for _ in range(6):
            past_year, past_month = current_date.year, current_date.month
            db_month = f"{past_month:02d}-{past_year}"
            record = fetch_demand_from_db(state, crop, db_month)
            demand = record.iloc[0]["demand"] if not record.empty else predict_demand_value(state, crop, db_month)

            past_demand.append({"month": f"{calendar.month_name[past_month]} {past_year}", "demand": demand})
            current_date = datetime(past_year if past_month > 1 else past_year - 1, past_month - 1 if past_month > 1 else 12, 1)
        
        return jsonify(past_demand)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def fetch_demand_from_db(state, crop, db_month):
    try:
        state_doc_ref = db.collection('crop_data').document(state)
        crop_collection_ref = state_doc_ref.collection(crop)
        month_doc_ref = crop_collection_ref.document(db_month)
        month_doc_snapshot = month_doc_ref.get()
        
        if month_doc_snapshot.exists and 'demand' in month_doc_snapshot.to_dict():
            return pd.DataFrame({"demand": [month_doc_snapshot.to_dict()['demand']], "month": [db_month]})
        
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching demand from database: {e}")
        return pd.DataFrame()

def predict_demand_value(state, crop, month):
    try:
        state_encoded, crop_encoded = le_state.transform([state])[0], le_crop.transform([crop])[0]
        year, month = int(month.split('-')[1]), int(month.split('-')[0])

        additional_features = get_additional_features(state, crop, month, year)
        price = get_price_or_predict(state, crop, year, month)

        # Function to safely encode categorical values
        def safe_encode(label_encoder, value):
            """Encodes a categorical value using LabelEncoder, handling unseen labels."""
            if value in label_encoder.classes_:
                return label_encoder.transform([value])[0]  # Encode if it exists
            else:
                print(f"⚠ Warning: Unseen label '{value}' detected. Using default encoding (0).")
                return 0  # Assign a default value (e.g., 0) for unseen labels
            
        # Define numerical columns (same as in training)
        numerical_cols = ["price", "marketing_spend", "competitor_price", "supply"]

        # Encode categorical variables safely
        state_encoded = safe_encode(label_encoders["state"], state)
        crop_encoded = safe_encode(label_encoders["crop"], crop)

        # Prepare input data as a dictionary
        input_data = {
            "state": state_encoded, 
            "crop": crop_encoded, 
            "year": year, 
            "month": month,
            "seasonality": additional_features["seasonality"],
            "special_event": additional_features["special_event"],
            "price": price,
            "marketing_spend": additional_features["marketing_spend"],
            "competitor_price": additional_features["competitor_price"],
            "supply": additional_features["supply"]
        }

        # Convert to DataFrame with correct column names
        input_df = pd.DataFrame([input_data])

        # Apply scaling to numerical features
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Ensure input order matches what the model expects
        input_df = input_df.reindex(columns=demand_model.feature_names_in_)

        # Predict demand
        predicted_demand = float(demand_model.predict(input_df)[0])  # Convert to float

        return round(predicted_demand, 2)
    except Exception as e:
        print(f"Error predicting demand: {e}")
        return 100


def get_price_or_predict(state, crop, year, month):
    try:
        db_month = f"{month:02d}-{year}"

        # Fetch price from database if not present then predict
        record = fetch_price_from_db(state, crop, db_month)

        if not record.empty:
            return record.iloc[0]["price"]
        else:
            return predict_price(state, crop, db_month)
    
    except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route('/detectDisease', methods=['POST'])
def detect_disease():
    try:
        data = request.get_json()
        image_base64 = data.get("image_base64")
        lang = data.get("language")

        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

        headers = {
            "Content-Type": "application/json"
        }

        prompt = f"""You are an agronomist. Identify any visible crop disease and suggest an appropriate chemical or organic treatment.
        Respond ONLY in JSON using this schema:
        {{"disease":string, "treatment":string, "crop_name":string}}
        Translate all values (not keys) into {lang}. If could not translate then give in English"""

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 10,
                "topP": 0.95,
                "maxOutputTokens": 512
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=15)
        return jsonify(response.json()), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/fertCalculator', methods=['POST'])
def fert_calculator():
    try:
        data = request.get_json(force=True)
        crop = data.get("crop", "Wheat").title()
        area = float(data.get("area_ha", 1))
        soil = data.get("soil_type", "loamy")
        stage = data.get("growth_stage", "vegetative")

        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

        headers = {
            "Content-Type": "application/json"
        }

        prompt = f"""
            You are an agronomy assistant. 
            Given a crop, {crop} in {area} hectare, with {soil} soil, and at {stage} stage. Suggest fertilizer requirements and example products.
            Respond ONLY in JSON using this structure:
            {{
            "crop": "<crop>",
            "area_ha": <float>,
            "soil_type": "<soil>",
            "growth_stage": "<stage>",
            "fertilizer": {{
                "N_kg": <float>,
                "P_kg": <float>,
                "K_kg": <float>,
                "products": [
                    {{"name": "Urea", "amount_kg": <float>}},
                    {{"name": "DAP", "amount_kg": <float>}}
                ]
            }},
            "notes": "<short agronomy suggestion>"
            }}
            """

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 10,
                "topP": 0.95,
                "maxOutputTokens": 512
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=15)
        gen = response.json()
        raw_text = gen['candidates'][0]['content']['parts'][0]['text']

        # Extract the JSON from inside the raw string
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        json_text = raw_text[start:end]

        return jsonify(json.loads(json_text)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)