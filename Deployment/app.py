from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# 1) Load the pipeline (model + preprocessing) and threshold
pipeline = joblib.load("models/final_logreg_pipeline.pkl")
threshold_dict = joblib.load("models/final_threshold.pkl")
best_threshold = threshold_dict["threshold"]

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON data for one or multiple customers in the format:
    {
      "data": [
        {"Age": 30, "City": "Brea", "Monthly Charge": 45.85, ...},
        {"Age": 59, "City": "Santa Cruz", "Monthly Charge": 19.6, ...}
      ]
    }
    Returns predicted churn = 0 or 1
    """
    # 2) Parse incoming JSON
    payload = request.get_json(force=True)
    if "data" not in payload:
        return jsonify({"error": "No 'data' field found in JSON"}), 400
    
    # 3) Convert input to a DataFrame
    #    'data' should be a list of dictionaries, each row is a customer's features
    input_df = pd.DataFrame(payload["data"])

    # 4) Get predicted probabilities from the pipeline
    probabilities = pipeline.predict_proba(input_df)[:, 1]
    
    # 5) Apply the best threshold
    predictions = (probabilities >= best_threshold).astype(int)

    # 6) Return results in JSON
    # You might also include the probabilities if desired
    results = []
    for prob, pred in zip(probabilities, predictions):
        results.append({
            "churn_probability": float(prob),
            "churn_prediction": int(pred)
        })

    return jsonify({"predictions": results})

@app.route("/", methods=["GET"])
def home():
    return "<h3>Churn Prediction API is Running!</h3>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
