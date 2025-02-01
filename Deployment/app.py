import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import FunctionTransformer

# ✅ **Define log_transform before loading the model**
def log_transform(X):
    """Log transform skewed numeric features."""
    return np.log1p(X)

# **Load trained model and preprocessing pipeline**
model_data = joblib.load("Models/final_best_model.pkl")

# ✅ **Ensure the custom function is reattached to the preprocessor**
preprocessor = model_data["preprocessor"]
preprocessor.named_transformers_["num"].steps[0] = ("log_transform", FunctionTransformer(log_transform))

model = model_data["model"]

# Load threshold
threshold_data = joblib.load("models/final_threshold.pkl")
threshold = threshold_data["threshold"]

# **Feature Selection (Ensure consistency with trained model)**
SELECTED_FEATURES = [
    "Tenure in Months", "Monthly Charge", "Total Charges", "CLTV",
    "Contract_One Year", "Contract_Two Year",
    "Online Security", "Streaming TV",
    "Payment Method_Credit Card", "Payment Method_Mailed Check",
    "Paperless Billing", "RevenuePerMonth"
]

# **Streamlit UI**
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability.")

# **User Input Form**
with st.form("customer_form"):
    tenure = st.number_input("📅 Tenure in Months", min_value=0, max_value=100, value=12)
    monthly_charge = st.number_input("💰 Monthly Charge", min_value=0.0, value=50.0)
    total_charges = st.number_input("💳 Total Charges", min_value=0.0, value=600.0)
    cltv = st.number_input("🔮 Customer Lifetime Value (CLTV)", min_value=0.0, value=5000.0)
    
    contract_one_year = st.checkbox("📄 Contract: One Year")
    contract_two_year = st.checkbox("📄 Contract: Two Year")
    
    online_security = st.checkbox("🔐 Online Security")
    streaming_tv = st.checkbox("📺 Streaming TV")
    
    payment_credit_card = st.checkbox("💳 Payment Method: Credit Card")
    payment_mailed_check = st.checkbox("📬 Payment Method: Mailed Check")
    
    paperless_billing = st.checkbox("🧾 Paperless Billing")
    
    revenue_per_month = st.number_input("📈 Revenue Per Month", min_value=0.0, value=80.0)

    submit_button = st.form_submit_button("🚀 Predict Churn")

# **Process Input & Make Prediction**
if submit_button:
    # **Prepare input for model**
    input_data = pd.DataFrame({
        "Tenure in Months": [tenure],
        "Monthly Charge": [monthly_charge],
        "Total Charges": [total_charges],
        "CLTV": [cltv],
        "Contract_One Year": [int(contract_one_year)],
        "Contract_Two Year": [int(contract_two_year)],
        "Online Security": [int(online_security)],
        "Streaming TV": [int(streaming_tv)],
        "Payment Method_Credit Card": [int(payment_credit_card)],
        "Payment Method_Mailed Check": [int(payment_mailed_check)],
        "Paperless Billing": [int(paperless_billing)],
        "RevenuePerMonth": [revenue_per_month],
    })

    # **Preprocess input**
    input_preprocessed = preprocessor.transform(input_data)

    # **Make prediction**
    churn_probability = model.predict_proba(input_preprocessed)[:, 1][0]
    churn_prediction = int(churn_probability >= threshold)

    # **Display Results**
    st.subheader("🔍 Prediction Result:")
    st.write(f"**Churn Probability:** {churn_probability:.2%}")
    
    if churn_prediction == 1:
        st.error("⚠️ **High Risk of Churn!** Take action to retain this customer.")
    else:
        st.success("✅ **Low Risk of Churn.** The customer is likely to stay.")
