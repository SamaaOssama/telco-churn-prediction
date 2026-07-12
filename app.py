import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Telco Customer Churn Predictor")
st.markdown(
    "Enter a customer's information below to predict whether they are likely to **churn** "
    "(leave the company). This app uses a **Naive Bayes** model, selected as the best-performing "
    "classifier (F1-score = 0.620, Recall = 76.7%) in the underlying analysis."
)
st.divider()

# ---------------------------------------------------------------
# Load model + expected feature schema
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    feature_names = joblib.load("model_features.pkl")
    return model, feature_names

model, feature_names = load_artifacts()

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
with st.form("customer_form"):
    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    with col2:
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)

    st.subheader("Account & Billing")
    col3, col4 = st.columns(2)
    with col3:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    with col4:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    st.subheader("Services Subscribed")
    col5, col6 = st.columns(2)
    with col5:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], help="Only applicable if Phone Service is Yes")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col6:
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])

    col7, col8 = st.columns(2)
    with col7:
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    with col8:
        pass
    col9, col10 = st.columns(2)
    with col9:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    with col10:
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

# ---------------------------------------------------------------
# Preprocessing (mirrors the notebook's Parts 2 & 4 pipeline exactly)
# ---------------------------------------------------------------
def build_feature_row(inputs: dict, feature_names: list) -> pd.DataFrame:
    row = {}

    # --- Label-encoded binary columns (Yes/No -> 1/0, Male/Female -> 1/0) ---
    row["Gender"] = 1 if inputs["gender"] == "Male" else 0
    row["SeniorCitizen"] = 1 if inputs["senior_citizen"] == "Yes" else 0
    row["Partner"] = 1 if inputs["partner"] == "Yes" else 0
    row["Dependents"] = 1 if inputs["dependents"] == "Yes" else 0
    row["PhoneService"] = 1 if inputs["phone_service"] == "Yes" else 0
    row["MultipleLines"] = 1 if inputs["multiple_lines"] == "Yes" else 0
    row["OnlineSecurity"] = 1 if inputs["online_security"] == "Yes" else 0
    row["OnlineBackup"] = 1 if inputs["online_backup"] == "Yes" else 0
    row["DeviceProtection"] = 1 if inputs["device_protection"] == "Yes" else 0
    row["TechSupport"] = 1 if inputs["tech_support"] == "Yes" else 0
    row["StreamingTV"] = 1 if inputs["streaming_tv"] == "Yes" else 0
    row["StreamingMovies"] = 1 if inputs["streaming_movies"] == "Yes" else 0
    row["PaperlessBilling"] = 1 if inputs["paperless_billing"] == "Yes" else 0

    # --- Numerical features ---
    row["Tenure"] = inputs["tenure"]
    row["MonthlyCharges"] = inputs["monthly_charges"]
    row["TotalCharges"] = inputs["total_charges"]

    # --- Engineered features (Part 4.1) ---
    row["IsNewCustomer"] = 1 if inputs["tenure"] <= 12 else 0
    row["NumAdditionalServices"] = sum(
        1 for col_input_key in ["online_security", "online_backup", "device_protection",
                                 "tech_support", "streaming_tv", "streaming_movies"]
        if inputs[col_input_key] == "Yes"
    )

    # --- One-hot encoded nominal columns (drop_first=True, matching training) ---
    row["InternetService_Fiber optic"] = 1 if inputs["internet_service"] == "Fiber optic" else 0
    row["InternetService_No"] = 1 if inputs["internet_service"] == "No" else 0
    row["Contract_One year"] = 1 if inputs["contract"] == "One year" else 0
    row["Contract_Two year"] = 1 if inputs["contract"] == "Two year" else 0
    row["PaymentMethod_Credit card (automatic)"] = 1 if inputs["payment_method"] == "Credit card (automatic)" else 0
    row["PaymentMethod_Electronic check"] = 1 if inputs["payment_method"] == "Electronic check" else 0
    row["PaymentMethod_Mailed check"] = 1 if inputs["payment_method"] == "Mailed check" else 0

    # Ensure column order exactly matches what the model was trained on
    return pd.DataFrame([row])[feature_names]

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if submitted:
    user_inputs = {
        "gender": gender, "senior_citizen": senior_citizen, "partner": partner, "dependents": dependents,
        "tenure": tenure, "monthly_charges": monthly_charges, "total_charges": total_charges,
        "contract": contract, "paperless_billing": paperless_billing, "payment_method": payment_method,
        "phone_service": phone_service, "multiple_lines": multiple_lines, "internet_service": internet_service,
        "online_security": online_security, "online_backup": online_backup, "device_protection": device_protection,
        "tech_support": tech_support, "streaming_tv": streaming_tv, "streaming_movies": streaming_movies,
    }

    X_input = build_feature_row(user_inputs, feature_names)

    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]  # probability of class "1" (Churn)

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn**.")
    else:
        st.success(f"✅ This customer is **likely to stay**.")

    st.metric(label="Churn Probability", value=f"{probability * 100:.1f}%")
    st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("See input feature values used for this prediction"):
        st.dataframe(X_input.T.rename(columns={0: "Value"}))
