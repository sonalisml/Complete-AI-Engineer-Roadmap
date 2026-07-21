import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("Heart_disease_model.pkl")

# Page
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below.")

# -------------------------
# User Inputs
# -------------------------

age = st.number_input("Age", 1, 100, 45)

sex = st.selectbox("Gender", [
    "Male",
    "Female"
])

cp = st.selectbox("Chest Pain", [
    "typical angina",
    "atypical angina",
    "non-anginal",
    "asymptomatic"
])

trestbps = st.number_input(
    "Resting Blood Pressure",
    80,
    250,
    120
)

chol = st.number_input(
    "Cholesterol",
    100,
    600,
    200
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    [True, False]
)

restecg = st.selectbox(
    "Resting ECG",
    [
        "normal",
        "st-t abnormality",
        "lv hypertrophy"
    ]
)

thalch = st.number_input(
    "Maximum Heart Rate",
    50,
    250,
    150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [True, False]
)

oldpeak = st.number_input(
    "Old Peak",
    0.0,
    10.0,
    1.0
)

slope = st.selectbox(
    "Slope",
    [
        "upsloping",
        "flat",
        "downsloping"
    ]
)

ca = st.number_input(
    "Major Vessels (CA)",
    0,
    4,
    0
)

thal = st.selectbox(
    "Thal",
    [
        "normal",
        "fixed defect",
        "reversable defect"
    ]
)

# -------------------------
# Prediction
# -------------------------

if st.button("❤️ Predict Heart Disease"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalch": [thalch],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"Probability of Heart Disease: **{probability[0][1]*100:.2f}%**")