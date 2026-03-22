import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts
model = joblib.load('loan_default_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')  # e.g., ['income', 'loan_amount', 'income_to_loan', 'loan_to_income', 'emp_Unemployed']

st.title("Loan Default Predictor")
st.write("Enter the applicant details:")

# Input fields
income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0)
loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, value=15000.0)
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed"])   # only two options

# Build raw input DataFrame
input_df = pd.DataFrame({
    'income': [income],
    'loan_amount': [loan_amount],
    'employment_status': [employment_status]
})

# Feature engineering (same as training)
input_df['income_to_loan'] = input_df['income'] / input_df['loan_amount']
input_df['loan_to_income'] = input_df['loan_amount'] / input_df['income']

# Create dummy column for Unemployed (1 if Unemployed, 0 otherwise)
input_df['emp_Unemployed'] = (input_df['employment_status'] == 'Unemployed').astype(int)

# Scale numerical features
numerical_cols = ['income', 'loan_amount', 'income_to_loan', 'loan_to_income']
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Ensure column order matches training
input_df = input_df[feature_columns]

# Predict
if st.button("Predict Default Probability"):
    pred = model.predict(input_df)[0]
    prob = np.clip(pred, 0, 1)   # clamp to [0,1]
    st.metric("Estimated Default Probability", f"{prob:.2%}")
    if prob > 0.5:
        st.error("High risk: Likely to default")
    else:
        st.success("Low risk: Unlikely to default")