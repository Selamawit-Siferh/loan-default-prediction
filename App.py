# app.py - Loan Default Prediction Web App
# --------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="centered"
)

# Load saved artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('loan_default_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.info("Please ensure 'loan_default_model.pkl', 'scaler.pkl', and 'feature_columns.pkl' are in the same directory.")
    st.stop()

# App title and description
st.title("Loan Default Predictor")
st.markdown("Enter applicant details to predict default probability")

st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    income = st.text_input(
        "Monthly Income",
        placeholder="Enter monthly income",
        help="Enter the applicant's monthly income"
    )

with col2:
    loan_amount = st.text_input(
        "Loan Amount",
        placeholder="Enter loan amount",
        help="Enter the requested loan amount"
    )

# Employment Status with placeholder
employment_options = ["Select status", "Employed", "Unemployed"]
employment_status = st.selectbox(
    "Employment Status",
    employment_options,
    index=0,
    help="Select the applicant's current employment status"
)

st.divider()

# Check if all inputs are provided
if not income:
    st.warning("Please enter a valid Monthly Income (greater than 0)")
    st.stop()

if not loan_amount:
    st.warning("Please enter a valid Loan Amount (greater than 0)")
    st.stop()

if employment_status == "Select status":
    st.warning("Please select an Employment Status")
    st.stop()

# Convert to float
try:
    income_val = float(income)
    loan_val = float(loan_amount)
except ValueError:
    st.warning("Please enter numeric values for Income and Loan Amount")
    st.stop()

if income_val <= 0:
    st.warning("Please enter a valid Monthly Income (greater than 0)")
    st.stop()

if loan_val <= 0:
    st.warning("Please enter a valid Loan Amount (greater than 0)")
    st.stop()

# Build input DataFrame
input_df = pd.DataFrame({
    'income': [income_val],
    'loan_amount': [loan_val],
    'employment_status': [employment_status]
})

# Feature engineering (same as training)
input_df['income_to_loan'] = input_df['income'] / input_df['loan_amount']
input_df['loan_to_income'] = input_df['loan_amount'] / input_df['income']

# Create dummy column for Unemployed (1 if Unemployed, 0 otherwise)
input_df['emp_Unemployed'] = (input_df['employment_status'] == 'Unemployed').astype(int)

# Scale numerical features
numerical_cols = ['income', 'loan_amount', 'income_to_loan', 'loan_to_income']
try:
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
except Exception as e:
    st.error(f"Error scaling features: {e}")
    st.stop()

# Ensure column order matches training
try:
    input_df = input_df[feature_columns]
except KeyError as e:
    st.error(f"Missing columns: {e}")
    st.stop()

# Predict button
if st.button("Predict Default Probability", type="primary", use_container_width=True):
    with st.spinner("Calculating prediction..."):
        prob = model.predict_proba(input_df)[0][1]
        
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric("Estimated Default Probability", f"{prob:.2%}")
        
        st.progress(prob)
        
        if prob > 0.5:
            st.error("HIGH RISK: Applicant is likely to default")
            st.warning("Recommendation: Review application carefully or consider declining.")
        else:
            st.success("LOW RISK: Applicant is unlikely to default")
            st.info("Recommendation: Application appears favorable.")
        
        st.markdown("### Risk Meter")
        if prob < 0.3:
            st.markdown("Low Risk Favorable")
        elif prob < 0.7:
            st.markdown("Medium Risk Needs Review")
        else:
            st.markdown("High Risk Caution Advised")
        
        with st.expander("View Input Summary"):
            st.write(f"Monthly Income: {income_val:.2f}")
            st.write(f"Loan Amount: {loan_val:.2f}")
            st.write(f"Loan to Income Ratio: {loan_val/income_val:.2f}")
            st.write(f"Employment Status: {employment_status}")

st.divider()
st.caption("""
Disclaimer: This is a predictive model. Final lending decisions should consider additional factors 
and be made by qualified professionals.
""")