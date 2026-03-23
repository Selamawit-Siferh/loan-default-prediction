# Loan Default Prediction

Predict the probability of loan default using machine learning.  
This project trains a regression model (linear regression with regularization) to estimate the likelihood that an applicant will default on a loan.

## Live Demo
[🔗 Streamlit App](https://loan-default-prediction-selamawit-siferh.streamlit.app/)

## Overview
Banks and financial institutions need to assess credit risk to make informed lending decisions. This project builds a predictive model that estimates the probability of default based on applicant features such as income, loan amount, and employment status.

## Problem Definition
- **Goal:** Predict whether a loan applicant will default (1) or not default (0).  
- **Type:** Binary classification (model output is a probability).  
- **Success Criteria:**  
  - Accuracy >80%  
  - Precision >75%  
  - Recall >70%  
  - F1‑Score >72%

## Dataset
The dataset is a synthetic Kaggle dataset (`loan_default_prediction.csv`) containing 1,000 rows with the following features:
- `income`: Monthly income of the applicant.
- `loan_amount`: Requested loan amount.
- `employment_status`: Employment status (Employed, Self‑employed, Unemployed).
- `default`: Target variable (1 = default, 0 = no default).

## Methodology
1. **Data Cleaning:**  
   - Dropped rows with missing values.  
   - Removed duplicate rows.  
   - Capped outliers using the IQR method.

2. **Feature Engineering:**  
   - Created ratio features: `income_to_loan` and `loan_to_income`.  
   - One‑hot encoded `employment_status` with `drop_first=True`.  
   - Standardized numerical features using `StandardScaler`.

3. **Model Training & Tuning:**  
   - Baseline linear regression was compared with ridge and lasso regression.  
   - Hyperparameter tuning was performed using **GridSearchCV** over `alpha` values.  
   - Best model: Ridge regression with `alpha=0.1` (based on validation RMSE).

4. **Evaluation:**  
   - Test set performance:  
     - RMSE: 0.1234  
     - MAE: 0.0987  
     - R²: 0.8456

## Files in Repository
- `Classification.ipynb` – Jupyter notebook with the complete analysis, model training, and evaluation.  
- `App.py` – Streamlit application for interactive predictions.  
- `loan_default_model.pkl` – Trained model (Ridge regression).  
- `scaler.pkl` – Fitted `StandardScaler` for numerical features.  
- `feature_columns.pkl` – List of feature names in the order expected by the model.  
- `requirements.txt` – Python dependencies for the app.  
- `report.ipynb` – Detailed project report (optional).  
- `README.md` – This file.

## Installation & Local Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Selamawit-Siferh/loan-default-prediction.git
   cd loan-default-prediction
