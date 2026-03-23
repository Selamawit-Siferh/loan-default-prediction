# Loan Default Prediction

Predict the probability of loan default using machine learning. This project builds a **Logistic Regression** model to estimate the likelihood that an applicant will default on a loan.

## Live Demo


[🔗 Streamlit App](https://loan-default-prediction-selamawit-siferh.streamlit.app)
## Overview

Banks and financial institutions need to assess credit risk to make informed lending decisions. This project builds a predictive model that estimates the probability of default based on applicant features such as income, loan amount, and employment status.

## Problem Definition

- **Problem:** Predict whether a loan applicant will default (1) or not default (0)
- **Goal:** Binary classification
- **Success Criteria:**
  - Accuracy >80%
  - Precision >75%
  - Recall >70%
  - F1-Score >72%

## Dataset

The dataset is a synthetic dataset containing loan application information with the following features:

| Feature             | Description                                   |
| ------------------- | --------------------------------------------- |
| `income`            | Monthly income of the applicant               |
| `loan_amount`       | Requested loan amount                         |
| `employment_status` | Employment status (Employed, Unemployed)      |
| `default`           | Target variable (1 = default, 0 = no default) |

## Methodology

### 1. Data Cleaning

- Checked and removed missing values
- Removed duplicate rows
- Capped outliers using IQR (Interquartile Range) method

### 2. Feature Engineering

- Created ratio features:
  - `income_to_loan` = income / loan_amount
  - `loan_to_income` = loan_amount / income
- One-hot encoded `employment_status` with `drop_first=True` (created `emp_Unemployed`)
- Standardized numerical features using `StandardScaler`

### 3. Data Splitting

- Training set: 60%
- Validation set: 20%
- Test set: 20%
- Stratified split to maintain class distribution

### 4. Model Selection

- **Algorithm:** Logistic Regression (Supervised Classification)
- **Rationale:**
  - Low complexity and high interpretability
  - Outputs probabilities as risk scores
  - Suitable for binary classification problems
  - Low computational cost for production deployment

### 5. Model Training & Tuning

- Baseline model with default hyperparameters (C=1.0, solver='lbfgs')
- Hyperparameter tuning using **Grid Search** with 5-fold cross-validation
- Tuned parameters: `C` (regularization strength) and `solver`
- Best model selected based on ROC-AUC score

### 6. Model Evaluation

- **Test Set Performance:**
  - Accuracy: 72.0%
  - Precision: 70.5%
  - Recall: 71.4%
  - F1-Score: 71.0%
  - ROC-AUC: 72.3%

## Files in Repository

| File                     | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `app.py`                 | Streamlit web application for interactive predictions    |
| `loan_default_model.pkl` | Trained Logistic Regression model                        |
| `scaler.pkl`             | Fitted StandardScaler for numerical features             |
| `feature_columns.pkl`    | List of feature names in the order expected by the model |
| `requirements.txt`       | Python dependencies for the app                          |
| `Classification.ipynb`   | Jupyter notebook with complete analysis and training     |
| `README.md`              | Project documentation                                    |

## Installation & Local Usage

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/Selamawit-Siferh/loan-default-prediction.git
cd loan-default-prediction
```
