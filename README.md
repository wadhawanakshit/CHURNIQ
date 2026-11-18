# CHURNIQ | Customer Churn Prediction & Retention Intelligence

CHURNIQ is a machine-learning-powered customer analytics platform
designed to estimate customer churn probability and identify
customers who may require retention attention.

## Features

- Customer churn exploratory data analysis
- Customer segmentation and interactive visualisations
- Machine learning model comparison
- Validation-based decision threshold selection
- SHAP-based model explainability
- Individual customer churn prediction
- Bulk customer prediction through CSV upload
- Interactive Streamlit dashboard

## Technology Stack

Python, Pandas, NumPy, Scikit-learn, SHAP, Plotly, Streamlit and Joblib.

## Project Structure

- app/: Streamlit dashboard
- models/: Saved model and preprocessing bundle
- notebooks/: Model development notebook
- reports/: Model evaluation and analysis reports
- assets/: Explainability visualisations

## Run Locally

1. Clone this repository.
2. Install dependencies:

   pip install -r requirements.txt

3. Start the dashboard from the project root:

   streamlit run app/dashboard.py

## Dataset

IBM Telco Customer Churn dataset.

## Important Note

This project is an analytical demonstration using a public dataset.
Predictions are estimates and should support, not replace,
business decisions.

## Author

Akshit Wadhawan
