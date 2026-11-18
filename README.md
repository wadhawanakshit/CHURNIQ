# CHURNIQ — Customer Churn Prediction & Retention Intelligence

An end-to-end machine learning project that predicts customer churn, identifies high-risk customers, and provides actionable insights through an interactive analytics dashboard.

**Live Dashboard:** [https://wadhawanakshit-churniq-appdashboard-ts9usz.streamlit.app/](https://wadhawanakshit-churniq-appdashboard-ts9usz.streamlit.app/)

## Project Overview

Customer churn is a major challenge for subscription-based businesses. CHURNIQ uses historical customer data to predict which customers are likely to discontinue a service and helps businesses understand the factors associated with churn.

The project combines exploratory data analysis, machine learning, model evaluation, explainable AI, and an interactive Streamlit dashboard.

## Key Features

- Customer churn analysis and interactive visualisations
- Customer-level churn probability prediction
- Batch prediction through CSV uploads
- Machine learning model comparison
- SHAP-based model explainability
- Downloadable prediction results

## Technology Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Deployment | Streamlit |
| Version Control | Git, GitHub |

## Dataset

**IBM Telco Customer Churn Dataset**

- 7,043 customer records
- 21 original columns
- Customer demographics, account details, subscribed services, and churn status

The dataset is used for exploratory analysis, model training, and evaluation.

## Machine Learning Workflow

1. Data loading and preprocessing
2. Exploratory data analysis
3. Feature engineering and encoding
4. Training and comparison of classification models
5. Model selection using validation performance
6. Decision-threshold tuning
7. Final test-set evaluation
8. SHAP explainability
9. Model serialization and deployment

## Project Structure

```text
CHURNIQ/
├── app/
│   └── dashboard.py
├── models/
│   └── churniq_model_bundle.joblib
├── notebook/
│   └── ChurnIQ_ML_Analysis.ipynb
├── reports/
│   ├── churniq_eda_insights.csv
│   ├── churniq_final_metrics.csv
│   ├── churniq_model_comparison.csv
│   └── churniq_shap_feature_importance.csv
├── assets/
│   ├── churniq_global_feature_importance.png
│   ├── churniq_shap_beeswarm.png
│   └── churniq_individual_explanation.png
├── requirements.txt
└── README.md
```

## Explainable AI

SHAP (SHapley Additive exPlanations) is used to understand how individual features influence model predictions.

The project includes:
- Global feature importance
- SHAP summary visualisation
- Individual customer-level explanations

## Running Locally

Clone the repository:

```bash
git clone https://github.com/wadhawanakshit/CHURNIQ.git
cd CHURNIQ
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app/dashboard.py
```

## Author

**Akshit Wadhawan**  
B.Tech — Artificial Intelligence & Data Science

[GitHub](https://github.com/wadhawanakshit)

---

*Built as an end-to-end machine learning and customer analytics portfolio project.*
