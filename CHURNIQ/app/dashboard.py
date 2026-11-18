
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CHURNIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# DESIGN SYSTEM
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0B1020;
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background: #11182B;
    border-right: 1px solid #26334D;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

p, label, span {
    color: #CBD5E1;
}

.metric-card {
    background: linear-gradient(135deg, #17243B, #111A2D);
    border: 1px solid #293B58;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #94A3B8;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: #F8FAFC;
    font-size: 29px;
    font-weight: 800;
}

.metric-note {
    color: #2DD4BF;
    font-size: 12px;
    margin-top: 5px;
}

.hero {
    background: linear-gradient(110deg, #172B48, #17213A, #28204B);
    border: 1px solid #344767;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    color: #B7C7DE;
    font-size: 14px;
}

div[data-testid="stForm"] {
    background: #111A2D;
    border: 1px solid #293B58;
    padding: 20px;
    border-radius: 14px;
}

.stButton > button {
    background: #14B8A6;
    color: #071521;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}

.stButton > button:hover {
    background: #2DD4BF;
    color: #071521;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293B58;
    border-radius: 10px;
}

hr {
    border-color: #26334D;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL ENGINE
# --------------------------------------------------

MODEL_PATH = "models/churniq_model_bundle.joblib"

@st.cache_resource
def load_engine():
    return joblib.load(MODEL_PATH)

try:
    engine = load_engine()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

FEATURES = engine["feature_names"]
MODEL = engine["model"]
PREPROCESSOR = engine["preprocessor"]
THRESHOLD = engine["threshold"]

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_customers(data):
    data = data[FEATURES].copy()

    processed = PREPROCESSOR.transform(data)

    probabilities = MODEL.predict_proba(processed)[:, 1]

    predictions = (probabilities >= THRESHOLD).astype(int)

    results = pd.DataFrame({
        "Churn Probability": probabilities,
        "Prediction": [
            "Likely to Churn" if p else "Likely to Stay"
            for p in predictions
        ],
        "Risk Classification": [
            "HIGH RISK" if p else "BELOW THRESHOLD"
            for p in predictions
        ]
    })

    return results

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("# CHURNIQ")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Customer Prediction",
            "Batch Analysis",
            "About the Model"
        ]
    )

    st.markdown("---")
    st.markdown("**MODEL STATUS**")
    st.success("Prediction engine loaded")
    st.caption(f"Model: {engine['model_name']}")
    st.caption(f"Decision threshold: {THRESHOLD:.1%}")

    st.markdown("---")
    st.caption("CHURNIQ • Customer Retention Intelligence")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-title">Customer Intelligence, Simplified.</div>
    <div class="hero-subtitle">
        Predict churn. Identify high-risk customers.
        Make smarter retention decisions.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.subheader("Business Overview")
    st.caption("Explore the IBM Telco customer dataset.")

    @st.cache_data
    def load_dataset():
        url = (
            "https://raw.githubusercontent.com/IBM/"
            "telco-customer-churn-on-icp4d/master/"
            "data/Telco-Customer-Churn.csv"
        )
        data = pd.read_csv(url)
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"], errors="coerce"
        )
        return data

    try:
        data = load_dataset()
    except Exception:
        st.warning(
            "The sample dataset could not be loaded. "
            "Check your internet connection."
        )
        st.stop()

    total_customers = len(data)
    churn_count = (data["Churn"] == "Yes").sum()
    churn_rate = churn_count / total_customers * 100
    avg_monthly = data["MonthlyCharges"].mean()
    avg_tenure = data["tenure"].mean()

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "TOTAL CUSTOMERS", f"{total_customers:,}", "Dataset records"),
        (c2, "CHURNED CUSTOMERS", f"{churn_count:,}", "Customers lost"),
        (c3, "CHURN RATE", f"{churn_rate:.1f}%", "Historical churn"),
        (c4, "AVG. MONTHLY CHARGE", f"${avg_monthly:.2f}", "Per customer")
    ]

    for col, label, value, note in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns(2)

    with left:
        st.markdown("### Customer Churn Distribution")
        churn_chart = data["Churn"].value_counts().reset_index()
        churn_chart.columns = ["Status", "Customers"]

        fig = px.pie(
            churn_chart,
            names="Status",
            values="Customers",
            hole=0.68,
            color="Status",
            color_discrete_map={
                "Yes": "#FB7185",
                "No": "#2DD4BF"
            }
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=10, r=10),
            legend_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Churn by Contract Type")

        contract_data = (
            data.groupby("Contract", observed=True)["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Churn Rate",
            color="Contract",
            text_auto=".1f",
            color_discrete_sequence=["#2DD4BF", "#818CF8", "#FB7185"]
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis_title="Churn rate (%)",
            xaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Customer Tenure Distribution")

    fig = px.histogram(
        data,
        x="tenure",
        color="Churn",
        nbins=30,
        barmode="overlay",
        opacity=0.8,
        color_discrete_map={
            "Yes": "#FB7185",
            "No": "#2DD4BF"
        }
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Tenure (months)",
        yaxis_title="Number of customers"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "These overview metrics describe the historical dataset. "
        "They are not live company customer records."
    )

# --------------------------------------------------
# SINGLE CUSTOMER PREDICTION
# --------------------------------------------------

elif page == "Customer Prediction":

    st.subheader("Individual Customer Risk Assessment")
    st.caption(
        "Enter customer details to estimate their probability of churn."
    )

    with st.form("customer_form"):

        st.markdown("### Personal Information")
        a, b, c = st.columns(3)

        with a:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with b:
            senior = st.selectbox("Senior Citizen", [0, 1])
        with c:
            partner = st.selectbox("Has Partner", ["Yes", "No"])

        a, b, c = st.columns(3)

        with a:
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        with b:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with c:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        st.markdown("### Services")
        a, b, c = st.columns(3)

        with a:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
        with b:
            multiple = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
        with c:
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )

        a, b, c = st.columns(3)

        with a:
            security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
        with b:
            protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
        with c:
            tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )

        st.markdown("### Billing & Contract")
        a, b, c = st.columns(3)

        with a:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
        with b:
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
        with c:
            monthly = st.number_input(
                "Monthly Charges ($)", 0.0, 200.0, 70.0
            )

        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=float(monthly * tenure),
            step=10.0
        )

        submitted = st.form_submit_button(
            "Analyse Customer Risk",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        try:
            result = predict_customers(customer).iloc[0]
            probability = float(result["Churn Probability"])

            st.markdown("---")
            st.markdown("### Prediction Results")

            a, b = st.columns(2)

            with a:
                st.metric(
                    "Churn Probability",
                    f"{probability:.1%}"
                )
                st.progress(probability)

            with b:
                st.metric(
                    "Model Decision",
                    result["Prediction"]
                )

            if probability >= THRESHOLD:
                st.error(
                    "HIGH RISK: This customer meets the model's "
                    "churn decision threshold."
                )
                st.markdown(
                    "**Suggested action:** Review customer concerns "
                    "and consider a personalised retention offer."
                )
            else:
                st.success(
                    "BELOW THRESHOLD: This customer is not flagged "
                    "by the current model decision threshold."
                )

            st.caption(
                "Predictions are estimates, not guarantees. "
                "Retention actions should also consider business context."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# --------------------------------------------------
# BATCH ANALYSIS
# --------------------------------------------------

elif page == "Batch Analysis":

    st.subheader("Bulk Customer Analysis")
    st.caption(
        "Upload a CSV containing customer records for batch predictions."
    )

    st.info(
        "Your CSV must contain the same 19 customer input columns "
        "used to train the model. A customerID column is optional."
    )

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            batch_data = pd.read_csv(uploaded_file)

            if "customerID" in batch_data.columns:
                customer_ids = batch_data["customerID"]
            else:
                customer_ids = pd.Series(
                    [f"Customer {i+1}" for i in range(len(batch_data))]
                )

            missing = [
                col for col in FEATURES
                if col not in batch_data.columns
            ]

            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                batch_results = predict_customers(batch_data)

                report = batch_results.copy()
                report.insert(0, "Customer ID", customer_ids.values)

                st.markdown("### Analysis Summary")

                a, b, c = st.columns(3)

                a.metric("Customers Analysed", len(report))
                high_risk = (
                    report["Risk Classification"] == "HIGH RISK"
                ).sum()
                b.metric("High-Risk Customers", int(high_risk))
                c.metric(
                    "High-Risk Share",
                    f"{high_risk / len(report):.1%}" if len(report) else "0%"
                )

                st.markdown("### Prediction Report")
                st.dataframe(report, use_container_width=True)

                csv = report.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Prediction Report",
                    data=csv,
                    file_name="churniq_prediction_report.csv",
                    mime="text/csv"
                )

        except Exception as error:
            st.error(f"Could not process the file: {error}")

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

elif page == "About the Model":

    st.subheader("Model Information")

    a, b, c = st.columns(3)

    a.metric("Selected Algorithm", engine["model_name"])
    b.metric("Decision Threshold", f"{THRESHOLD:.2%}")
    c.metric("Input Features", len(FEATURES))

    st.markdown("### Technology Stack")

    st.write("""
    - **Python:** Core programming language
    - **Pandas & NumPy:** Data manipulation
    - **Scikit-learn:** Machine learning and preprocessing
    - **SHAP:** Model explainability (developed in the notebook)
    - **Plotly:** Interactive visualisations
    - **Streamlit:** Interactive web application
    """)

    st.markdown("### Prediction Workflow")

    st.write("""
    1. Customer information is received.
    2. The saved preprocessing pipeline transforms the data.
    3. The trained model estimates churn probability.
    4. The validation-selected threshold determines the model decision.
    5. The dashboard displays the result.
    """)

    st.markdown("### Model Input Features")
    st.dataframe(
        pd.DataFrame({"Feature": FEATURES}),
        use_container_width=True
    )

    st.warning(
        "The model's predictions are probabilistic estimates. "
        "They should support, not replace, business decisions."
    )
