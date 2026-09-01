import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


st.set_page_config(
    page_title="Banking Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
    }

    .risk-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .small-text {
        font-size: 14px;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_models():

    credit_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "credit_risk_model.pkl"
        )
    )

    credit_preprocessor = joblib.load(
        os.path.join(
            MODEL_DIR,
            "credit_preprocessor.pkl"
        )
    )

    credit_threshold = float(
        joblib.load(
            os.path.join(
                MODEL_DIR,
                "credit_risk_threshold.pkl"
            )
        )
    )

    fraud_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "fraud_detection_model.pkl"
        )
    )

    fraud_scaler = joblib.load(
        os.path.join(
            MODEL_DIR,
            "fraud_scaler.pkl"
        )
    )

    fraud_threshold = float(
        joblib.load(
            os.path.join(
                MODEL_DIR,
                "fraud_threshold.pkl"
            )
        )
    )

    return (
        credit_model,
        credit_preprocessor,
        credit_threshold,
        fraud_model,
        fraud_scaler,
        fraud_threshold
    )


(
    credit_model,
    credit_preprocessor,
    credit_threshold,
    fraud_model,
    fraud_scaler,
    fraud_threshold
) = load_models()


# ============================================================
# CREDIT RISK PREDICTION
# ============================================================

def predict_credit_risk(
    person_age,
    person_income,
    person_emp_length,
    person_home_ownership,
    loan_intent,
    loan_grade,
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    cb_person_default_on_file,
    cb_person_cred_hist_length
):

    person_income_log = np.log1p(
        person_income
    )

    employment_age_ratio = (
        person_emp_length / person_age
        if person_age > 0
        else 0
    )

    credit_history_age_ratio = (
        cb_person_cred_hist_length / person_age
        if person_age > 0
        else 0
    )

    input_data = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_income_log": person_income_log,
        "person_emp_length": person_emp_length,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "employment_age_ratio": employment_age_ratio,
        "credit_history_age_ratio": credit_history_age_ratio
    }])

    processed_input = credit_preprocessor.transform(
        input_data
    )

    probability = credit_model.predict_proba(
        processed_input
    )[0, 1]

    prediction = int(
        probability >= credit_threshold
    )

    return prediction, probability


# ============================================================
# FRAUD DETECTION
# ============================================================

def predict_fraud(
    time_value,
    v_values,
    amount
):

    if len(v_values) != 28:
        raise ValueError(
            "Exactly 28 V features are required."
        )

    feature_names = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount"
    ]

    transaction = pd.DataFrame(
        [[
            time_value,
            *v_values,
            amount
        ]],
        columns=feature_names
    )

    scaled_time_amount = fraud_scaler.transform(
        transaction[
            ["Time", "Amount"]
        ]
    )

    transaction["Time"] = (
        scaled_time_amount[:, 0]
    )

    transaction["Amount"] = (
        scaled_time_amount[:, 1]
    )

    transaction = transaction[
        fraud_model.feature_names_in_
    ]

    probability = fraud_model.predict_proba(
        transaction
    )[0, 1]

    prediction = int(
        probability >= fraud_threshold
    )

    return prediction, probability


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏦 Banking Risk Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    End-to-end machine learning platform for
    <b>Credit Risk Assessment</b> and
    <b>Fraud Detection</b>.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Overview",
        "💳 Credit Risk",
        "🚨 Fraud Detection"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning Decision Support"
)

st.sidebar.caption(
    "Banking Credit Risk & Fraud Analytics"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This application demonstrates an end-to-end banking
        analytics workflow combining machine learning,
        feature engineering, threshold optimization and
        business-oriented risk assessment.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Credit Risk Model",
            "Active"
        )

        st.caption(
            "Loan default probability assessment"
        )

    with col2:

        st.metric(
            "Fraud Detection",
            "Active"
        )

        st.caption(
            "Transaction fraud probability assessment"
        )

    with col3:

        st.metric(
            "ML Models",
            "2"
        )

        st.caption(
            "Credit Risk + Fraud Detection"
        )

    st.divider()

    st.subheader("Analytics Workflow")

    st.code(
        """
Data Understanding
        ↓
Data Cleaning
        ↓
EDA
        ↓
Statistical Analysis
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Model Evaluation
        ↓
Threshold Optimization
        ↓
Business Risk Assessment
        """,
        language="text"
    )

    st.info(
        "Use the sidebar to access the Credit Risk "
        "and Fraud Detection modules."
    )


# ============================================================
# CREDIT RISK
# ============================================================

elif page == "💳 Credit Risk":

    st.markdown(
        '<div class="section-title">💳 Credit Risk Assessment</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Estimate the probability that a loan applicant "
        "will default."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Applicant")

        person_age = st.number_input(
            "Applicant Age",
            min_value=18,
            max_value=100,
            value=30
        )

        person_income = st.number_input(
            "Annual Income",
            min_value=0,
            value=60000,
            step=1000
        )

        person_emp_length = st.number_input(
            "Employment Length (Years)",
            min_value=0.0,
            max_value=60.0,
            value=5.0,
            step=0.5
        )

        person_home_ownership = st.selectbox(
            "Home Ownership",
            [
                "RENT",
                "OWN",
                "MORTGAGE",
                "OTHER"
            ]
        )

    with col2:

        st.subheader("Loan")

        loan_intent = st.selectbox(
            "Loan Intent",
            [
                "DEBTCONSOLIDATION",
                "EDUCATION",
                "HOMEIMPROVEMENT",
                "MEDICAL",
                "PERSONAL",
                "VENTURE"
            ]
        )

        loan_grade = st.selectbox(
            "Loan Grade",
            [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G"
            ]
        )

        loan_amnt = st.number_input(
            "Loan Amount",
            min_value=0,
            value=10000,
            step=500
        )

        loan_int_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=10.5,
            step=0.1
        )

    with col3:

        st.subheader("Credit History")

        loan_percent_income = st.number_input(
            "Loan Percent of Income",
            min_value=0.0,
            max_value=1.0,
            value=0.17,
            step=0.01
        )

        cb_person_default_on_file = st.selectbox(
            "Previous Default on File",
            ["N", "Y"]
        )

        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (Years)",
            min_value=0,
            max_value=50,
            value=8
        )

    st.divider()

    if st.button(
        "🔍 Assess Credit Risk",
        type="primary",
        use_container_width=True
    ):

        if loan_amnt > person_income:
            st.warning(
                "Loan amount is greater than annual income. "
                "Please verify the inputs."
            )

        prediction, probability = predict_credit_risk(
            person_age,
            person_income,
            person_emp_length,
            person_home_ownership,
            loan_intent,
            loan_grade,
            loan_amnt,
            loan_int_rate,
            loan_percent_income,
            cb_person_default_on_file,
            cb_person_cred_hist_length
        )

        st.divider()

        st.subheader("Risk Assessment")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Default Probability",
                f"{probability:.2%}"
            )

        with col2:

            st.metric(
                "Model Threshold",
                f"{credit_threshold:.2%}"
            )

        with col3:

            if prediction == 1:
                st.error("⚠️ Higher Default Risk")
            else:
                st.success("✅ Lower Default Risk")

        st.progress(
            min(
                max(
                    float(probability),
                    0.0
                ),
                1.0
            )
        )

        if prediction == 1:

            st.warning(
                "The predicted probability is above the "
                "model's optimized decision threshold."
            )

        else:

            st.info(
                "The predicted probability is below the "
                "model's optimized decision threshold."
            )

        with st.expander(
            "View calculated model features"
        ):

            st.write(
                {
                    "person_income_log":
                        float(
                            np.log1p(person_income)
                        ),
                    "employment_age_ratio":
                        float(
                            person_emp_length /
                            person_age
                        ),
                    "credit_history_age_ratio":
                        float(
                            cb_person_cred_hist_length /
                            person_age
                        )
                }
            )


# ============================================================
# FRAUD DETECTION
# ============================================================

elif page == "🚨 Fraud Detection":

    st.markdown(
        '<div class="section-title">🚨 Fraud Detection</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Estimate the probability that a transaction "
        "is fraudulent."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        time_value = st.number_input(
            "Transaction Time",
            min_value=0.0,
            value=0.0
        )

    with col2:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

    st.subheader(
        "Transaction Features"
    )

    st.caption(
        "Enter the 28 PCA-transformed transaction features."
    )

    v_values = []

    columns = st.columns(4)

    for i in range(1, 29):

        with columns[(i - 1) % 4]:

            value = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f",
                key=f"v_{i}"
            )

            v_values.append(value)

    st.divider()

    if st.button(
        "🔍 Detect Fraud",
        type="primary",
        use_container_width=True
    ):

        prediction, probability = predict_fraud(
            time_value,
            v_values,
            amount
        )

        st.subheader("Fraud Assessment")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

        with col2:

            st.metric(
                "Model Threshold",
                f"{fraud_threshold:.2%}"
            )

        with col3:

            if prediction == 1:
                st.error("🚨 Potential Fraud")
            else:
                st.success("✅ Likely Legitimate")

        st.progress(
            min(
                max(
                    float(probability),
                    0.0
                ),
                1.0
            )
        )

        if prediction == 1:

            st.warning(
                "The predicted probability is above the "
                "optimized fraud decision threshold. "
                "Further investigation may be appropriate."
            )

        else:

            st.info(
                "The predicted probability is below the "
                "optimized fraud decision threshold."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Banking Credit Risk & Fraud Analytics | "
    "Machine Learning Decision Support"
)