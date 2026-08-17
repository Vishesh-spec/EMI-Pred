import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💳",
    layout="wide",
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "EMI_dataset_sample.csv")


@st.cache_data
def load_meta():
    with open(os.path.join(MODELS_DIR, "classifier_meta.json")) as f:
        clf_meta = json.load(f)
    with open(os.path.join(MODELS_DIR, "regressor_meta.json")) as f:
        reg_meta = json.load(f)
    return clf_meta, reg_meta


@st.cache_data
def load_sample():
    return pd.read_csv(DATA_PATH)


st.title("💳 EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")

st.markdown(
    """
Welcome! **EMIPredict AI** helps financial institutions, FinTech platforms and loan
officers instantly assess a customer's **EMI eligibility** and recommend the
**maximum safe monthly EMI amount** — powered by machine learning models trained
on 400,000 real-style lending records and tracked end-to-end with **MLflow**.

👈 Use the sidebar to navigate:
- **Predict** — run a real-time eligibility + EMI check for a customer
- **Data Explorer** — explore the training dataset
- **Model Performance** — compare all 6 trained models & see why the best ones were selected
- **Admin Data Manager** — full CRUD (create/read/update/delete) for financial records
"""
)

try:
    clf_meta, reg_meta = load_meta()
    df = load_sample()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Records", "400,000")
    col2.metric("Best Classifier", clf_meta["best_model"],
                f"{clf_meta['results'][clf_meta['best_model']]['accuracy']*100:.1f}% accuracy")
    col3.metric("Best Regressor", reg_meta["best_model"],
                f"RMSE ₹{reg_meta['results'][reg_meta['best_model']]['rmse']:.0f}")
    col4.metric("EMI Scenarios Covered", df["emi_scenario"].nunique())

    st.divider()
    st.markdown("### Business Use Cases")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**Financial Institutions & Banks**
- Automate loan approval & cut manual underwriting time
- Risk-based pricing across EMI scenarios
- Real-time eligibility checks for walk-in customers
"""
        )
    with c2:
        st.markdown(
            """
**FinTech Companies & Loan Officers**
- Instant pre-qualification inside digital lending apps
- AI-backed loan-amount recommendations
- Portfolio-level risk & default monitoring
"""
        )
except FileNotFoundError:
    st.warning("Model artifacts not found yet — run `scripts/train_models.py` (or the stage scripts) first.")

st.divider()
st.caption("Built with Python, scikit-learn, XGBoost, MLflow & Streamlit · EMIPredict AI")
