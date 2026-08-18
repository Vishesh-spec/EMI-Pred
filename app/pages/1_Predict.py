import json
import os

import joblib
import pandas as pd
import streamlit as st

from feature_engineering import build_model_matrix


st.set_page_config(page_title="Predict — EMIPredict AI", page_icon="🔮", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

EDUCATION_LEVELS = ["High School", "Graduate", "Post Graduate", "Professional"]
EMPLOYMENT_TYPES = ["Private", "Government", "Self-employed"]
COMPANY_TYPES = ["Startup", "SME", "MNC", "Public Sector", "Not Applicable"]
HOUSE_TYPES = ["Rented", "Own", "Family"]
SCENARIOS = {
    "E-commerce Shopping EMI": dict(amount=(10_000, 200_000), tenure=(3, 24)),
    "Home Appliances EMI":     dict(amount=(20_000, 300_000), tenure=(6, 36)),
    "Vehicle EMI":             dict(amount=(80_000, 1_500_000), tenure=(12, 84)),
    "Personal Loan EMI":       dict(amount=(50_000, 1_000_000), tenure=(12, 60)),
    "Education EMI":           dict(amount=(50_000, 500_000), tenure=(6, 48)),
}


@st.cache_resource
def load_artifacts():
    clf = joblib.load(f"{MODELS_DIR}/best_classifier.pkl")
    clf_scaler = joblib.load(f"{MODELS_DIR}/classifier_scaler.pkl")
    clf_encoders = joblib.load(f"{MODELS_DIR}/classifier_encoders.pkl")
    label_encoder = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")
    reg = joblib.load(f"{MODELS_DIR}/best_regressor.pkl")
    reg_scaler = joblib.load(f"{MODELS_DIR}/regressor_scaler.pkl")
    reg_encoders = joblib.load(f"{MODELS_DIR}/regressor_encoders.pkl")
    with open(f"{MODELS_DIR}/classifier_meta.json") as f:
        clf_meta = json.load(f)
    with open(f"{MODELS_DIR}/regressor_meta.json") as f:
        reg_meta = json.load(f)
    return clf, clf_scaler, clf_encoders, label_encoder, reg, reg_scaler, reg_encoders, clf_meta, reg_meta


st.title("🔮 Real-Time EMI Prediction")
st.caption("Fill in a customer's financial profile to get an instant eligibility decision and recommended EMI.")

try:
    (clf, clf_scaler, clf_encoders, label_encoder,
     reg, reg_scaler, reg_encoders, clf_meta, reg_meta) = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run the training pipeline first (see README).")
    st.stop()

with st.form("customer_form"):
    st.markdown("#### Personal Demographics")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", 25, 60, 35)
    gender = c2.selectbox("Gender", ["Male", "Female"])
    marital_status = c3.selectbox("Marital Status", ["Single", "Married"])
    education = c4.selectbox("Education", EDUCATION_LEVELS, index=1)

    st.markdown("#### Employment & Income")
    c1, c2, c3, c4 = st.columns(4)
    monthly_salary = c1.number_input("Monthly Salary (₹)", 15_000, 200_000, 45_000, step=1_000)
    employment_type = c2.selectbox("Employment Type", EMPLOYMENT_TYPES)
    years_of_employment = c3.number_input("Years of Employment", 0.0, 40.0, 5.0, step=0.5)
    company_type = c4.selectbox("Company Type", COMPANY_TYPES)

    st.markdown("#### Housing & Family")
    c1, c2, c3, c4 = st.columns(4)
    house_type = c1.selectbox("House Type", HOUSE_TYPES)
    monthly_rent = c2.number_input("Monthly Rent (₹)", 0, 60_000, 8_000, step=500)
    family_size = c3.number_input("Family Size", 1, 10, 3)
    dependents = c4.number_input("Dependents", 0, 8, 1)

    st.markdown("#### Monthly Financial Obligations")
    c1, c2, c3, c4, c5 = st.columns(5)
    school_fees = c1.number_input("School Fees (₹)", 0, 20_000, 0, step=500)
    college_fees = c2.number_input("College Fees (₹)", 0, 30_000, 0, step=500)
    travel_expenses = c3.number_input("Travel Expenses (₹)", 0, 20_000, 3_000, step=250)
    groceries_utilities = c4.number_input("Groceries & Utilities (₹)", 0, 50_000, 8_000, step=500)
    other_monthly_expenses = c5.number_input("Other Expenses (₹)", 0, 30_000, 2_000, step=250)

    st.markdown("#### Financial Status & Credit History")
    c1, c2, c3, c4, c5 = st.columns(5)
    existing_loans = c1.selectbox("Existing Loans?", ["No", "Yes"])
    current_emi_amount = c2.number_input("Current EMI Amount (₹)", 0, 50_000, 0, step=500)
    credit_score = c3.number_input("Credit Score", 300, 850, 700)
    bank_balance = c4.number_input("Bank Balance (₹)", 0, 2_000_000, 80_000, step=5_000)
    emergency_fund = c5.number_input("Emergency Fund (₹)", 0, 500_000, 20_000, step=1_000)

    st.markdown("#### Loan Application Details")
    c1, c2, c3 = st.columns(3)
    emi_scenario = c1.selectbox("EMI Scenario", list(SCENARIOS.keys()))
    amt_lo, amt_hi = SCENARIOS[emi_scenario]["amount"]
    ten_lo, ten_hi = SCENARIOS[emi_scenario]["tenure"]
    requested_amount = c2.number_input(
        f"Requested Amount (₹{amt_lo:,} – ₹{amt_hi:,})", amt_lo, amt_hi, int((amt_lo + amt_hi) / 4), step=1_000
    )
    requested_tenure = c3.number_input(
        f"Requested Tenure (months, {ten_lo}–{ten_hi})", ten_lo, ten_hi, min(ten_lo + 12, ten_hi)
    )

    submitted = st.form_submit_button("Assess Risk & Predict", use_container_width=True, type="primary")

if submitted:
    row = pd.DataFrame([{
        "age": age, "gender": gender, "marital_status": marital_status, "education": education,
        "monthly_salary": monthly_salary, "employment_type": employment_type,
        "years_of_employment": years_of_employment, "company_type": company_type,
        "house_type": house_type, "monthly_rent": monthly_rent, "family_size": family_size,
        "dependents": dependents, "school_fees": school_fees, "college_fees": college_fees,
        "travel_expenses": travel_expenses, "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses, "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount, "credit_score": credit_score,
        "bank_balance": bank_balance, "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario, "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
    }])

    # Classification
    X_clf, _ = build_model_matrix(row, encoders=clf_encoders, fit=False)
    X_clf_in = clf_scaler.transform(X_clf) if clf_meta["scaled_input"] else X_clf
    pred_class_idx = clf.predict(X_clf_in)[0]
    pred_proba = clf.predict_proba(X_clf_in)[0]
    pred_label = label_encoder.inverse_transform([pred_class_idx])[0]
    proba_map = dict(zip(label_encoder.classes_, pred_proba))

    # Regression
    X_reg, _ = build_model_matrix(row, encoders=reg_encoders, fit=False)
    X_reg_in = reg_scaler.transform(X_reg) if reg_meta["scaled_input"] else X_reg
    pred_max_emi = float(reg.predict(X_reg_in)[0])

    st.divider()
    st.markdown("### Result")

    colA, colB = st.columns([1, 1])
    with colA:
        badge = {"Eligible": "🟢", "High_Risk": "🟡", "Not_Eligible": "🔴"}[pred_label]
        st.markdown(f"#### {badge} EMI Eligibility: **{pred_label.replace('_', ' ')}**")
        st.progress(float(proba_map.get(pred_label, 0)), text=f"Model confidence: {proba_map.get(pred_label, 0)*100:.1f}%")
        prob_df = pd.DataFrame({"Class": list(proba_map.keys()), "Probability": list(proba_map.values())})
        st.bar_chart(prob_df.set_index("Class"))

    with colB:
        st.markdown("#### 💰 Recommended Maximum Monthly EMI")
        st.metric("Max Safe EMI", f"₹{pred_max_emi:,.0f} / month")
        # Rough affordability check against what they requested
        annual_rate = 0.13
        r = annual_rate / 12
        n = requested_tenure
        p = requested_amount
        est_installment = p * r * (1 + r) ** n / ((1 + r) ** n - 1)
        st.metric("Estimated Installment for Requested Loan", f"₹{est_installment:,.0f} / month",
                  delta=f"{est_installment - pred_max_emi:+,.0f} vs. max safe EMI", delta_color="inverse")
        if est_installment <= pred_max_emi:
            st.success("The requested EMI fits comfortably within the customer's safe affordability limit.")
        else:
            st.warning("The requested EMI exceeds the recommended safe limit — consider a longer tenure or lower amount.")

    with st.expander("View engineered features used by the model"):
        st.dataframe(X_clf.T.rename(columns={0: "value"}))
