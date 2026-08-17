import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Explorer — EMIPredict AI", page_icon="📊", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "EMI_dataset_sample.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.title("📊 Data Explorer")
st.caption("Exploring a 20,000-row sample of the 400,000-record EMI training dataset.")

df = load_data()

col1, col2, col3 = st.columns(3)
col1.metric("Sample Size", f"{len(df):,} rows")
col2.metric("Full Dataset Size", "400,000 rows")
col3.metric("Features", "22 input + 2 target")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Target Distributions", "EMI Scenarios", "Financial Profile", "Correlations"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**EMI Eligibility Distribution**")
        st.bar_chart(df["emi_eligibility"].value_counts())
    with c2:
        st.markdown("**Max Monthly EMI Distribution**")
        emi_bins = pd.cut(df["max_monthly_emi"], bins=15).value_counts().sort_index()
        emi_bins.index = emi_bins.index.astype(str)
        st.bar_chart(emi_bins)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Records per EMI Scenario**")
        st.bar_chart(df["emi_scenario"].value_counts())
    with c2:
        st.markdown("**Avg. Max EMI by Scenario**")
        st.bar_chart(df.groupby("emi_scenario")["max_monthly_emi"].mean())

    st.markdown("**Eligibility Rate by Scenario**")
    elig_by_scenario = pd.crosstab(df["emi_scenario"], df["emi_eligibility"], normalize="index") * 100
    st.bar_chart(elig_by_scenario)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Monthly Salary Distribution**")
        salary_bins = pd.cut(df["monthly_salary"], bins=15).value_counts().sort_index()
        salary_bins.index = salary_bins.index.astype(str)
        st.bar_chart(salary_bins)
        st.markdown("**Credit Score Distribution**")
        credit_bins = pd.cut(df["credit_score"], bins=15).value_counts().sort_index()
        credit_bins.index = credit_bins.index.astype(str)
        st.bar_chart(credit_bins)
    with c2:
        st.markdown("**Eligibility by Employment Type**")
        et = pd.crosstab(df["employment_type"], df["emi_eligibility"], normalize="index") * 100
        st.bar_chart(et)
        st.markdown("**Eligibility by Education**")
        ed = pd.crosstab(df["education"], df["emi_eligibility"], normalize="index") * 100
        st.bar_chart(ed)

with tab4:
    numeric_cols = [
        "age", "monthly_salary", "years_of_employment", "monthly_rent", "credit_score",
        "bank_balance", "emergency_fund", "current_emi_amount", "requested_amount",
        "requested_tenure", "max_monthly_emi",
    ]
    corr = df[numeric_cols].corr()
    st.markdown("**Correlation Heatmap (numeric features vs. max_monthly_emi)**")
    st.dataframe(corr.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1).format("{:.2f}"))

st.divider()
st.markdown("### Raw Data Sample")
st.dataframe(df.head(200), use_container_width=True)
