"""
Admin — Data Manager
=====================
Complete CRUD (Create / Read / Update / Delete) interface for the customer
financial-profile records used by EMIPredict AI, as required by the project
brief ("Complete CRUD operations for financial data management" /
"Administrative interface for data management operations").

Operates on a local *working copy* (data/EMI_dataset_admin.csv) seeded from
the 20,000-row sample, so admins can freely add/edit/delete records without
touching the original training data. Changes persist across reruns within
the same session and can be downloaded as CSV.
"""
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Admin Data Manager — EMIPredict AI", page_icon="🛠️", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SEED_PATH = os.path.join(DATA_DIR, "EMI_dataset_sample.csv")
WORKING_PATH = os.path.join(DATA_DIR, "EMI_dataset_admin.csv")

st.title("🛠️ Admin — Financial Data Manager")
st.caption("Full CRUD interface: Create, Read, Update, and Delete customer financial records.")


@st.cache_data
def load_seed():
    return pd.read_csv(SEED_PATH)


def get_working_df():
    if "admin_df" not in st.session_state:
        if os.path.exists(WORKING_PATH):
            st.session_state.admin_df = pd.read_csv(WORKING_PATH)
        else:
            st.session_state.admin_df = load_seed().copy()
    return st.session_state.admin_df


def save_working_df(df):
    st.session_state.admin_df = df
    df.to_csv(WORKING_PATH, index=False)


df = get_working_df()

tab_read, tab_create, tab_update, tab_delete = st.tabs(
    ["📖 Read / Search", "➕ Create", "✏️ Update", "🗑️ Delete"]
)

# ---------------------------------------------------------------- READ ----
with tab_read:
    st.markdown(f"**Total records: {len(df):,}**")
    c1, c2, c3 = st.columns(3)
    scenario_filter = c1.selectbox(
        "Filter by EMI scenario", ["All"] + sorted(df["emi_scenario"].dropna().unique().tolist())
    )
    eligibility_filter = c2.selectbox(
        "Filter by eligibility", ["All"] + sorted(df["emi_eligibility"].dropna().unique().tolist())
    )
    search_id = c3.text_input("Search by row index (optional)")

    view = df.copy()
    if scenario_filter != "All":
        view = view[view["emi_scenario"] == scenario_filter]
    if eligibility_filter != "All":
        view = view[view["emi_eligibility"] == eligibility_filter]
    if search_id.strip().isdigit():
        idx = int(search_id.strip())
        view = view.loc[[idx]] if idx in view.index else view.iloc[0:0]

    st.dataframe(view, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Download current view as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="emi_records_export.csv",
        mime="text/csv",
    )

# -------------------------------------------------------------- CREATE ----
with tab_create:
    st.markdown("Add a new customer financial record.")
    with st.form("create_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 25, 60, 30)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        marital_status = c3.selectbox("Marital Status", ["Single", "Married"])

        education = c1.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
        employment_type = c2.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        monthly_salary = c3.number_input("Monthly Salary (₹)", 15_000, 200_000, 40_000, step=1000)

        credit_score = c1.number_input("Credit Score", 300, 850, 650)
        bank_balance = c2.number_input("Bank Balance (₹)", 0, 5_000_000, 50_000, step=1000)
        emergency_fund = c3.number_input("Emergency Fund (₹)", 0, 2_000_000, 20_000, step=1000)

        emi_scenario = c1.selectbox(
            "EMI Scenario",
            ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI",
             "Personal Loan EMI", "Education EMI"],
        )
        requested_amount = c2.number_input("Requested Amount (₹)", 10_000, 1_500_000, 100_000, step=1000)
        requested_tenure = c3.number_input("Requested Tenure (months)", 3, 84, 24)

        emi_eligibility = c1.selectbox("EMI Eligibility (label)", ["Eligible", "High_Risk", "Not_Eligible"])
        max_monthly_emi = c2.number_input("Max Monthly EMI (₹, label)", 500, 50_000, 5000, step=100)

        submitted = st.form_submit_button("➕ Add Record")
        if submitted:
            new_row = {c: None for c in df.columns}
            new_row.update({
                "age": age, "gender": gender, "marital_status": marital_status,
                "education": education, "employment_type": employment_type,
                "monthly_salary": monthly_salary, "credit_score": credit_score,
                "bank_balance": bank_balance, "emergency_fund": emergency_fund,
                "emi_scenario": emi_scenario, "requested_amount": requested_amount,
                "requested_tenure": requested_tenure, "emi_eligibility": emi_eligibility,
                "max_monthly_emi": max_monthly_emi,
            })
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_working_df(df)
            st.success(f"Record added. New total: {len(df):,} rows.")

# -------------------------------------------------------------- UPDATE ----
with tab_update:
    st.markdown("Edit an existing record by row index.")
    if len(df) == 0:
        st.info("No records available.")
    else:
        edit_idx = st.number_input("Row index to edit", 0, len(df) - 1, 0)
        record = df.loc[edit_idx].to_dict()
        with st.form("update_form"):
            c1, c2, c3 = st.columns(3)
            new_salary = c1.number_input("Monthly Salary (₹)", 15_000, 200_000,
                                          int(record.get("monthly_salary", 40000)), step=1000)
            new_credit = c2.number_input("Credit Score", 300, 850, int(record.get("credit_score", 650)))
            new_eligibility = c3.selectbox(
                "EMI Eligibility", ["Eligible", "High_Risk", "Not_Eligible"],
                index=["Eligible", "High_Risk", "Not_Eligible"].index(record.get("emi_eligibility", "Eligible"))
                if record.get("emi_eligibility") in ["Eligible", "High_Risk", "Not_Eligible"] else 0,
            )
            new_max_emi = c1.number_input("Max Monthly EMI (₹)", 500, 50_000,
                                           int(record.get("max_monthly_emi", 5000)), step=100)
            update_submitted = st.form_submit_button("✏️ Save Changes")
            if update_submitted:
                df.loc[edit_idx, "monthly_salary"] = new_salary
                df.loc[edit_idx, "credit_score"] = new_credit
                df.loc[edit_idx, "emi_eligibility"] = new_eligibility
                df.loc[edit_idx, "max_monthly_emi"] = new_max_emi
                save_working_df(df)
                st.success(f"Row {edit_idx} updated.")
        st.markdown("**Current record:**")
        st.json({k: record[k] for k in list(record)[:12]})

# -------------------------------------------------------------- DELETE ----
with tab_delete:
    st.markdown("Remove a record by row index.")
    if len(df) == 0:
        st.info("No records available.")
    else:
        del_idx = st.number_input("Row index to delete", 0, len(df) - 1, 0, key="del_idx")
        st.dataframe(df.loc[[del_idx]], use_container_width=True)
        if st.button("🗑️ Delete Record", type="primary"):
            df = df.drop(index=del_idx).reset_index(drop=True)
            save_working_df(df)
            st.success("Record deleted.")
            st.rerun()

st.divider()
if st.button("↺ Reset working copy to original sample data"):
    if os.path.exists(WORKING_PATH):
        os.remove(WORKING_PATH)
    st.session_state.pop("admin_df", None)
    st.rerun()
