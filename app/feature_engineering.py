"""
EMIPredict AI — Feature Engineering
Shared preprocessing so training and the Streamlit app stay in sync.
"""
import numpy as np
import pandas as pd

CATEGORICAL_COLS = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario",
]

NUMERIC_BASE_COLS = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure",
]

ENGINEERED_COLS = [
    "total_monthly_expenses", "disposable_income", "debt_to_income_ratio",
    "expense_to_income_ratio", "affordability_ratio", "estimated_monthly_installment",
    "savings_to_income_ratio", "dependents_per_income", "credit_score_norm",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive financial ratios & risk features from the raw EMI dataset."""
    df = df.copy()

    df["total_monthly_expenses"] = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"]
        + df["travel_expenses"] + df["groceries_utilities"]
        + df["other_monthly_expenses"] + df["current_emi_amount"]
    )
    df["disposable_income"] = df["monthly_salary"] - df["total_monthly_expenses"]
    df["debt_to_income_ratio"] = df["current_emi_amount"] / (df["monthly_salary"] + 1)
    df["expense_to_income_ratio"] = df["total_monthly_expenses"] / (df["monthly_salary"] + 1)

    annual_rate = 0.13
    r = annual_rate / 12
    n_months = df["requested_tenure"].values
    p = df["requested_amount"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        installment = p * r * (1 + r) ** n_months / ((1 + r) ** n_months - 1)
    df["estimated_monthly_installment"] = installment
    df["affordability_ratio"] = df["estimated_monthly_installment"] / (df["disposable_income"] + 1)

    df["savings_to_income_ratio"] = df["emergency_fund"] / (df["monthly_salary"] * 3 + 1)
    df["dependents_per_income"] = df["dependents"] / (df["monthly_salary"] / 10_000 + 1)
    df["credit_score_norm"] = (df["credit_score"] - 300) / 550

    return df


def get_feature_columns():
    return NUMERIC_BASE_COLS + ENGINEERED_COLS + CATEGORICAL_COLS


def build_model_matrix(df: pd.DataFrame, encoders=None, fit=False):
    """
    One-hot encodes categoricals (aligned to a fixed column set) and returns
    a numeric feature matrix ready for scikit-learn / XGBoost models.
    """
    df = engineer_features(df)
    feature_cols = NUMERIC_BASE_COLS + ENGINEERED_COLS
    numeric_part = df[feature_cols].astype(float)

    cat_part = pd.get_dummies(df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)

    if fit:
        dummy_columns = cat_part.columns.tolist()
    else:
        dummy_columns = encoders["dummy_columns"]
        cat_part = cat_part.reindex(columns=dummy_columns, fill_value=0)

    X = pd.concat([numeric_part.reset_index(drop=True), cat_part.reset_index(drop=True)], axis=1)
    if fit:
        encoders = {"dummy_columns": dummy_columns, "feature_order": X.columns.tolist()}
    else:
        X = X.reindex(columns=encoders["feature_order"], fill_value=0)

    return X, encoders
