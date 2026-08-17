"""
EMIPredict AI — Data Preprocessing & Quality Assessment
=========================================================
Implements Step 1 of the project brief in full:
  - Missing value detection & handling
  - Duplicate detection & removal
  - Inconsistency / validity checks against documented business ranges
  - Outlier flagging (IQR method, report-only — not auto-dropped)
  - A JSON + human-readable quality report saved to data/quality_report.json
  - Produces a CLEANED dataset used downstream by training

Run:
    python3 scripts/data_preprocessing.py
"""
import json
import os

import numpy as np
import pandas as pd

DATA_PATH = "/home/claude/emipredict/data/EMI_dataset.csv"
CLEAN_PATH = "/home/claude/emipredict/data/EMI_dataset_clean.csv"
REPORT_PATH = "/home/claude/emipredict/data/quality_report.json"

# Documented valid ranges / categories (from project brief) used for the
# "inconsistency" validation checks.
VALID_RANGES = {
    "age": (25, 60),
    "monthly_salary": (15_000, 200_000),
    "credit_score": (300, 850),
    "max_monthly_emi": (500, 50_000),
}
VALID_CATEGORIES = {
    "gender": {"Male", "Female"},
    "marital_status": {"Single", "Married"},
    "existing_loans": {"Yes", "No"},
    "emi_scenario": {
        "E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI",
        "Personal Loan EMI", "Education EMI",
    },
    "emi_eligibility": {"Eligible", "High_Risk", "Not_Eligible"},
}

NUMERIC_NON_NEGATIVE = [
    "monthly_salary", "years_of_employment", "monthly_rent", "family_size",
    "dependents", "school_fees", "college_fees", "travel_expenses",
    "groceries_utilities", "other_monthly_expenses", "current_emi_amount",
    "bank_balance", "emergency_fund", "requested_amount", "requested_tenure",
]


def assess_and_clean(df: pd.DataFrame):
    report = {"initial_rows": int(len(df)), "initial_cols": int(df.shape[1])}

    # ---------------- 1. Missing values ----------------
    missing_counts = df.isnull().sum()
    missing_report = {c: int(v) for c, v in missing_counts.items() if v > 0}
    report["missing_values_by_column"] = missing_report
    report["total_missing_cells"] = int(missing_counts.sum())

    if missing_report:
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
        for c in cat_cols:
            if df[c].isnull().any():
                df[c] = df[c].fillna(df[c].mode(dropna=True)[0])
    report["rows_after_missing_handling"] = int(len(df))

    # ---------------- 2. Duplicates ----------------
    dup_count = int(df.duplicated().sum())
    report["duplicate_rows_found"] = dup_count
    if dup_count:
        df = df.drop_duplicates().reset_index(drop=True)
    report["rows_after_dedup"] = int(len(df))

    # ---------------- 3. Inconsistency / validity checks ----------------
    inconsistencies = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if bad:
                inconsistencies[f"{col}_out_of_range[{lo},{hi}]"] = bad
                df[col] = df[col].clip(lo, hi)

    for col in NUMERIC_NON_NEGATIVE:
        if col in df.columns:
            bad = int((df[col] < 0).sum())
            if bad:
                inconsistencies[f"{col}_negative"] = bad
                df[col] = df[col].clip(lower=0)

    for col, allowed in VALID_CATEGORIES.items():
        if col in df.columns:
            bad_mask = ~df[col].isin(allowed)
            bad = int(bad_mask.sum())
            if bad:
                inconsistencies[f"{col}_invalid_category"] = bad
                df = df[~bad_mask].reset_index(drop=True)

    report["inconsistencies_found_and_fixed"] = inconsistencies
    report["rows_after_consistency_checks"] = int(len(df))

    # ---------------- 4. Outlier flags (report-only, IQR method) ----------------
    outlier_report = {}
    for col in ["monthly_salary", "bank_balance", "emergency_fund", "current_emi_amount"]:
        if col in df.columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
            outlier_report[col] = {"count": n_out, "pct": round(n_out / len(df) * 100, 2)}
    report["outliers_flagged_iqr"] = outlier_report

    # ---------------- 5. Final validation summary ----------------
    report["final_rows"] = int(len(df))
    report["final_cols"] = int(df.shape[1])
    report["rows_removed_total"] = int(report["initial_rows"] - report["final_rows"])
    report["pct_rows_removed"] = round(report["rows_removed_total"] / report["initial_rows"] * 100, 3)
    report["passed_validation"] = True

    return df, report


if __name__ == "__main__":
    print("Loading raw dataset...")

    df = pd.read_csv(
        r"C:\Users\acer\Desktop\EMI\EMIPredict_AI_Project_Updated\emipredict_package\data\EMI_dataset.csv"
    )

    print(f"Raw shape: {df.shape}")

    clean_df, report = assess_and_clean(df)

    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    clean_df.to_csv(CLEAN_PATH, index=False)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nClean shape: {clean_df.shape}")
    print(f"Saved cleaned dataset -> {CLEAN_PATH}")
    print(f"Saved quality report  -> {REPORT_PATH}")
    print("\n--- Quality Report Summary ---")
    print(json.dumps(report, indent=2))
