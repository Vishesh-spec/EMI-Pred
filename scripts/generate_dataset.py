"""
EMIPredict AI — Synthetic Dataset Generator
Generates 400,000 realistic financial records across 5 EMI scenarios,
matching the project brief's 22 input features + 2 target variables.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_PER_SCENARIO = 80_000

SCENARIOS = {
    "E-commerce Shopping EMI": dict(amount=(10_000, 200_000), tenure=(3, 24)),
    "Home Appliances EMI":     dict(amount=(20_000, 300_000), tenure=(6, 36)),
    "Vehicle EMI":             dict(amount=(80_000, 1_500_000), tenure=(12, 84)),
    "Personal Loan EMI":       dict(amount=(50_000, 1_000_000), tenure=(12, 60)),
    "Education EMI":           dict(amount=(50_000, 500_000), tenure=(6, 48)),
}

EDUCATION_LEVELS = ["High School", "Graduate", "Post Graduate", "Professional"]
EMPLOYMENT_TYPES = ["Private", "Government", "Self-employed"]
COMPANY_TYPES = ["Startup", "SME", "MNC", "Public Sector", "Not Applicable"]
HOUSE_TYPES = ["Rented", "Own", "Family"]


def gen_scenario_block(scenario_name, cfg, n):
    age = RNG.integers(25, 61, n)
    gender = RNG.choice(["Male", "Female"], n, p=[0.58, 0.42])
    marital_status = RNG.choice(["Single", "Married"], n, p=[0.42, 0.58])
    education = RNG.choice(EDUCATION_LEVELS, n, p=[0.15, 0.45, 0.28, 0.12])

    # Employment & income — education/age lift income
    edu_lift = pd.Series(education).map(
        {"High School": 0.0, "Graduate": 0.15, "Post Graduate": 0.35, "Professional": 0.55}
    ).values
    base_salary = RNG.lognormal(mean=10.6, sigma=0.45, size=n)
    monthly_salary = np.clip(base_salary * (1 + edu_lift) * (1 + (age - 25) * 0.01), 15_000, 200_000)

    employment_type = RNG.choice(EMPLOYMENT_TYPES, n, p=[0.6, 0.15, 0.25])
    years_of_employment = np.clip(RNG.normal((age - 22) * 0.55, 3, n), 0, age - 21)
    company_type = np.where(
        employment_type == "Self-employed", "Not Applicable",
        RNG.choice(["Startup", "SME", "MNC", "Public Sector"], n, p=[0.15, 0.35, 0.35, 0.15]),
    )

    # Housing & family
    house_type = RNG.choice(HOUSE_TYPES, n, p=[0.42, 0.33, 0.25])
    monthly_rent = np.where(
        house_type == "Rented",
        np.clip(monthly_salary * RNG.uniform(0.12, 0.3, n), 3_000, 60_000),
        np.where(house_type == "Own", 0.0, np.clip(monthly_salary * RNG.uniform(0.0, 0.05, n), 0, 5_000)),
    )
    family_size = RNG.integers(1, 7, n)
    dependents = np.clip(family_size - RNG.integers(1, 3, n), 0, None)

    # Monthly obligations
    school_fees = np.where(dependents > 0, RNG.uniform(0, 8_000, n) * (dependents > 0), 0.0)
    college_fees = np.where((dependents > 0) & (age > 40), RNG.uniform(0, 15_000, n), 0.0)
    travel_expenses = np.clip(monthly_salary * RNG.uniform(0.02, 0.08, n), 500, 15_000)
    groceries_utilities = np.clip(2_000 * family_size + RNG.normal(0, 1_500, n), 2_000, 40_000)
    other_monthly_expenses = np.clip(monthly_salary * RNG.uniform(0.02, 0.1, n), 500, 20_000)

    # Financial status & credit history
    existing_loans = RNG.choice(["Yes", "No"], n, p=[0.4, 0.6])
    current_emi_amount = np.where(
        existing_loans == "Yes",
        np.clip(monthly_salary * RNG.uniform(0.05, 0.3, n), 500, 40_000),
        0.0,
    )
    credit_score = np.clip(
        RNG.normal(650, 90, n) + years_of_employment * 2 - (existing_loans == "Yes") * 15,
        300, 850,
    )
    bank_balance = np.clip(RNG.lognormal(mean=10.2, sigma=0.9, size=n), 0, 2_000_000)
    emergency_fund = np.clip(bank_balance * RNG.uniform(0.05, 0.4, n), 0, 500_000)

    # Loan application details
    amt_lo, amt_hi = cfg["amount"]
    ten_lo, ten_hi = cfg["tenure"]
    requested_amount = RNG.uniform(amt_lo, amt_hi, n)
    requested_tenure = RNG.integers(ten_lo, ten_hi + 1, n)
    emi_scenario = np.full(n, scenario_name)

    df = pd.DataFrame({
        "age": age.astype(int),
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary.round(2),
        "employment_type": employment_type,
        "years_of_employment": years_of_employment.round(1),
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent.round(2),
        "family_size": family_size.astype(int),
        "dependents": dependents.astype(int),
        "school_fees": school_fees.round(2),
        "college_fees": college_fees.round(2),
        "travel_expenses": travel_expenses.round(2),
        "groceries_utilities": groceries_utilities.round(2),
        "other_monthly_expenses": other_monthly_expenses.round(2),
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount.round(2),
        "credit_score": credit_score.round(0).astype(int),
        "bank_balance": bank_balance.round(2),
        "emergency_fund": emergency_fund.round(2),
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount.round(2),
        "requested_tenure": requested_tenure.astype(int),
    })
    return df


def add_targets(df):
    n = len(df)
    total_expenses = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"] + df["travel_expenses"]
        + df["groceries_utilities"] + df["other_monthly_expenses"] + df["current_emi_amount"]
    )
    disposable_income = np.clip(df["monthly_salary"] - total_expenses, -50_000, None)

    # Simple flat-rate estimate of the requested monthly installment
    annual_rate = 0.13
    r = annual_rate / 12
    n_months = df["requested_tenure"].values
    p = df["requested_amount"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        requested_monthly_installment = p * r * (1 + r) ** n_months / ((1 + r) ** n_months - 1)

    # A composite affordability / risk score (higher = safer)
    credit_component = (df["credit_score"] - 300) / 550          # 0..1
    dti_ratio = np.clip(df["current_emi_amount"] / (df["monthly_salary"] + 1), 0, 2)
    savings_component = np.clip(df["emergency_fund"] / (df["monthly_salary"] * 3 + 1), 0, 1)
    employment_bonus = np.where(df["employment_type"] == "Government", 0.08,
                          np.where(df["employment_type"] == "Private", 0.04, 0.0))
    tenure_bonus = np.clip(df["years_of_employment"] / 20, 0, 0.2)

    affordability_ratio = requested_monthly_installment / (disposable_income + 1)

    noise = RNG.normal(0, 0.05, n)
    risk_score = (
        0.35 * credit_component
        + 0.25 * (1 - np.clip(dti_ratio, 0, 1))
        + 0.15 * savings_component
        + employment_bonus
        + tenure_bonus
        - 0.35 * np.clip(affordability_ratio, 0, 2)
        + noise
    )

    eligibility = np.where(
        (risk_score > 0.28) & (affordability_ratio < 0.55),
        "Eligible",
        np.where(
            (risk_score > 0.05) & (affordability_ratio < 0.9),
            "High_Risk",
            "Not_Eligible",
        ),
    )

    # Max safe monthly EMI: a risk-adjusted fraction of disposable income
    safe_fraction = np.clip(0.35 + 0.25 * credit_component - 0.2 * dti_ratio, 0.1, 0.6)
    max_monthly_emi = np.clip(
        disposable_income * safe_fraction * (1 + RNG.normal(0, 0.05, n)),
        500, 50_000,
    )

    df = df.copy()
    df["emi_eligibility"] = eligibility
    df["max_monthly_emi"] = max_monthly_emi.round(2)
    return df


def main(out_path="/home/claude/emipredict/data/EMI_dataset.csv"):
    blocks = []
    for scenario, cfg in SCENARIOS.items():
        block = gen_scenario_block(scenario, cfg, N_PER_SCENARIO)
        blocks.append(block)
    df = pd.concat(blocks, ignore_index=True)
    df = add_targets(df)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    df.insert(0, "customer_id", [f"CUST{100000 + i}" for i in range(len(df))])
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} columns -> {out_path}")
    print(df["emi_eligibility"].value_counts(normalize=True))
    return df


if __name__ == "__main__":
    main()
