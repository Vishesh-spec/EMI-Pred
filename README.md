# 💳 EMIPredict AI — Intelligent Financial Risk Assessment Platform

An end-to-end FinTech ML platform that predicts **EMI eligibility** (classification)
and the **maximum safe monthly EMI amount** (regression) from a customer's
financial profile — with full **MLflow** experiment tracking and a real-time
**Streamlit** web application.

## 📌 Problem Statement
People often struggle to repay EMIs due to poor financial planning and inadequate
risk assessment. This platform gives lenders data-driven, real-time insight into
whether a customer can safely take on a loan, and how much they can afford —
before the loan is approved.

## 🗂️ Project Structure
```
emipredict/
├── data/
│   ├── EMI_dataset.csv            # full 400,000-row RAW dataset
│   ├── EMI_dataset_clean.csv      # cleaned & validated dataset (output of data_preprocessing.py)
│   ├── quality_report.json        # missing/duplicate/range/outlier audit report
│   ├── EMI_dataset_sample.csv     # 20,000-row sample used by the Data Explorer page
│   └── EMI_dataset_admin.csv      # working copy edited via the Admin Data Manager (generated)
├── scripts/
│   ├── generate_dataset.py        # synthesizes the 400K-record EMI dataset
│   ├── data_preprocessing.py      # Step 1: missing-value/duplicate/range/outlier cleaning + quality report
│   ├── feature_engineering.py     # shared feature engineering (used by training + app)
│   ├── train_models.py            # single-script pipeline: trains all 6 models + MLflow (registry-enabled)
│   ├── precompute.py              # caches train/val/test splits (used by train_stage.py)
│   ├── train_stage.py             # trains ONE model at a time (fast, resumable) + MLflow registry
│   └── finalize.py                # picks best classifier/regressor, promotes them to @production in MLflow
├── models/                        # trained models + metadata (generated)
│   ├── best_classifier.pkl / best_regressor.pkl
│   ├── clf_logreg.pkl / clf_rf.pkl / clf_xgb.pkl        # all 3 individual classifiers
│   ├── reg_linreg.pkl / reg_rf.pkl / reg_xgb.pkl        # all 3 individual regressors
│   ├── classifier_meta.json / regressor_meta.json       # val + test metrics for every model tried
│   └── ...scalers, encoders, label encoder
├── mlflow.db                      # SQLite MLflow tracking store + Model Registry (6 registered models)
├── app/
│   ├── app.py                     # Streamlit home page
│   ├── feature_engineering.py     # copy used at runtime by the app
│   └── pages/
│       ├── 1_Predict.py                 # real-time prediction form
│       ├── 2_Data_Explorer.py           # dataset EDA & charts
│       ├── 3_Model_Performance.py       # model comparison + live MLflow Model Registry view
│       └── 4_Admin_Data_Manager.py      # full CRUD: create / read / update / delete records
├── requirements.txt
└── README.md
```

## 📊 Dataset — `EMI_dataset` (400,000 records)
5 EMI scenarios × 80,000 records each:

| Scenario | Amount Range | Tenure |
|---|---|---|
| E-commerce Shopping EMI | ₹10K – ₹200K | 3–24 months |
| Home Appliances EMI | ₹20K – ₹300K | 6–36 months |
| Vehicle EMI | ₹80K – ₹1,500K | 12–84 months |
| Personal Loan EMI | ₹50K – ₹1,000K | 12–60 months |
| Education EMI | ₹50K – ₹500K | 6–48 months |

**22 input features** across demographics, employment, housing/family,
monthly obligations, credit history, and loan application details, plus
**2 targets**: `emi_eligibility` (Eligible / High_Risk / Not_Eligible) and
`max_monthly_emi`.

> The raw dataset wasn't provided alongside the brief, so it was **synthetically
> generated** (`scripts/generate_dataset.py`) to exactly match every field,
> range, and scenario described in the project spec, with realistic
> correlations (income → affordability → eligibility, credit score → risk, etc.)
> baked in so the ML models have genuine signal to learn from.

## 🧹 Step 1 — Data Preprocessing & Quality Assessment
`scripts/data_preprocessing.py` runs a full audit on the 400,000-row raw
dataset before any modeling begins:
- Missing-value detection + median/mode imputation as a safety net
- Duplicate-row detection & removal
- Range validation against documented business rules (age, salary, credit
  score, EMI) — out-of-range values are clipped
- Categorical validity checks — invalid category rows are dropped
- Outlier flagging (IQR method) on key monetary columns — reported, not
  auto-removed

Output: `data/EMI_dataset_clean.csv` (the dataset all models are trained on)
and `data/quality_report.json` (full audit trail). On this dataset, the audit
found **0 missing values, 0 duplicates, 0 validation failures** — confirming
the data was fit for modeling.
```bash
python scripts/data_preprocessing.py
```

## 🤖 Models Trained (≥3 each, as required)
All models use a **70% train / 15% validation / 15% test** split. Validation
metrics guide model selection sanity-checks; test metrics (below) are the
final, held-out numbers.

**Classification — EMI Eligibility**
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 93.43% | 0.948 | 0.934 | 0.938 | 0.994 |
| **Random Forest ⭐ (best)** | **99.73%** | **0.997** | **0.997** | **0.997** | **1.000** |
| XGBoost | 99.64% | 0.996 | 0.996 | 0.996 | 1.000 |

**Regression — Maximum Monthly EMI**
| Model | RMSE (₹) | MAE (₹) | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | 2,427 | ~1,450 | 0.961 | ~39% |
| Random Forest | 1,290 | ~880 | 0.989 | ~9% |
| **XGBoost ⭐ (best)** | **950** | **~620** | **0.994** | **~4%** |

Both exceed the brief's targets: **classification accuracy > 90%** and
**regression RMSE < ₹2,000**. All 6 runs (params + val/test metrics + the
fitted model artifact) are logged to MLflow.

Model selection: classification models are ranked by **weighted F1-score**
(accuracy alone is misleading given the class imbalance), regression models
by **RMSE** (direct rupee-level error).

## 🧪 MLflow Experiment Tracking & Model Registry
Every model run logs hyperparameters, validation + test metrics, **and the
fitted model itself as a versioned artifact** to a local MLflow tracking
store (`mlflow.db`, SQLite backend). All 6 models are registered in the
**MLflow Model Registry** (e.g. `EMIPredict-RandomForestClassifier`,
`EMIPredict-XGBoostRegressor`), and `scripts/finalize.py` tags the winning
classifier + regressor with a `production` alias for clear, auditable
version control of what's actually deployed.

To explore it:
```bash
cd emipredict
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open `http://localhost:5000` to compare runs, view metric charts, and
browse the model registry (Models tab). Two experiments are created:
`EMIPredict-Classification` and `EMIPredict-Regression`.

## 🚀 How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset
```bash
python scripts/generate_dataset.py
```

### 3. Clean & validate the data
```bash
python scripts/data_preprocessing.py
```

### 4. Train all models
Either the single-script pipeline:
```bash
python scripts/train_models.py
```
…or the faster, resumable stage-by-stage approach used to build this project
(recommended for large datasets / limited compute):
```bash
python scripts/precompute.py
python scripts/train_stage.py clf logreg
python scripts/train_stage.py clf rf
python scripts/train_stage.py clf xgb
python scripts/train_stage.py reg linreg
python scripts/train_stage.py reg rf
python scripts/train_stage.py reg xgb
python scripts/finalize.py
```

### 5. Launch the Streamlit app
```bash
cd app
streamlit run app.py
```

## ✅ Brief → Deliverables Checklist
| Brief Requirement | Where it's covered |
|---|---|
| Data cleaning: missing values, duplicates, inconsistencies | `scripts/data_preprocessing.py` → `data/quality_report.json` |
| Train/Validation/Test splits | `scripts/precompute.py`, `train_stage.py`, `train_models.py` (70/15/15) |
| EDA report with business insights | `EMIPredict_EDA_Report.docx` (10 charts on the full 400K dataset) |
| ≥3 classification + ≥3 regression models | `models/training_results.json` (6 models, val+test metrics) |
| MLflow experiment tracking (params + metrics) | `mlflow.db`, both experiments, all 6 runs |
| MLflow model artifact storage + Model Registry | `mlflow.sklearn.log_model` / `mlflow.xgboost.log_model`, 6 registered models, `production` alias on the winners |
| Real-time classification + regression predictions | `app/pages/1_Predict.py` |
| CRUD operations / admin data management | `app/pages/4_Admin_Data_Manager.py` |
| Technical documentation (architecture + methodology) | `EMIPredict_Technical_Documentation.docx` |
| Cloud deployment on Streamlit Cloud | See "Deploy on Streamlit Community Cloud" below |

## 📤 Push to GitHub

This folder is **already a git repo with an initial commit** — you just need
to connect it to GitHub and push:

```bash
# 1. Create a new repository on github.com (e.g. "emipredict-ai"), keep it empty (no README/license)
# 2. From inside this project folder:
git remote add origin https://github.com/<your-username>/emipredict-ai.git
git branch -M main
git push -u origin main
```

If `git init` wasn't already run (e.g. you re-downloaded a fresh copy), do this first:
```bash
git init
git add .
git commit -m "Initial commit"
```

> **Note on repo size:** the full dataset (`data/EMI_dataset.csv`, ~78MB) and the
> best classifier (`models/best_classifier.pkl`, ~55MB) are both under GitHub's
> 100MB single-file limit, so a normal `git push` works fine. If you ever add
> larger files, use [Git LFS](https://git-lfs.com/).

## ☁️ Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (see above).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"** → select your repo → branch `main`.
4. Set **Main file path** to: `app/app.py` (the app lives in the `app/` subfolder, not the repo root).
5. Click **Deploy**. Streamlit Cloud auto-detects `requirements.txt` at the repo
   root and installs everything — first deploy takes ~2–5 minutes.
6. You'll get a public URL like `https://your-app-name.streamlit.app` — share
   it with anyone, no installation needed on their end.

**If deployment fails:** check the app logs on Streamlit Cloud (bottom-right
"Manage app" → logs) — the most common issues are a missing dependency in
`requirements.txt` or a wrong "Main file path". Both are already handled
correctly in this repo.

## 🖥️ Application Pages
- **Home** — platform overview, key metrics, business use cases
- **Predict** — full customer-profile form → instant eligibility + max EMI, with
  confidence scores and an affordability check against the requested loan
- **Data Explorer** — distributions, scenario breakdowns, correlation heatmap
- **Model Performance** — side-by-side comparison of all 6 trained models
  (validation + test metrics), best-model selection rationale, tuned
  hyperparameters, and a live view of the MLflow Model Registry
- **Admin Data Manager** — full **CRUD** interface: create new customer
  records, read/search/filter/export existing ones, update fields, and
  delete records

## 💼 Business Impact
- Automates loan eligibility checks, cutting manual underwriting time
- Gives loan officers instant, explainable risk assessments
- Flags at-risk applicants (`High_Risk` class) for tailored pricing instead of
  blanket rejection
- Recommends a safe EMI ceiling per customer — reducing default risk while
  maximizing approved loan volume

## 🏷️ Tech Stack
Python · pandas · scikit-learn · XGBoost · MLflow · Streamlit · joblib
