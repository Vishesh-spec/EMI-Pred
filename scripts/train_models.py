"""
EMIPredict AI — Model Training Pipeline
Trains >=3 classification models (EMI eligibility) and >=3 regression models
(max monthly EMI), tunes them, logs everything to MLflow, and saves the best
of each to models/ for the Streamlit app.
"""
import json
import time
import warnings
warnings.filterwarnings("ignore")

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from emipredict_package.scripts.feature_engineering import build_model_matrix

# NOTE: points at the CLEANED / validated dataset produced by
# scripts/data_preprocessing.py — run that script first.
DATA_PATH = "/home/claude/emipredict/data/EMI_dataset_clean.csv"
MODELS_DIR = "/home/claude/emipredict/models"
MLFLOW_URI = "sqlite:////home/claude/emipredict/mlflow.db"
TUNE_SAMPLE_SIZE = 60_000   # subsample for RandomizedSearchCV to keep runtime sane
RANDOM_STATE = 42

mlflow.set_tracking_uri(MLFLOW_URI)


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def log_model_to_registry(model, flavor, name, X_example):
    """Log the fitted model as an MLflow artifact and register it in the
    MLflow Model Registry under a unique, versioned name."""
    log_fn = mlflow.xgboost.log_model if flavor == "xgboost" else mlflow.sklearn.log_model
    log_fn(model, name="model", registered_model_name=f"EMIPredict-{name}", input_example=X_example)


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ----------------------------------------------------------------------------
# Classification: EMI eligibility (Eligible / High_Risk / Not_Eligible)
# ----------------------------------------------------------------------------
def train_classification(df, experiment_name="EMIPredict-Classification"):
    print("\n" + "=" * 70)
    print("CLASSIFICATION — EMI Eligibility")
    print("=" * 70)
    mlflow.set_experiment(experiment_name)

    X, encoders = build_model_matrix(df, fit=True)
    y_raw = df["emi_eligibility"].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 70% train / 15% validation / 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Stratified subsample for hyperparameter tuning
    tune_idx, _ = train_test_split(
        np.arange(len(X_train)), train_size=min(TUNE_SAMPLE_SIZE, len(X_train)),
        stratify=y_train, random_state=RANDOM_STATE,
    )

    results = {}
    fitted_models = {}

    # --- Model 1: Logistic Regression -----------------------------------
    with mlflow.start_run(run_name="LogisticRegression"):
        t0 = time.time()
        param_dist = {"C": uniform(0.01, 10), "penalty": ["l2"], "solver": ["lbfgs"]}
        search = RandomizedSearchCV(
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE),
            param_dist, n_iter=6, cv=3, scoring="f1_weighted", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train_scaled[tune_idx], y_train[tune_idx])
        best_params = search.best_params_
        model = LogisticRegression(max_iter=500, class_weight="balanced",
                                    random_state=RANDOM_STATE, **best_params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)

        metrics = log_clf_metrics(y_test, y_pred, y_proba, best_params, time.time() - t0, "LogisticRegression")
        results["LogisticRegression"] = metrics
        fitted_models["LogisticRegression"] = (model, "scaled")
        log_model_to_registry(model, "sklearn", "LogisticRegression", X_train_scaled[:2])

    # --- Model 2: Random Forest ------------------------------------------
    with mlflow.start_run(run_name="RandomForestClassifier"):
        t0 = time.time()
        param_dist = {
            "n_estimators": randint(150, 400),
            "max_depth": randint(6, 24),
            "min_samples_leaf": randint(1, 8),
            "max_features": ["sqrt", "log2"],
        }
        search = RandomizedSearchCV(
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            param_dist, n_iter=6, cv=3, scoring="f1_weighted", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train.iloc[tune_idx], y_train[tune_idx])
        best_params = search.best_params_
        model = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                        n_jobs=-1, **best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = log_clf_metrics(y_test, y_pred, y_proba, best_params, time.time() - t0, "RandomForestClassifier")
        results["RandomForestClassifier"] = metrics
        fitted_models["RandomForestClassifier"] = (model, "raw")
        log_model_to_registry(model, "sklearn", "RandomForestClassifier", X_train.iloc[:2])

    # --- Model 3: XGBoost --------------------------------------------------
    with mlflow.start_run(run_name="XGBoostClassifier"):
        t0 = time.time()
        param_dist = {
            "n_estimators": randint(150, 400),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.03, 0.25),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.7, 0.3),
        }
        sample_weight_tune = compute_sample_weights(y_train[tune_idx])
        search = RandomizedSearchCV(
            XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                          random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"),
            param_dist, n_iter=6, cv=3, scoring="f1_weighted", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train.iloc[tune_idx], y_train[tune_idx], sample_weight=sample_weight_tune)
        best_params = search.best_params_
        sample_weight_full = compute_sample_weights(y_train)
        model = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                              random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", **best_params)
        model.fit(X_train, y_train, sample_weight=sample_weight_full)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = log_clf_metrics(y_test, y_pred, y_proba, best_params, time.time() - t0, "XGBoostClassifier")
        results["XGBoostClassifier"] = metrics
        fitted_models["XGBoostClassifier"] = (model, "raw")
        log_model_to_registry(model, "xgboost", "XGBoostClassifier", X_train.iloc[:2])

    # --- Select & save best model ------------------------------------------
    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    best_model, mode = fitted_models[best_name]
    print(f"\nBest classification model: {best_name}  (F1={results[best_name]['f1_weighted']:.4f})")

    joblib.dump(best_model, f"{MODELS_DIR}/best_classifier.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/classifier_scaler.pkl")
    joblib.dump(label_encoder, f"{MODELS_DIR}/label_encoder.pkl")
    joblib.dump(encoders, f"{MODELS_DIR}/classifier_encoders.pkl")
    with open(f"{MODELS_DIR}/classifier_meta.json", "w") as f:
        json.dump({"best_model": best_name, "scaled_input": mode == "scaled", "results": results}, f, indent=2)

    return results, best_name


def compute_sample_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[v] for v in y])


def log_clf_metrics(y_test, y_pred, y_proba, params, elapsed, model_name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_weighted", prec)
    mlflow.log_metric("recall_weighted", rec)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("roc_auc_ovr", auc)
    mlflow.log_metric("train_time_sec", elapsed)

    print(f"{model_name:28s} acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  ({elapsed:.1f}s)")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_weighted": f1, "roc_auc": auc}


# ----------------------------------------------------------------------------
# Regression: Maximum monthly EMI
# ----------------------------------------------------------------------------
def train_regression(df, experiment_name="EMIPredict-Regression"):
    print("\n" + "=" * 70)
    print("REGRESSION — Maximum Monthly EMI")
    print("=" * 70)
    mlflow.set_experiment(experiment_name)

    X, encoders = build_model_matrix(df, fit=True)
    y = df["max_monthly_emi"].values

    # 70% train / 15% validation / 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tune_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_train), size=min(TUNE_SAMPLE_SIZE, len(X_train)), replace=False
    )

    results = {}
    fitted_models = {}

    # --- Model 1: Linear Regression -----------------------------------
    with mlflow.start_run(run_name="LinearRegression"):
        t0 = time.time()
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = log_reg_metrics(y_test, y_pred, {}, time.time() - t0, "LinearRegression")
        results["LinearRegression"] = metrics
        fitted_models["LinearRegression"] = (model, "scaled")
        log_model_to_registry(model, "sklearn", "LinearRegression", X_train_scaled[:2])

    # --- Model 2: Random Forest Regressor -------------------------------
    with mlflow.start_run(run_name="RandomForestRegressor"):
        t0 = time.time()
        param_dist = {
            "n_estimators": randint(150, 400),
            "max_depth": randint(6, 24),
            "min_samples_leaf": randint(1, 8),
            "max_features": ["sqrt", "log2", 1.0],
        }
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            param_dist, n_iter=6, cv=3, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train.iloc[tune_idx], y_train[tune_idx])
        best_params = search.best_params_
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = log_reg_metrics(y_test, y_pred, best_params, time.time() - t0, "RandomForestRegressor")
        results["RandomForestRegressor"] = metrics
        fitted_models["RandomForestRegressor"] = (model, "raw")
        log_model_to_registry(model, "sklearn", "RandomForestRegressor", X_train.iloc[:2])

    # --- Model 3: XGBoost Regressor -----------------------------------
    with mlflow.start_run(run_name="XGBoostRegressor"):
        t0 = time.time()
        param_dist = {
            "n_estimators": randint(150, 400),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.03, 0.25),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.7, 0.3),
        }
        search = RandomizedSearchCV(
            XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                        n_jobs=-1, tree_method="hist"),
            param_dist, n_iter=6, cv=3, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train.iloc[tune_idx], y_train[tune_idx])
        best_params = search.best_params_
        model = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                             n_jobs=-1, tree_method="hist", **best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = log_reg_metrics(y_test, y_pred, best_params, time.time() - t0, "XGBoostRegressor")
        results["XGBoostRegressor"] = metrics
        fitted_models["XGBoostRegressor"] = (model, "raw")
        log_model_to_registry(model, "xgboost", "XGBoostRegressor", X_train.iloc[:2])

    # --- Select & save best model ------------------------------------------
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model, mode = fitted_models[best_name]
    print(f"\nBest regression model: {best_name}  (RMSE={results[best_name]['rmse']:.2f})")

    joblib.dump(best_model, f"{MODELS_DIR}/best_regressor.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/regressor_scaler.pkl")
    joblib.dump(encoders, f"{MODELS_DIR}/regressor_encoders.pkl")
    with open(f"{MODELS_DIR}/regressor_meta.json", "w") as f:
        json.dump({"best_model": best_name, "scaled_input": mode == "scaled", "results": results}, f, indent=2)

    return results, best_name


def log_reg_metrics(y_test, y_pred, params, elapsed, model_name):
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    mape_val = mape(y_test, y_pred)

    mlflow.log_params(params)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape_val)
    mlflow.log_metric("train_time_sec", elapsed)

    print(f"{model_name:28s} rmse={rmse:.2f}  mae={mae:.2f}  r2={r2:.4f}  mape={mape_val:.2f}%  ({elapsed:.1f}s)")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape_val}


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df):,} records, {df.shape[1]} columns")

    clf_results, clf_best = train_classification(df)
    reg_results, reg_best = train_regression(df)

    summary = {
        "classification": {"results": clf_results, "best": clf_best},
        "regression": {"results": reg_results, "best": reg_best},
    }
    with open(f"{MODELS_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\nDone. Models saved to", MODELS_DIR)
