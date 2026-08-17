"""
Train a single model (one stage) using the cached train/val/test splits, log
params + metrics + the fitted model artifact to MLflow (with Model Registry
registration), and append the result to a shared results JSON.

Usage: python3 train_stage.py <task> <model>
  task  in {clf, reg}
  model in {logreg, rf, xgb} for clf   OR   {linreg, rf, xgb} for reg
"""
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor

CACHE_DIR = "/home/claude/emipredict/cache"
MODELS_DIR = "/home/claude/emipredict/models"
RESULTS_PATH = f"{MODELS_DIR}/training_results.json"
MLFLOW_URI = "sqlite:////home/claude/emipredict/mlflow.db"
RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_URI)


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_sample_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    wmap = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([wmap[v] for v in y])


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"classification": {}, "regression": {}}


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def run_clf(model_key):
    cache = joblib.load(f"{CACHE_DIR}/clf_cache.pkl")
    X_train, X_val, X_test = cache["X_train"], cache["X_val"], cache["X_test"]
    y_train, y_val, y_test = cache["y_train"], cache["y_val"], cache["y_test"]
    X_train_scaled, X_val_scaled, X_test_scaled = (
        cache["X_train_scaled"], cache["X_val_scaled"], cache["X_test_scaled"]
    )
    tune_idx = cache["tune_idx"]

    mlflow.set_experiment("EMIPredict-Classification")
    t0 = time.time()

    if model_key == "logreg":
        name = "LogisticRegression"
        param_dist = {"C": uniform(0.01, 10)}
        search = RandomizedSearchCV(
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE),
            param_dist, n_iter=5, cv=3, scoring="f1_weighted", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train_scaled[tune_idx], y_train[tune_idx])
        model = LogisticRegression(max_iter=500, class_weight="balanced",
                                    random_state=RANDOM_STATE, **search.best_params_)
        model.fit(X_train_scaled, y_train)
        y_val_pred, y_val_proba = model.predict(X_val_scaled), model.predict_proba(X_val_scaled)
        y_pred, y_proba = model.predict(X_test_scaled), model.predict_proba(X_test_scaled)
        scaled = True
        flavor = "sklearn"

    elif model_key == "rf":
        name = "RandomForestClassifier"
        param_dist = {"n_estimators": randint(120, 260), "max_depth": randint(8, 20),
                      "min_samples_leaf": randint(1, 6), "max_features": ["sqrt", "log2"]}
        search = RandomizedSearchCV(
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            param_dist, n_iter=4, cv=3, scoring="f1_weighted", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train.iloc[tune_idx], y_train[tune_idx])
        model = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                        n_jobs=-1, **search.best_params_)
        model.fit(X_train, y_train)
        y_val_pred, y_val_proba = model.predict(X_val), model.predict_proba(X_val)
        y_pred, y_proba = model.predict(X_test), model.predict_proba(X_test)
        scaled = False
        flavor = "sklearn"

    elif model_key == "xgb":
        name = "XGBoostClassifier"
        param_dist = {"n_estimators": randint(120, 260), "max_depth": randint(3, 8),
                      "learning_rate": uniform(0.05, 0.2), "subsample": uniform(0.7, 0.3),
                      "colsample_bytree": uniform(0.7, 0.3)}
        sw_tune = compute_sample_weights(y_train[tune_idx])
        search = RandomizedSearchCV(
            XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                          random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"),
            param_dist, n_iter=4, cv=3, scoring="f1_weighted", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train.iloc[tune_idx], y_train[tune_idx], sample_weight=sw_tune)
        sw_full = compute_sample_weights(y_train)
        model = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                              random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", **search.best_params_)
        model.fit(X_train, y_train, sample_weight=sw_full)
        y_val_pred, y_val_proba = model.predict(X_val), model.predict_proba(X_val)
        y_pred, y_proba = model.predict(X_test), model.predict_proba(X_test)
        scaled = False
        flavor = "xgboost"
    else:
        raise ValueError(model_key)

    elapsed = time.time() - t0

    def clf_scores(yt, yp, ypr):
        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, average="weighted", zero_division=0)
        rec = recall_score(yt, yp, average="weighted", zero_division=0)
        f1 = f1_score(yt, yp, average="weighted", zero_division=0)
        try:
            auc = roc_auc_score(yt, ypr, multi_class="ovr", average="weighted")
        except Exception:
            auc = float("nan")
        return acc, prec, rec, f1, auc

    val_acc, val_prec, val_rec, val_f1, val_auc = clf_scores(y_val, y_val_pred, y_val_proba)
    acc, prec, rec, f1, auc = clf_scores(y_test, y_pred, y_proba)

    with mlflow.start_run(run_name=name):
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({
            "val_accuracy": val_acc, "val_f1_weighted": val_f1, "val_roc_auc_ovr": val_auc,
            "accuracy": acc, "precision_weighted": prec, "recall_weighted": rec,
            "f1_weighted": f1, "roc_auc_ovr": auc, "train_time_sec": elapsed,
        })
        # --- Log the fitted model as an MLflow artifact + register it ---
        log_fn = mlflow.xgboost.log_model if flavor == "xgboost" else mlflow.sklearn.log_model
        log_fn(
            model,
            name="model",
            registered_model_name=f"EMIPredict-{name}",
            input_example=X_train.iloc[:2] if not scaled else X_train_scaled[:2],
        )

    joblib.dump(model, f"{MODELS_DIR}/clf_{model_key}.pkl")
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_weighted": f1,
               "roc_auc": auc, "val_accuracy": val_acc, "val_f1_weighted": val_f1,
               "val_roc_auc": val_auc, "train_time_sec": elapsed, "scaled_input": scaled,
               "best_params": search.best_params_}

    results = load_results()
    results["classification"][name] = metrics
    save_results(results)

    print(f"{name:28s} val_f1={val_f1:.4f}  test_acc={acc:.4f}  test_f1={f1:.4f}  test_auc={auc:.4f}  ({elapsed:.1f}s)  [registered in MLflow Model Registry as EMIPredict-{name}]")


def run_reg(model_key):
    cache = joblib.load(f"{CACHE_DIR}/reg_cache.pkl")
    X_train, X_val, X_test = cache["X_train"], cache["X_val"], cache["X_test"]
    y_train, y_val, y_test = cache["y_train"], cache["y_val"], cache["y_test"]
    X_train_scaled, X_val_scaled, X_test_scaled = (
        cache["X_train_scaled"], cache["X_val_scaled"], cache["X_test_scaled"]
    )
    tune_idx = cache["tune_idx"]

    mlflow.set_experiment("EMIPredict-Regression")
    t0 = time.time()
    best_params = {}

    if model_key == "linreg":
        name = "LinearRegression"
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
        y_pred = model.predict(X_test_scaled)
        scaled = True
        flavor = "sklearn"

    elif model_key == "rf":
        name = "RandomForestRegressor"
        param_dist = {"n_estimators": randint(120, 260), "max_depth": randint(8, 20),
                      "min_samples_leaf": randint(1, 6), "max_features": ["sqrt", "log2"]}
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            param_dist, n_iter=4, cv=3, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train.iloc[tune_idx], y_train[tune_idx])
        best_params = search.best_params_
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_pred = model.predict(X_test)
        scaled = False
        flavor = "sklearn"

    elif model_key == "xgb":
        name = "XGBoostRegressor"
        param_dist = {"n_estimators": randint(120, 260), "max_depth": randint(3, 8),
                      "learning_rate": uniform(0.05, 0.2), "subsample": uniform(0.7, 0.3),
                      "colsample_bytree": uniform(0.7, 0.3)}
        search = RandomizedSearchCV(
            XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"),
            param_dist, n_iter=4, cv=3, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train.iloc[tune_idx], y_train[tune_idx])
        best_params = search.best_params_
        model = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                             n_jobs=-1, tree_method="hist", **best_params)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_pred = model.predict(X_test)
        scaled = False
        flavor = "xgboost"
    else:
        raise ValueError(model_key)

    elapsed = time.time() - t0

    def reg_scores(yt, yp):
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))
        return rmse, mae, r2, mape(yt, yp)

    val_rmse, val_mae, val_r2, val_mape = reg_scores(y_val, y_val_pred)
    rmse, mae, r2, mape_val = reg_scores(y_test, y_pred)

    with mlflow.start_run(run_name=name):
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2, "val_mape": val_mape,
            "rmse": rmse, "mae": mae, "r2": r2, "mape": mape_val, "train_time_sec": elapsed,
        })
        log_fn = mlflow.xgboost.log_model if flavor == "xgboost" else mlflow.sklearn.log_model
        log_fn(
            model,
            name="model",
            registered_model_name=f"EMIPredict-{name}",
            input_example=X_train.iloc[:2] if not scaled else X_train_scaled[:2],
        )

    joblib.dump(model, f"{MODELS_DIR}/reg_{model_key}.pkl")
    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape_val,
               "val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2, "val_mape": val_mape,
               "train_time_sec": elapsed, "scaled_input": scaled, "best_params": best_params}

    results = load_results()
    results["regression"][name] = metrics
    save_results(results)

    print(f"{name:28s} val_rmse={val_rmse:.2f}  test_rmse={rmse:.2f}  test_r2={r2:.4f}  ({elapsed:.1f}s)  [registered in MLflow Model Registry as EMIPredict-{name}]")


if __name__ == "__main__":
    task, model_key = sys.argv[1], sys.argv[2]
    if task == "clf":
        run_clf(model_key)
    elif task == "reg":
        run_reg(model_key)
    else:
        raise ValueError(task)
