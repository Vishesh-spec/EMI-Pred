"""Pick the best classifier & regressor from training_results.json, copy them
to best_classifier.pkl / best_regressor.pkl, save all supporting artifacts
(scalers, encoders, label encoder) for the Streamlit app, AND promote the
winning models to the "production" alias in the MLflow Model Registry."""
import json
import shutil

import joblib
import mlflow
from mlflow import MlflowClient

MODELS_DIR = "/home/claude/emipredict/models"
CACHE_DIR = "/home/claude/emipredict/cache"
MLFLOW_URI = "sqlite:////home/claude/emipredict/mlflow.db"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()


def promote_to_production(registered_name):
    """Tag the latest version of a registered model with the 'production' alias."""
    try:
        versions = client.search_model_versions(f"name='{registered_name}'")
        latest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(registered_name, "production", latest.version)
        print(f"  -> MLflow Registry: '{registered_name}' v{latest.version} tagged as @production")
    except Exception as e:
        print(f"  -> Could not promote '{registered_name}' in MLflow registry: {e}")

with open(f"{MODELS_DIR}/training_results.json") as f:
    results = json.load(f)

MODEL_KEY_MAP_CLF = {
    "LogisticRegression": "logreg", "RandomForestClassifier": "rf", "XGBoostClassifier": "xgb",
}
MODEL_KEY_MAP_REG = {
    "LinearRegression": "linreg", "RandomForestRegressor": "rf", "XGBoostRegressor": "xgb",
}

# --- classification ---
clf_results = results["classification"]
best_clf_name = max(clf_results, key=lambda k: clf_results[k]["f1_weighted"])
best_clf_key = MODEL_KEY_MAP_CLF[best_clf_name]
shutil.copy(f"{MODELS_DIR}/clf_{best_clf_key}.pkl", f"{MODELS_DIR}/best_classifier.pkl")

clf_cache = joblib.load(f"{CACHE_DIR}/clf_cache.pkl")
joblib.dump(clf_cache["scaler"], f"{MODELS_DIR}/classifier_scaler.pkl")
joblib.dump(clf_cache["label_encoder"], f"{MODELS_DIR}/label_encoder.pkl")
joblib.dump(clf_cache["encoders"], f"{MODELS_DIR}/classifier_encoders.pkl")

with open(f"{MODELS_DIR}/classifier_meta.json", "w") as f:
    json.dump({
        "best_model": best_clf_name,
        "scaled_input": clf_results[best_clf_name]["scaled_input"],
        "results": clf_results,
    }, f, indent=2)

print(f"Best classifier: {best_clf_name}  (F1={clf_results[best_clf_name]['f1_weighted']:.4f}, "
      f"Accuracy={clf_results[best_clf_name]['accuracy']:.4f})")
promote_to_production(f"EMIPredict-{best_clf_name}")

# --- regression ---
reg_results = results["regression"]
best_reg_name = min(reg_results, key=lambda k: reg_results[k]["rmse"])
best_reg_key = MODEL_KEY_MAP_REG[best_reg_name]
shutil.copy(f"{MODELS_DIR}/reg_{best_reg_key}.pkl", f"{MODELS_DIR}/best_regressor.pkl")

reg_cache = joblib.load(f"{CACHE_DIR}/reg_cache.pkl")
joblib.dump(reg_cache["scaler"], f"{MODELS_DIR}/regressor_scaler.pkl")
joblib.dump(reg_cache["encoders"], f"{MODELS_DIR}/regressor_encoders.pkl")

with open(f"{MODELS_DIR}/regressor_meta.json", "w") as f:
    json.dump({
        "best_model": best_reg_name,
        "scaled_input": reg_results[best_reg_name]["scaled_input"],
        "results": reg_results,
    }, f, indent=2)

print(f"Best regressor: {best_reg_name}  (RMSE={reg_results[best_reg_name]['rmse']:.2f}, "
      f"R2={reg_results[best_reg_name]['r2']:.4f})")
promote_to_production(f"EMIPredict-{best_reg_name}")

print("\nFinal artifacts saved to", MODELS_DIR)
