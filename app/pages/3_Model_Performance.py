import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Performance — EMIPredict AI", page_icon="📈", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


@st.cache_data
def load_meta():
    with open(f"{MODELS_DIR}/classifier_meta.json") as f:
        clf_meta = json.load(f)
    with open(f"{MODELS_DIR}/regressor_meta.json") as f:
        reg_meta = json.load(f)
    return clf_meta, reg_meta


st.title("📈 Model Performance & MLflow Experiment Comparison")
st.caption("All 6 models were trained with RandomizedSearchCV (5-fold CV) and logged to MLflow "
           "for full experiment tracking, comparison, and model registry.")

try:
    clf_meta, reg_meta = load_meta()
except FileNotFoundError:
    st.error("Training results not found. Run the training pipeline first.")
    st.stop()

st.markdown("## Classification — EMI Eligibility")
st.caption("Trained on a 70% train / 15% validation / 15% test split — Test columns shown below are the final, held-out evaluation.")
clf_df = pd.DataFrame(clf_meta["results"]).T
clf_df.index.name = "Model"
clf_df = clf_df[["val_f1_weighted", "accuracy", "precision", "recall", "f1_weighted", "roc_auc", "train_time_sec"]]
clf_df.columns = ["Val F1", "Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score", "Test ROC-AUC", "Train Time (s)"]

st.dataframe(
    clf_df.style.highlight_max(subset=["Val F1", "Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score", "Test ROC-AUC"], color="#c6f6c6")
    .format({"Val F1": "{:.4f}", "Test Accuracy": "{:.4f}", "Test Precision": "{:.4f}", "Test Recall": "{:.4f}",
             "Test F1-Score": "{:.4f}", "Test ROC-AUC": "{:.4f}", "Train Time (s)": "{:.1f}"}),
    use_container_width=True,
)
st.bar_chart(clf_df[["Test Accuracy", "Test F1-Score", "Test ROC-AUC"]])
st.success(f"🏆 **Best classification model: {clf_meta['best_model']}** — selected by highest weighted F1-score on the "
           "held-out test set, the right metric here since accuracy alone is misleading on an imbalanced 3-class target.")

st.divider()

st.markdown("## Regression — Maximum Monthly EMI")
st.caption("Trained on a 70% train / 15% validation / 15% test split — Test columns shown below are the final, held-out evaluation.")
reg_df = pd.DataFrame(reg_meta["results"]).T
reg_df.index.name = "Model"
reg_df = reg_df[["val_rmse", "rmse", "mae", "r2", "mape", "train_time_sec"]]
reg_df.columns = ["Val RMSE (₹)", "Test RMSE (₹)", "Test MAE (₹)", "Test R²", "Test MAPE (%)", "Train Time (s)"]

st.dataframe(
    reg_df.style.highlight_min(subset=["Val RMSE (₹)", "Test RMSE (₹)", "Test MAE (₹)", "Test MAPE (%)"], color="#c6f6c6")
    .highlight_max(subset=["Test R²"], color="#c6f6c6")
    .format({"Val RMSE (₹)": "{:.1f}", "Test RMSE (₹)": "{:.1f}", "Test MAE (₹)": "{:.1f}", "Test R²": "{:.4f}",
             "Test MAPE (%)": "{:.2f}", "Train Time (s)": "{:.1f}"}),
    use_container_width=True,
)
st.bar_chart(reg_df[["Test RMSE (₹)", "Test MAE (₹)"]])
st.success(f"🏆 **Best regression model: {reg_meta['best_model']}** — selected by lowest RMSE on the held-out test set, "
           "directly minimizing rupee-level prediction error on the maximum safe EMI.")

st.divider()
st.markdown("## Selected Model Hyperparameters")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**{clf_meta['best_model']} (Classifier)**")
    st.json(clf_meta["results"][clf_meta["best_model"]]["best_params"])
with c2:
    st.markdown(f"**{reg_meta['best_model']} (Regressor)**")
    st.json(reg_meta["results"][reg_meta["best_model"]]["best_params"])

st.divider()
st.markdown("## MLflow Model Registry")
try:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(f"sqlite:///{MODELS_DIR}/../mlflow.db")
    client = MlflowClient()
    rows = []
    for rm in client.search_registered_models():
        for v in client.search_model_versions(f"name='{rm.name}'"):
            rows.append({"Registered Model": rm.name, "Version": v.version,
                         "Alias": ", ".join(v.aliases) if v.aliases else "—"})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No registered models found yet — run the training pipeline first.")
except Exception:
    st.info(
        "All 6 trained models (params, metrics, and the fitted model artifact) are logged and "
        "versioned in the **MLflow Model Registry** — the best classifier and regressor are "
        "additionally tagged with a `production` alias."
    )

st.info(
    "💡 Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` from the project root to explore "
    "the full experiment history, compare runs side-by-side, and browse the model registry visually."
)
