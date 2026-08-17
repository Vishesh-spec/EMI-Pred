"""Precompute & cache train/test splits for classification + regression so
each model-training stage can run independently and quickly."""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from emipredict_package.scripts.feature_engineering import build_model_matrix

DATA_PATH = "/home/claude/emipredict/data/EMI_dataset_clean.csv"
CACHE_DIR = "/home/claude/emipredict/cache"
RANDOM_STATE = 42
TUNE_SAMPLE_SIZE = 40_000

import os
os.makedirs(CACHE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows (from cleaned/validated dataset)")

# ---------------- classification ----------------
# 3-way split: 70% train / 15% validation / 15% test
X, encoders = build_model_matrix(df, fit=True)
y_raw = df["emi_eligibility"].values
le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

tune_idx, _ = train_test_split(
    np.arange(len(X_train)), train_size=min(TUNE_SAMPLE_SIZE, len(X_train)),
    stratify=y_train, random_state=RANDOM_STATE,
)

joblib.dump(
    dict(X_train=X_train, X_val=X_val, X_test=X_test,
         y_train=y_train, y_val=y_val, y_test=y_test,
         X_train_scaled=X_train_scaled, X_val_scaled=X_val_scaled, X_test_scaled=X_test_scaled,
         tune_idx=tune_idx, encoders=encoders, label_encoder=le, scaler=scaler),
    f"{CACHE_DIR}/clf_cache.pkl",
)
print("Classification cache saved — train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

# ---------------- regression ----------------
Xr, encoders_r = build_model_matrix(df, fit=True)
yr = df["max_monthly_emi"].values

Xr_train, Xr_temp, yr_train, yr_temp = train_test_split(
    Xr, yr, test_size=0.30, random_state=RANDOM_STATE
)
Xr_val, Xr_test, yr_val, yr_test = train_test_split(
    Xr_temp, yr_temp, test_size=0.50, random_state=RANDOM_STATE
)
scaler_r = StandardScaler()
Xr_train_scaled = scaler_r.fit_transform(Xr_train)
Xr_val_scaled = scaler_r.transform(Xr_val)
Xr_test_scaled = scaler_r.transform(Xr_test)

tune_idx_r = np.random.RandomState(RANDOM_STATE).choice(
    len(Xr_train), size=min(TUNE_SAMPLE_SIZE, len(Xr_train)), replace=False
)

joblib.dump(
    dict(X_train=Xr_train, X_val=Xr_val, X_test=Xr_test,
         y_train=yr_train, y_val=yr_val, y_test=yr_test,
         X_train_scaled=Xr_train_scaled, X_val_scaled=Xr_val_scaled, X_test_scaled=Xr_test_scaled,
         tune_idx=tune_idx_r, encoders=encoders_r, scaler=scaler_r),
    f"{CACHE_DIR}/reg_cache.pkl",
)
print("Regression cache saved — train:", Xr_train.shape, "val:", Xr_val.shape, "test:", Xr_test.shape)
