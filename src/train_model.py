"""
train_model.py

Refactored from the original notebooks.

Pipeline:
- Load train and test CSVs
- Build engineered features (CCI, RSI, MACD, etc.)
- Upsample to balance classes on train
- Scale features
- Train tuned RandomForest model
- Evaluate on test set
- Save model, scaler, and feature list into models/

Run:
    python -m src.train_model
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

from .data_loader import load_train_data, load_test_data
from .features import build_features

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _upsample_binary(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Upsample minority class to match majority class.

    This mirrors the notebook:
    - split by label
    - sample with replacement from minority
    - concat and shuffle
    """
    df = features_df.copy()
    label_col = "label"

    zeros = df[df[label_col] == 0]
    ones = df[df[label_col] == 1]

    if len(zeros) == 0 or len(ones) == 0:
        print("[WARN] Only one class present; skipping upsampling.")
        return df

    if len(zeros) > len(ones):
        majority, minority = zeros, ones
    else:
        majority, minority = ones, zeros

    minority_upsampled = minority.sample(
        n=len(majority), replace=True, random_state=42
    )

    balanced_df = (
        pd.concat([majority, minority_upsampled], axis=0)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    return balanced_df


def main():
    # ---------------- Load data ----------------
    print("[INFO] Loading training and test data...")
    BTC_train, USDIRT_train, Wallex_train = load_train_data()
    BTC_test, USDIRT_test, Wallex_test = load_test_data()

    # ---------------- Build features ----------------
    print("[INFO] Building training features...")
    features_train = build_features(BTC_train, USDIRT_train, Wallex_train)
    print(f"[INFO] Train features shape: {features_train.shape}")
    print(f"[INFO] Train label distribution:\n{features_train['label'].value_counts()}")

    print("[INFO] Building test features...")
    features_test = build_features(BTC_test, USDIRT_test, Wallex_test)
    print(f"[INFO] Test features shape: {features_test.shape}")
    print(f"[INFO] Test label distribution:\n{features_test['label'].value_counts()}")

    # ---------------- Upsample ----------------
    print("[INFO] Upsampling training data to balance classes...")
    balanced_train = _upsample_binary(features_train)
    print(f"[INFO] Balanced train shape: {balanced_train.shape}")
    print(f"[INFO] Balanced label distribution:\n{balanced_train['label'].value_counts()}")

    # ---------------- Split X, y ----------------
    X_train = balanced_train.drop(columns=["label"])
    y_train = balanced_train["label"].astype(int)

    X_test = features_test.drop(columns=["label"])
    y_test = features_test["label"].astype(int)

    feature_names = X_train.columns.tolist()

    # ---------------- Scale ----------------
    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- Cross-validation ----------------
    print("[INFO] Running cross-validation on RandomForest...")
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    scores = cross_val_score(base_rf, X_train_scaled, y_train, cv=5)
    print(f"[INFO] Mean CV accuracy: {scores.mean():.4f}")

    # ---------------- Train final model ----------------
    print("[INFO] Training final tuned RandomForest model...")
    model = RandomForestClassifier(
        max_depth=40,
        min_samples_leaf=4,
        n_estimators=1000,
        min_samples_split=6,
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # ---------------- Evaluate on test ----------------
    print("[INFO] Evaluating on held-out test set...")
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, digits=3)
    print(report)

    # ---------------- Save artifacts ----------------
    model_path = MODELS_DIR / "tether_toman_rf_model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    features_path = MODELS_DIR / "feature_columns.txt"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    features_path.write_text("\n".join(feature_names))

    print(f"[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved scaler to: {scaler_path}")
    print(f"[INFO] Saved feature columns to: {features_path}")
    print("[INFO] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
