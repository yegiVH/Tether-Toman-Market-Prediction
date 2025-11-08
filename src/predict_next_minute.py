"""
predict_next_minute.py

Use the trained RandomForest model to:
- Load test data
- Rebuild features in the same way
- Apply the saved scaler
- Predict direction (0/1) for each row
- Print a classification report
- Save predictions to results/predictions.csv

Run:
    python -m src.predict_next_minute
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report

from .data_loader import load_test_data
from .features import build_features

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    model_path = MODELS_DIR / "tether_toman_rf_model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    features_path = MODELS_DIR / "feature_columns.txt"

    if not model_path.exists() or not scaler_path.exists() or not features_path.exists():
        raise FileNotFoundError(
            "Model, scaler, or feature_columns.txt not found. "
            "Run `python -m src.train_model` first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = features_path.read_text().splitlines()

    return model, scaler, feature_names


def main():
    print("[INFO] Loading artifacts...")
    model, scaler, feature_names = load_artifacts()

    print("[INFO] Loading and processing test data...")
    BTC_test, USDIRT_test, Wallex_test = load_test_data()
    features_test = build_features(BTC_test, USDIRT_test, Wallex_test)

    X_test = features_test.drop(columns=["label"])
    y_test = features_test["label"].astype(int)

    # Align columns with training order
    X_test = X_test[feature_names]

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print("[INFO] Classification report on test set:")
    print(classification_report(y_test, y_pred, digits=3))

    # Optional: save predictions with index if Datetime column exists
    if "Datetime" in Wallex_test.columns:
        idx = Wallex_test["Datetime"].iloc[-len(y_pred) :]
    else:
        idx = range(len(y_pred))

    output = pd.DataFrame({"index": idx, "prediction": y_pred})
    out_path = RESULTS_DIR / "predictions.csv"
    output.to_csv(out_path, index=False)
    print(f"[INFO] Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
