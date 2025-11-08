
# 📈 Tether–Toman Market Prediction

This project predicts **short-term price movements** in the **Tether to Toman (USDT–IRT)** exchange rate using classical machine learning techniques.

It includes data cleaning, feature engineering (CCI, RSI, MACD), model training, evaluation, and prediction — all organized as a reproducible Python pipeline.

---

##  Setup

### 1️⃣ Create and activate a virtual environment (Windows example)
```bash
python -m venv venv
venv\Scripts\Activate
````

If PowerShell blocks activation:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Prepare data

Place the following `.csv` files inside the `data/` folder:

* `BTCUSDT.csv`
* `USDIRT.csv`
* `Wallex_USDIRT.csv`
* `BTC_TEST.csv`
* `DOLLAR_TEST.csv`
* `TETHER_TEST.csv`

---

## Run the Pipeline

### Train the model

```bash
python -m src.train_model
```

This will:

* Clean and load raw data
* Generate technical indicators (CCI, RSI, MACD)
* Upsample to balance classes
* Scale features
* Train a tuned **RandomForestClassifier**
* Evaluate performance
* Save model, scaler, and feature list in the `models/` folder

Example output:

```text
[INFO] Train features shape: (67206, 14)
[INFO] Mean CV accuracy: 0.8391
[INFO] Evaluating on held-out test set...
              precision    recall  f1-score   support

           0      0.510     0.832     0.632       691
           1      0.504     0.176     0.261       671

[INFO] Saved model to: models/tether_toman_rf_model.joblib
[INFO] Saved scaler to: models/scaler.joblib
[INFO] Saved feature columns to: models/feature_columns.txt
[INFO] Training pipeline completed successfully.
```

### ▶️ Generate predictions

```bash
python -m src.predict_next_minute
```

This will:

* Build features from test data
* Load the trained model and scaler
* Evaluate model performance
* Save predictions to:

```
results/predictions.csv
```

---

## 🗂 Project Structure

```text
Tether-Toman-Market-Prediction/
│
├─ data/
│   ├─ BTCUSDT.csv
│   ├─ USDIRT.csv
│   ├─ Wallex_USDIRT.csv
│   ├─ BTC_TEST.csv
│   ├─ DOLLAR_TEST.csv
│   └─ TETHER_TEST.csv
│
├─ src/
│   ├─ __init__.py
│   ├─ data_loader.py
│   ├─ features.py
│   ├─ train_model.py
│   └─ predict_next_minute.py
│
├─ models/
│   ├─ tether_toman_rf_model.joblib
│   ├─ scaler.joblib
│   └─ feature_columns.txt
│
├─ results/
│   └─ predictions.csv
│
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## 🧩 Feature Engineering

| Feature                                             | Description                                           |
| --------------------------------------------------- | ----------------------------------------------------- |
| `BTCUSDT_CCI`, `USDIRT_CCI`, `Wallex_USDIRT_CCI`    | Commodity Channel Index (20-period)                   |
| `BTCUSDT_RSI`, `USDIRT_RSI`, `Wallex_USDIRT_RSI`    | Relative Strength Index (20-period)                   |
| `BTCUSDT_MACD`, `USDIRT_MACD`, `Wallex_USDIRT_MACD` | MACD (EMA12 − EMA26)                                  |
| `Wallex_USDIRT_Avg_Volume_hour`                     | Rolling average of trading volume over 61 points      |
| `Wallex_USDIRT_return`                              | One-step return (Close − Open)                        |
| `Tether/Dollar_close`                               | Ratio Wallex_Close / USDIRT_Close                     |
| `Tether-Dollar/Tether_close`                        | (Wallex_Close − USDIRT_Close) / Wallex_Close          |
| `label`                                             | Binary target: 1 if next Wallex Close > current Close |

---

## 📊 Model Summary

| Parameter                 | Value                  |
| ------------------------- | ---------------------- |
| Algorithm                 | RandomForestClassifier |
| n_estimators              | 1000                   |
| max_depth                 | 40                     |
| min_samples_split         | 6                      |
| min_samples_leaf          | 4                      |
| bootstrap                 | False                  |
| Cross-validation accuracy | ~0.84                  |
| Test accuracy             | ~0.51                  |

---

## 🧠 Next Steps

* Add more financial indicators (moving averages, volatility, lag features)
* Compare models (XGBoost, LightGBM, CatBoost)
* Tune hyperparameters with `GridSearchCV` or `Optuna`
* Deploy the model using a REST API (FastAPI or Flask)

