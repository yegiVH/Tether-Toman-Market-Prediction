"""
data_loader.py

Loading and cleaning data 

This mirrors the logic used in the notebooks:
- BTCUSDT.csv
- USDIRT.csv
- Wallex_USDIRT.csv
- BTC_TEST.csv
- DOLLAR_TEST.csv
- TETHER_TEST.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def _check_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")


def _clean_usdirt(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns to Open/High/Low/Close if using uppercase
    rename_map = {
        "OPEN": "Open",
        "HIGH": "High",
        "LOW": "Low",
        "CLOSE": "Close",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Drop VOL if present
    if "VOL" in df.columns:
        df = df.drop(columns=["VOL"])

    # Drop Unnamed: 0 if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Fill missing Close with mean
    if "Close" in df.columns:
        df["Close"] = df["Close"].astype(float)
        df["Close"].fillna(df["Close"].mean(), inplace=True)

    return df


def _clean_btcusdt(df: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Close" in df.columns:
        df["Close"] = df["Close"].astype(float)
        df["Close"].fillna(df["Close"].mean(), inplace=True)
    return df


def _clean_wallex(df: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Clean Volume: replace '?' with NaN, convert to float, fill with mean
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].replace({"?": np.nan})
        df["Volume"] = df["Volume"].astype(float)
        df["Volume"].fillna(df["Volume"].mean(), inplace=True)

    return df


def load_train_data():
    """
    Load and clean training data:
    - BTCUSDT.csv
    - USDIRT.csv
    - Wallex_USDIRT.csv
    """
    btc_path = DATA_DIR / "BTCUSDT.csv"
    usdirt_path = DATA_DIR / "USDIRT.csv"
    wallex_path = DATA_DIR / "Wallex_USDIRT.csv"

    _check_exists(btc_path)
    _check_exists(usdirt_path)
    _check_exists(wallex_path)

    BTCUSDT = pd.read_csv(btc_path)
    USDIRT = pd.read_csv(usdirt_path)
    Wallex = pd.read_csv(wallex_path)

    BTCUSDT = _clean_btcusdt(BTCUSDT)
    USDIRT = _clean_usdirt(USDIRT)
    Wallex = _clean_wallex(Wallex)

    return BTCUSDT, USDIRT, Wallex


def load_test_data():
    """
    Load and clean test data:
    - BTC_TEST.csv
    - DOLLAR_TEST.csv
    - TETHER_TEST.csv

    This mirrors the logic in the second notebook.
    """
    btc_path = DATA_DIR / "BTC_TEST.csv"
    usdirt_path = DATA_DIR / "DOLLAR_TEST.csv"
    wallex_path = DATA_DIR / "TETHER_TEST.csv"

    _check_exists(btc_path)
    _check_exists(usdirt_path)
    _check_exists(wallex_path)

    BTCUSDT = pd.read_csv(btc_path)
    USDIRT = pd.read_csv(usdirt_path)
    Wallex = pd.read_csv(wallex_path)

    BTCUSDT = _clean_btcusdt(BTCUSDT)
    USDIRT = _clean_usdirt(USDIRT)
    Wallex = _clean_wallex(Wallex)

    return BTCUSDT, USDIRT, Wallex


if __name__ == "__main__":
    btc, usd, wal = load_train_data()
    print("[INFO] Train shapes:", btc.shape, usd.shape, wal.shape)
    btc_t, usd_t, wal_t = load_test_data()
    print("[INFO] Test shapes:", btc_t.shape, usd_t.shape, wal_t.shape)