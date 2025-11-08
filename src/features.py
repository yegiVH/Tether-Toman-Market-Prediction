"""
features.py

Feature engineering for:
- BTCUSDT
- USDIRT
- Wallex_USDIRT

Implements:
- CCI (with custom MAD, no TA-Lib)
- RSI (custom)
- MACD
- Wallex volume features
- Price ratios
- Binary label: next Wallex close up/down
"""

from typing import Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "BTCUSDT_CCI",
    "USDIRT_CCI",
    "Wallex_USDIRT_CCI",
    "BTCUSDT_RSI",
    "USDIRT_RSI",
    "Wallex_USDIRT_RSI",
    "BTCUSDT_MACD",
    "USDIRT_MACD",
    "Wallex_USDIRT_MACD",
    "Wallex_USDIRT_Avg_Volume_hour",
    "Wallex_USDIRT_return",
    "Tether/Dollar_close",
    "Tether-Dollar/Tether_close",
    "label",
]


# -------------------------------------------------------------------
# Alignment helper
# -------------------------------------------------------------------
def _align_lengths(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Cut all dataframes to the same length (min length), aligning from the end.
    This ensures BTCUSDT, USDIRT, Wallex_USDIRT line up row-wise.
    """
    min_len = min(len(df) for df in dfs)
    return tuple(df.iloc[-min_len:].reset_index(drop=True) for df in dfs)


# -------------------------------------------------------------------
# Indicator implementations
# -------------------------------------------------------------------
def _mad(values: np.ndarray) -> float:
    """
    Mean Absolute Deviation around the mean.
    Used for CCI denominator.
    """
    if values.size == 0:
        return np.nan
    m = values.mean()
    return np.mean(np.abs(values - m))


def _cci(table: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI):
        TP = (High + Low + Close) / 3
        CCI = (TP - SMA(TP)) / (0.015 * MAD(TP))
    """
    tp = (table["High"] + table["Low"] + table["Close"]) / 3.0
    sma = tp.rolling(period).mean()

    # rolling MAD (custom, since Series.mad is deprecated/awkward)
    mad = tp.rolling(period).apply(lambda x: _mad(x.values), raw=False)

    cci = (tp - sma) / (0.015 * mad)
    cci = cci.replace([np.inf, -np.inf], np.nan)
    cci = cci.fillna(cci.mean())
    return cci


def _rsi(table: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Relative Strength Index (RSI) without TA-Lib.
    """
    close = table["Close"].astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    rsi = rsi.fillna(rsi.mean())
    return rsi


def _macd(table: pd.DataFrame) -> pd.Series:
    """
    MACD = EMA_12(Close) - EMA_26(Close)
    """
    close = table["Close"].astype(float)
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd = macd.fillna(macd.mean())
    return macd


# -------------------------------------------------------------------
# Safety helpers for OHLC
# -------------------------------------------------------------------
def _ensure_close(df: pd.DataFrame, context: str) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise KeyError(f"[{context}] Missing 'Close' column in input data.")
    out = df.copy()
    out["Close"] = out["Close"].astype(float)
    return out


def _ensure_ohlc(df: pd.DataFrame, context: str) -> pd.DataFrame:
    out = df.copy()
    needed = ["Open", "High", "Low", "Close"]

    if all(c in out.columns for c in needed):
        for c in needed:
            out[c] = out[c].astype(float)
        return out

    # Fallback for BTCUSDT: use Close as OHLC if only Close exists
    if "Close" in out.columns:
        c = out["Close"].astype(float)
        out["Open"] = c
        out["High"] = c
        out["Low"] = c
        out["Close"] = c
        return out

    raise KeyError(f"[{context}] Missing OHLC/Close columns in input data.")


# -------------------------------------------------------------------
# Main feature builder
# -------------------------------------------------------------------
def build_features(
    BTCUSDT: pd.DataFrame, USDIRT: pd.DataFrame, Wallex: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the full feature matrix as in the notebooks.

    Assumes cleaned inputs:
    - BTCUSDT: Close (and optionally OHLC)
    - USDIRT: Open, High, Low, Close
    - Wallex: Open, High, Low, Close, Volume

    Steps:
    - Align all three by length
    - Compute: CCI, RSI, MACD for each
    - Volume rolling mean (61)
    - Wallex return (Close - Open)
    - Ratios vs USDIRT
    - Binary label based on next Wallex Close
    """
    BTCUSDT, USDIRT, Wallex = _align_lengths(BTCUSDT, USDIRT, Wallex)
    n = len(Wallex)
    features = pd.DataFrame(index=range(n))

    # --- CCI ---
    features["BTCUSDT_CCI"] = _cci(_ensure_ohlc(BTCUSDT, "BTCUSDT_CCI"))
    features["USDIRT_CCI"] = _cci(_ensure_ohlc(USDIRT, "USDIRT_CCI"))
    features["Wallex_USDIRT_CCI"] = _cci(_ensure_ohlc(Wallex, "Wallex_USDIRT_CCI"))

    # --- RSI ---
    features["BTCUSDT_RSI"] = _rsi(_ensure_close(BTCUSDT, "BTCUSDT_RSI"))
    features["USDIRT_RSI"] = _rsi(_ensure_close(USDIRT, "USDIRT_RSI"))
    features["Wallex_USDIRT_RSI"] = _rsi(_ensure_close(Wallex, "Wallex_USDIRT_RSI"))

    # --- MACD ---
    features["BTCUSDT_MACD"] = _macd(_ensure_close(BTCUSDT, "BTCUSDT_MACD"))
    features["USDIRT_MACD"] = _macd(_ensure_close(USDIRT, "USDIRT_MACD"))
    features["Wallex_USDIRT_MACD"] = _macd(_ensure_close(Wallex, "Wallex_USDIRT_MACD"))

    # --- Avg Wallex volume (~1 hour, 61 steps) ---
    if "Volume" in Wallex.columns:
        vol = Wallex["Volume"].astype(float)
        vol_avg = vol.rolling(61).mean()
        vol_avg = vol_avg.fillna(vol_avg.mean())
        features["Wallex_USDIRT_Avg_Volume_hour"] = vol_avg
    else:
        features["Wallex_USDIRT_Avg_Volume_hour"] = 0.0

    # --- Wallex return (Close - Open) ---
    features["Wallex_USDIRT_return"] = (
        Wallex["Close"].astype(float) - Wallex["Open"].astype(float)
    )

    # --- Ratios vs USDIRT ---
    features["Tether/Dollar_close"] = (
        Wallex["Close"].astype(float) / USDIRT["Close"].astype(float)
    )
    features["Tether-Dollar/Tether_close"] = (
        Wallex["Close"].astype(float) - USDIRT["Close"].astype(float)
    ) / Wallex["Close"].astype(float)

    # --- Label: 1 if next Wallex Close > current Close, else 0 ---
    close = Wallex["Close"].astype(float)
    next_close = close.shift(-1)
    label = (next_close > close).astype(int)
    label.iloc[-1] = 0  # last row has no future close
    features["label"] = label

    # Final NaN cleanup
    for col in features.columns:
        if features[col].isna().any():
            features[col].fillna(features[col].mean(), inplace=True)

    return features


if __name__ == "__main__":
    print("[INFO] features.py loaded successfully.")