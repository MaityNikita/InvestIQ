"""
=============================================================================
  feature_engineering.py — Financial Feature Pipeline
  AI Financial Advisory Chatbot — InvestIQ
=============================================================================
  Transforms raw OHLCV DataFrames into rich ML-ready feature sets:
    - Lag features         : lag_1, lag_3, lag_5, lag_10, lag_20
    - Moving averages      : MA-7, MA-14, MA-21, MA-50
    - Exponential MAs      : EMA-12, EMA-26, MACD
    - Volatility           : 10-day & 20-day annualised std
    - RSI (14-day)
    - Bollinger Bands      : upper, lower, bandwidth, %B
    - Stochastic Oscillator: %K, %D
    - On-Balance Volume    : OBV
    - Price ratios         : 52-week high/low ratio
    - Momentum             : 5-day & 20-day momentum
    - Calendar features    : day_of_week, month, year
=============================================================================
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to a raw OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        Original columns + all engineered features.  Rows with NaN values
        (due to rolling windows) are dropped before returning.
    """
    d = df.copy()
    c = d["Close"]
    h = d["High"]
    l = d["Low"]
    v = d["Volume"]

    # ── Lag features ──────────────────────────────────────────────────────────
    for lag in [1, 3, 5, 10, 20]:
        d[f"lag_{lag}"] = c.shift(lag)

    # ── Simple moving averages ────────────────────────────────────────────────
    for w in [7, 14, 21, 50]:
        d[f"ma_{w}"] = c.rolling(w).mean()

    # ── Exponential moving averages & MACD ────────────────────────────────────
    d["ema_12"] = c.ewm(span=12, adjust=False).mean()
    d["ema_26"] = c.ewm(span=26, adjust=False).mean()
    d["macd"]   = d["ema_12"] - d["ema_26"]
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]

    # ── Volatility (annualised) ───────────────────────────────────────────────
    ret = c.pct_change()
    d["volatility_10"] = ret.rolling(10).std() * np.sqrt(252)
    d["volatility_20"] = ret.rolling(20).std() * np.sqrt(252)

    # ── RSI (14-day) ──────────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands (20-day, 2σ) ─────────────────────────────────────────
    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    d["bb_upper"]    = bb_upper
    d["bb_lower"]    = bb_lower
    d["bb_width"]    = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    d["bb_pct"]      = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ── Stochastic Oscillator (%K, %D) ────────────────────────────────────────
    low_14  = l.rolling(14).min()
    high_14 = h.rolling(14).max()
    d["stoch_k"] = 100 * (c - low_14) / (high_14 - low_14 + 1e-9)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # ── On-Balance Volume (OBV) ───────────────────────────────────────────────
    obv_vals = np.where(c.diff() > 0, v, np.where(c.diff() < 0, -v, 0))
    d["obv"] = pd.Series(obv_vals, index=d.index).cumsum()

    # ── 52-week high/low ratios ───────────────────────────────────────────────
    d["ratio_52w_high"] = c / h.rolling(252).max().shift(1).clip(lower=1e-9)
    d["ratio_52w_low"]  = c / l.rolling(252).min().shift(1).clip(lower=1e-9)

    # ── Price range & daily return ───────────────────────────────────────────
    d["high_low_range"] = (h - l) / c.clip(lower=1e-9)
    d["daily_return"]   = ret

    # ── Momentum ──────────────────────────────────────────────────────────────
    d["momentum_5"]  = c / c.shift(5).clip(lower=1e-9) - 1
    d["momentum_20"] = c / c.shift(20).clip(lower=1e-9) - 1

    # ── Calendar features ─────────────────────────────────────────────────────
    d["day_of_week"] = d.index.dayofweek.astype(int)
    d["month"]       = d.index.month.astype(int)
    d["year"]        = d.index.year.astype(int)

    d.dropna(inplace=True)
    return d


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of engineered feature column names
    (i.e., everything except raw OHLCV).
    """
    raw = {"Open", "High", "Low", "Close", "Volume"}
    return [c for c in df.columns if c not in raw]


def prepare_xy(df: pd.DataFrame):
    """
    Split a feature-engineered DataFrame into X, y, feature_cols.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        X : feature matrix
        y : Close price target vector
        feature_cols : ordered list of feature names
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values.astype(float)
    y = df["Close"].values.astype(float)
    return X, y, feature_cols


# ── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_ingestion import download_data
    raw = download_data(verbose=True)
    for name, df_raw in raw.items():
        df_feat = add_features(df_raw)
        feat_cols = get_feature_columns(df_feat)
        print(f"\n  {name}: {len(feat_cols)} features, {len(df_feat)} rows")
        print(f"  Features: {feat_cols[:8]} …")
