"""
=============================================================================
  predictor.py — Prediction Orchestrator
  AI Financial Advisory Chatbot — InvestIQ
=============================================================================
  Ties together data_ingestion → feature_engineering → models
  and returns JSON-serialisable prediction results.

  Usage (standalone):
      python predictor.py --ticker TCS.NS --days 30
=============================================================================
"""

import argparse
import json
import numpy as np
import pandas as pd

from data_ingestion      import get_single_ticker
from feature_engineering import add_features, prepare_xy
from models              import (
    train_random_forest, train_xgboost, train_lstm,
    classical_future_forecast, lstm_future_forecast,
    weighted_ensemble, compute_metrics, LSTM_LOOKBACK,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PREDICT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict(
    ticker: str,
    name:   str   = None,
    start:  str   = "2018-01-01",
    end:    str   = "2024-12-31",
    forecast_days: int = 30,
    models_to_use: list = None,
    verbose: bool = True,
) -> dict:
    """
    Full prediction pipeline for one ticker.

    Returns a dict suitable for JSON serialisation:
    {
        "ticker":      str,
        "name":        str,
        "last_close":  float,
        "trend":       "bullish" | "bearish" | "neutral",
        "confidence":  float (0–1),
        "metrics":     {model_name: {MAE, RMSE, R2, MAPE%}},
        "forecast":    {model_name: [price_day_1 … price_day_N]},
        "ensemble_forecast": [price_day_1 … price_day_N],
        "future_dates": [YYYY-MM-DD …],
        "recommendation": str,
    }
    """
    if models_to_use is None:
        models_to_use = ["Random Forest", "XGBoost", "LSTM"]

    company_label = name or ticker.replace(".NS", "").replace("^", "")

    if verbose:
        print(f"\n🔍  Predicting {company_label} ({ticker}) …")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    df_raw = get_single_ticker(ticker, start=start, end=end)
    df     = add_features(df_raw)
    X, y, feat_cols = prepare_xy(df)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    results  = {}
    test_preds = {}

    # ── 2. Train models ───────────────────────────────────────────────────────
    if "Random Forest" in models_to_use:
        if verbose: print("  🌲  Random Forest …")
        rf_pred, rf_model, _ = train_random_forest(
            X_train, y_train, X_test, feat_cols, company=company_label)
        rf_future = classical_future_forecast(
            rf_model, df, feat_cols, forecast_days, "rf")
        results["Random Forest"] = {
            "metrics": compute_metrics(y_test, rf_pred, "Random Forest"),
            "forecast": rf_future.tolist(),
        }
        test_preds["Random Forest"] = rf_pred

    if "XGBoost" in models_to_use:
        if verbose: print("  ⚡  XGBoost …")
        xgb_pred, xgb_model = train_xgboost(
            X_train, y_train, X_test, company=company_label)
        import xgboost as xgb_lib
        xgb_future = classical_future_forecast(
            xgb_model, df, feat_cols, forecast_days, "xgb")
        results["XGBoost"] = {
            "metrics": compute_metrics(y_test, xgb_pred, "XGBoost"),
            "forecast": xgb_future.tolist(),
        }
        test_preds["XGBoost"] = xgb_pred

    if "LSTM" in models_to_use:
        if verbose: print("  🧠  LSTM (may take ~60s) …")
        close_arr = df["Close"].values
        lstm_pred, lstm_model, lstm_scaler, lstm_n_train = train_lstm(
            close_arr, split=0.8, company=company_label)
        last_seq   = close_arr[-LSTM_LOOKBACK:]
        lstm_future = lstm_future_forecast(lstm_model, lstm_scaler,
                                           last_seq, forecast_days)
        actual_lstm = close_arr[lstm_n_train:lstm_n_train + len(lstm_pred)]
        results["LSTM"] = {
            "metrics": compute_metrics(actual_lstm, lstm_pred, "LSTM"),
            "forecast": lstm_future.tolist(),
        }
        test_preds["LSTM"] = lstm_pred

    # ── 3. Ensemble ───────────────────────────────────────────────────────────
    weights  = {"Random Forest": 0.25, "XGBoost": 0.35, "LSTM": 0.40}
    ensemble = weighted_ensemble(
        {k: np.array(v["forecast"]) for k, v in results.items()},
        weights={k: weights.get(k, 1/len(results)) for k in results},
    )

    # ── 4. Trend & confidence ─────────────────────────────────────────────────
    last_close = float(df["Close"].iloc[-1])
    ens_end    = float(ensemble[-1]) if len(ensemble) else last_close
    change_pct = (ens_end - last_close) / (last_close + 1e-9) * 100

    if change_pct >  2.0:
        trend = "bullish"
    elif change_pct < -2.0:
        trend = "bearish"
    else:
        trend = "neutral"

    # Confidence: mean R² across models (clipped 0–1)
    r2_vals = [v["metrics"]["R2"] for v in results.values()]
    confidence = float(np.clip(np.mean(r2_vals), 0, 1))

    # ── 5. Future dates ───────────────────────────────────────────────────────
    last_date    = df.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=forecast_days
    )

    # ── 6. Recommendation string ──────────────────────────────────────────────
    recommendation = _make_recommendation(trend, confidence, change_pct,
                                          company_label)

    output = {
        "ticker":            ticker,
        "name":              company_label,
        "last_close":        last_close,
        "trend":             trend,
        "change_pct":        round(change_pct, 2),
        "confidence":        round(confidence, 4),
        "metrics":           {k: v["metrics"] for k, v in results.items()},
        "forecast":          {k: v["forecast"] for k, v in results.items()},
        "ensemble_forecast": ensemble.tolist(),
        "future_dates":      [d.strftime("%Y-%m-%d") for d in future_dates],
        "recommendation":    recommendation,
    }

    if verbose:
        print(f"  ✅  Trend: {trend} | Change: {change_pct:+.2f}% | "
              f"Confidence: {confidence:.2%}")

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_recommendation(trend: str, confidence: float,
                          change_pct: float, name: str) -> str:
    if trend == "bullish" and confidence > 0.7:
        return (f"Our models show {name} on a strong upward trajectory "
                f"(+{change_pct:.1f}% over {30} days, confidence "
                f"{confidence:.0%}). This may be a good opportunity to "
                f"consider a long position or increase existing holdings. "
                f"Always pair predictions with your own risk assessment.")
    elif trend == "bullish":
        return (f"{name} shows a moderate bullish signal (+{change_pct:.1f}%). "
                f"Consider dollar-cost averaging rather than a lump-sum entry, "
                f"as confidence is moderate ({confidence:.0%}).")
    elif trend == "bearish" and confidence > 0.7:
        return (f"Models indicate a {abs(change_pct):.1f}% decline risk for "
                f"{name} over {30} days. Consider reducing exposure, "
                f"setting tighter stop-losses, or hedging your position.")
    elif trend == "bearish":
        return (f"{name} shows a mild bearish signal. Monitor closely and "
                f"avoid adding to positions until trend stabilises.")
    else:
        return (f"{name} is trending sideways (±{abs(change_pct):.1f}%). "
                f"Hold existing positions and wait for a clearer signal "
                f"before making new commitments.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InvestIQ Stock Predictor")
    parser.add_argument("--ticker", default="TCS.NS")
    parser.add_argument("--days",   type=int, default=30)
    parser.add_argument("--start",  default="2018-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    args = parser.parse_args()

    result = predict(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        forecast_days=args.days,
    )
    print("\n" + "="*60)
    print(json.dumps({
        "ticker":         result["ticker"],
        "trend":          result["trend"],
        "change_pct":     result["change_pct"],
        "confidence":     result["confidence"],
        "recommendation": result["recommendation"],
    }, indent=2))
