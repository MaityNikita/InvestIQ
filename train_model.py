"""
=============================================================================
  train_model.py — Model Retraining Entry Point
  AI Financial Advisory Chatbot — InvestIQ
=============================================================================
  Retrains all ML models for all configured IT-sector companies and saves
  them to ~/InvestIQ_Models/.

  Usage:
      python train_model.py
      python train_model.py --companies TCS INFY --start 2019-01-01
      python train_model.py --skip-lstm   (faster, skips LSTM training)
=============================================================================
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from data_ingestion      import download_data, DEFAULT_COMPANIES
from feature_engineering import add_features, prepare_xy, get_feature_columns
from models              import (
    train_linear_poly, train_random_forest, train_xgboost, train_lstm,
    classical_future_forecast, lstm_future_forecast,
    compute_metrics, MODEL_DIR, LSTM_LOOKBACK,
)


def retrain_all(
    companies:    dict  = None,
    start:        str   = "2018-01-01",
    end:          str   = "2024-12-31",
    forecast_days: int  = 30,
    skip_lstm:    bool  = False,
    verbose:      bool  = True,
) -> pd.DataFrame:
    """
    Download data, engineer features, train all models, and save artefacts.

    Returns a DataFrame of all model metrics for reporting.
    """
    if companies is None:
        companies = DEFAULT_COMPANIES

    print("\n" + "="*65)
    print("  🏋️  InvestIQ — Model Training Pipeline")
    print("="*65)

    # ── 1. Download ───────────────────────────────────────────────────────────
    raw_data = download_data(
        companies={k: v for k, v in DEFAULT_COMPANIES.items()
                   if k in companies} if isinstance(companies, list)
        else companies,
        start=start,
        end=end,
        verbose=verbose,
    )

    all_metrics = []
    t0 = time.time()

    for company, df_raw in raw_data.items():
        print(f"\n{'─'*55}")
        print(f"  🔧  Training: {company}")
        print(f"{'─'*55}")

        # ── 2. Features ───────────────────────────────────────────────────────
        df         = add_features(df_raw)
        X, y, feat = prepare_xy(df)
        split      = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        # Future dates
        future_dates = pd.bdate_range(
            start=df.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days
        )

        # ── 3a. Linear / Polynomial ───────────────────────────────────────────
        print("  📐  Linear/Poly Regression …", end=" ", flush=True)
        p_pred, _ = train_linear_poly(X_tr, y_tr, X_te, company=company)
        m = compute_metrics(y_te, p_pred, "Linear/Poly")
        m["Company"] = company
        all_metrics.append(m)
        print(f"R²={m['R2']:.4f}  MAPE={m['MAPE%']:.2f}%")

        # ── 3b. Random Forest ─────────────────────────────────────────────────
        print("  🌲  Random Forest …", end=" ", flush=True)
        rf_pred, _, importances = train_random_forest(
            X_tr, y_tr, X_te, feat, company=company)
        m = compute_metrics(y_te, rf_pred, "Random Forest")
        m["Company"] = company
        all_metrics.append(m)
        print(f"R²={m['R2']:.4f}  MAPE={m['MAPE%']:.2f}%")

        # ── 3c. XGBoost ───────────────────────────────────────────────────────
        print("  ⚡  XGBoost …", end=" ", flush=True)
        xgb_pred, _ = train_xgboost(X_tr, y_tr, X_te, company=company)
        m = compute_metrics(y_te, xgb_pred, "XGBoost")
        m["Company"] = company
        all_metrics.append(m)
        print(f"R²={m['R2']:.4f}  MAPE={m['MAPE%']:.2f}%")

        # ── 3d. LSTM ──────────────────────────────────────────────────────────
        if not skip_lstm:
            print("  🧠  LSTM (Bidirectional, ~60s) …", end=" ", flush=True)
            close_arr = df["Close"].values
            lstm_pred, _, _, lstm_n_train = train_lstm(
                close_arr, split=0.8, company=company)
            actual_lstm = close_arr[lstm_n_train:lstm_n_train + len(lstm_pred)]
            m = compute_metrics(actual_lstm, lstm_pred, "LSTM")
            m["Company"] = company
            all_metrics.append(m)
            print(f"R²={m['R2']:.4f}  MAPE={m['MAPE%']:.2f}%")
        else:
            print("  🧠  LSTM — skipped (--skip-lstm flag set)")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    metrics_df = pd.DataFrame(all_metrics)

    print("\n" + "="*65)
    print(f"  📋  TRAINING COMPLETE  ({elapsed:.1f}s)")
    print("="*65)

    summary = (
        metrics_df.groupby("Model")[["MAE","RMSE","R2","MAPE%"]]
        .mean()
        .round(3)
    )
    print(summary.to_string())

    # Save metrics CSV
    csv_path = os.path.join(MODEL_DIR, "training_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  💾  Metrics saved → {csv_path}")
    print(f"  📁  Models saved  → {MODEL_DIR}")
    print("="*65 + "\n")

    return metrics_df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InvestIQ — Retrain all models")
    parser.add_argument(
        "--companies", nargs="*", default=None,
        help="Company names to train (default: all). E.g. TCS Infosys"
    )
    parser.add_argument("--start",     default="2018-01-01")
    parser.add_argument("--end",       default="2024-12-31")
    parser.add_argument("--days",      type=int, default=30,
                        help="Forecast horizon (days)")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM training (much faster)")
    args = parser.parse_args()

    retrain_all(
        companies    = args.companies,
        start        = args.start,
        end          = args.end,
        forecast_days= args.days,
        skip_lstm    = args.skip_lstm,
    )
