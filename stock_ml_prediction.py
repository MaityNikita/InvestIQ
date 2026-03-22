"""
=============================================================================
  ML-BASED STOCK PRICE PREDICTION — IT Sector (NSE India)
  Models: LSTM · Random Forest · XGBoost · Linear/Polynomial Regression
  Output: Forecast Charts + Model Accuracy Comparison
=============================================================================

  INSTALL DEPENDENCIES (run once in terminal / Spyder console):
      pip install yfinance pandas numpy matplotlib seaborn scikit-learn
      pip install xgboost tensorflow keras

  HOW IT WORKS:
  1. Downloads historical data for TCS, Infosys, Wipro, HCL Tech,
     Tech Mahindra and NIFTY IT via yfinance
  2. Engineers features: lag prices, moving averages, RSI, volatility
  3. Trains 4 ML models per company on 80% data
  4. Evaluates on 20% test data (MAE, RMSE, R², MAPE)
  5. Plots: Actual vs Predicted, 30-day Future Forecast, Model Comparison

  OUTPUT FILES saved to ~/StockML_Results/
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.ensemble      import RandomForestRegressor
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models   import Sequential
from tensorflow.keras.layers   import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── Configuration ─────────────────────────────────────────────────────────────
COMPANIES = {
    "TCS":           "TCS.NS",
    "Infosys":       "INFY.NS",
    "Wipro":         "WIPRO.NS",
    "HCL Tech":      "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "NIFTY IT":      "^CNXIT",
}

PALETTE = {
    "TCS":           "#00C9FF",
    "Infosys":       "#FF6B6B",
    "Wipro":         "#FFD93D",
    "HCL Tech":      "#6BCB77",
    "Tech Mahindra": "#C77DFF",
    "NIFTY IT":      "#FF9F1C",
}

MODEL_COLORS = {
    "Linear/Poly Reg": "#00E5FF",
    "Random Forest":   "#69FF47",
    "XGBoost":         "#FF6B6B",
    "LSTM":            "#FFD93D",
    "Actual":          "#FFFFFF",
}

START        = "2018-01-01"
END          = "2024-12-31"
FORECAST_DAYS = 30
LSTM_LOOKBACK = 60          # days of history LSTM sees at once
POLY_DEGREE   = 3
RF_TREES      = 200
XGB_ROUNDS    = 300
RANDOM_STATE  = 42

OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "StockML_Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def mape(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def metrics(y_true, y_pred, label=""):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    # Make sure same length
    min_len = min(len(y_true), len(y_pred))
    y_true  = y_true[:min_len]
    y_pred  = y_pred[:min_len]
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mp   = mape(y_true, y_pred)
    return {"Model": label, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE%": mp}

def dark_fig(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    fig.patch.set_facecolor("#0D1117")
    return fig

def dark_ax(ax):
    ax.set_facecolor("#141B24")
    for spine in ax.spines.values():
        spine.set_color("#2a2a2a")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.grid(color="#1e2433", linewidth=0.6, linestyle="--")
    return ax


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

def download_data():
    import yfinance as yf
    all_data = {}
    print("\n📡  Downloading historical data …")
    for name, ticker in COMPANIES.items():
        df = yf.download(ticker, start=START, end=END,
                         progress=False, auto_adjust=True)
        if df.empty:
            print(f"  ⚠️  {name} — no data, skipping.")
            continue
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.dropna(inplace=True)
        all_data[name] = df
        print(f"  ✅  {name:15s} — {len(df)} rows")
    return all_data


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def add_features(df):
    d = df.copy()
    c = d["Close"]

    # Lag features
    for lag in [1, 3, 5, 10, 20]:
        d[f"lag_{lag}"] = c.shift(lag)

    # Moving averages
    for w in [7, 14, 21, 50]:
        d[f"ma_{w}"] = c.rolling(w).mean()

    # Exponential moving averages
    d["ema_12"] = c.ewm(span=12).mean()
    d["ema_26"] = c.ewm(span=26).mean()
    d["macd"]   = d["ema_12"] - d["ema_26"]

    # Volatility
    d["volatility_10"] = c.pct_change().rolling(10).std() * np.sqrt(252)
    d["volatility_20"] = c.pct_change().rolling(20).std() * np.sqrt(252)

    # RSI (14-day)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # Price range & momentum
    d["high_low_range"] = (df["High"] - df["Low"]) / c
    d["daily_return"]   = c.pct_change()
    d["momentum_5"]     = c / c.shift(5) - 1
    d["momentum_20"]    = c / c.shift(20) - 1

    # Day / month / year numerics
    d["day_of_week"] = d.index.dayofweek
    d["month"]       = d.index.month
    d["year"]        = d.index.year

    d.dropna(inplace=True)
    return d


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — CLASSICAL ML MODELS (Linear/Poly, RF, XGB)
# ═══════════════════════════════════════════════════════════════════════════

def prepare_xy(df):
    feature_cols = [c for c in df.columns if c not in
                    ["Open","High","Low","Close","Volume"]]
    X = df[feature_cols].values
    y = df["Close"].values
    return X, y, feature_cols

def train_linear_poly(X_train, y_train, X_test):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg",  LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test).flatten(), pipe

def train_rf(X_train, y_train, X_test, feature_cols):
    model = RandomForestRegressor(n_estimators=RF_TREES,
                                   max_depth=12,
                                   min_samples_leaf=3,
                                   random_state=RANDOM_STATE,
                                   n_jobs=-1)
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    return model.predict(X_test).flatten(), model, importances

def train_xgb(X_train, y_train, X_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test)
    params = {
        "objective":        "reg:squarederror",
        "learning_rate":    0.05,
        "max_depth":        6,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "seed":             RANDOM_STATE,
        "verbosity":        0,
    }
    model = xgb.train(params, dtrain, num_boost_round=XGB_ROUNDS,
                      evals=[(dtrain, "train")], verbose_eval=False)
    return model.predict(dtest), model


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4 — LSTM MODEL
# ═══════════════════════════════════════════════════════════════════════════

def build_lstm_dataset(series, lookback):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

def train_lstm(close_prices, split=0.8):
    n_train   = int(len(close_prices) * split)
    train_ser = close_prices[:n_train]
    test_ser  = close_prices[n_train - LSTM_LOOKBACK:]

    X_tr, y_tr, scaler = build_lstm_dataset(train_ser, LSTM_LOOKBACK)
    X_te, y_te, _      = build_lstm_dataset(test_ser,  LSTM_LOOKBACK)

    X_tr = X_tr.reshape(-1, LSTM_LOOKBACK, 1)
    X_te = X_te.reshape(-1, LSTM_LOOKBACK, 1)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(LSTM_LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber")
    es = EarlyStopping(patience=8, restore_best_weights=True, verbose=0)
    model.fit(X_tr, y_tr, epochs=60, batch_size=32,
              validation_split=0.1, callbacks=[es], verbose=0)

    pred_scaled = model.predict(X_te, verbose=0)
    pred = scaler.inverse_transform(pred_scaled).flatten()
    return pred, model, scaler, n_train

def lstm_future_forecast(model, scaler, last_sequence, n_days):
    seq = last_sequence.copy().reshape(-1, 1)
    seq = scaler.transform(seq)
    preds = []
    inp   = list(seq[-LSTM_LOOKBACK:, 0])
    for _ in range(n_days):
        x = np.array(inp[-LSTM_LOOKBACK:]).reshape(1, LSTM_LOOKBACK, 1)
        p = model.predict(x, verbose=0)[0, 0]
        preds.append(p)
        inp.append(p)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5 — FUTURE FORECAST (classical models via last known features)
# ═══════════════════════════════════════════════════════════════════════════

def classical_future(model, df, feature_cols, n_days, model_type="rf"):
    """
    Simple iterative forecast: repeat last feature row, update lag/price.
    """
    preds = []
    last_row = df[feature_cols].iloc[-1].values.copy()
    last_close = df["Close"].iloc[-1]

    for _ in range(n_days):
        x = last_row.reshape(1, -1)
        if model_type == "poly":
            p = float(model.predict(x)[0])
        elif model_type == "xgb":
            p = float(model.predict(xgb.DMatrix(x))[0])
        else:
            p = float(model.predict(x)[0])
        preds.append(p)
        # Update lag_1 in last_row (index of lag_1 in feature_cols)
        if "lag_1" in feature_cols:
            idx = feature_cols.index("lag_1")
            last_row[idx] = last_close
        last_close = p
    return np.array(preds)


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 6 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_predictions(company, df, results, n_train, future_dates):
    """
    One big figure per company:
      Row 1: Actual vs Predicted (test period) — 4 models
      Row 2: 30-day future forecast
    """
    fig = dark_fig(figsize=(20, 10))
    fig.suptitle(f"  {company} — ML Stock Price Prediction",
                 color="white", fontsize=15, fontweight="bold", y=1.0)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    test_index = df.index[n_train:]
    actual_test = df["Close"].iloc[n_train:].values

    model_list = ["Linear/Poly Reg", "Random Forest", "XGBoost", "LSTM"]

    for i, mname in enumerate(model_list):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        dark_ax(ax)

        pred = results[mname]["test_pred"]
        # Align lengths
        min_len = min(len(test_index), len(actual_test), len(pred))
        t_idx = test_index[:min_len]
        act   = actual_test[:min_len]
        pr    = pred[:min_len]

        ax.plot(t_idx, act, color=MODEL_COLORS["Actual"],
                linewidth=1.4, alpha=0.85, label="Actual")
        ax.plot(t_idx, pr,  color=MODEL_COLORS[mname],
                linewidth=1.4, alpha=0.85, linestyle="--", label=mname)

        # Error fill
        ax.fill_between(t_idx, act, pr,
                        alpha=0.12, color=MODEL_COLORS[mname])

        m = results[mname]["metrics"]
        ax.set_title(f"{mname}  |  R²={m['R2']:.3f}  MAPE={m['MAPE%']:.1f}%",
                     color=MODEL_COLORS[mname], fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Price (₹)", color="#888", fontsize=8)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
        ax.legend(frameon=False, labelcolor="white", fontsize=8, loc="upper left")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{company.replace(' ','_')}_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    print(f"  💾  Saved → {path}")
    plt.show()


def plot_forecast(company, df, results, future_dates):
    """
    Combined 30-day forecast from all models on one chart.
    """
    last_actual = df["Close"].tail(120)

    fig, ax = plt.subplots(figsize=(14, 6))
    dark_ax(ax)
    fig.patch.set_facecolor("#0D1117")

    ax.plot(last_actual.index, last_actual.values,
            color="white", linewidth=2.0, alpha=0.9, label="Historical (last 120d)")
    ax.axvline(df.index[-1], color="#444", linewidth=1.0, linestyle=":")

    for mname in ["Linear/Poly Reg", "Random Forest", "XGBoost", "LSTM"]:
        fp = results[mname].get("future_pred")
        if fp is None:
            continue
        ax.plot(future_dates, fp,
                color=MODEL_COLORS[mname],
                linewidth=2.0, linestyle="--",
                alpha=0.9, marker="o", markersize=2,
                label=f"{mname} Forecast")

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.set_title(f"{company} — 30-Day Price Forecast",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", color="#888")
    ax.set_ylabel("Price (₹)", color="#888")
    ax.legend(frameon=False, labelcolor="white", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{company.replace(' ','_')}_forecast.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    print(f"  💾  Saved → {path}")
    plt.show()


def plot_accuracy_comparison(all_metrics_df):
    """
    Grid of 4 metric bar charts across all companies & models.
    """
    metric_list = [("MAE", "Lower is Better ↓"),
                   ("RMSE","Lower is Better ↓"),
                   ("R2",  "Higher is Better ↑"),
                   ("MAPE%","Lower is Better ↓")]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Model Accuracy Comparison — All Companies & Models",
                 color="white", fontsize=14, fontweight="bold")

    model_order  = ["Linear/Poly Reg","Random Forest","XGBoost","LSTM"]
    m_colors     = [MODEL_COLORS[m] for m in model_order]

    for idx, (metric, subtitle) in enumerate(metric_list):
        ax = axes[idx // 2][idx % 2]
        dark_ax(ax)

        pivot = all_metrics_df.pivot(index="Company",
                                     columns="Model",
                                     values=metric)
        pivot = pivot.reindex(columns=model_order)

        x      = np.arange(len(pivot))
        width  = 0.18
        offsets= np.linspace(-1.5, 1.5, 4) * width

        for j, (mname, col) in enumerate(zip(model_order, m_colors)):
            if mname not in pivot.columns:
                continue
            ax.bar(x + offsets[j], pivot[mname].values,
                   width=width, color=col, alpha=0.85,
                   label=mname, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=20,
                           ha="right", color="#aaa", fontsize=8)
        ax.set_title(f"{metric}  ({subtitle})",
                     color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel(metric, color="#888", fontsize=9)
        ax.legend(frameon=False, labelcolor="white",
                  fontsize=8, loc="upper right")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    print(f"  💾  Saved → {path}")
    plt.show()


def plot_feature_importance(importances_dict):
    """Random Forest feature importances per company."""
    n    = len(importances_dict)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("Random Forest — Top Feature Importances",
                 color="white", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for idx, (company, imp) in enumerate(importances_dict.items()):
        ax = axes[idx]
        dark_ax(ax)
        top = imp.nlargest(12)
        color = PALETTE.get(company, "#00C9FF")
        bars  = ax.barh(top.index[::-1], top.values[::-1],
                        color=color, alpha=0.85)
        ax.set_title(company, color=color, fontsize=10, fontweight="bold")
        ax.set_xlabel("Importance", color="#888", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7.5)

    for i in range(len(importances_dict), len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    print(f"  💾  Saved → {path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run():
    print("\n" + "="*65)
    print("   ML STOCK PREDICTION — IT SECTOR (Spyder)")
    print("="*65)

    # ── 1. Download ──────────────────────────────────────────────────────
    raw_data = download_data()
    if not raw_data:
        raise RuntimeError("No data fetched. Check internet connection.")

    all_metrics   = []
    importances_d = {}

    for company, df_raw in raw_data.items():
        print(f"\n{'─'*55}")
        print(f"  🔧  Training models for: {company}")
        print(f"{'─'*55}")

        # ── 2. Features ──────────────────────────────────────────────────
        df = add_features(df_raw)
        X, y, feat_cols = prepare_xy(df)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        n_train = split_idx + (len(df_raw) - len(df))   # account for NaN rows dropped

        results = {}

        # ── 3a. Linear/Poly Regression ───────────────────────────────────
        print("  📐  Linear / Polynomial Regression …")
        poly_pred, poly_model = train_linear_poly(X_train, y_train, X_test)
        results["Linear/Poly Reg"] = {
            "test_pred":   poly_pred,
            "metrics":     metrics(y_test, poly_pred, "Linear/Poly Reg"),
            "future_pred": classical_future(poly_model, df, feat_cols,
                                            FORECAST_DAYS, "poly")
        }
        m = results["Linear/Poly Reg"]["metrics"]
        print(f"     MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
              f"R²={m['R2']:.4f}  MAPE={m['MAPE%']:.2f}%")

        # ── 3b. Random Forest ─────────────────────────────────────────────
        print("  🌲  Random Forest …")
        rf_pred, rf_model, imp = train_rf(X_train, y_train, X_test, feat_cols)
        results["Random Forest"] = {
            "test_pred":   rf_pred,
            "metrics":     metrics(y_test, rf_pred, "Random Forest"),
            "future_pred": classical_future(rf_model, df, feat_cols,
                                            FORECAST_DAYS, "rf")
        }
        importances_d[company] = imp
        m = results["Random Forest"]["metrics"]
        print(f"     MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
              f"R²={m['R2']:.4f}  MAPE={m['MAPE%']:.2f}%")

        # ── 3c. XGBoost ───────────────────────────────────────────────────
        print("  ⚡  XGBoost …")
        xgb_pred, xgb_model = train_xgb(X_train, y_train, X_test)
        results["XGBoost"] = {
            "test_pred":   xgb_pred,
            "metrics":     metrics(y_test, xgb_pred, "XGBoost"),
            "future_pred": classical_future(xgb_model, df, feat_cols,
                                            FORECAST_DAYS, "xgb")
        }
        m = results["XGBoost"]["metrics"]
        print(f"     MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
              f"R²={m['R2']:.4f}  MAPE={m['MAPE%']:.2f}%")

        # ── 3d. LSTM ──────────────────────────────────────────────────────
        print("  🧠  LSTM (this may take ~30–60s) …")
        close_arr     = df["Close"].values
        lstm_pred, lstm_model, lstm_scaler, lstm_n_train = \
            train_lstm(close_arr, split=0.8)
        last_seq      = close_arr[-LSTM_LOOKBACK:]
        lstm_future   = lstm_future_forecast(lstm_model, lstm_scaler,
                                             last_seq, FORECAST_DAYS)
        # Align with test set
        actual_lstm   = close_arr[lstm_n_train:lstm_n_train + len(lstm_pred)]
        results["LSTM"] = {
            "test_pred":   lstm_pred,
            "metrics":     metrics(actual_lstm, lstm_pred, "LSTM"),
            "future_pred": lstm_future
        }
        m = results["LSTM"]["metrics"]
        print(f"     MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
              f"R²={m['R2']:.4f}  MAPE={m['MAPE%']:.2f}%")

        # ── 4. Collect metrics ────────────────────────────────────────────
        for mname, res in results.items():
            row = res["metrics"].copy()
            row["Company"] = company
            all_metrics.append(row)

        # ── 5. Future dates ───────────────────────────────────────────────
        last_date    = df.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                      periods=FORECAST_DAYS)

        # ── 6. Per-company plots ──────────────────────────────────────────
        print(f"\n  📊  Plotting {company} …")
        plot_predictions(company, df, results, split_idx, future_dates)
        plot_forecast(company, df, results, future_dates)

    # ── 7. Global comparison charts ──────────────────────────────────────────
    print("\n📊  Generating global accuracy comparison …")
    all_metrics_df = pd.DataFrame(all_metrics)
    plot_accuracy_comparison(all_metrics_df)
    plot_feature_importance(importances_d)

    # ── 8. Summary table ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  📋  FINAL MODEL SUMMARY")
    print("="*65)
    summary = all_metrics_df.pivot_table(
        index="Model", values=["MAE","RMSE","R2","MAPE%"],
        aggfunc="mean").round(3)
    print(summary.to_string())

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    all_metrics_df.to_csv(csv_path, index=False)

    print(f"\n✅  All outputs saved to: {OUTPUT_DIR}")
    print("="*65 + "\n")
    return all_metrics_df

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results_df = run()
