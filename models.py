"""
=============================================================================
  models.py — ML Model Training Functions
  AI Financial Advisory Chatbot — InvestIQ
=============================================================================
  Provides modular, cacheable training functions for:
    - Linear / Polynomial Regression
    - Random Forest Regressor
    - XGBoost Regressor (with early-stopping)
    - LSTM (Keras) — multi-step look-back sequence model
    - Ensemble (weighted average of RF + XGB + LSTM)

  Each trainer returns (predictions_on_test, fitted_model, optional_extras).
  Models are cached to disk via joblib / Keras save so they can be loaded
  without retraining.
=============================================================================
"""

import os
import warnings
import numpy as np
import joblib

from sklearn.linear_model    import LinearRegression
from sklearn.preprocessing   import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.ensemble        import RandomForestRegressor
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── Constants ─────────────────────────────────────────────────────────────────
POLY_DEGREE   = 3
RF_TREES      = 200
XGB_ROUNDS    = 400
LSTM_LOOKBACK = 60
RANDOM_STATE  = 42
MODEL_DIR     = os.path.join(os.path.expanduser("~"), "InvestIQ_Models")
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str = "") -> dict:
    """Return MAE, RMSE, R², MAPE% for a prediction pair."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) /
                                y_true[mask])) * 100)
    return {"Model": label, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE%": mape}


# ═══════════════════════════════════════════════════════════════════════════════
#  1. LINEAR / POLYNOMIAL REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def train_linear_poly(X_train, y_train, X_test, degree: int = POLY_DEGREE,
                      company: str = "model"):
    """
    Train a Polynomial → StandardScaler → LinearRegression pipeline.

    Returns
    -------
    (y_pred, pipeline)
    """
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg",    LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test).flatten()

    path = os.path.join(MODEL_DIR, f"{company}_poly.pkl")
    joblib.dump(pipe, path)

    return y_pred, pipe


# ═══════════════════════════════════════════════════════════════════════════════
#  2. RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════

def train_random_forest(X_train, y_train, X_test, feature_cols: list,
                        company: str = "model"):
    """
    Train a Random Forest regressor.

    Returns
    -------
    (y_pred, rf_model, feature_importances_series)
    """
    import pandas as pd
    rf = RandomForestRegressor(
        n_estimators  = RF_TREES,
        max_depth      = 14,
        min_samples_leaf = 2,
        max_features   = "sqrt",
        random_state   = RANDOM_STATE,
        n_jobs         = -1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test).flatten()
    importances = pd.Series(rf.feature_importances_, index=feature_cols)

    path = os.path.join(MODEL_DIR, f"{company}_rf.pkl")
    joblib.dump(rf, path)

    return y_pred, rf, importances


# ═══════════════════════════════════════════════════════════════════════════════
#  3. XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_test, company: str = "model"):
    """
    Train an XGBoost regressor with early-stopping on a validation split.

    Returns
    -------
    (y_pred, xgb_booster)
    """
    # 10% of training data → validation for early stopping
    val_idx   = int(len(X_train) * 0.9)
    X_tr, X_v = X_train[:val_idx], X_train[val_idx:]
    y_tr, y_v = y_train[:val_idx], y_train[val_idx:]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_v,  label=y_v)
    dtest  = xgb.DMatrix(X_test)

    params = {
        "objective":        "reg:squarederror",
        "learning_rate":    0.04,
        "max_depth":        6,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "seed":             RANDOM_STATE,
        "verbosity":        0,
    }
    model = xgb.train(
        params, dtrain,
        num_boost_round = XGB_ROUNDS,
        evals           = [(dtrain, "train"), (dval, "val")],
        early_stopping_rounds = 30,
        verbose_eval    = False,
    )
    y_pred = model.predict(dtest).flatten()

    path = os.path.join(MODEL_DIR, f"{company}_xgb.ubj")
    model.save_model(path)

    return y_pred, model


# ═══════════════════════════════════════════════════════════════════════════════
#  4. LSTM (Bidirectional)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sequences(series: np.ndarray, scaler: MinMaxScaler,
                     lookback: int):
    """Convert a 1-D price series → (X, y) sequence arrays."""
    scaled = scaler.transform(series.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y)


def train_lstm(close_prices: np.ndarray, split: float = 0.8,
               company: str = "model"):
    """
    Train a Bidirectional LSTM on closing prices.

    Returns
    -------
    (y_pred_unscaled, lstm_model, scaler, n_train_rows)
    """
    n_train = int(len(close_prices) * split)
    train_series = close_prices[:n_train]
    test_series  = close_prices[n_train - LSTM_LOOKBACK:]

    scaler = MinMaxScaler()
    scaler.fit(train_series.reshape(-1, 1))

    X_tr, y_tr = _build_sequences(train_series, scaler, LSTM_LOOKBACK)
    X_te, y_te = _build_sequences(test_series,  scaler, LSTM_LOOKBACK)

    X_tr = X_tr.reshape(-1, LSTM_LOOKBACK, 1)
    X_te = X_te.reshape(-1, LSTM_LOOKBACK, 1)

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True),
                      input_shape=(LSTM_LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="huber")

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    ]
    model.fit(
        X_tr, y_tr,
        epochs          = 80,
        batch_size      = 32,
        validation_split= 0.1,
        callbacks       = callbacks,
        verbose         = 0,
    )

    pred_scaled = model.predict(X_te, verbose=0)
    y_pred = scaler.inverse_transform(pred_scaled).flatten()

    model_path = os.path.join(MODEL_DIR, f"{company}_lstm.keras")
    model.save(model_path)

    return y_pred, model, scaler, n_train


# ═══════════════════════════════════════════════════════════════════════════════
#  5. FUTURE FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

def classical_future_forecast(model, df, feature_cols: list,
                               n_days: int, model_type: str = "rf") -> np.ndarray:
    """Iterative n-day-ahead forecast for classical (non-LSTM) models."""
    preds      = []
    last_row   = df[feature_cols].iloc[-1].values.copy().astype(float)
    last_close = float(df["Close"].iloc[-1])

    for _ in range(n_days):
        x = last_row.reshape(1, -1)
        if model_type == "poly":
            p = float(model.predict(x)[0])
        elif model_type == "xgb":
            p = float(model.predict(xgb.DMatrix(x))[0])
        else:
            p = float(model.predict(x)[0])
        preds.append(p)
        if "lag_1" in feature_cols:
            last_row[feature_cols.index("lag_1")] = last_close
        last_close = p

    return np.array(preds)


def lstm_future_forecast(model, scaler: MinMaxScaler,
                         last_sequence: np.ndarray, n_days: int) -> np.ndarray:
    """Auto-regressive n-day-ahead LSTM forecast."""
    seq  = scaler.transform(last_sequence.reshape(-1, 1))
    inp  = list(seq[-LSTM_LOOKBACK:, 0])
    outs = []
    for _ in range(n_days):
        x = np.array(inp[-LSTM_LOOKBACK:]).reshape(1, LSTM_LOOKBACK, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        outs.append(p)
        inp.append(p)
    return scaler.inverse_transform(np.array(outs).reshape(-1, 1)).flatten()


# ═══════════════════════════════════════════════════════════════════════════════
#  6. ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def weighted_ensemble(predictions: dict, weights: dict = None) -> np.ndarray:
    """
    Compute a weighted average ensemble from multiple model predictions.

    Parameters
    ----------
    predictions : dict[str, np.ndarray]
        e.g. {"Random Forest": arr, "XGBoost": arr, "LSTM": arr}
    weights : dict[str, float] | None
        If None, equal weights are used.

    Returns
    -------
    np.ndarray  ensemble prediction
    """
    keys = list(predictions.keys())
    if weights is None:
        weights = {k: 1.0 / len(keys) for k in keys}

    min_len = min(len(v) for v in predictions.values())
    result  = np.zeros(min_len)
    total_w = sum(weights.get(k, 0) for k in keys)

    for k in keys:
        w = weights.get(k, 0) / total_w
        result += w * predictions[k][:min_len]

    return result


# ── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("models.py — import this module to use the training functions.")
    print(f"Model cache → {MODEL_DIR}")
