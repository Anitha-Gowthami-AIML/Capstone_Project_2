"""
train_models.py
----------------
End-to-end training pipeline for the Melbourne daily minimum temperature
project. Trains every model that appeared in the original notebook
(ARIMA, SARIMA, SimpleRNN, LSTM, GRU, CNN+RNN hybrid, Prophet, NeuralProphet),
evaluates each one on a chronological train/test split, and writes all
results to ./artifacts so the Streamlit app (app.py) can load them instantly
without re-training anything.

Run:
    python train_models.py

Notes on the NeuralProphet error from the source notebook
-----------------------------------------------------------
The original notebook crashed/behaved inconsistently around NeuralProphet
because of three separate problems, all fixed below:

1. `!pip install pandas==1.5.3` / `!pip install neuralprophet==0.6.2` were run
   in the MIDDLE of a live kernel, AFTER pandas/numpy had already been
   imported and used by every other model. Re-installing a package mid-session
   does nothing until the kernel restarts, so the "fix" never actually took
   effect and instead left the environment in a half-upgraded, half-downgraded
   state. -> We simply pin compatible versions in requirements.txt *before*
   the process starts, and never reinstall packages at runtime.
2. Newer NumPy removed `np.NaN` (capital N), which old NeuralProphet/Prophet
   internals reference. -> We keep the `np.NaN = np.nan` shim, applied once,
   at import time, before NeuralProphet is imported.
3. The notebook left NeuralProphet's default `n_lags=0`, which turns it into
   a pure curve-fitting model (trend + seasonality only, like Prophet) and
   silently ignores autoregression -- not wrong, but not "neural" in any
   meaningful sense and easy to confuse with a bug when results look just
   like Prophet's. -> We explicitly set `n_lags=window_size` so the model
   actually uses an AR-Net on past observations, and we turn off the
   ipywidgets progress bar (`progress=None`) so the script runs cleanly
   outside of Jupyter.
"""

import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

# --- NeuralProphet / Prophet compatibility shim (must run before they import numpy internals) ---
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from data_utils import load_clean_data, train_test_split_series, create_sequences, TARGET_COL

ARTIFACT_DIR = "artifacts"
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
WINDOW_SIZE = 30
EPOCHS = 60          # trimmed from the notebook's 500 -- converges well before that on this data
BATCH_SIZE = 32
RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)


def metrics_dict(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 1. Load + EDA artifacts (stationarity test, decomposition) -- used by the
#    "Data Exploration" tab in the Streamlit app.
# ----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller

    log("Running EDA: stationarity test + seasonal decomposition")
    adf_result = adfuller(df[TARGET_COL].dropna())
    adf_summary = {
        "ADF_statistic": float(adf_result[0]),
        "p_value": float(adf_result[1]),
        "is_stationary": bool(adf_result[1] < 0.05),
    }

    decomp = seasonal_decompose(df[TARGET_COL], model="additive", period=365)
    decomp_df = pd.DataFrame(
        {
            "observed": decomp.observed,
            "trend": decomp.trend,
            "seasonal": decomp.seasonal,
            "resid": decomp.resid,
        }
    )
    decomp_df.to_csv(os.path.join(ARTIFACT_DIR, "decomposition.csv"))

    with open(os.path.join(ARTIFACT_DIR, "adf_test.json"), "w") as f:
        json.dump(adf_summary, f, indent=2)

    df.to_csv(os.path.join(ARTIFACT_DIR, "clean_data.csv"))
    return adf_summary


# ----------------------------------------------------------------------------
# 2. Classical statistical models: ARIMA + SARIMA
# ----------------------------------------------------------------------------
def run_arima_family(train_data, test_data, results, predictions, run_sarima=False):
    from statsmodels.tsa.arima.model import ARIMA

    log("Training ARIMA(1,0,2) [best small-grid order found in EDA]")
    order = (1, 0, 2)
    model = ARIMA(train_data[TARGET_COL], order=order)
    fit = model.fit()

    train_pred = fit.predict(start=0, end=len(train_data) - 1)
    test_pred = fit.forecast(steps=len(test_data))

    results["ARIMA"] = {
        "description": f"ARIMA{order} fit by maximum likelihood on the training set.",
        "params": {"order": order, "AIC": float(fit.aic), "BIC": float(fit.bic)},
        "train": metrics_dict(train_data[TARGET_COL], train_pred),
        "test": metrics_dict(test_data[TARGET_COL], test_pred),
    }
    predictions["ARIMA"] = {
        "train": pd.Series(train_pred.values, index=train_data.index),
        "test": pd.Series(test_pred.values, index=test_data.index),
    }
    joblib.dump(fit, os.path.join(MODEL_DIR, "arima.joblib"))

    if not run_sarima:
        log("Skipping SARIMA by default -- a full period=365 seasonal SARIMAX "
            "is extremely slow on 4 years of daily data (can take 30+ minutes). "
            "Pass run_sarima=True to train_models.main() / run_arima_family() "
            "if you want it included.")
        return

    log("Training SARIMA(1,0,1)x(1,1,1,365)... (this is the slowest model, please wait)")
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        sorder = (1, 0, 1)
        seasonal_order = (1, 1, 1, 365)
        smodel = SARIMAX(
            train_data[TARGET_COL],
            order=sorder,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sfit = smodel.fit(disp=False)
        strain_pred = sfit.predict(start=0, end=len(train_data) - 1)
        stest_pred = sfit.forecast(steps=len(test_data))

        results["SARIMA"] = {
            "description": f"SARIMA{sorder}x{seasonal_order} -- adds an explicit yearly seasonal term on top of ARIMA.",
            "params": {"order": sorder, "seasonal_order": seasonal_order, "AIC": float(sfit.aic)},
            "train": metrics_dict(train_data[TARGET_COL], strain_pred),
            "test": metrics_dict(test_data[TARGET_COL], stest_pred),
        }
        predictions["SARIMA"] = {
            "train": pd.Series(strain_pred.values, index=train_data.index),
            "test": pd.Series(stest_pred.values, index=test_data.index),
        }
        joblib.dump(sfit, os.path.join(MODEL_DIR, "sarima.joblib"))
    except Exception as e:
        log(f"SARIMA failed/skipped ({e}); continuing without it.")


# ----------------------------------------------------------------------------
# 3. Deep learning models: SimpleRNN, LSTM, GRU, CNN+RNN hybrid
# ----------------------------------------------------------------------------
def build_dl_models(window_size):
    from keras.models import Sequential
    from keras.layers import (
        LSTM, GRU, SimpleRNN, Dense, Conv1D, MaxPooling1D,
    )

    archs = {}

    archs["RNN"] = Sequential([
        SimpleRNN(64, activation="tanh", input_shape=(window_size, 1)),
        Dense(1),
    ])

    archs["LSTM"] = Sequential([
        LSTM(64, activation="tanh", input_shape=(window_size, 1)),
        Dense(1),
    ])

    archs["GRU"] = Sequential([
        GRU(64, activation="tanh", input_shape=(window_size, 1)),
        Dense(1),
    ])

    archs["CNN+RNN Hybrid"] = Sequential([
        Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(window_size, 1)),
        MaxPooling1D(pool_size=2),
        SimpleRNN(64, activation="tanh"),
        Dense(1),
    ])

    for m in archs.values():
        m.compile(optimizer="adam", loss="mse")
    return archs


def run_deep_learning(train_data, test_data, results, predictions, window_size=WINDOW_SIZE):
    from keras.callbacks import EarlyStopping

    log("Building sliding-window sequences for the deep learning models")
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data[[TARGET_COL]]).flatten()
    test_scaled = scaler.transform(test_data[[TARGET_COL]]).flatten()
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test, y_test = create_sequences(
        np.concatenate([train_scaled[-window_size:], test_scaled]), window_size
    )
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Dates aligned to each prediction (predictions start window_size days
    # into each split, since the first window_size points are only ever used
    # as input context, never as a target).
    train_dates = train_data.index[window_size:]
    test_dates = test_data.index

    models = build_dl_models(window_size)
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    descriptions = {
        "RNN": "A SimpleRNN reads the last 30 days one day at a time, carrying a hidden state forward, and predicts day 31.",
        "LSTM": "An LSTM is a SimpleRNN with gated memory cells, designed to retain useful signal over longer windows without vanishing gradients.",
        "GRU": "A GRU is a lighter-weight gated alternative to LSTM with fewer parameters but similar long-range memory behaviour.",
        "CNN+RNN Hybrid": "A 1-D convolution first extracts local shape features (e.g. short warm/cold streaks) from the 30-day window, then a SimpleRNN summarizes those features over time before predicting day 31.",
    }

    for name, model in models.items():
        log(f"Training {name} ...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            verbose=0, validation_split=0.1, callbacks=[es],
        )

        train_pred_scaled = model.predict(X_train, verbose=0).flatten()
        test_pred_scaled = model.predict(X_test, verbose=0).flatten()

        # Inverse-transform back to real degrees Celsius.
        train_pred = scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
        test_pred = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
        y_train_real = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        results[name] = {
            "description": descriptions[name],
            "params": {
                "window_size": window_size,
                "epochs_run": len(history.history["loss"]),
                "final_train_loss": float(history.history["loss"][-1]),
                "final_val_loss": float(history.history["val_loss"][-1]),
            },
            "train": metrics_dict(y_train_real, train_pred),
            "test": metrics_dict(y_test_real, test_pred),
            "loss_curve": [float(x) for x in history.history["loss"]],
            "val_loss_curve": [float(x) for x in history.history["val_loss"]],
        }
        predictions[name] = {
            "train": pd.Series(train_pred, index=train_dates),
            "test": pd.Series(test_pred, index=test_dates),
        }
        model.save(os.path.join(MODEL_DIR, f"{name.replace('+', '_').replace(' ', '_')}.keras"))


# ----------------------------------------------------------------------------
# 4. Prophet
# ----------------------------------------------------------------------------
def run_prophet(train_data, test_data, results, predictions):
    from prophet import Prophet

    log("Training Prophet ...")
    train_df = train_data.reset_index().rename(columns={"Date": "ds", TARGET_COL: "y"})
    test_df = test_data.reset_index().rename(columns={"Date": "ds", TARGET_COL: "y"})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df), freq="D")
    forecast = model.predict(future)
    forecast = forecast.set_index("ds")

    train_pred = forecast.loc[train_df["ds"], "yhat"]
    test_pred = forecast.loc[test_df["ds"], "yhat"]

    results["Prophet"] = {
        "description": "Facebook/Meta Prophet decomposes the series into trend + yearly seasonality using Fourier terms and changepoints, fit with a Bayesian curve-fitting procedure (no autoregression).",
        "params": {"yearly_fourier_terms": 10},
        "train": metrics_dict(train_df["y"], train_pred.values),
        "test": metrics_dict(test_df["y"], test_pred.values),
    }
    predictions["Prophet"] = {
        "train": pd.Series(train_pred.values, index=train_data.index),
        "test": pd.Series(test_pred.values, index=test_data.index),
    }
    components = forecast[["trend", "yearly"]] if "yearly" in forecast.columns else forecast[["trend"]]
    components.to_csv(os.path.join(ARTIFACT_DIR, "prophet_components.csv"))
    joblib.dump(model, os.path.join(MODEL_DIR, "prophet.joblib"))


# ----------------------------------------------------------------------------
# 5. NeuralProphet  (fixed version -- see module docstring)
# ----------------------------------------------------------------------------
def run_neuralprophet(train_data, test_data, results, predictions, window_size=WINDOW_SIZE):
    from neuralprophet import NeuralProphet

    log("Training NeuralProphet (with autoregression enabled) ...")
    train_df = train_data.reset_index().rename(columns={"Date": "ds", TARGET_COL: "y"})
    test_df = test_data.reset_index().rename(columns={"Date": "ds", TARGET_COL: "y"})
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        n_lags=window_size,        # <-- THE FIX: actually use autoregression
        n_forecasts=1,
        learning_rate=0.01,
        epochs=80,
        batch_size=64,
    )
    # progress=None avoids the ipywidgets/Jupyter progress-bar dependency
    # that caused environment noise in the original notebook.
    model.fit(train_df, freq="D", progress=None)

    forecast = model.predict(full_df)
    forecast = forecast.set_index("ds")
    yhat_col = "yhat1"  # single-step forecast column with n_forecasts=1

    train_pred = forecast.loc[train_df["ds"][window_size:], yhat_col]
    test_pred = forecast.loc[test_df["ds"], yhat_col]

    results["NeuralProphet"] = {
        "description": (
            "NeuralProphet extends Prophet's trend + seasonality decomposition with an AR-Net "
            f"that looks back {window_size} days, giving it genuine autoregressive memory that "
            "plain Prophet lacks."
        ),
        "params": {"n_lags": window_size, "n_forecasts": 1, "epochs": 80},
        "train": metrics_dict(train_df["y"].iloc[window_size:], train_pred.values),
        "test": metrics_dict(test_df["y"], test_pred.values),
    }
    predictions["NeuralProphet"] = {
        "train": pd.Series(train_pred.values, index=train_data.index[window_size:]),
        "test": pd.Series(test_pred.values, index=test_data.index),
    }
    joblib.dump(model, os.path.join(MODEL_DIR, "neuralprophet.joblib"))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    log("Loading and cleaning data ...")
    df = load_clean_data()
    run_eda(df)

    train_data, test_data = train_test_split_series(df, train_frac=0.8)
    log(f"Train size: {len(train_data)} | Test size: {len(test_data)}")

    results = {}
    predictions = {}

    run_arima_family(train_data, test_data, results, predictions, run_sarima=False)
    run_deep_learning(train_data, test_data, results, predictions)

    try:
        run_prophet(train_data, test_data, results, predictions)
    except Exception as e:
        log(f"Prophet failed/skipped: {e}")

    try:
        run_neuralprophet(train_data, test_data, results, predictions)
    except Exception as e:
        log(f"NeuralProphet failed/skipped: {e}")

    # --- Persist metrics ---
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Persist predictions as one wide CSV, aligned on the full date index ---
    pred_df = pd.DataFrame(index=df.index)
    pred_df["actual"] = df[TARGET_COL]
    pred_df["split"] = ["train" if d in train_data.index else "test" for d in df.index]
    for name, splits in predictions.items():
        col = pd.concat([splits["train"], splits["test"]])
        pred_df[f"pred_{name}"] = col
    pred_df.to_csv(os.path.join(ARTIFACT_DIR, "predictions.csv"))

    # --- Leaderboard ---
    leaderboard = (
        pd.DataFrame(
            {name: {"Test_RMSE": r["test"]["RMSE"], "Test_MAE": r["test"]["MAE"], "Test_R2": r["test"]["R2"]}
             for name, r in results.items()}
        )
        .T.sort_values("Test_RMSE")
    )
    leaderboard.to_csv(os.path.join(ARTIFACT_DIR, "leaderboard.csv"))
    log("Done. Leaderboard:")
    print(leaderboard)


if __name__ == "__main__":
    main()
