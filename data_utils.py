"""
data_utils.py
--------------
Shared, reusable data loading / cleaning utilities for the Melbourne daily
minimum temperature forecasting project. Both train_models.py and app.py
import from here so the data seen by every model is guaranteed identical.
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/daily-minimum-temperatures-in-me.csv"
TARGET_COL = "Temperature"


def load_clean_data(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Reproduces (and hardens) the cleaning steps from the original notebook:
      1. Read CSV, parse Date as index.
      2. Coerce the temperature column to numeric (the raw Kaggle file has a
         trailing footer row that is not numeric -> becomes NaN -> dropped).
      3. Rename the long column name to 'Temperature'.
      4. Reindex to a complete daily calendar (the raw file is missing two
         Dec-31 rows, in leap-adjacent years) and interpolate the gaps.
    Returns a DataFrame indexed by date with a single 'Temperature' column.
    """
    df = pd.read_csv(
        path,
        parse_dates=["Date"],
        index_col="Date",
        delimiter=",",
        on_bad_lines="skip",
    )
    long_col = df.columns[0]
    df[long_col] = pd.to_numeric(df[long_col], errors="coerce")
    df = df.rename(columns={long_col: TARGET_COL})
    df = df[df.index.notna()]

    # Fill in any missing calendar days, then interpolate the values.
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range)
    df[TARGET_COL] = df[TARGET_COL].interpolate(method="time", limit_direction="both")
    df.index.name = "Date"
    return df[[TARGET_COL]]


def train_test_split_series(df: pd.DataFrame, train_frac: float = 0.8):
    """Chronological 80/20 split — never shuffle time series data."""
    n = len(df)
    split = int(n * train_frac)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def create_sequences(values: np.ndarray, window_size: int = 30):
    """
    Turn a 1-D array into sliding-window (X, y) pairs for the deep learning
    models: X[i] = values[i : i+window_size], y[i] = values[i+window_size].
    """
    xs, ys = [], []
    for i in range(len(values) - window_size):
        xs.append(values[i : i + window_size])
        ys.append(values[i + window_size])
    return np.array(xs), np.array(ys)
