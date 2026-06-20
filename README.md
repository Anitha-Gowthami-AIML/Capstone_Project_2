# Melbourne Daily Minimum Temperature — Forecasting Explainability Lab

A complete, runnable continuation of `predict-daily-temp-timeseries.ipynb`:
it fixes the NeuralProphet issue from the notebook, trains and fairly
compares **8 forecasting models** (ARIMA, SARIMA, SimpleRNN, LSTM, GRU,
CNN+RNN Hybrid, Prophet, NeuralProphet), and ships a multi-tab **Streamlit**
app that explains the whole project — problem, data, concepts, comparison,
per-model deep-dive, interactive forecasting, and conclusions — in plain
language for people who don't already know what a time series or an LSTM is.

## Files

```
project/
├── data/
│   └── daily-minimum-temperatures-in-me.csv   # your uploaded dataset
├── data_utils.py        # shared cleaning / splitting / windowing helpers
├── train_models.py      # trains all 8 models, writes everything to artifacts/
├── app.py                # Streamlit explainability UI (reads artifacts/)
├── requirements.txt
└── artifacts/            # created by train_models.py (metrics, predictions, models)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

1. **Train everything** (writes results to `artifacts/`, takes a few
   minutes — the deep learning models are the slowest part):

   ```bash
   python train_models.py
   ```

2. **Launch the explainability app**:

   ```bash
   streamlit run app.py
   ```

   The app reads only from `artifacts/` — it never re-trains anything live,
   so it opens instantly and stays responsive no matter how slow the
   original training was.

## The NeuralProphet error from the original notebook — what was wrong, and the fix

The original notebook's NeuralProphet section had three separate, stacked
problems:

1. **`!pip install pandas==1.5.3` / `!pip install neuralprophet==0.6.2` were
   run mid-kernel, after pandas/numpy/keras/etc. had already been imported
   and used by every earlier model in the notebook.** Re-installing a
   package while the Python process is already running does *nothing* until
   the kernel restarts — so this "fix" never took effect, and instead risked
   leaving the live environment in a half-upgraded, half-downgraded,
   inconsistent state for every subsequent cell.
   **Fix:** pin compatible versions once in `requirements.txt`, *before* the
   process starts, and never call `pip install` from inside a running
   script/notebook.

2. **Newer NumPy removed the deprecated `np.NaN` alias** (only `np.nan`
   remains), but older NeuralProphet/Prophet internals still reference
   `np.NaN`, raising an `AttributeError` on import.
   **Fix:** a tiny compatibility shim, applied once before NeuralProphet is
   imported:
   ```python
   if not hasattr(np, "NaN"):
       np.NaN = np.nan
   ```

3. **`n_lags` was left at its default of `0`**, which turns NeuralProphet
   into a pure trend+seasonality curve-fitter — functionally almost
   identical to plain Prophet, with no actual autoregression. This wasn't a
   crash, but it silently defeated the entire point of using
   "Neural"Prophet over Prophet, and made results confusingly similar to the
   Prophet cell right above it.
   **Fix:** explicitly set `n_lags=window_size` (30, matching the other deep
   learning models) so the AR-Net component is actually used, and set
   `progress=None` when fitting so the run doesn't depend on
   Jupyter/ipywidgets (which is what produced all of the widget-state noise
   you'll see if you inspect the original `.ipynb` JSON).

All three fixes are implemented in `train_models.py::run_neuralprophet`.

## What the app covers (tab by tab)

| Tab | What it's for |
|---|---|
| 📋 Problem Statement | What we're forecasting, why it's hard, evaluation methodology |
| 🔍 Data Exploration | Raw series, year-over-year overlay, seasonal decomposition, stationarity test |
| 📚 Concepts 101 | Plain-language explainer for ARIMA, SARIMA, RNN, LSTM, GRU, CNN+RNN, Prophet, NeuralProphet |
| 🏆 Model Comparison | Sortable leaderboard, RMSE/R² bar charts, overlaid prediction lines, train/test generalization gap |
| 🔬 Model Deep-Dive | Per-model description, metrics, actual-vs-predicted plot, residual diagnostics, training curves, Prophet component breakdown |
| 🔮 Interactive Forecast | Pick a model + date window, see zoomed-in performance and a plain-English confidence statement |
| 💡 Insights & Conclusion | Best/worst model, key takeaways, suggested next steps |
| 🗺️ Map View | Melbourne plotted on an OpenStreetMap base layer, color/size-coded by month, plus a 12-month climatology bar chart |

## Hero banner & color theme

The app opens with a colorful hero banner (sunset-orange → storm-navy →
city-night gradient) and now has a **rich purple / lavender / sky-blue
wallpaper** behind the whole app — a blurred, dimmed, original SVG
night-skyline-and-river silhouette drawn to evoke the mood of the Melbourne
photos you shared, without embedding the actual copyrighted photos directly.

To keep everything legible against the darker backdrop:
- Headings and body text switched to light lavender/white.
- Metrics, expanders, and alert boxes float as solid white "glass cards" on
  top of the wallpaper, so numbers and explanatory text always have strong
  contrast — **nothing is sacrificed for the visual styling**.
- The previously-truncated "Date range" metric was fixed two ways: the
  displayed text is shorter (`Jan 1981 – Dec 1990`), and metric values now
  wrap instead of being clipped, so this can't happen again even with longer
  text elsewhere in the app.

If you'd like to use one of your own **licensed photos** as the literal hero
background instead of the CSS gradient, just drop it at:

```
assets/hero.jpg
```

`app.py` checks for that file automatically and switches the hero background
to it (with a dark overlay so the title text stays readable) — no code
changes needed. (We didn't embed the specific photos you shared in this
chat directly into the app, since they appear to be copyrighted stock
images; the CSS gradient and wallpaper mimic their sunset/storm/night-skyline
mood using original shapes and colors instead.)

## Notes / customization

- `train_models.py` skips the seasonal (`period=365`) SARIMA fit by default
  because it is extremely slow on ~4 years of daily data (state-space size
  scales with the seasonal period). Pass `run_sarima=True` into
  `run_arima_family(...)` inside `main()` if you have time and want it
  included in the comparison.
- Deep learning epochs are trimmed from the notebook's 500 down to 60 with
  early stopping (`patience=8`) — this converges to essentially the same
  result much faster; raise `EPOCHS` in `train_models.py` if you want to
  match the original notebook more closely.
- All models are evaluated on the **same chronological 80/20 split** so
  comparisons are apples-to-apples.