"""
app.py
------
Streamlit explainability app for the Melbourne Daily Minimum Temperature
forecasting project. Built so that someone with ZERO time-series / deep
learning background can still come away understanding:
  - what the problem is and why it's hard
  - what each model family does, in plain language
  - how the models actually compare on this data, and why
  - how to read a forecast and trust (or distrust) it

Run with:
    streamlit run app.py

The app reads everything from ./artifacts, which is produced by
`python train_models.py`. If artifacts are missing, the app will tell you
exactly what to run.
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data_utils import load_clean_data, TARGET_COL

ARTIFACT_DIR = "artifacts"

st.set_page_config(
    page_title="Melbourne Temperature Forecasting — Explainability Lab",
    page_icon="🌡️",
    layout="wide",
)

MODEL_GROUPS = {
    "Statistical": ["ARIMA", "SARIMA"],
    "Deep Learning": ["RNN", "LSTM", "GRU", "CNN+RNN Hybrid"],
    "Specialized Forecasting": ["Prophet", "NeuralProphet"],
}

MODEL_COLORS = {
    "ARIMA": "#1f77b4", "SARIMA": "#17becf",
    "RNN": "#ff7f0e", "LSTM": "#d62728", "GRU": "#9467bd", "CNN+RNN Hybrid": "#8c564b",
    "Prophet": "#2ca02c", "NeuralProphet": "#e377c2",
}

MELBOURNE_LAT, MELBOURNE_LON = -37.8136, 144.9631

# ---------------------------------------------------------------------------
# Theme / Hero banner
# ---------------------------------------------------------------------------
# Optional: drop your own licensed photo at assets/hero.jpg and it will be
# used as the hero background automatically. If it's not there, we fall back
# to an original CSS/SVG gradient (sunset-orange -> storm-blue -> city-night)
# inspired by the references you shared, drawn with plain CSS/SVG rather than
# any copyrighted photograph.
import base64


def _hero_background_css():
    custom_path = "assets/hero.jpg"
    if os.path.exists(custom_path):
        with open(custom_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"linear-gradient(120deg, rgba(20,10,0,0.35), rgba(5,15,40,0.55)), url('data:image/jpeg;base64,{b64}')"
    # Original gradient: warm sunset (left) -> storm navy (right) -> deep night
    return (
        "linear-gradient(100deg,"
        " #ffb347 0%, #ff7e29 14%, #e85d2c 24%,"
        " #4a3b3a 38%, #2c3e5e 52%, #1b2a4a 66%,"
        " #15203c 80%, #0c1430 100%)"
    )


def _wallpaper_background_css():
    """
    Wallpaper behind the whole app. If the user has dropped their own image
    at assets/background.jpg, use that directly (as a base64 data URI so no
    extra HTTP round-trip / static file serving is needed). Otherwise fall
    back to the original CSS/SVG skyline drawing below.
    """
    custom_path = "assets/background.jpg"
    if os.path.exists(custom_path):
        with open(custom_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"url('data:image/jpeg;base64,{b64}')"
    return f"url('{_wallpaper_svg_data_uri()}')"


def _wallpaper_svg_data_uri():
    """
    Original (non-photographic) city-skyline-at-night silhouette, drawn as
    inline SVG, in a purple / lavender / sky-blue palette. Used as a blurred,
    dimmed wallpaper behind the whole app. We deliberately don't embed the
    actual photo you shared (it reads as a copyrighted stock photograph) —
    this recreates the same mood (dusk sky, lit towers, river reflection)
    with original shapes/colors instead.
    """
    import urllib.parse

    svg = """
    <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 900' preserveAspectRatio='xMidYMax slice'>
      <defs>
        <linearGradient id='sky' x1='0' y1='0' x2='0' y2='1'>
          <stop offset='0%' stop-color='#241b4e'/>
          <stop offset='45%' stop-color='#3b2f6b'/>
          <stop offset='75%' stop-color='#4d5d9a'/>
          <stop offset='100%' stop-color='#6fa8c9'/>
        </linearGradient>
        <linearGradient id='water' x1='0' y1='0' x2='0' y2='1'>
          <stop offset='0%' stop-color='#1c1640'/>
          <stop offset='100%' stop-color='#0c0a22'/>
        </linearGradient>
      </defs>
      <rect width='1600' height='900' fill='url(#sky)'/>
      <rect y='620' width='1600' height='280' fill='url(#water)'/>
      <g fill='#1a1438'>
        <rect x='30'  y='430' width='70'  height='200'/>
        <rect x='120' y='380' width='55'  height='250'/>
        <rect x='200' y='250' width='90'  height='380'/>
        <rect x='310' y='460' width='60'  height='170'/>
        <rect x='390' y='320' width='75'  height='310'/>
        <rect x='490' y='500' width='50'  height='130'/>
        <rect x='560' y='400' width='65'  height='230'/>
        <rect x='650' y='470' width='55'  height='160'/>
        <rect x='730' y='280' width='95'  height='350'/>
        <rect x='850' y='440' width='60'  height='190'/>
        <rect x='930' y='360' width='70' height='270'/>
        <rect x='1020' y='480' width='55' height='150'/>
        <rect x='1100' y='300' width='85' height='330'/>
        <rect x='1210' y='420' width='65' height='210'/>
        <rect x='1300' y='350' width='75' height='280'/>
        <rect x='1400' y='460' width='60' height='170'/>
        <rect x='1480' y='390' width='80' height='240'/>
      </g>
      <g fill='#ffd9a0' opacity='0.85'>
        <circle cx='55'  cy='470' r='3'/><circle cx='80'  cy='510' r='3'/>
        <circle cx='235' cy='300' r='3'/><circle cx='260' cy='340' r='3'/><circle cx='235' cy='400' r='3'/>
        <circle cx='415' cy='370' r='3'/><circle cx='440' cy='420' r='3'/><circle cx='415' cy='470' r='3'/>
        <circle cx='760' cy='330' r='3'/><circle cx='790' cy='380' r='3'/><circle cx='760' cy='440' r='3'/>
        <circle cx='1130' cy='350' r='3'/><circle cx='1160' cy='400' r='3'/><circle cx='1130' cy='460' r='3'/>
        <circle cx='1320' cy='400' r='3'/><circle cx='1350' cy='450' r='3'/>
        <circle cx='1500' cy='430' r='3'/><circle cx='1525' cy='480' r='3'/>
      </g>
      <g fill='#9ad6ff' opacity='0.5'>
        <rect x='200' y='620' width='4' height='180'/>
        <rect x='400' y='620' width='4' height='160'/>
        <rect x='760' y='620' width='4' height='200'/>
        <rect x='1130' y='620' width='4' height='170'/>
      </g>
    </svg>
    """.strip()
    return "data:image/svg+xml," + urllib.parse.quote(svg)


def inject_theme():
    hero_bg = _hero_background_css()
    wallpaper = _wallpaper_background_css()
    st.markdown(
        f"""
        <style>
        /* ---- App-wide base wallpaper ----
           NOTE: layered directly on .stApp's own `background` (gradient +
           image in one declaration) instead of a ::before/filter/blur
           pseudo-element trick — that trick can interact badly with
           Streamlit's internal stacking contexts and blank the page out, so
           we deliberately keep this simple and robust.
           The gradient is a darkening overlay so light text stays readable
           on top of a bright sky/field photo. */
        .stApp {{
            background:
                linear-gradient(180deg, rgba(15,20,40,0.8) 0%, rgba(20,30,55,0.78) 40%,
                                         rgba(20,35,28,0.8) 75%, rgba(15,25,18,0.85) 100%),
                {wallpaper};
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        html, body, [class*="css"] {{
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        }}

        /* Headings & markdown body text, SCOPED to our own content area only
           -- never touch Streamlit's internal error/alert/code/table text,
           which has its own contrast-safe styling we don't want to fight. */
        .block-container h1, .block-container h2,
        .block-container h3, .block-container h4 {{
            color: #f3efff !important;
            font-weight: 700 !important;
            text-shadow: 0 1px 6px rgba(0,0,0,0.45);
        }}
        [data-testid="stMarkdown"] p,
        [data-testid="stMarkdown"] li {{
            color: #f1edff;
        }}
        [data-testid="stCaptionContainer"] p {{
            color: #d8d2f2 !important;
        }}
        .stRadio label, .stSelectbox label, .stMultiSelect label, .stSlider label {{
            color: #f1edff !important;
        }}

        /* ---- Hero banner (unchanged on request) ---- */
        .hero-banner {{
            background: {hero_bg};
            background-size: cover;
            background-position: center;
            border-radius: 18px;
            padding: 3rem 2.5rem 2.5rem 2.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        }}
        .hero-banner h1 {{
            color: #ffffff !important;
            font-size: 2.6rem !important;
            text-shadow: 0 2px 10px rgba(0,0,0,0.6);
            margin-bottom: 0.4rem !important;
        }}
        .hero-banner p {{
            color: #f5f5f5 !important;
            font-size: 1.15rem;
            text-shadow: 0 1px 6px rgba(0,0,0,0.6);
            max-width: 800px;
            margin-bottom: 0;
        }}
        .hero-pills {{
            margin-top: 1.2rem;
        }}
        .hero-pill {{
            display: inline-block;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.45);
            color: #ffffff;
            font-weight: 600;
            font-size: 0.85rem;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            margin: 0.2rem 0.4rem 0.2rem 0;
            text-shadow: 0 1px 4px rgba(0,0,0,0.5);
        }}

        /* ---- Tabs: readable against the dark wallpaper ---- */
        button[data-baseweb="tab"] {{
            font-size: 1.0rem !important;
            font-weight: 600 !important;
            color: #e3ddff !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: #ffd166 !important;
            border-bottom-color: #ffd166 !important;
        }}

        /* ---- Metric cards: solid light background, dark text, never truncated ---- */
        div[data-testid="stMetric"] {{
            background: #ffffff;
            border: 1px solid #d9d3ee;
            border-radius: 12px;
            padding: 0.8rem 0.8rem 0.6rem 0.8rem;
            box-shadow: 0 4px 14px rgba(20,10,50,0.25);
        }}
        div[data-testid="stMetricLabel"] {{
            color: #4b4470 !important;
            font-size: 0.85rem !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: #221c45 !important;
        }}
        /* Fix long values (e.g. date ranges) getting clipped/ellipsized */
        div[data-testid="stMetricValue"] > div {{
            font-size: 1.05rem !important;
            line-height: 1.35rem !important;
            white-space: normal !important;
            overflow-wrap: break-word !important;
            word-break: break-word !important;
        }}

        /* ---- Expanders: white "glass cards" floating on the wallpaper ---- */
        [data-testid="stExpander"] {{
            background: #ffffff;
            border-radius: 10px;
            border: 1px solid #e2dcf3;
            box-shadow: 0 4px 16px rgba(20,10,50,0.2);
        }}
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] li,
        [data-testid="stExpander"] span,
        [data-testid="stExpander"] summary {{
            color: #1f1f1f !important;
        }}

        /* ---- Dataframes / tables: keep their native light background + dark text ---- */
        [data-testid="stDataFrame"] {{
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(20,10,50,0.2);
        }}

        /* ---- Alert boxes (st.info / st.success / st.warning / st.error):
               same white "glass card" treatment as expanders. Without this
               they're fully transparent and float unreadable text directly
               on top of the wallpaper photo. ---- */
        [data-testid="stAlert"] {{
            background: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 16px rgba(20,10,50,0.2);
        }}
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] li,
        [data-testid="stAlert"] span,
        [data-testid="stAlert"] strong {{
            color: #1f1f1f !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🌡️ Melbourne Daily Minimum Temperature</h1>
            <p>Forecasting Explainability Lab — sunset-to-storm seasonality, ten years of
            real weather, and eight models (statistical, deep learning, and modern
            forecasting libraries) compared side by side, explained for everyone.</p>
            <div class="hero-pills">
                <span class="hero-pill">📍 Melbourne, Australia</span>
                <span class="hero-pill">📅 1981 – 1990</span>
                <span class="hero-pill">🤖 8 models compared</span>
                <span class="hero-pill">🔍 Full explainability</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Artifact loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_artifacts():
    missing = []
    paths = {
        "metrics": os.path.join(ARTIFACT_DIR, "metrics.json"),
        "predictions": os.path.join(ARTIFACT_DIR, "predictions.csv"),
        "leaderboard": os.path.join(ARTIFACT_DIR, "leaderboard.csv"),
        "adf": os.path.join(ARTIFACT_DIR, "adf_test.json"),
        "decomposition": os.path.join(ARTIFACT_DIR, "decomposition.csv"),
    }
    for name, p in paths.items():
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        return None, missing

    with open(paths["metrics"]) as f:
        metrics = json.load(f)
    predictions = pd.read_csv(paths["predictions"], index_col=0, parse_dates=True)
    leaderboard = pd.read_csv(paths["leaderboard"], index_col=0)
    with open(paths["adf"]) as f:
        adf = json.load(f)
    decomposition = pd.read_csv(paths["decomposition"], index_col=0, parse_dates=True)

    prophet_components_path = os.path.join(ARTIFACT_DIR, "prophet_components.csv")
    prophet_components = (
        pd.read_csv(prophet_components_path, index_col=0, parse_dates=True)
        if os.path.exists(prophet_components_path) else None
    )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "leaderboard": leaderboard,
        "adf": adf,
        "decomposition": decomposition,
        "prophet_components": prophet_components,
    }, []


@st.cache_data
def get_raw_data():
    return load_clean_data()


artifacts, missing = load_artifacts()

inject_theme()
render_hero()

if artifacts is None:
    st.error(
        "No trained model artifacts found yet.\n\n"
        "This app reads pre-computed results from the `artifacts/` folder. "
        "Please run the training pipeline first:\n\n"
        "```\npython train_models.py\n```\n\n"
        f"Missing files:\n" + "\n".join(f"- {m}" for m in missing)
    )
    st.stop()

raw_df = get_raw_data()
metrics = artifacts["metrics"]
predictions = artifacts["predictions"]
leaderboard = artifacts["leaderboard"]
available_models = list(metrics.keys())

tabs = st.tabs([
    "📋 Problem Statement",
    "🔍 Data Exploration",
    "📚 Concepts 101",
    "🏆 Model Comparison",
    "🔬 Model Deep-Dive",
    "🔮 Interactive Forecast",
    "💡 Insights & Conclusion",
    "🗺️ Map View",
])

# ===========================================================================
TAB_PROBLEM = tabs[0]
with TAB_PROBLEM:
    st.header("What problem are we solving?")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
**The dataset.** Daily minimum temperatures recorded in Melbourne, Australia,
from **1 Jan 1981 to 31 Dec 1990** — 3,650 consecutive daily readings in °C.

**The task.** Given everything we know up to a certain day, predict the
minimum temperature for the *next* day (and, in the interactive tab, several
days ahead). This is a classic **univariate time-series forecasting**
problem: one variable, measured at regular intervals, where order matters.

**Why it's a genuinely useful test case.**
- It has **strong, repeating yearly seasonality** (Southern Hemisphere
  summer/winter) but very **weak day-to-day memory** — today's temperature
  barely predicts tomorrow's beyond "it's currently winter".
- That combination makes it a great benchmark: models that lean on
  *seasonality* (Prophet, SARIMA) should do reasonably well, while models
  built primarily for *short-term autoregressive memory* (plain RNN/LSTM/GRU)
  are fighting an uphill battle, because the *true* signal-to-noise ratio at
  a 1-day horizon is low.

**Why we compare 8 models instead of just picking one.** In real forecasting
work there is no universally "best" algorithm — performance depends on the
shape of the signal (trend vs. seasonality vs. noise), how much history is
needed, and how interpretable the result needs to be. This project trains
representatives from three different philosophies and judges them on the
exact same train/test split so the comparison is fair.
            """
        )
    with col2:
        st.info(
            "**Models trained:**\n\n"
            "**Statistical**\n- ARIMA\n- SARIMA\n\n"
            "**Deep Learning**\n- SimpleRNN\n- LSTM\n- GRU\n- CNN + RNN Hybrid\n\n"
            "**Specialized forecasting libraries**\n- Prophet\n- NeuralProphet"
        )
        st.metric("Total days", f"{len(raw_df):,}")
        st.metric("Date range", f"{raw_df.index.min().strftime('%b %Y')} – {raw_df.index.max().strftime('%b %Y')}")

    st.markdown("---")
    st.subheader("Evaluation methodology")
    st.markdown(
        """
- **Chronological 80/20 split.** The first ~8 years train the models; the
  last ~2 years are held out and never seen during fitting. We never shuffle
  time series data — that would let models "peek" at the future.
- **Metrics.**
  - **RMSE** (Root Mean Squared Error): average prediction error, in °C,
    penalizing large misses more heavily. Lower is better.
  - **MAE** (Mean Absolute Error): average absolute error in °C. Easier to
    interpret directly ("the model is typically off by X degrees"). Lower is
    better.
  - **R²**: the fraction of variance in the actual temperature the model's
    predictions explain, relative to just guessing the mean every time.
    1.0 is a perfect model; 0.0 is no better than the historical average;
    negative means *worse* than that average.
        """
    )

# ===========================================================================
TAB_EDA = tabs[1]
with TAB_EDA:
    st.header("Exploring the data before modeling anything")

    st.subheader("1. The raw series")
    fig = px.line(raw_df, y=TARGET_COL, title="Daily Minimum Temperature, 1981–1990")
    fig.update_layout(xaxis_title="Date", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Notice the repeating wave: every year, temperatures dip in the Southern "
        "Hemisphere winter (~June–August) and peak in summer (~Dec–Feb)."
    )

    st.subheader("2. Year-over-year overlay")
    overlay_df = raw_df.copy()
    overlay_df["year"] = overlay_df.index.year
    overlay_df["day_of_year"] = overlay_df.index.dayofyear
    fig2 = px.line(
        overlay_df, x="day_of_year", y=TARGET_COL, color=overlay_df["year"].astype(str),
        title="Every year plotted on top of each other",
    )
    fig2.update_layout(xaxis_title="Day of year", yaxis_title="Temperature (°C)", legend_title="Year")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Every single year follows almost the same downward-then-upward shape. "
        "**This is the single most exploitable pattern in the whole dataset** — "
        "any model that can learn 'what month is it' captures most of the "
        "explainable variance."
    )

    st.subheader("3. Seasonal decomposition")
    st.markdown(
        "We can mathematically split the series into three additive pieces: "
        "`Observed = Trend + Seasonal + Residual`."
    )
    decomp = artifacts["decomposition"]
    fig3 = go.Figure()
    for col in ["observed", "trend", "seasonal", "resid"]:
        fig3.add_trace(go.Scatter(x=decomp.index, y=decomp[col], mode="lines", name=col.capitalize()))
    fig3.update_layout(title="Additive Decomposition", xaxis_title="Date", yaxis_title="°C",
                        legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)
    cols = st.columns(3)
    cols[0].markdown("**Trend** — slow multi-year drift in the average level.")
    cols[1].markdown("**Seasonal** — the repeating yearly summer/winter wave.")
    cols[2].markdown("**Residual** — whatever is left over: short-term noise the model can't explain structurally.")

    st.subheader("4. Is the series stationary?")
    adf = artifacts["adf"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ADF statistic", f"{adf['ADF_statistic']:.3f}")
    c2.metric("p-value", f"{adf['p_value']:.4g}")
    c3.metric("Stationary?", "✅ Yes" if adf["is_stationary"] else "❌ No")
    st.markdown(
        """
**What is "stationary" and why does it matter?** A stationary series has a
mean, variance, and autocorrelation structure that don't change over time
(after removing the known yearly cycle). The **Augmented Dickey-Fuller (ADF)
test** checks for this statistically:
- Null hypothesis: the series has a "unit root" (is *not* stationary).
- If **p-value < 0.05**, we reject the null → the series **is** stationary.

This dataset comes back stationary, which is good news for ARIMA-style models:
it means we don't need extra differencing (the `d` parameter in ARIMA(p,d,q))
to make the mean/variance behave — the seasonal wave is large but it's a
*deterministic*, low-noise pattern, not a wandering random walk.
        """
    )

# ===========================================================================
TAB_CONCEPTS = tabs[2]
with TAB_CONCEPTS:
    st.header("Time series & forecasting, explained from scratch")
    st.markdown(
        "No background assumed. Read top to bottom, or jump to whichever model "
        "you're curious about."
    )

    with st.expander("📈 What is a time series, and what makes forecasting it hard?", expanded=True):
        st.markdown(
            """
A **time series** is just a sequence of numbers measured at regular time
intervals — here, one temperature reading per day. What makes forecasting it
different from ordinary prediction is that **order matters** and **the
future depends on the past**, so we can't shuffle the data or split it
randomly into train/test — we always train on the past and test on the
future, just like a real forecaster would.

Forecasting is hard because the future is a mix of:
1. **Signal** — predictable structure (trends, seasons, cycles).
2. **Noise** — genuinely unpredictable day-to-day variation (weather fronts,
   measurement error).
A model's job is to capture as much signal as possible without "memorizing"
the noise (overfitting).
            """
        )

    with st.expander("🔢 ARIMA — AutoRegressive Integrated Moving Average"):
        st.markdown(
            """
ARIMA predicts the next value as a weighted combination of:
- **AR(p)** — the last *p* actual values ("today looks like the last few days").
- **I(d)** — how many times we *difference* the series (today minus yesterday)
  to remove trend and make it stationary.
- **MA(q)** — the last *q* forecast *errors* ("we under-shot yesterday, so
  nudge today's forecast up a bit").

It's a linear model with a long, successful track record, fit using maximum
likelihood. Its main weakness here: plain ARIMA has **no concept of yearly
seasonality** — it only sees the last few days, so it can't "remember" that
August is winter.
            """
        )

    with st.expander("🔁 SARIMA — Seasonal ARIMA"):
        st.markdown(
            """
SARIMA is ARIMA plus a second, *seasonal* AR/I/MA block tied to a fixed
period (here, 365 days). It explicitly compares today to "the same day last
year" in addition to "a few days ago" — directly addressing ARIMA's blind
spot for yearly seasonality, at the cost of being much more expensive to fit
on long seasonal periods like 365.
            """
        )

    with st.expander("🔄 RNN (SimpleRNN) — Recurrent Neural Network"):
        st.markdown(
            """
An RNN reads a window of past days **one day at a time**, updating an
internal "memory" vector (the hidden state) at each step, and uses that final
memory to predict the next value. Conceptually: *"having read the last 30
days in order, what comes next?"*

Weakness: over long sequences, plain RNNs struggle to retain information from
many steps back — gradients used to train it tend to vanish, so it leans
mostly on the most recent few days.
            """
        )

    with st.expander("🧠 LSTM — Long Short-Term Memory"):
        st.markdown(
            """
An LSTM is an RNN with an added **memory cell** plus three learned "gates"
(input, forget, output) that explicitly control what to remember, what to
discard, and what to expose at each time step. This solves the vanishing-memory
problem that plain RNNs have, letting it use longer windows of history
effectively — useful for series with longer cycles than a few days.
            """
        )

    with st.expander("⚡ GRU — Gated Recurrent Unit"):
        st.markdown(
            """
A GRU is a streamlined cousin of the LSTM: it merges some of the gates
together, so it has fewer parameters and trains faster, while keeping most of
the same long-memory benefits. In practice GRU and LSTM often perform very
similarly; GRU is the "lighter" choice.
            """
        )

    with st.expander("🧩 CNN + RNN Hybrid"):
        st.markdown(
            """
This model first runs a **1-D convolution** over the 30-day input window —
like sliding a small "shape detector" along the sequence to pick out local
patterns (e.g. "a sharp 3-day cold snap"). Those extracted features are then
fed into a SimpleRNN, which summarizes them across the whole window before
making the final prediction. The intuition: let the CNN do cheap local
pattern-spotting first, so the RNN only has to reason about already-summarized
features instead of raw daily noise.
            """
        )

    with st.expander("📊 Prophet"):
        st.markdown(
            """
Prophet (originally from Meta) is **not** a neural network — it's a curve-
fitting model: `y(t) = trend(t) + seasonality(t) + holidays(t) + noise`. The
yearly seasonal wave is represented with a Fourier series (a sum of sine/cosine
waves), and the trend is piecewise-linear with automatically detected
changepoints. It was designed to be robust and easy to use even with
missing data or irregular spacing, and it produces directly interpretable
trend/seasonality components — but, by default, **it has no memory of
recent actual values** (no autoregression).
            """
        )

    with st.expander("🧠📊 NeuralProphet"):
        st.markdown(
            """
NeuralProphet keeps Prophet's interpretable trend + seasonality decomposition
but adds a small neural network ("AR-Net") that looks back at recent actual
values to add a genuine autoregressive component — combining the best of
both worlds: interpretable seasonal structure *and* short-term memory. In this
project we explicitly set `n_lags=30` so it actually uses that AR-Net (the
default, `n_lags=0`, would otherwise behave just like plain Prophet).
            """
        )

    with st.expander("🤔 So which type of model *should* win here?"):
        st.markdown(
            """
Given that this series has **strong yearly seasonality but very weak 1-day
autocorrelation**, you'd predict, before even running anything:
- Models that explicitly encode the yearly cycle (**SARIMA, Prophet,
  NeuralProphet**) have a structural advantage.
- Pure short-memory models (**plain RNN/LSTM/GRU/ARIMA** without seasonal
  terms) are forced to *implicitly* learn "what month is it" purely from a
  30-day window, which is possible but inefficient.
Check the **Model Comparison** tab to see whether the actual results bear
this out.
            """
        )

# ===========================================================================
TAB_COMPARE = tabs[3]
with TAB_COMPARE:
    st.header("How do all 8 models actually compare?")

    sort_by = st.radio("Rank models by:", ["Test_RMSE", "Test_MAE", "Test_R2"], horizontal=True)
    ascending = sort_by != "Test_R2"
    ranked = leaderboard.sort_values(sort_by, ascending=ascending)

    st.dataframe(
        ranked.style.format({"Test_RMSE": "{:.3f}", "Test_MAE": "{:.3f}", "Test_R2": "{:.3f}"})
        .background_gradient(cmap="RdYlGn_r", subset=["Test_RMSE", "Test_MAE"])
        .background_gradient(cmap="RdYlGn", subset=["Test_R2"]),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            ranked, x=ranked.index, y="Test_RMSE", color=ranked.index,
            color_discrete_map=MODEL_COLORS, title="Test RMSE by model (lower = better)",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="RMSE (°C)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            ranked, x=ranked.index, y="Test_R2", color=ranked.index,
            color_discrete_map=MODEL_COLORS, title="Test R² by model (higher = better)",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="R²")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Overlay: actual vs. every model's test-period predictions")
    show_models = st.multiselect(
        "Models to overlay", available_models, default=available_models[: min(4, len(available_models))]
    )
    test_actual = predictions[predictions["split"] == "test"]["actual"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_actual.index, y=test_actual.values, mode="lines",
                              name="Actual", line=dict(color="black", width=2)))
    for m in show_models:
        col = f"pred_{m}"
        if col in predictions.columns:
            series = predictions.loc[predictions["split"] == "test", col].dropna()
            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                                      name=m, line=dict(color=MODEL_COLORS.get(m))))
    fig.update_layout(title="Held-out test period: Actual vs. Predicted", xaxis_title="Date",
                       yaxis_title="Temperature (°C)", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Why the gap between train and test performance matters")
    train_test_df = pd.DataFrame({
        m: {"Train RMSE": metrics[m]["train"]["RMSE"], "Test RMSE": metrics[m]["test"]["RMSE"]}
        for m in available_models
    }).T
    train_test_df["Gap (Test - Train)"] = train_test_df["Test RMSE"] - train_test_df["Train RMSE"]
    st.dataframe(train_test_df.style.format("{:.3f}"), use_container_width=True)
    st.caption(
        "A small gap means the model **generalizes** well to unseen data. A large "
        "gap is a sign of **overfitting** — the model memorized quirks of the "
        "training period that don't hold up on the held-out years."
    )

# ===========================================================================
TAB_DEEPDIVE = tabs[4]
with TAB_DEEPDIVE:
    st.header("Pick a model and inspect it closely")

    selected = st.selectbox("Model", available_models)
    m = metrics[selected]

    st.subheader(f"{selected}: what it's doing")
    st.markdown(m["description"])

    cols = st.columns(4)
    cols[0].metric("Test RMSE", f"{m['test']['RMSE']:.3f} °C")
    cols[1].metric("Test MAE", f"{m['test']['MAE']:.3f} °C")
    cols[2].metric("Test R²", f"{m['test']['R2']:.3f}")
    cols[3].metric("Train RMSE", f"{m['train']['RMSE']:.3f} °C")

    with st.expander("Model configuration / hyperparameters"):
    params = m.get("params", None)
    
    if params:
        st.json(params)
    else:
        st.info("No hyperparameters logged for this model.")

    st.subheader("Actual vs. Predicted")
    split_choice = st.radio("Show:", ["Test (held-out)", "Train", "Both"], horizontal=True, key=f"split_{selected}")
    col = f"pred_{selected}"
    df_plot = predictions[["actual", "split", col]].dropna(subset=[col])
    if split_choice == "Test (held-out)":
        df_plot = df_plot[df_plot["split"] == "test"]
    elif split_choice == "Train":
        df_plot = df_plot[df_plot["split"] == "train"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["actual"], mode="lines", name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], mode="lines", name=f"{selected} Prediction",
                              line=dict(color=MODEL_COLORS.get(selected, "red"))))
    fig.update_layout(xaxis_title="Date", yaxis_title="Temperature (°C)", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Residuals (Actual − Predicted)")
    resid = df_plot["actual"] - df_plot[col]
    c1, c2 = st.columns(2)
    with c1:
        fig_r = px.histogram(resid, nbins=40, title="Residual distribution")
        fig_r.update_layout(xaxis_title="Error (°C)", showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)
    with c2:
        fig_r2 = px.scatter(x=df_plot["actual"], y=resid, title="Residual vs. Actual value",
                             labels={"x": "Actual Temperature (°C)", "y": "Residual (°C)"})
        fig_r2.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_r2, use_container_width=True)
    st.caption(
        "**How to read this:** a residual histogram centered near 0 with no strong "
        "skew means the model isn't systematically over- or under-predicting. A "
        "funnel or trend shape in the scatter plot would suggest the model's error "
        "depends on the temperature level itself (e.g. it might be worse on hot "
        "days vs. cold days)."
    )

    if "loss_curve" in m:
        st.subheader("Training curve")
        loss_df = pd.DataFrame({"Train loss": m["loss_curve"], "Validation loss": m["val_loss_curve"]})
        fig_l = px.line(loss_df, title=f"{selected} — Loss per epoch (MSE on scaled data)")
        fig_l.update_layout(xaxis_title="Epoch", yaxis_title="MSE Loss")
        st.plotly_chart(fig_l, use_container_width=True)
        st.caption(
            "Train and validation loss decreasing together = the model is learning "
            "genuine structure. If validation loss starts rising while train loss "
            "keeps falling, that's the textbook sign of overfitting (early stopping "
            "was used here specifically to catch and avoid that)."
        )

    if selected == "Prophet" and artifacts["prophet_components"] is not None:
        st.subheader("Prophet's decomposed components")
        pc = artifacts["prophet_components"]
        fig_pc = go.Figure()
        for c in pc.columns:
            fig_pc.add_trace(go.Scatter(x=pc.index, y=pc[c], mode="lines", name=c.capitalize()))
        fig_pc.update_layout(title="Prophet trend / yearly seasonality components", xaxis_title="Date")
        st.plotly_chart(fig_pc, use_container_width=True)
        st.caption(
            "This is Prophet's biggest interpretability advantage: you can see "
            "*exactly* how much of any prediction came from the long-term trend "
            "vs. the time of year, in physically meaningful units (°C)."
        )

# ===========================================================================
TAB_FORECAST = tabs[5]
with TAB_FORECAST:
    st.header("Try it yourself: pick a model, pick a date, see the forecast in context")

    selected_fc = st.selectbox("Model to inspect", available_models, key="forecast_model")
    col = f"pred_{selected_fc}"
    df_fc = predictions[["actual", "split", col]].dropna(subset=[col])

    min_d, max_d = df_fc.index.min().date(), df_fc.index.max().date()
    date_range = st.slider(
        "Date window to display",
        min_value=min_d, max_value=max_d,
        value=(df_fc.index[-180].date(), max_d),
        format="YYYY-MM-DD",
    )
    window_df = df_fc.loc[str(date_range[0]) : str(date_range[1])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window_df.index, y=window_df["actual"], mode="lines+markers",
                              name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=window_df.index, y=window_df[col], mode="lines+markers",
                              name=f"{selected_fc} Prediction", line=dict(color=MODEL_COLORS.get(selected_fc))))
    fig.update_layout(title=f"{selected_fc}: zoomed-in window", xaxis_title="Date", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig, use_container_width=True)

    err = (window_df["actual"] - window_df[col]).abs()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg. absolute error in window", f"{err.mean():.2f} °C")
    c2.metric("Worst single-day error", f"{err.max():.2f} °C")
    c3.metric("Days shown", f"{len(window_df)}")

    st.markdown("---")
    st.subheader("How confident should you be in any single day's forecast?")
    st.markdown(
        f"""
Across the full held-out test period, **{selected_fc}** misses by an average
of **{metrics[selected_fc]['test']['MAE']:.2f}°C** (MAE) and explains
**{metrics[selected_fc]['test']['R2']*100:.1f}%** of the variance in actual
temperatures (R²) it never saw during training. Treat any single-day forecast
as "the actual value is most likely within about ±{metrics[selected_fc]['test']['MAE']:.1f}°C
of this prediction", not as an exact number — that uncertainty is inherent
to weather, not a flaw unique to this model.
        """
    )

# ===========================================================================
TAB_CONCLUSION = tabs[6]
with TAB_CONCLUSION:
    st.header("Putting it all together")

    best_model = leaderboard.sort_values("Test_RMSE").index[0]
    worst_model = leaderboard.sort_values("Test_RMSE").index[-1]

    st.success(
        f"**Best performer (lowest test RMSE): {best_model}** — "
        f"RMSE {leaderboard.loc[best_model, 'Test_RMSE']:.3f}°C, "
        f"R² {leaderboard.loc[best_model, 'Test_R2']:.3f}"
    )
    st.warning(
        f"**Weakest performer: {worst_model}** — "
        f"RMSE {leaderboard.loc[worst_model, 'Test_RMSE']:.3f}°C, "
        f"R² {leaderboard.loc[worst_model, 'Test_R2']:.3f}"
    )

    st.markdown(
        """
### Key takeaways

1. **Bigger / fancier ≠ better, automatically.** Deep learning models
   (LSTM/GRU/CNN-RNN) have far more capacity than ARIMA, but capacity only
   helps if the data has complex structure for it to capture. On a series
   dominated by one simple repeating yearly wave, a model that's *told about*
   that seasonality directly (SARIMA, Prophet, NeuralProphet) can match or
   beat a generic sequence model that has to discover the cycle from scratch
   using only a short window of recent days.

2. **Match the model to the structure you found in EDA.** This is why we did
   the seasonal decomposition and stationarity test *before* modeling — they
   told us in advance that seasonality dominates and trend/noise are
   secondary, which is exactly the kind of insight that should drive model
   selection.

3. **Interpretability is a real cost/benefit trade-off.** Prophet and
   NeuralProphet hand you a clean trend/seasonality breakdown you can show to
   a non-technical stakeholder. A LSTM's internal gates are not meaningfully
   inspectable in the same way — you mostly have to trust its aggregate
   error metrics.

4. **A 1-day-ahead forecast on noisy daily weather will never be perfect.**
   Even the best model here leaves meaningful residual error, because daily
   minimum temperature has a real random weather component that no amount of
   historical pattern-matching can eliminate.

### Suggested next steps if you wanted to push this further
- Add exogenous weather features (humidity, pressure, wind) as model inputs.
- Try multi-step-ahead forecasting (7, 14, 30 days) instead of just 1 day.
- Ensemble the top 2–3 models (e.g. average SARIMA + NeuralProphet).
- Quantify forecast *uncertainty* (prediction intervals), not just a point estimate.
        """
    )

# ===========================================================================
TAB_MAP = tabs[7]
with TAB_MAP:
    st.header("🗺️ Where this data actually comes from")
    st.markdown(
        "The dataset has a single weather station — **Melbourne, Victoria,\n"
        "Australia** — so this tab maps that one location and lets you scrub\n"
        "through the calendar year to see how its *climatological* average\n"
        "minimum temperature swings between seasons."
    )

    raw_df_map = raw_df.copy()
    raw_df_map["month"] = raw_df_map.index.month
    monthly_clim = raw_df_map.groupby("month")[TARGET_COL].agg(["mean", "min", "max", "std"])
    month_names = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]
    monthly_clim.index = [month_names[m - 1] for m in monthly_clim.index]
    # restore numeric month order after renaming
    monthly_clim = monthly_clim.loc[month_names]

    col_map, col_info = st.columns([2, 1])

    with col_map:
        month_choice = st.select_slider("Month", options=month_names, value="January")
        avg_temp = monthly_clim.loc[month_choice, "mean"]

        # Color scale: cold blue -> warm orange/red, fixed range across all months
        vmin, vmax = monthly_clim["mean"].min(), monthly_clim["mean"].max()
        norm = (avg_temp - vmin) / (vmax - vmin + 1e-9)
        marker_color = px.colors.sample_colorscale("RdYlBu_r", [norm])[0]

        fig_map = go.Figure(go.Scattermapbox(
            lat=[MELBOURNE_LAT],
            lon=[MELBOURNE_LON],
            mode="markers+text",
            marker=dict(size=38, color=marker_color, opacity=0.9),
            text=[f"Melbourne<br>{avg_temp:.1f}°C avg"],
            textposition="top center",
            textfont=dict(size=14, color="#1a1a2e"),
            hovertemplate=f"<b>Melbourne</b><br>{month_choice} avg min temp: {avg_temp:.1f}°C<extra></extra>",
        ))
        fig_map.update_layout(
            mapbox=dict(style="open-street-map", center=dict(lat=MELBOURNE_LAT, lon=MELBOURNE_LON), zoom=4),
            margin=dict(l=0, r=0, t=10, b=0),
            height=480,
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            "Marker color encodes how warm/cold this month is *relative to "
            "Melbourne's own yearly range* (blue = coldest months, red = warmest), "
            "using the open street map base layer — no API key required."
        )

    with col_info:
        st.subheader(f"{month_choice} climatology")
        st.metric("Average minimum temp", f"{avg_temp:.1f} °C")
        st.metric("Coldest on record (this month)", f"{monthly_clim.loc[month_choice, 'min']:.1f} °C")
        st.metric("Warmest on record (this month)", f"{monthly_clim.loc[month_choice, 'max']:.1f} °C")
        st.metric("Day-to-day variability (std dev)", f"{monthly_clim.loc[month_choice, 'std']:.1f} °C")

    st.markdown("---")
    st.subheader("All 12 months at a glance")
    fig_bar = px.bar(
        monthly_clim.reset_index().rename(columns={"index": "Month"}),
        x="Month", y="mean", color="mean", color_continuous_scale="RdYlBu_r",
        labels={"mean": "Avg min temp (°C)"}, title="Average minimum temperature by month",
    )
    fig_bar.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="°C")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        "This is the exact seasonal wave every seasonality-aware model "
        "(SARIMA, Prophet, NeuralProphet) is learning to exploit — and the "
        "main reason those models have a structural head start on this "
        "particular dataset, as discussed in **Concepts 101**."
    )

    st.info(
        "💡 **Working with multiple weather stations?** If you extend this "
        "project with more cities, add their `(name, lat, lon, avg_temp)` "
        "rows to the `Scattermapbox` call above — the map and color scale "
        "will handle any number of locations automatically."
    )
