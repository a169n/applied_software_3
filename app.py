"""Minimal web UI for trying the analyzer interactively.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import streamlit as st

from climate_analyzer import plotting
from climate_analyzer.cleaning import OutlierMethod
from climate_analyzer.io import DataValidationError, datetime_index, load_csv, load_sample
from climate_analyzer.pipeline import AnalysisConfig, default_aggregation, run_analysis
from climate_analyzer.sources import DataSourceError, fetch_city

SAMPLE = "Astana sample (synthetic, offline)"
LIVE = "Astana real data (Open-Meteo)"
UPLOAD = "Upload CSV"


@st.cache_data
def _sample() -> pd.DataFrame:
    return load_sample()


@st.cache_data
def _uploaded(content: bytes) -> pd.DataFrame:
    return load_csv(io.BytesIO(content))


@st.cache_data(ttl="1d", show_spinner="Downloading Astana weather history...")
def _astana(start_year: int) -> pd.DataFrame:
    return fetch_city("astana", date(start_year, 1, 1), date(date.today().year - 1, 12, 31))


def _default_units(column: str) -> str:
    return "mm" if default_aggregation(column) == "sum" else "°C"


st.set_page_config(page_title="Climate Analyzer", layout="wide")
st.title("Climate Analyzer")
st.caption(
    "Clean daily weather data for Astana (or your own station), study long-term trends, "
    "anomalies and extremes, and forecast the coming months."
)

with st.sidebar:
    st.header("Data")
    source = st.radio("Source", [SAMPLE, LIVE, UPLOAD])
    df: pd.DataFrame | None = None
    try:
        if source == UPLOAD:
            upload = st.file_uploader("CSV with a date column", type="csv")
            if upload is not None:
                df = _uploaded(upload.getvalue())
        elif source == LIVE:
            df = _astana(1994)
        else:
            df = _sample()
    except DataValidationError as exc:
        st.error(f"Could not read the file: {exc}")
    except DataSourceError as exc:
        st.error(f"Download failed: {exc}")

    if df is not None:
        column = st.selectbox("Variable", list(df.columns))
        units = st.text_input("Units", _default_units(column))

        st.header("Cleaning")
        outlier_method: OutlierMethod = st.selectbox("Outlier detection", ["zscore", "iqr", "none"])
        max_gap = st.slider("Interpolate gaps up to (days)", 0, 14, 3)

        st.header("Analysis")
        years = datetime_index(df).year
        first, last = int(years.min()), int(years.max())
        baseline = st.slider("Baseline period", first, last, (first, min(first + 19, last)))
        smoothing = st.slider("Moving average (years)", 1, 15, 5)

        st.header("Forecast")
        forecast_months = st.slider("Forecast horizon (months)", 1, 60, 24)
        backtest_months = st.slider("Backtest period (months)", 12, 120, 36, step=12)

if df is None:
    st.info("Choose the sample dataset or upload a CSV file to start.")
    st.stop()

config = AnalysisConfig(
    column=column,
    units=units,
    outlier_method=outlier_method,
    max_gap=max_gap,
    baseline=baseline,
    smoothing_years=smoothing,
    forecast_months=forecast_months,
    backtest_months=backtest_months,
)
try:
    result = run_analysis(df, config)
except ValueError as exc:
    st.error(f"Analysis failed: {exc}")
    st.stop()

trend = result.trend
cols = st.columns(5)
cols[0].metric("Trend per decade", f"{trend.slope_per_decade:+.2f} {units}" if trend else "n/a")
cols[1].metric("Mann-Kendall p-value", f"{trend.mann_kendall_p_value:.3g}" if trend else "n/a")
cols[2].metric("Mean", f"{result.summary['mean']:.2f} {units}")
cols[3].metric("Completeness", f"{result.cleaning.completeness:.1%}")
if result.backtest is not None:
    cols[4].metric("Forecast skill", f"{result.backtest.skill:+.1%}")
for warning in result.warnings:
    st.warning(warning)

tabs = st.tabs(
    ["Trend", "Forecast", "Anomalies", "Climatology", "Decomposition", "Extremes", "Report"]
)
with tabs[0]:
    st.pyplot(
        plotting.plot_time_series(result.cleaned, result.annual, trend, result.smoothed, units)
    )
with tabs[1]:
    if result.forecast is None:
        st.info("Not enough data to fit the forecast model.")
    else:
        st.pyplot(
            plotting.plot_forecast(result.monthly, result.forecast, result.backtest, units=units)
        )
        if result.backtest is not None:
            b = result.backtest
            st.markdown(
                f"Backtest on the last {b.n_test} months: mean absolute error "
                f"**{b.mae:.2f} {units}** vs **{b.climatology_mae:.2f} {units}** for "
                f"climatology; {b.coverage:.0%} of observations inside the 95% interval."
            )
        st.dataframe(result.forecast.round(2))
with tabs[2]:
    st.pyplot(plotting.plot_anomalies(result.annual_anomalies, units))
with tabs[3]:
    st.pyplot(plotting.plot_climatology(result.climatology, units=units))
with tabs[4]:
    if result.decomposition is None:
        st.info("Not enough data for a seasonal decomposition.")
    else:
        st.pyplot(plotting.plot_decomposition(result.decomposition, units))
with tabs[5]:
    if result.extremes is None:
        st.info("Extreme-day counts are only computed for averaged variables like temperature.")
    else:
        st.pyplot(plotting.plot_extremes(result.extremes))
        st.dataframe(result.extremes)
with tabs[6]:
    st.markdown(result.to_markdown())
    st.download_button("Download report (Markdown)", result.to_markdown(), "report.md")
    st.download_button("Download summary (JSON)", result.to_json(), "summary.json")
    st.download_button(
        "Download cleaned data (CSV)", result.cleaned.to_csv(), f"{column}_cleaned.csv"
    )
