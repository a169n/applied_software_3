"""Matplotlib figures for the analysis results.

All functions build a :class:`matplotlib.figure.Figure` directly instead of
going through ``pyplot``.  That avoids global state, works without a display
(CI servers) and is safe to call from the Streamlit web UI.
"""

from __future__ import annotations

import calendar

import pandas as pd
from matplotlib.figure import Figure

from .analysis import Decomposition, TrendResult, trend_line
from .forecast import BacktestResult
from .io import datetime_index

WARM = "#c0392b"
COLD = "#2471a3"
NEUTRAL = "#7f8c8d"
ACCENT = "#1e8449"


def _new_figure(height: float = 4.0, nrows: int = 1) -> tuple[Figure, list]:
    fig = Figure(figsize=(10, height), layout="constrained")
    axes = fig.subplots(nrows, 1, sharex=nrows > 1, squeeze=False)[:, 0]
    for ax in axes:
        ax.grid(alpha=0.3)
    return fig, list(axes)


def plot_time_series(
    daily: pd.Series,
    annual: pd.Series,
    trend: TrendResult | None = None,
    smoothed: pd.Series | None = None,
    units: str = "",
) -> Figure:
    """Top: all observations. Bottom: annual values, moving average and trend."""
    fig, (top, bottom) = _new_figure(height=6, nrows=2)
    label = units or str(daily.name)
    top.plot(daily.index, daily, color=NEUTRAL, lw=0.4)
    top.set_title(f"{daily.name}: observations")
    top.set_ylabel(label)

    mid_year = datetime_index(annual) + pd.DateOffset(months=6)
    bottom.plot(mid_year, annual, "o-", color=COLD, ms=3, lw=1, label="Annual value")
    if smoothed is not None:
        bottom.plot(mid_year, smoothed, color=ACCENT, lw=2, label="Moving average")
    if trend is not None:
        line = trend_line(annual.set_axis(mid_year), trend)
        bottom.plot(
            line.index,
            line,
            "--",
            color=WARM,
            lw=2,
            label=f"Trend {trend.slope_per_decade:+.2f} {units}/decade",
        )
    bottom.set_title("Annual values and long-term trend")
    bottom.set_ylabel(label)
    bottom.legend(loc="upper left", fontsize=8)
    return fig


def plot_anomalies(annual_anomalies: pd.Series, units: str = "") -> Figure:
    """Warming-stripes style bar chart of annual anomalies."""
    fig, (ax,) = _new_figure()
    colors = [WARM if v >= 0 else COLD for v in annual_anomalies.fillna(0)]
    ax.bar(datetime_index(annual_anomalies).year, annual_anomalies, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Annual anomaly relative to the baseline climatology")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Anomaly {units}".strip())
    return fig


def plot_climatology(clim: pd.Series, recent: pd.Series | None = None, units: str = "") -> Figure:
    """Mean annual cycle; optionally compared with a more recent period."""
    fig, (ax,) = _new_figure()
    labels = [calendar.month_abbr[m] for m in clim.index]
    ax.plot(labels, clim, "o-", color=COLD, label="Baseline")
    if recent is not None:
        ax.plot(labels, recent.reindex(clim.index), "o-", color=WARM, label="Recent period")
        ax.legend(fontsize=8)
    ax.set_title("Mean annual cycle (climatology)")
    ax.set_ylabel(units or str(clim.name))
    return fig


def plot_decomposition(decomposition: Decomposition, units: str = "") -> Figure:
    """Four stacked panels: observed, trend, seasonal and residual components."""
    fig, axes = _new_figure(height=8, nrows=4)
    parts = decomposition.to_frame()
    colors = [NEUTRAL, WARM, COLD, ACCENT]
    for ax, column, color in zip(axes, parts.columns, colors, strict=True):
        ax.plot(parts.index, parts[column], color=color, lw=1)
        ax.set_ylabel(column.capitalize())
    axes[0].set_title(f"Seasonal decomposition {f'({units})' if units else ''}".strip())
    return fig


def plot_extremes(extremes: pd.DataFrame) -> Figure:
    """Number of unusually warm and cold days per year."""
    fig, (ax,) = _new_figure()
    years = extremes.index
    width = 0.4
    ax.bar(years - width / 2, extremes["warm_days"], width, color=WARM, label="Warm days")
    ax.bar(years + width / 2, extremes["cold_days"], width, color=COLD, label="Cold days")
    ax.set_title("Extreme days per year (relative to baseline percentiles)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Days")
    ax.legend(fontsize=8)
    return fig


def plot_forecast(
    monthly: pd.Series,
    forecast: pd.DataFrame,
    backtest: BacktestResult | None = None,
    history_years: int = 6,
    units: str = "",
) -> Figure:
    """Recent monthly history, the hold-out backtest and the forecast with its interval."""
    fig, (ax,) = _new_figure()
    end = datetime_index(monthly)[-1]
    start = end - pd.DateOffset(years=history_years)
    recent = monthly[monthly.index > start]
    ax.plot(recent.index, recent, "o-", color=NEUTRAL, ms=2, lw=1, label="Observed")

    if backtest is not None:
        bt = backtest.predictions[backtest.predictions.index > start]
        ax.plot(bt.index, bt["predicted"], color=COLD, lw=1.5, label="Backtest prediction")

    ax.fill_between(
        forecast.index,
        forecast["lower"],
        forecast["upper"],
        color=WARM,
        alpha=0.2,
        label="95% prediction interval",
    )
    ax.plot(forecast.index, forecast["forecast"], color=WARM, lw=2, label="Forecast")
    ax.axvline(end, color="black", lw=0.8, ls=":")
    ax.set_title("Monthly forecast")
    ax.set_ylabel(units or str(monthly.name))
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    return fig
