"""Web UI for trying the analyzer interactively.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import calendar
import io
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from climate_analyzer.analysis import trend_line
from climate_analyzer.cleaning import OutlierMethod
from climate_analyzer.io import DataValidationError, datetime_index, load_csv, load_sample
from climate_analyzer.pipeline import (
    AnalysisConfig,
    AnalysisResult,
    default_aggregation,
    run_analysis,
)
from climate_analyzer.plotting import ACCENT, COLD, NEUTRAL, WARM
from climate_analyzer.sources import DataSourceError, fetch_city

SAMPLE = "Example data"
LIVE = "Real Astana weather"
UPLOAD = "My own CSV file"

VARIABLE_NAMES = {
    "tavg": "Average daily temperature",
    "tmin": "Coldest temperature of the day",
    "tmax": "Warmest temperature of the day",
    "prcp": "Precipitation (rain and snow)",
}
OUTLIER_LABELS = {
    "zscore": "Standard: far from the monthly average",
    "iqr": "Robust: far outside the usual monthly range",
    "none": "Off: keep every value",
}
WARM_FILL = "rgba(192, 57, 43, 0.2)"
CHART_TIP = (
    "Hover over the chart for exact values, drag to zoom in, double-click to zoom back out, "
    "and click a name in the legend to hide or show that line."
)


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


def _variable_label(column: str) -> str:
    return f"{VARIABLE_NAMES[column]} ({column})" if column in VARIABLE_NAMES else column


def _style(fig: go.Figure, y_title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
    )
    fig.update_yaxes(hoverformat=".2f")
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


def daily_chart(result: AnalysisResult, units: str) -> go.Figure:
    daily = result.cleaned
    fig = go.Figure()
    fig.add_scatter(
        x=daily.index,
        y=daily,
        mode="lines",
        name="Daily value",
        line={"color": NEUTRAL, "width": 1},
    )
    fig.update_xaxes(rangeslider_visible=True)
    return _style(fig, units)


def annual_chart(result: AnalysisResult, units: str, yearly_name: str) -> go.Figure:
    annual, trend = result.annual, result.trend
    index = datetime_index(annual)
    fig = go.Figure()
    fig.add_scatter(
        x=index.year,
        y=annual,
        mode="lines+markers",
        name=yearly_name,
        line={"color": COLD, "width": 1},
    )
    fig.add_scatter(
        x=index.year,
        y=result.smoothed,
        mode="lines",
        name=f"Smoothed ({result.config.smoothing_years}-year average)",
        line={"color": ACCENT, "width": 3},
    )
    if trend is not None:
        line = trend_line(annual.set_axis(index + pd.DateOffset(months=6)), trend)
        fig.add_scatter(
            x=index.year,
            y=line,
            mode="lines",
            name=f"Straight-line trend ({trend.slope_per_decade:+.2f} {units} per decade)",
            line={"color": WARM, "width": 2, "dash": "dash"},
        )
    fig.update_xaxes(title_text="Year")
    return _style(fig, units)


def forecast_chart(result: AnalysisResult, units: str, history_years: int = 6) -> go.Figure:
    monthly, forecast, backtest = result.monthly, result.forecast, result.backtest
    assert forecast is not None
    end = datetime_index(monthly)[-1]
    start = end - pd.DateOffset(years=history_years)
    recent = monthly[monthly.index > start]

    fig = go.Figure()
    fig.add_scatter(
        x=recent.index,
        y=recent,
        mode="lines+markers",
        name="Measured",
        line={"color": NEUTRAL, "width": 1},
        marker={"size": 4},
    )
    if backtest is not None:
        test = backtest.predictions[backtest.predictions.index > start]
        fig.add_scatter(
            x=test.index,
            y=test["predicted"],
            mode="lines",
            name="Test forecast (made without seeing these months)",
            line={"color": COLD, "width": 2},
        )
    # Two invisible lines; the area between them is the prediction interval.
    fig.add_scatter(
        x=forecast.index,
        y=forecast["upper"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=forecast.index,
        y=forecast["lower"],
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor=WARM_FILL,
        name="Likely range (95%)",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=forecast.index,
        y=forecast["forecast"],
        mode="lines",
        name="Forecast",
        line={"color": WARM, "width": 3},
        customdata=forecast[["lower", "upper"]].to_numpy(),
        hovertemplate="%{y:.1f} (likely between %{customdata[0]:.1f} and %{customdata[1]:.1f})",
    )
    fig.add_vline(x=end, line_dash="dot", line_color=NEUTRAL)
    return _style(fig, units)


def anomaly_chart(result: AnalysisResult, units: str) -> go.Figure:
    anomalies = result.annual_anomalies
    colors = [WARM if value >= 0 else COLD for value in anomalies.fillna(0)]
    fig = go.Figure()
    fig.add_bar(
        x=datetime_index(anomalies).year,
        y=anomalies,
        marker_color=colors,
        name="Difference from normal",
    )
    fig.update_xaxes(title_text="Year")
    return _style(fig, f"Difference from normal ({units})")


def climatology_chart(result: AnalysisResult, units: str) -> go.Figure:
    clim = result.climatology
    fig = go.Figure()
    fig.add_scatter(
        x=[calendar.month_abbr[month] for month in clim.index],
        y=clim,
        mode="lines+markers",
        name="Typical value",
        line={"color": COLD, "width": 3},
    )
    return _style(fig, units)


def decomposition_chart(result: AnalysisResult, units: str) -> go.Figure:
    assert result.decomposition is not None
    parts = result.decomposition.to_frame()
    titles = [
        "Monthly values (what was measured)",
        "Long-term direction",
        "Pattern that repeats every year",
        "What is left over (irregular ups and downs)",
    ]
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.07
    )
    colors = [NEUTRAL, WARM, COLD, ACCENT]
    for row, (column, color) in enumerate(zip(parts.columns, colors, strict=True), start=1):
        fig.add_scatter(
            x=parts.index,
            y=parts[column],
            mode="lines",
            name=titles[row - 1],
            line={"color": color, "width": 1.5},
            row=row,
            col=1,
        )
    fig.update_layout(showlegend=False)
    return _style(fig, units, height=820)


def extremes_chart(result: AnalysisResult, high_name: str, low_name: str) -> go.Figure:
    extremes = result.extremes
    assert extremes is not None
    fig = go.Figure()
    fig.add_bar(x=extremes.index, y=extremes["warm_days"], name=high_name, marker_color=WARM)
    fig.add_bar(x=extremes.index, y=extremes["cold_days"], name=low_name, marker_color=COLD)
    fig.update_xaxes(title_text="Year")
    return _style(fig, "Days per year")


def _headline(result: AnalysisResult, name: str, units: str) -> str:
    trend = result.trend
    if trend is None:
        return f"There are not enough complete years to tell whether {name.lower()} is changing."
    direction = "rising" if trend.slope_per_decade > 0 else "falling"
    change = f"{abs(trend.slope_per_decade):.2f} {units} every ten years"
    if trend.is_significant():
        return (
            f"**{name} has been {direction} by about {change}.** The change is steady enough "
            "that it is very unlikely to be a coincidence."
        )
    return (
        f"**{name} looks like it is {direction} by about {change}, but this is not certain.** "
        "The year-to-year ups and downs are large enough that it could be a coincidence."
    )


def _explain(summary: str, details: str) -> None:
    st.markdown(summary)
    with st.expander("How is this calculated?"):
        st.markdown(details)


st.set_page_config(page_title="Climate Analyzer", layout="wide")
st.title("Climate Analyzer")
st.caption(
    "See how the weather in Astana (or at your own station) has changed over the years, "
    "what a typical year looks like, and what the coming months may bring."
)

with st.sidebar:
    st.header("1. Choose the data")
    source = st.radio(
        "Where should the data come from?",
        [SAMPLE, LIVE, UPLOAD],
        captions=[
            "Made-up but realistic Astana weather, 1994–2023. Works offline.",
            "Real measurements downloaded from Open-Meteo. Needs internet.",
            "One row per day, a date column and at least one column of numbers.",
        ],
    )
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
        column = st.selectbox(
            "What do you want to look at?",
            list(df.columns),
            format_func=_variable_label,
            help="Each option is one column of the data. The short code in brackets is the "
            "column name in the file.",
        )

        st.header("2. Set the comparison")
        years = datetime_index(df).year
        first, last = int(years.min()), int(years.max())
        baseline = st.slider(
            "Years that count as 'normal'",
            first,
            last,
            (first, min(first + 19, last)),
            help="Every year is compared with the average of these years. Climate scientists "
            "call this the baseline period and usually use 20–30 years.",
        )
        smoothing = st.slider(
            "Smooth the yearly line over (years)",
            1,
            15,
            5,
            help="Single years jump up and down a lot. Averaging each year with its neighbours "
            "(a moving average) hides those jumps so the long-term direction is easier to see. "
            "1 means no smoothing.",
        )

        st.header("3. Forecast")
        forecast_months = st.slider(
            "How many months ahead?",
            1,
            60,
            24,
            help="The further ahead, the wider and less certain the forecast becomes.",
        )

        with st.expander("Advanced settings (the defaults are fine)"):
            units = st.text_input(
                "Units",
                _default_units(column),
                help="Only used for labels on charts and in the report.",
            )
            outlier_method: OutlierMethod = st.selectbox(
                "Remove suspicious values",
                ["zscore", "iqr", "none"],
                format_func=OUTLIER_LABELS.__getitem__,
                help="Sensors sometimes record impossible numbers (outliers), like +60 °C in "
                "January. **Standard** removes values more than 4 standard deviations from that "
                "month's average (z-score). **Robust** removes values far outside the range that "
                "holds the middle half of that month's values (IQR); it is less easily fooled "
                "when there are many bad values. **Off** keeps everything.",
            )
            max_gap = st.slider(
                "Fill in missing days, up to (days in a row)",
                0,
                14,
                3,
                help="Short gaps are filled by drawing a straight line between the days before "
                "and after (interpolation). Longer gaps are left empty, because inventing that "
                "much weather would distort the results. 0 means never fill.",
            )
            backtest_months = st.slider(
                "Check the forecast against the last (months)",
                12,
                120,
                36,
                step=12,
                help="To see how trustworthy the forecast is, the app hides this many of the most "
                "recent months, predicts them from the earlier data, and compares the prediction "
                "with what really happened (a backtest).",
            )

if df is None:
    st.info("Choose the example data or upload a CSV file in the sidebar to start.")
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
name = VARIABLE_NAMES.get(column, column)
is_total = config.resolved_aggregation() == "sum"
yearly_name = "Yearly total" if is_total else "Yearly average"
normal_years = f"{baseline[0]}–{baseline[1]}"

st.markdown(_headline(result, name, units))

cols = st.columns(5)
cols[0].metric(
    "Change per decade",
    f"{trend.slope_per_decade:+.2f} {units}" if trend else "n/a",
    help="How much the yearly value goes up (+) or down (−) every ten years, on average.",
)
cols[1].metric(
    "Is the change real?",
    ("Yes" if trend.is_significant() else "Not sure") if trend else "n/a",
    help="A statistical test (Mann-Kendall) estimates how likely it is to see a change like "
    "this purely by chance. Below 5% counts as a real change. "
    + (f"Here the chance is {trend.mann_kendall_p_value:.2%} (the p-value)." if trend else ""),
)
cols[2].metric(
    "Average day",
    f"{result.summary['mean']:.2f} {units}",
    help="The average of all daily values in the whole record.",
)
cols[3].metric(
    "Usable data",
    f"{result.cleaning.completeness:.1%}",
    help="Share of days that have a value after suspicious values were removed and short "
    "gaps were filled. Below 90% the results become less reliable.",
)
if result.backtest is not None:
    cols[4].metric(
        "Forecast quality",
        f"{result.backtest.skill:+.1%}",
        help="How much smaller the forecast's error is than simply guessing the usual value "
        "for each month. 0% means no better than that guess, 100% would be perfect, and a "
        "negative number means worse than guessing.",
    )
for warning in result.warnings:
    st.warning(warning)

with st.expander("New here? How to use this page"):
    st.markdown(
        f"""
1. **Pick the data** in the sidebar on the left, then what you want to look at.
2. **Read the sentence and numbers above** for the short answer. Hover over the **?** next to
   any number or setting for an explanation.
3. **Open the tabs below** for the details. Each tab starts with what the chart shows and how
   to read it.

{CHART_TIP}

**Words used on this page**

- **Normal / baseline** – the average of the years you chose as 'normal' ({normal_years}).
- **Trend** – the overall direction over many years, ignoring single warm or cold years.
- **Difference from normal (anomaly)** – how far a year is above or below normal.
- **Typical year (climatology)** – the usual value for each calendar month.
- **Moving average** – each year averaged with its neighbours to smooth out jumps.
- **Suspicious value (outlier)** – a measurement so far from the usual range that it is
  probably a sensor or typing error.
- **Likely range (prediction interval)** – the band the real value should fall in 95 times
  out of 100.
"""
    )

tabs = st.tabs(
    [
        "Long-term trend",
        "Forecast",
        "Difference from normal",
        "Typical year",
        "Trend and seasons",
        "Extreme days",
        "Report and downloads",
    ]
)
with tabs[0]:
    _explain(
        "**Is it changing over the years?** The first chart has one point per year. The dashed "
        "red line is the overall direction: if it slopes upward, values are rising. The thick "
        "green line follows the yearly points more closely, so you can see warmer and colder "
        "stretches.",
        f"""
- **{yearly_name}** – all daily values of a year combined into one number. Years with less
  than 80% of their days available are left out.
- **Smoothed line** – each year averaged with the years around it ({smoothing} years in total).
- **Straight-line trend** – the straight line that fits the yearly points best (ordinary least
  squares). Its slope is the *change per decade*.
- **Is it real?** – the Mann-Kendall test checks whether later years are higher (or lower) than
  earlier years more often than chance would explain.
""",
    )
    st.caption(CHART_TIP)
    st.plotly_chart(annual_chart(result, units, yearly_name))
    st.markdown(
        "**Every single day.** This is the cleaned daily data everything else is built from. "
        "The regular waves are the seasons. Drag the small chart underneath to move through "
        "the years."
    )
    st.plotly_chart(daily_chart(result, units))
with tabs[1]:
    if result.forecast is None:
        st.info("Not enough data to fit the forecast model.")
    else:
        _explain(
            "**What may the coming months bring?** Grey is what was measured, the red line is "
            "the forecast, and the red band is the range the real value should fall in 95 times "
            "out of 100. The blue line is a test: a forecast for months the model was not "
            "allowed to see. The closer blue is to grey, the more you can trust the red line.",
            """
The forecast adds three things together:

1. **The long-term direction** – a straight line through all monthly values.
2. **The seasons** – a smooth wave that repeats every 12 months.
3. **Memory** – if last month was unusually warm or cold, part of that usually carries over
   into the next month.

It is a statistical forecast of monthly values, not a weather forecast: it cannot say what
a particular day will be like.
""",
        )
        st.plotly_chart(forecast_chart(result, units))
        if result.backtest is not None:
            b = result.backtest
            st.markdown(
                f"**How good was the test?** For the last {b.n_test} months, the forecast was "
                f"off by **{b.mae:.2f} {units}** on average. Simply guessing the usual value for "
                f"each month would have been off by **{b.climatology_mae:.2f} {units}**. "
                f"{b.coverage:.0%} of the real values fell inside the likely range "
                "(ideally about 95%)."
            )
        with st.expander("Show the forecast as a table"):
            table = result.forecast.round(2).rename(
                columns={
                    "forecast": f"Forecast ({units})",
                    "lower": "Lowest likely",
                    "upper": "Highest likely",
                }
            )
            st.dataframe(table.set_axis(datetime_index(table).strftime("%b %Y")))
with tabs[2]:
    _explain(
        f"**Which years were unusual?** Each bar is one year compared with normal "
        f"({normal_years}). Red bars are years above normal, blue bars are years below normal. "
        "More and taller red bars towards the right mean recent years are higher than they "
        "used to be.",
        f"""
The **normal** value is the average of the {yearly_name.lower()}s in {normal_years}. Each bar
is that year's value minus the normal value. Scientists call this difference an *anomaly*.
You can change the normal years in the sidebar.
""",
    )
    st.plotly_chart(anomaly_chart(result, units))
with tabs[3]:
    _explain(
        f"**What does a normal year look like?** The usual value for each month, based on "
        f"{normal_years}. Use it to see how big the difference between the seasons is.",
        f"""
All Januaries in {normal_years} are averaged into one January value, all Februaries into one
February value, and so on. Scientists call the result the *climatology*.
""",
    )
    st.plotly_chart(climatology_chart(result, units))
with tabs[4]:
    if result.decomposition is None:
        st.info("Not enough data to separate the trend from the seasons.")
    else:
        _explain(
            "**What is hiding behind the seasons?** Summer and winter differ so much that slow "
            "changes are hard to see. Here the monthly values (top) are split into three parts "
            "that add up to them again: the slow long-term direction, the pattern that repeats "
            "every year, and the irregular rest.",
            """
- **Long-term direction** – the average of the 12 months around each month, which cancels
  out the seasons.
- **Yearly pattern** – what is left after removing the direction, averaged for each calendar
  month.
- **Left over** – monthly values minus the two parts above. Large spikes are unusually warm,
  cold, wet or dry months.

This is called a *classical additive seasonal decomposition*.
""",
        )
        st.plotly_chart(decomposition_chart(result, units))
with tabs[5]:
    if result.extremes is None:
        st.info(
            "Extreme days are only counted for values that are averaged, like temperature, "
            "not for totals like precipitation."
        )
    else:
        _explain(
            f"**Are extreme days becoming more common?** For each year, the number of days "
            f"that were unusually high (red) or unusually low (blue) for their time of year. "
            f"In the normal years ({normal_years}) about 1 day in 10 is of each kind, roughly "
            "36 a year.",
            f"""
A day counts as **unusually high** when it is above 9 out of 10 days of the same calendar
month in {normal_years} (above the 90th percentile), and as **unusually low** when it is
below 9 out of 10 of them (below the 10th percentile). Comparing with the same month means a
mild January day can count as unusually high even though it is colder than any July day.
""",
        )
        st.plotly_chart(extremes_chart(result, "Unusually high days", "Unusually low days"))
        with st.expander("Show the counts as a table"):
            st.dataframe(
                result.extremes.rename(
                    columns={"warm_days": "Unusually high days", "cold_days": "Unusually low days"}
                )
            )
with tabs[6]:
    st.markdown(
        "**Take the results with you.** The full report with every number, in the technical "
        "terms a scientist would use, plus the cleaned data."
    )
    downloads = st.columns(3)
    downloads[0].download_button(
        "Download report (Markdown)", result.to_markdown(), "report.md", help="Readable text."
    )
    downloads[1].download_button(
        "Download summary (JSON)",
        result.to_json(),
        "summary.json",
        help="All numbers in a format other programs can read.",
    )
    downloads[2].download_button(
        "Download cleaned data (CSV)",
        result.cleaned.to_csv(),
        f"{column}_cleaned.csv",
        help="The daily values after cleaning. Opens in Excel.",
    )
    st.divider()
    st.markdown(result.to_markdown())
