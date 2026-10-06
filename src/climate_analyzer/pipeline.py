"""End-to-end analysis pipeline shared by the command line and the web UI."""

from __future__ import annotations

import calendar
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from . import analysis
from .cleaning import CleaningReport, OutlierMethod, clean
from .forecast import BacktestResult, SeasonalTrendModel, backtest
from .io import datetime_index, get_series

#: Column name fragments that identify accumulated quantities (summed, not averaged).
ACCUMULATED_HINTS = ("prcp", "precip", "rain", "snow")


def default_aggregation(column: str) -> analysis.Aggregation:
    """Precipitation-like variables are summed per period, everything else averaged."""
    name = column.lower()
    return "sum" if any(hint in name for hint in ACCUMULATED_HINTS) else "mean"


@dataclass(frozen=True)
class AnalysisConfig:
    """User-tunable settings of one analysis run."""

    column: str = "tavg"
    units: str = "°C"
    outlier_method: OutlierMethod = "zscore"
    outlier_threshold: float | None = None
    max_gap: int = 3
    baseline: analysis.Baseline | None = None
    smoothing_years: int = 5
    aggregation: analysis.Aggregation | None = None
    min_coverage: float = 0.8
    forecast_months: int = 24
    backtest_months: int = 36
    harmonics: int = 2

    def resolved_aggregation(self) -> analysis.Aggregation:
        return self.aggregation or default_aggregation(self.column)


@dataclass
class AnalysisResult:
    """Everything produced by :func:`run_analysis`."""

    config: AnalysisConfig
    cleaned: pd.Series
    cleaning: CleaningReport
    monthly: pd.Series
    annual: pd.Series
    smoothed: pd.Series
    trend: analysis.TrendResult | None
    climatology: pd.Series
    annual_anomalies: pd.Series
    summary: dict[str, Any]
    decomposition: analysis.Decomposition | None = None
    extremes: pd.DataFrame | None = None
    model: SeasonalTrendModel | None = None
    forecast: pd.DataFrame | None = None
    backtest: BacktestResult | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary of the numerical results (no full time series)."""
        cfg = asdict(self.config)
        cfg["aggregation"] = self.config.resolved_aggregation()
        return {
            "config": cfg,
            "summary": self.summary,
            "cleaning": {**asdict(self.cleaning), "completeness": self.cleaning.completeness},
            "trend": self.trend.as_dict() if self.trend else None,
            "trend_significant": self.trend.is_significant() if self.trend else None,
            "climatology": {
                m: _round(v) for m, v in zip(range(1, 13), self.climatology, strict=True)
            },
            "annual": _by_year(self.annual),
            "annual_anomalies": _by_year(self.annual_anomalies),
            "model": self._model_dict(),
            "warnings": self.warnings,
        }

    def _model_dict(self) -> dict[str, Any] | None:
        if self.model is None:
            return None
        info: dict[str, Any] = {
            "harmonics": self.model.harmonics,
            "trend_per_decade": _round(self.model.trend_per_decade),
            "seasonal_amplitude": _round(self.model.seasonal_amplitude),
            "ar1_phi": _round(self.model.phi_),
            "residual_std": _round(self.model.sigma_),
            "r_squared": _round(self.model.r_squared()),
        }
        if self.backtest is not None:
            b = self.backtest
            info["backtest"] = {
                "train_end": b.train_end.date().isoformat(),
                "months": b.n_test,
                "mae": _round(b.mae),
                "rmse": _round(b.rmse),
                "bias": _round(b.bias),
                "climatology_mae": _round(b.climatology_mae),
                "skill": _round(b.skill),
                "interval_coverage": _round(b.coverage),
            }
        if self.forecast is not None:
            months = datetime_index(self.forecast).strftime("%Y-%m")
            info["forecast"] = {
                month: {str(k): _round(v) for k, v in row.items()}
                for month, (_, row) in zip(months, self.forecast.iterrows(), strict=True)
            }
        return info

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Human-readable report."""
        cfg, s, c, t = self.config, self.summary, self.cleaning, self.trend
        u = cfg.units
        baseline = f"{cfg.baseline[0]}–{cfg.baseline[1]}" if cfg.baseline else "full record"
        lines = [
            f"# Climate analysis report: `{cfg.column}`",
            "",
            f"Period **{s['start']} – {s['end']}**, baseline: {baseline}.",
            "",
            "## Data quality",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Observations | {c.total} |",
            f"| Missing in input | {c.missing_before} |",
            f"| Outliers removed ({cfg.outlier_method}) | {c.outliers_removed} |",
            f"| Short gaps interpolated | {c.gaps_filled} |",
            f"| Completeness after cleaning | {c.completeness:.1%} |",
            "",
            "## Summary statistics",
            "",
            f"- Mean: **{s['mean']:.2f} {u}** (standard deviation {s['std']:.2f} {u})",
            f"- Minimum: {s['min']:.1f} {u} on {s['min_date']}",
            f"- Maximum: {s['max']:.1f} {u} on {s['max_date']}",
            "",
            "## Long-term trend (annual values)",
            "",
        ]
        if t is None:
            lines.append("- Not enough complete years to estimate a trend.")
        else:
            verdict = (
                "statistically significant (Mann-Kendall p < 0.05)"
                if t.is_significant()
                else "not statistically significant (Mann-Kendall p ≥ 0.05)"
            )
            lines += [
                f"- OLS slope: **{t.slope_per_decade:+.3f} {u}/decade** "
                f"(± {t.stderr_per_decade:.3f}, R² = {t.r_squared:.2f}, p = {t.p_value:.3g})",
                f"- Sen's slope: {t.sen_slope_per_decade:+.3f} {u}/decade",
                f"- Mann-Kendall τ = {t.mann_kendall_tau:.2f}, p = {t.mann_kendall_p_value:.3g}",
                f"- The trend is {verdict}.",
            ]
        if self.extremes is not None and not self.extremes.empty:
            ex = self.extremes
            half = len(ex) // 2
            lines += [
                "",
                "## Extreme days",
                "",
                f"- Warm days per year: {ex['warm_days'].iloc[:half].mean():.1f} in the first half "
                f"of the record vs {ex['warm_days'].iloc[half:].mean():.1f} in the second half.",
                f"- Cold days per year: {ex['cold_days'].iloc[:half].mean():.1f} in the first half "
                f"of the record vs {ex['cold_days'].iloc[half:].mean():.1f} in the second half.",
            ]
        lines += self._model_markdown()
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines) + "\n"

    def _model_markdown(self) -> list[str]:
        if self.model is None:
            return []
        m, u = self.model, self.config.units
        lines = [
            "",
            "## Statistical model and forecast",
            "",
            f"Monthly values are modelled as linear trend + {m.harmonics} annual harmonic(s) "
            "+ AR(1) residuals.",
            "",
            f"- Trend term: {m.trend_per_decade:+.3f} {u}/decade",
            f"- Amplitude of the annual cycle: {m.seasonal_amplitude:.1f} {u}",
            f"- Residual autocorrelation φ = {m.phi_:.2f}, residual std = {m.sigma_:.2f} {u}",
            f"- Variance explained (R²): {m.r_squared():.3f}",
        ]
        if self.backtest is not None:
            b = self.backtest
            lines += [
                "",
                f"**Backtest** (trained until {b.train_end:%Y-%m}, tested on {b.n_test} months):",
                "",
                "| Metric | Model | Climatology |",
                "|---|---|---|",
                f"| Mean absolute error | {b.mae:.2f} {u} | {b.climatology_mae:.2f} {u} |",
                f"| Root mean square error | {b.rmse:.2f} {u} | |",
                f"| Bias | {b.bias:+.2f} {u} | |",
                "",
                f"Skill score vs climatology: **{b.skill:+.1%}**; "
                f"{b.coverage:.0%} of observations fell inside the 95% prediction interval.",
            ]
        if self.forecast is not None:
            lines += [
                "",
                "**Forecast** (95% prediction interval):",
                "",
                "| Month | Forecast | Lower | Upper |",
                "|---|---|---|---|",
            ]
            for ts, row in self.forecast.head(12).iterrows():
                lines.append(
                    f"| {calendar.month_abbr[ts.month]} {ts.year} | {row['forecast']:.1f} | "
                    f"{row['lower']:.1f} | {row['upper']:.1f} |"
                )
        return lines


def _round(value: Any, digits: int = 3) -> float | None:
    return None if pd.isna(value) else round(float(value), digits)


def _by_year(series: pd.Series) -> dict[int, float | None]:
    years = datetime_index(series).year
    return {int(y): _round(v) for y, v in zip(years, series.to_numpy(), strict=True)}


def _mid_year(series: pd.Series) -> pd.Series:
    """Place annual values in the middle of their year for trend fitting."""
    return series.set_axis(series.index + pd.DateOffset(months=6))


def run_analysis(df: pd.DataFrame, config: AnalysisConfig | None = None) -> AnalysisResult:
    """Clean one variable of ``df`` and compute all statistics.

    ``df`` must be indexed by date, as returned by :func:`climate_analyzer.io.load_csv`.
    """
    config = config or AnalysisConfig()
    how = config.resolved_aggregation()
    warnings: list[str] = []

    raw = get_series(df, config.column)
    cleaned, cleaning_report = clean(
        raw,
        outlier_method=config.outlier_method,
        threshold=config.outlier_threshold,
        max_gap=config.max_gap,
    )
    if cleaning_report.completeness < 0.9:
        warnings.append(
            f"Only {cleaning_report.completeness:.0%} of the observations are usable; "
            "results may be biased."
        )

    monthly = analysis.resample(cleaned, "MS", how=how, min_coverage=config.min_coverage)
    annual = analysis.resample(cleaned, "YS", how=how, min_coverage=config.min_coverage)
    dropped = int(annual.isna().sum())
    if dropped:
        warnings.append(f"{dropped} year(s) excluded because of insufficient coverage.")

    trend = None
    try:
        trend = analysis.linear_trend(_mid_year(annual))
    except ValueError:
        warnings.append("Trend not estimated: fewer than 3 complete years of data.")
    smoothed = analysis.moving_average(annual, config.smoothing_years)

    climatology = analysis.climatology(monthly, config.baseline)
    # Annual values all fall in January, so this is "annual value minus baseline mean".
    annual_anomalies = analysis.anomalies(annual, config.baseline)

    decomposition = None
    try:
        decomposition = analysis.seasonal_decompose(monthly, period=12)
    except ValueError as exc:
        warnings.append(f"Seasonal decomposition skipped: {exc}.")

    extremes = None
    if how == "mean":
        extremes = analysis.extreme_days(cleaned, config.baseline)

    model = forecast = backtest_result = None
    non_negative = how == "sum"
    try:
        model = SeasonalTrendModel(config.harmonics, non_negative=non_negative).fit(monthly)
        if config.forecast_months > 0:
            forecast = model.forecast(config.forecast_months)
        if config.backtest_months > 0:
            backtest_result = backtest(
                monthly, config.backtest_months, config.harmonics, non_negative=non_negative
            )
    except ValueError as exc:
        warnings.append(f"Forecast model skipped: {exc}.")

    return AnalysisResult(
        config=config,
        cleaned=cleaned,
        cleaning=cleaning_report,
        monthly=monthly,
        annual=annual,
        smoothed=smoothed,
        trend=trend,
        climatology=climatology,
        annual_anomalies=annual_anomalies,
        summary=analysis.summary_statistics(cleaned),
        decomposition=decomposition,
        extremes=extremes,
        model=model,
        forecast=forecast,
        backtest=backtest_result,
        warnings=warnings,
    )
