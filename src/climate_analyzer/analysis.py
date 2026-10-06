"""Statistical analysis of climate time series."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from scipy import stats

from .io import datetime_index

Aggregation = Literal["mean", "sum"]
Baseline = tuple[int, int]


def decimal_year(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert timestamps to fractional years, e.g. 2000-07-02 -> ~2000.5."""
    days_in_year = np.where(index.is_leap_year, 366.0, 365.0)
    return index.year.to_numpy() + (index.dayofyear.to_numpy() - 1) / days_in_year


def resample(
    series: pd.Series,
    freq: str = "MS",
    how: Aggregation = "mean",
    min_coverage: float = 0.8,
    base_freq: str = "D",
) -> pd.Series:
    """Aggregate a series to a coarser frequency (``MS`` monthly, ``YS`` yearly).

    A period is set to ``NaN`` when less than ``min_coverage`` of its expected
    ``base_freq`` observations are present, so that e.g. a month with only five
    measured days does not distort a trend.
    """
    datetime_index(series)
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1")
    if series.empty:
        return series.copy()

    present = series.resample(freq).count()
    # Start-anchored bins (MS, QS, YS) are labelled with the first day of the
    # period, so the full calendar runs from the first label to the end of the
    # last period.
    end = present.index[-1] + to_offset(freq)
    full = pd.date_range(present.index[0], end, freq=base_freq, inclusive="left")
    expected = pd.Series(1, index=full).resample(freq).size()

    if how == "mean":
        aggregated = series.resample(freq).mean()
    elif how == "sum":
        aggregated = series.resample(freq).sum(min_count=1)
    else:
        raise ValueError(f"Unknown aggregation: {how!r}")

    coverage = present / expected.reindex(present.index)
    return aggregated.where(coverage >= min_coverage)


def moving_average(series: pd.Series, window: int, min_fraction: float = 0.5) -> pd.Series:
    """Centred rolling mean; needs at least ``min_fraction`` of the window present."""
    if window < 1:
        raise ValueError("window must be at least 1")
    min_periods = max(1, int(np.ceil(window * min_fraction)))
    return series.rolling(window, center=True, min_periods=min_periods).mean()


@dataclass(frozen=True)
class TrendResult:
    """Linear (OLS) and non-parametric (Mann-Kendall / Sen) trend estimates."""

    n: int
    slope_per_decade: float
    intercept: float
    r_squared: float
    p_value: float
    stderr_per_decade: float
    sen_slope_per_decade: float
    mann_kendall_tau: float
    mann_kendall_p_value: float

    def is_significant(self, alpha: float = 0.05) -> bool:
        """True if the Mann-Kendall test rejects 'no monotonic trend' at ``alpha``."""
        return self.mann_kendall_p_value < alpha

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def linear_trend(series: pd.Series) -> TrendResult:
    """Estimate the long-term trend of a series.

    Two complementary methods are reported:

    * ordinary least squares regression (slope, R^2, p-value), which is the
      textbook approach but assumes normally distributed residuals;
    * the Mann-Kendall test with Sen's slope estimator, the standard
      non-parametric method in climatology, which is robust to outliers and
      skewed data.

    Slopes are expressed in units of the series per decade.
    """
    index = datetime_index(series)
    valid = series.notna().to_numpy()
    if valid.sum() < 3:
        raise ValueError("at least 3 non-missing values are required to fit a trend")

    x = decimal_year(index)[valid]
    y = series.to_numpy(dtype=float)[valid]
    if np.ptp(x) == 0:
        raise ValueError("all observations have the same timestamp")

    ols = stats.linregress(x, y)
    sen = stats.theilslopes(y, x)
    tau = stats.kendalltau(x, y)

    return TrendResult(
        n=int(valid.sum()),
        slope_per_decade=float(ols.slope * 10),
        intercept=float(ols.intercept),
        r_squared=float(ols.rvalue**2),
        p_value=float(ols.pvalue),
        stderr_per_decade=float(ols.stderr * 10),
        sen_slope_per_decade=float(sen.slope * 10),
        mann_kendall_tau=float(tau.statistic),
        mann_kendall_p_value=float(tau.pvalue),
    )


def trend_line(series: pd.Series, trend: TrendResult) -> pd.Series:
    """Values of the fitted OLS line at each timestamp of ``series``."""
    x = decimal_year(datetime_index(series))
    return pd.Series(trend.intercept + trend.slope_per_decade / 10 * x, index=series.index)


def _baseline_slice(series: pd.Series, baseline: Baseline | None) -> pd.Series:
    if baseline is None:
        return series
    start, end = baseline
    if start > end:
        raise ValueError("baseline start year must not be after end year")
    years = datetime_index(series).year
    selected = series[(years >= start) & (years <= end)]
    if selected.dropna().empty:
        raise ValueError(f"no data in baseline period {start}-{end}")
    return selected


def climatology(series: pd.Series, baseline: Baseline | None = None) -> pd.Series:
    """Long-term mean for each calendar month (index 1..12) over ``baseline`` years."""
    ref = _baseline_slice(series, baseline)
    clim = ref.groupby(datetime_index(ref).month).mean()
    clim.index.name = "month"
    return clim.reindex(range(1, 13))


def anomalies(series: pd.Series, baseline: Baseline | None = None) -> pd.Series:
    """Deviation of every value from the climatological mean of its calendar month."""
    clim = climatology(series, baseline)
    expected = clim.reindex(datetime_index(series).month).to_numpy()
    return (series - expected).rename(f"{series.name}_anomaly" if series.name else None)


@dataclass(frozen=True)
class Decomposition:
    """Additive decomposition: ``observed = trend + seasonal + residual``."""

    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "observed": self.observed,
                "trend": self.trend,
                "seasonal": self.seasonal,
                "residual": self.residual,
            }
        )


def seasonal_decompose(series: pd.Series, period: int = 12) -> Decomposition:
    """Classical additive seasonal decomposition (moving-average method).

    The trend is a centred moving average over one full ``period`` (a 2xN
    average for even periods), the seasonal component is the average detrended
    value for each position in the cycle (normalised to sum to zero) and the
    residual is what remains.  Isolated single missing values are linearly
    interpolated first so that one bad month does not erase a whole year of
    the moving-average trend; the residual stays ``NaN`` at those positions.
    """
    if period < 2:
        raise ValueError("period must be at least 2")
    if series.dropna().size < 2 * period:
        raise ValueError(f"need at least two full cycles ({2 * period} values) to decompose")

    filled = series.interpolate(limit=1, limit_area="inside")
    if period % 2 == 0:
        trend = filled.rolling(period, center=True).mean().rolling(2).mean().shift(-1)
    else:
        trend = filled.rolling(period, center=True).mean()

    detrended = series - trend
    position = np.arange(len(series)) % period
    seasonal_index = detrended.groupby(position).mean()
    seasonal_index -= seasonal_index.mean()
    seasonal = pd.Series(seasonal_index.reindex(position).to_numpy(), index=series.index)
    residual = series - trend - seasonal
    return Decomposition(observed=series, trend=trend, seasonal=seasonal, residual=residual)


def extreme_days(
    series: pd.Series,
    baseline: Baseline | None = None,
    quantile: float = 0.9,
) -> pd.DataFrame:
    """Count unusually warm and cold days per year.

    A day is *warm* when it exceeds the ``quantile`` of the baseline period for
    the same calendar month and *cold* when it is below the ``1 - quantile``
    quantile (similar to the WMO ETCCDI indices TX90p / TN10p).
    """
    if not 0.5 < quantile < 1:
        raise ValueError("quantile must be between 0.5 and 1")
    index = datetime_index(series)
    ref = _baseline_slice(series, baseline)
    ref_months = ref.groupby(datetime_index(ref).month)
    upper = ref_months.quantile(quantile).reindex(index.month).to_numpy()
    lower = ref_months.quantile(1 - quantile).reindex(index.month).to_numpy()

    valid = series.notna()
    frame = pd.DataFrame(
        {
            "warm_days": (series > upper) & valid,
            "cold_days": (series < lower) & valid,
            "observed_days": valid,
        },
        index=index,
    )
    result = frame.groupby(index.year).sum().astype(int)
    result.index.name = "year"
    return result


def summary_statistics(series: pd.Series) -> dict[str, Any]:
    """Basic descriptive statistics, JSON serialisable."""
    index = datetime_index(series)
    valid = series.dropna()
    if valid.empty:
        raise ValueError("series contains no data")
    valid_index, values = datetime_index(valid), valid.to_numpy()
    return {
        "variable": str(series.name),
        "start": index.min().date().isoformat(),
        "end": index.max().date().isoformat(),
        "count": int(valid.size),
        "missing": int(series.isna().sum()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "min_date": valid_index[int(np.argmin(values))].date().isoformat(),
        "max": float(valid.max()),
        "max_date": valid_index[int(np.argmax(values))].date().isoformat(),
    }
