"""Quality control: outlier detection and gap filling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .io import datetime_index

OutlierMethod = Literal["zscore", "iqr", "none"]


@dataclass(frozen=True)
class CleaningReport:
    """What the cleaning step changed in a series."""

    total: int
    missing_before: int
    outliers_removed: int
    gaps_filled: int
    missing_after: int

    @property
    def completeness(self) -> float:
        """Share of non-missing values after cleaning, between 0 and 1."""
        return 1.0 - self.missing_after / self.total if self.total else 0.0


def _seasonal_baseline(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Per-calendar-month mean and spread of a series, aligned to its index.

    Temperatures differ by tens of degrees between seasons, so outliers must be
    judged against the expected value for that time of year, not against the
    global mean of the whole record.
    """
    grouped = series.groupby(datetime_index(series).month)
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return mean, std


def detect_outliers(
    series: pd.Series,
    method: OutlierMethod = "zscore",
    threshold: float | None = None,
) -> pd.Series:
    """Return a boolean mask that is ``True`` where a value is an outlier.

    ``zscore``
        Flags values more than ``threshold`` (default 4) standard deviations
        away from the mean of the same calendar month.
    ``iqr``
        Flags values outside ``[Q1 - k*IQR, Q3 + k*IQR]`` of the same calendar
        month, where ``k = threshold`` (default 3).
    ``none``
        Flags nothing.

    Missing values are never flagged.
    """
    months = datetime_index(series).month
    if method == "none":
        return pd.Series(False, index=series.index)

    if method == "zscore":
        k = 4.0 if threshold is None else threshold
        mean, std = _seasonal_baseline(series)
        z = (series - mean) / std.replace(0, np.nan)
        mask = z.abs() > k
    elif method == "iqr":
        k = 3.0 if threshold is None else threshold
        grouped = series.groupby(months)
        q1 = grouped.transform(lambda s: s.quantile(0.25))
        q3 = grouped.transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        mask = (series < q1 - k * iqr) | (series > q3 + k * iqr)
    else:
        raise ValueError(f"Unknown outlier method: {method!r}")

    return mask.fillna(False).astype(bool)


def fill_gaps(series: pd.Series, max_gap: int = 3) -> pd.Series:
    """Linearly interpolate runs of at most ``max_gap`` consecutive missing values.

    Longer gaps are left as ``NaN`` on purpose: inventing two weeks of weather
    would bias any statistic computed afterwards.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    if max_gap == 0:
        return series.copy()

    is_na = series.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_length = is_na.groupby(run_id).transform("sum")
    fillable = is_na & (run_length <= max_gap)

    interpolated = series.interpolate(method="time", limit_area="inside")
    return series.where(~fillable, interpolated)


def clean(
    series: pd.Series,
    outlier_method: OutlierMethod = "zscore",
    threshold: float | None = None,
    max_gap: int = 3,
) -> tuple[pd.Series, CleaningReport]:
    """Remove outliers, then fill short gaps. Returns the new series and a report."""
    missing_before = int(series.isna().sum())
    outliers = detect_outliers(series, method=outlier_method, threshold=threshold)
    without_outliers = series.mask(outliers)
    filled = fill_gaps(without_outliers, max_gap=max_gap)

    missing_mid = int(without_outliers.isna().sum())
    missing_after = int(filled.isna().sum())
    report = CleaningReport(
        total=len(series),
        missing_before=missing_before,
        outliers_removed=int(outliers.sum()),
        gaps_filled=missing_mid - missing_after,
        missing_after=missing_after,
    )
    return filled, report
