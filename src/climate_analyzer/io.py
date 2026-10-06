"""Loading and validating weather time series."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import IO

import pandas as pd

#: Values that weather archives commonly use to mean "no observation".
DEFAULT_NA_SENTINELS: tuple[float, ...] = (-999.0, -9999.0, 9999.0)

#: Column names that are recognised automatically as the timestamp column.
DATE_COLUMN_CANDIDATES: tuple[str, ...] = ("date", "time", "datetime", "timestamp", "day")

SAMPLE_DATASET = "astana_daily_sample.csv"


class DataValidationError(ValueError):
    """Raised when an input file cannot be interpreted as a weather time series."""


def _find_date_column(columns: Iterable[str]) -> str:
    lookup = {c.strip().lower(): c for c in columns}
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    raise DataValidationError(
        "Could not find a date column. Expected one of "
        f"{', '.join(DATE_COLUMN_CANDIDATES)} or pass date_column explicitly."
    )


def load_csv(
    source: str | Path | IO[str] | IO[bytes],
    date_column: str | None = None,
    na_sentinels: Iterable[float] = DEFAULT_NA_SENTINELS,
) -> pd.DataFrame:
    """Read a CSV file with one row per observation into a tidy DataFrame.

    The result is indexed by a sorted, duplicate-free :class:`~pandas.DatetimeIndex`
    and contains only numeric columns.  Sentinel values such as ``-999`` are
    converted to ``NaN``.

    Parameters
    ----------
    source:
        Path or open file object (e.g. an uploaded file in the web UI).
    date_column:
        Name of the timestamp column.  Detected automatically when omitted.
    na_sentinels:
        Numeric codes that mean "missing value".

    Raises
    ------
    DataValidationError
        If the file is empty, has no recognisable date column, has unparsable
        dates or contains no numeric data.
    """
    try:
        raw = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The file is empty.") from exc

    if raw.empty:
        raise DataValidationError("The file contains a header but no rows.")

    date_column = date_column or _find_date_column(raw.columns)
    if date_column not in raw.columns:
        raise DataValidationError(f"Date column {date_column!r} not found in the file.")

    try:
        index = pd.to_datetime(raw[date_column], format="mixed")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Column {date_column!r} contains invalid dates.") from exc

    data = raw.drop(columns=[date_column]).apply(pd.to_numeric, errors="coerce")
    data = data.dropna(axis="columns", how="all")
    if data.empty:
        raise DataValidationError("No numeric columns found besides the date column.")

    data.index = pd.DatetimeIndex(index, name="date")
    data = data.replace(list(na_sentinels), float("nan"))
    data = data[~data.index.duplicated(keep="first")].sort_index()
    return data


def load_sample() -> pd.DataFrame:
    """Return the bundled synthetic Astana dataset (1994-2023, daily)."""
    path = resources.files("climate_analyzer.data").joinpath(SAMPLE_DATASET)
    with path.open("r", encoding="utf-8") as handle:
        return load_csv(handle)


def datetime_index(data: pd.Series | pd.DataFrame) -> pd.DatetimeIndex:
    """Return the index of ``data``, checking that it holds timestamps."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must be indexed by a DatetimeIndex")
    return data.index


def get_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Extract one variable as a float series, with a helpful error message."""
    if column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise DataValidationError(f"Column {column!r} not found. Available: {available}.")
    return df[column].astype(float).rename(column)
