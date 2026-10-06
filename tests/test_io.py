from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from climate_analyzer.io import DataValidationError, get_series, load_csv, load_sample


def _csv(text: str) -> io.StringIO:
    return io.StringIO(text.strip() + "\n")


def test_load_sample_is_clean_daily_frame():
    df = load_sample()
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert list(df.columns) == ["tavg", "tmin", "tmax", "prcp"]
    assert df.index[0] == pd.Timestamp("1994-01-01")
    assert df.index[-1] == pd.Timestamp("2023-12-31")
    # The -999 sentinels in the raw file must have been turned into NaN.
    assert not (df == -999).any().any()


def test_load_csv_from_path(csv_file):
    df = load_csv(csv_file)
    assert list(df.columns) == ["tavg", "prcp"]
    assert df.index.name == "date"


def test_load_csv_from_bytes_buffer():
    df = load_csv(io.BytesIO(b"date,tavg\n2020-01-01,1.5\n2020-01-02,2.5\n"))
    assert df["tavg"].tolist() == [1.5, 2.5]


@pytest.mark.parametrize("header", ["date", "Date", "TIME", "datetime", " timestamp "])
def test_date_column_is_detected_case_insensitively(header):
    df = load_csv(_csv(f"{header},t\n2020-01-01,1\n2020-01-02,2"))
    assert len(df) == 2


def test_explicit_date_column():
    df = load_csv(_csv("obs_day,t\n2020-01-01,1"), date_column="obs_day")
    assert df.index[0] == pd.Timestamp("2020-01-01")


def test_explicit_date_column_missing():
    with pytest.raises(DataValidationError, match="not found"):
        load_csv(_csv("date,t\n2020-01-01,1"), date_column="when")


def test_rows_are_sorted_and_deduplicated():
    df = load_csv(_csv("date,t\n2020-01-03,3\n2020-01-01,1\n2020-01-01,99\n2020-01-02,2"))
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert df["t"].tolist() == [1.0, 2.0, 3.0]


def test_first_duplicate_wins():
    df = load_csv(_csv("date,t\n2020-01-01,1\n2020-01-01,99"))
    assert df["t"].tolist() == [1.0]


def test_sentinels_and_text_become_nan():
    df = load_csv(_csv("date,t,p\n2020-01-01,-999,1\n2020-01-02,n/a,-9999\n2020-01-03,4,2"))
    assert df["t"].isna().tolist() == [True, True, False]
    assert np.isnan(df.loc["2020-01-02", "p"])


def test_non_numeric_columns_are_dropped():
    df = load_csv(_csv("date,station,t\n2020-01-01,AST,1\n2020-01-02,AST,2"))
    assert list(df.columns) == ["t"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("date,t", "no rows"),
        ("x,t\n1,2", "date column"),
        ("date,t\nnot-a-date,1", "invalid dates"),
        ("date,name\n2020-01-01,abc", "No numeric"),
    ],
)
def test_invalid_input(text, message):
    with pytest.raises(DataValidationError, match=message):
        load_csv(io.StringIO(text))


def test_get_series():
    df = load_csv(_csv("date,t\n2020-01-01,1"))
    series = get_series(df, "t")
    assert series.name == "t"
    assert series.dtype == float
    with pytest.raises(DataValidationError, match="Available: t"):
        get_series(df, "rain")
