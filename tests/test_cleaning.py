from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climate_analyzer.cleaning import clean, detect_outliers, fill_gaps

from .conftest import make_daily


@pytest.mark.parametrize("method", ["zscore", "iqr"])
def test_injected_spikes_are_detected(method):
    series = make_daily(years=5, noise=1.0)
    spikes = series.index[[100, 500, 1200]]
    series[spikes] += [30, -30, 35]

    mask = detect_outliers(series, method=method)

    assert mask[spikes].all()
    assert mask.sum() <= 10  # almost nothing else is flagged


def test_outliers_are_judged_per_month():
    # -25 degC is normal in January but absurd in July.
    index = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    series = pd.Series(np.where(index.month == 1, -25.0, 20.0), index=index)
    series += np.random.default_rng(0).normal(0, 1, len(index))
    series[pd.Timestamp("2002-07-15")] = -25.0

    mask = detect_outliers(series)

    assert mask[pd.Timestamp("2002-07-15")]
    assert not mask[index.month == 1].any()


def test_none_method_and_missing_values():
    series = make_daily(years=2, noise=1.0)
    series.iloc[10] = np.nan
    assert not detect_outliers(series, method="none").any()
    assert not detect_outliers(series)[series.index[10]]


def test_custom_threshold_flags_more():
    series = make_daily(years=3, noise=2.0)
    assert detect_outliers(series, threshold=2).sum() > detect_outliers(series).sum()


def test_detect_outliers_validates_input():
    with pytest.raises(TypeError):
        detect_outliers(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="Unknown"):
        detect_outliers(make_daily(years=1), method="magic")  # type: ignore[arg-type]


def test_fill_gaps_interpolates_short_gaps_linearly():
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    series = pd.Series([0.0, np.nan, np.nan, 3.0, 4.0, 5.0], index=index)
    filled = fill_gaps(series, max_gap=2)
    assert filled.tolist() == pytest.approx([0, 1, 2, 3, 4, 5])


def test_fill_gaps_keeps_long_gaps_and_edges():
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    series = pd.Series([np.nan, 1, np.nan, np.nan, np.nan, 5, 6, np.nan], index=index)
    filled = fill_gaps(series, max_gap=2)
    assert filled.isna().tolist() == [True, False, True, True, True, False, False, True]


def test_fill_gaps_arguments():
    series = make_daily(years=1)
    series.iloc[5] = np.nan
    assert fill_gaps(series, max_gap=0).isna().sum() == 1
    with pytest.raises(ValueError):
        fill_gaps(series, max_gap=-1)


def test_clean_report():
    series = make_daily(years=3, noise=1.0)
    series.iloc[[50, 51]] = np.nan  # short gap, filled
    series.iloc[300:310] = np.nan  # long gap, kept
    series.iloc[600] += 40  # outlier, removed and then interpolated

    cleaned, report = clean(series, max_gap=3)

    assert report.total == len(series)
    assert report.missing_before == 12
    assert report.outliers_removed == 1
    assert report.gaps_filled == 3
    assert report.missing_after == 10
    assert cleaned.isna().sum() == 10
    assert report.completeness == pytest.approx(1 - 10 / len(series))
    assert abs(cleaned.iloc[600] - series.iloc[599]) < 10
