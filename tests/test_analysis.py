from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climate_analyzer import analysis

from .conftest import make_daily


def test_decimal_year():
    index = pd.DatetimeIndex(["2000-01-01", "2001-07-02", "2004-12-31"])
    result = analysis.decimal_year(index)
    assert result[0] == 2000.0
    assert result[1] == pytest.approx(2001.5, abs=0.01)
    assert result[2] == pytest.approx(2005.0, abs=0.01)


def test_resample_monthly_mean_and_sum():
    index = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    series = pd.Series(1.0, index=index)
    series["2020-02"] = 3.0
    assert analysis.resample(series, "MS").tolist() == [1.0, 3.0]
    assert analysis.resample(series, "MS", how="sum").tolist() == [31.0, 87.0]


def test_resample_masks_periods_with_low_coverage():
    index = pd.date_range("2020-01-01", "2020-03-31", freq="D")
    series = pd.Series(1.0, index=index)
    series["2020-02-01":"2020-02-20"] = np.nan  # Feb only 9/29 days present
    monthly = analysis.resample(series, "MS", min_coverage=0.8)
    assert monthly.isna().tolist() == [False, True, False]


def test_resample_counts_missing_rows_and_partial_last_period():
    # Rows absent from the file (not just NaN) must also reduce coverage, and a
    # record that stops on the 5th of a month leaves that month incomplete.
    index = pd.date_range("2020-01-01", "2020-03-05", freq="D")
    series = pd.Series(1.0, index=index).drop(pd.date_range("2020-02-01", "2020-02-15"))
    monthly = analysis.resample(series, "MS")
    assert monthly.isna().tolist() == [False, True, True]


def test_resample_validation():
    series = make_daily(years=1)
    with pytest.raises(ValueError):
        analysis.resample(series, how="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        analysis.resample(series, min_coverage=1.5)
    with pytest.raises(TypeError):
        analysis.resample(pd.Series([1.0]))
    assert analysis.resample(series.iloc[:0]).empty


def test_moving_average():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert analysis.moving_average(series, 3).tolist() == [1.5, 2.0, 3.0, 4.0, 4.5]
    with pytest.raises(ValueError):
        analysis.moving_average(series, 0)


def test_linear_trend_recovers_exact_slope():
    index = pd.date_range("1990-07-01", periods=30, freq="12MS")
    series = pd.Series(0.03 * np.arange(30) + 5.0, index=index)  # 0.3 per decade
    trend = analysis.linear_trend(series)
    assert trend.slope_per_decade == pytest.approx(0.3, abs=1e-3)
    assert trend.sen_slope_per_decade == pytest.approx(0.3, abs=1e-3)
    assert trend.r_squared == pytest.approx(1.0)
    assert trend.mann_kendall_tau == pytest.approx(1.0)
    assert trend.is_significant()
    assert trend.as_dict()["n"] == 30


def test_linear_trend_noise_is_not_significant():
    rng = np.random.default_rng(3)
    index = pd.date_range("1990-01-01", periods=30, freq="YS")
    trend = analysis.linear_trend(pd.Series(rng.normal(size=30), index=index))
    assert not trend.is_significant()


def test_linear_trend_on_annual_means_of_daily_data(daily):
    annual = analysis.resample(daily, "YS")
    trend = analysis.linear_trend(annual)
    assert trend.slope_per_decade == pytest.approx(0.5, abs=0.25)


def test_linear_trend_validation():
    index = pd.date_range("2000-01-01", periods=3, freq="YS")
    with pytest.raises(ValueError, match="at least 3"):
        analysis.linear_trend(pd.Series([1.0, np.nan, 2.0], index=index))
    same_day = pd.DatetimeIndex(["2000-01-01"] * 3)
    with pytest.raises(ValueError, match="same timestamp"):
        analysis.linear_trend(pd.Series([1.0, 2.0, 3.0], index=same_day))


def test_trend_line_matches_fit():
    index = pd.date_range("2000-01-01", periods=10, freq="YS")
    series = pd.Series(2.0 + 0.1 * np.arange(10), index=index)
    line = analysis.trend_line(series, analysis.linear_trend(series))
    np.testing.assert_allclose(line.to_numpy(), series.to_numpy(), atol=1e-2)


def test_climatology_and_anomalies():
    series = make_daily(years=10, amplitude=15)
    monthly = analysis.resample(series, "MS")
    clim = analysis.climatology(monthly)
    assert list(clim.index) == list(range(1, 13))
    assert clim.idxmax() == 7 and clim.idxmin() == 1

    anomalies = analysis.anomalies(monthly)
    assert anomalies.name == "tavg_anomaly"
    assert anomalies.mean() == pytest.approx(0, abs=1e-9)


def test_baseline_period_shifts_anomalies():
    index = pd.date_range("2000-01-01", periods=20, freq="YS")
    series = pd.Series(np.arange(20, dtype=float), index=index)
    anomalies = analysis.anomalies(series, baseline=(2000, 2004))
    assert anomalies.iloc[0] == -2.0  # baseline mean is 2
    assert anomalies.iloc[-1] == 17.0
    with pytest.raises(ValueError, match="no data"):
        analysis.climatology(series, baseline=(1900, 1910))
    with pytest.raises(ValueError, match="start year"):
        analysis.climatology(series, baseline=(2010, 2000))


def test_seasonal_decompose_recovers_components():
    index = pd.date_range("2000-01-01", periods=120, freq="MS")
    seasonal = 10 * np.sin(2 * np.pi * np.arange(120) / 12)
    trend = 0.05 * np.arange(120)
    series = pd.Series(seasonal + trend, index=index)
    series.iloc[50] = np.nan  # a single missing month must not break the trend

    result = analysis.seasonal_decompose(series)

    middle = slice(12, 108)
    np.testing.assert_allclose(result.trend.iloc[middle], trend[middle], atol=0.1)
    np.testing.assert_allclose(result.seasonal.iloc[middle], seasonal[middle], atol=0.3)
    assert result.trend.iloc[12:108].notna().all()
    assert result.residual.abs().max() < 0.5
    assert list(result.to_frame().columns) == ["observed", "trend", "seasonal", "residual"]


def test_seasonal_decompose_odd_period_and_validation():
    index = pd.date_range("2000-01-01", periods=21, freq="D")
    series = pd.Series(np.tile([1.0, 2.0, 3.0], 7), index=index)
    result = analysis.seasonal_decompose(series, period=3)
    assert result.seasonal.iloc[:3].tolist() == pytest.approx([-1, 0, 1])
    with pytest.raises(ValueError, match="two full cycles"):
        analysis.seasonal_decompose(series.iloc[:5], period=3)
    with pytest.raises(ValueError, match="at least 2"):
        analysis.seasonal_decompose(series, period=1)


def test_extreme_days_increase_with_warming():
    series = make_daily(years=20, trend_per_decade=3.0, noise=2.0)
    extremes = analysis.extreme_days(series, baseline=(2000, 2004))
    assert list(extremes.columns) == ["warm_days", "cold_days", "observed_days"]
    assert extremes["warm_days"].iloc[-5:].mean() > extremes["warm_days"].iloc[:5].mean()
    assert extremes["cold_days"].iloc[-5:].mean() < extremes["cold_days"].iloc[:5].mean()
    with pytest.raises(ValueError):
        analysis.extreme_days(series, quantile=0.4)


def test_summary_statistics():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    stats = analysis.summary_statistics(pd.Series([1.0, np.nan, -2.0, 5.0], index=index, name="t"))
    assert stats["count"] == 3
    assert stats["missing"] == 1
    assert stats["min_date"] == "2020-01-03"
    assert stats["max_date"] == "2020-01-04"
    assert stats["mean"] == pytest.approx(4 / 3)
    with pytest.raises(ValueError):
        analysis.summary_statistics(pd.Series([np.nan], index=index[:1]))
