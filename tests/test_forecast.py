from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climate_analyzer.forecast import SeasonalTrendModel, backtest


def monthly_series(years=20, trend_per_decade=0.5, noise=0.0, seed=0) -> pd.Series:
    index = pd.date_range("2000-01-01", periods=12 * years, freq="MS")
    t = np.arange(len(index)) / 12
    rng = np.random.default_rng(seed)
    values = (
        4.0
        + trend_per_decade / 10 * t
        - 17 * np.cos(2 * np.pi * t)
        + rng.normal(0, noise, len(index))
    )
    return pd.Series(values, index=index, name="tavg")


def test_model_recovers_trend_and_amplitude():
    model = SeasonalTrendModel().fit(monthly_series())
    assert model.trend_per_decade == pytest.approx(0.5, abs=0.01)
    assert model.seasonal_amplitude == pytest.approx(17, abs=0.5)
    # Not exactly 1: the test data uses equal-length months, the model real dates.
    assert model.r_squared() == pytest.approx(1.0, abs=1e-3)


def test_forecast_shape_and_intervals():
    model = SeasonalTrendModel().fit(monthly_series(noise=1.0))
    fc = model.forecast(24, level=0.9)
    assert len(fc) == 24
    assert fc.index[0] == pd.Timestamp("2020-01-01")
    assert pd.infer_freq(fc.index) == "MS"
    assert (fc["lower"] < fc["forecast"]).all()
    assert (fc["forecast"] < fc["upper"]).all()
    # January is cold, July is warm.
    assert fc.loc["2020-01-01", "forecast"] < -5 < 15 < fc.loc["2020-07-01", "forecast"]
    # Uncertainty never shrinks with the horizon.
    width = (fc["upper"] - fc["lower"]).to_numpy()
    assert np.all(np.diff(width) >= -1e-9)


def test_ar1_memory_decays():
    series = monthly_series(noise=0.0)
    index = series.index
    rng = np.random.default_rng(1)
    ar = np.zeros(len(series))
    for i in range(1, len(series)):
        ar[i] = 0.8 * ar[i - 1] + rng.normal()
    model = SeasonalTrendModel().fit(pd.Series(series.to_numpy() + ar, index=index))
    assert 0.4 < model.phi_ < 0.95
    assert model.last_residual_ != 0
    fc = model.forecast(36)
    gap = (fc["forecast"] - model.predict(fc.index)).abs()
    assert gap.is_monotonic_decreasing
    assert gap.iloc[-1] < 0.01 * gap.iloc[0]


def test_non_negative_model_clips_predictions():
    rain = monthly_series(trend_per_decade=-20).clip(lower=0)
    model = SeasonalTrendModel(non_negative=True).fit(rain)
    fc = model.forecast(24)
    assert (fc >= 0).all().all()
    assert (model.predict(fc.index) >= 0).all()


def test_model_validation():
    with pytest.raises(ValueError):
        SeasonalTrendModel(harmonics=-1)
    model = SeasonalTrendModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.forecast(3)
    with pytest.raises(TypeError):
        model.fit(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="at least"):
        model.fit(monthly_series().iloc[:5])
    irregular = monthly_series().iloc[[0, 1, 5, 6, 7, 20, 21, 40, 41, 60]]
    with pytest.raises(ValueError, match="frequency"):
        model.fit(irregular)
    model.fit(monthly_series())
    with pytest.raises(ValueError):
        model.forecast(0)
    with pytest.raises(ValueError):
        model.forecast(3, level=1.5)


def test_missing_last_value_disables_memory():
    series = monthly_series(noise=1.0)
    series.iloc[-1] = np.nan
    model = SeasonalTrendModel().fit(series)
    assert model.last_residual_ == 0.0


def test_backtest_beats_climatology_when_trend_is_strong():
    series = monthly_series(years=30, trend_per_decade=2.0, noise=0.5)
    result = backtest(series, test_periods=36)
    assert result.n_test == 36
    assert result.train_end == pd.Timestamp("2026-12-01")
    assert result.mae < result.climatology_mae
    assert result.skill > 0.2
    assert result.coverage > 0.8
    assert abs(result.bias) < 0.5
    assert set(result.predictions.columns) == {
        "actual",
        "predicted",
        "lower",
        "upper",
        "climatology",
    }


def test_backtest_validation():
    series = monthly_series(years=3)
    with pytest.raises(ValueError):
        backtest(series, 0)
    with pytest.raises(ValueError):
        backtest(series, len(series))
    gappy = series.copy()
    gappy.iloc[-6:] = np.nan
    with pytest.raises(ValueError, match="no observations"):
        backtest(gappy, 6)
