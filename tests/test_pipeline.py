from __future__ import annotations

import json

import numpy as np
import pytest

from climate_analyzer.io import DataValidationError, load_sample
from climate_analyzer.pipeline import AnalysisConfig, default_aggregation, run_analysis


@pytest.fixture(scope="module")
def astana():
    return load_sample()


@pytest.fixture(scope="module")
def astana_result(astana):
    return run_analysis(astana, AnalysisConfig(baseline=(1994, 2013)))


def test_default_aggregation():
    assert default_aggregation("tavg") == "mean"
    assert default_aggregation("PRCP") == "sum"
    assert default_aggregation("precipitation_sum") == "sum"
    assert AnalysisConfig(column="snow_depth").resolved_aggregation() == "sum"
    assert AnalysisConfig(column="prcp", aggregation="mean").resolved_aggregation() == "mean"


def test_astana_sample_end_to_end(astana_result):
    """The sample data contains a +0.35 degC/decade warming trend; we must find it."""
    r = astana_result
    assert r.trend.slope_per_decade == pytest.approx(0.35, abs=0.15)
    assert r.trend.is_significant()
    assert r.cleaning.outliers_removed >= 8  # the injected sensor spikes
    assert r.cleaning.completeness > 0.99
    assert len(r.annual) == 30
    assert r.climatology.idxmin() == 1 and r.climatology.idxmax() == 7
    assert r.decomposition is not None
    assert r.extremes is not None and len(r.extremes) == 30
    assert r.warnings == []


def test_astana_forecast(astana_result):
    r = astana_result
    assert r.model is not None and r.forecast is not None and r.backtest is not None
    assert len(r.forecast) == 24
    assert str(r.forecast.index[0].date()) == "2024-01-01"
    july = r.forecast.loc["2024-07-01", "forecast"]
    january = r.forecast.loc["2024-01-01", "forecast"]
    assert 15 < july < 28 and -20 < january < -5
    assert r.backtest.mae < r.backtest.climatology_mae * 1.1


def test_report_outputs(astana_result):
    data = json.loads(astana_result.to_json())
    assert data["config"]["aggregation"] == "mean"
    assert data["trend_significant"] is True
    assert set(data["climatology"]) == {str(m) for m in range(1, 13)}
    assert "backtest" in data["model"] and "2024-01" in data["model"]["forecast"]

    md = astana_result.to_markdown()
    for heading in ("# Climate analysis report", "## Data quality", "## Long-term trend"):
        assert heading in md
    assert "## Statistical model and forecast" in md
    assert "baseline: 1994–2013" in md


def test_precipitation_is_summed(astana):
    result = run_analysis(astana, AnalysisConfig(column="prcp", units="mm", outlier_method="none"))
    assert 200 < result.annual.mean() < 400  # annual total in mm
    assert result.extremes is None
    assert result.forecast is not None and (result.forecast >= 0).all().all()
    assert "## Extreme days" not in result.to_markdown()


def test_partial_year_and_disabled_forecast(astana):
    short = astana.loc["2018-01-01":"2022-06-30"]
    result = run_analysis(short, AnalysisConfig(backtest_months=0, forecast_months=0))
    assert result.forecast is None and result.backtest is None
    assert result.model is not None
    assert result.trend is not None
    assert any("insufficient coverage" in w for w in result.warnings)


def test_short_record_degrades_gracefully(astana):
    tiny = astana.loc["2020-01-01":"2021-03-31"]
    result = run_analysis(tiny)
    assert result.trend is None
    assert result.decomposition is None
    assert result.backtest is None
    assert any("decomposition skipped" in w for w in result.warnings)
    assert any("Trend not estimated" in w for w in result.warnings)
    assert any("Forecast model skipped" in w for w in result.warnings)
    assert "Not enough complete years" in result.to_markdown()
    assert json.loads(result.to_json())["trend"] is None


def test_poor_data_quality_warning(astana):
    gappy = astana.copy()
    gappy.loc[gappy.index.day > 20, "tavg"] = np.nan
    result = run_analysis(gappy, AnalysisConfig(max_gap=0))
    assert any("usable" in w for w in result.warnings)
    assert "## Warnings" in result.to_markdown()


def test_unknown_column(astana):
    with pytest.raises(DataValidationError):
        run_analysis(astana, AnalysisConfig(column="humidity"))
