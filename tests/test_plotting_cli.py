from __future__ import annotations

import json

import pandas as pd
import pytest
from matplotlib.figure import Figure

from climate_analyzer import __version__, cli, plotting
from climate_analyzer.io import load_sample
from climate_analyzer.pipeline import AnalysisConfig, run_analysis
from climate_analyzer.sources import DataSourceError


@pytest.fixture(scope="module")
def result():
    return run_analysis(load_sample(), AnalysisConfig(baseline=(1994, 2013)))


def test_all_figures_render(result, tmp_path):
    assert result.decomposition is not None and result.extremes is not None
    assert result.forecast is not None
    figures = [
        plotting.plot_time_series(result.cleaned, result.annual, result.trend, result.smoothed),
        plotting.plot_anomalies(result.annual_anomalies, "°C"),
        plotting.plot_climatology(result.climatology, recent=result.climatology + 1),
        plotting.plot_decomposition(result.decomposition, "°C"),
        plotting.plot_extremes(result.extremes),
        plotting.plot_forecast(result.monthly, result.forecast, result.backtest),
    ]
    expected_axes = [2, 1, 1, 4, 1, 1]
    for fig, n_axes in zip(figures, expected_axes, strict=True):
        assert isinstance(fig, Figure)
        assert len(fig.axes) == n_axes
        fig.savefig(tmp_path / "figure.png")


def test_cli_analyze_sample(tmp_path, capsys):
    out = tmp_path / "results"
    code = cli.main(["analyze", "sample", "-o", str(out), "--baseline", "1994", "2013"])
    assert code == 0
    expected = {
        "report.md",
        "summary.json",
        "annual.csv",
        "forecast.csv",
        "trend.png",
        "anomalies.png",
        "climatology.png",
        "decomposition.png",
        "extremes.png",
        "forecast.png",
    }
    assert {p.name for p in out.iterdir()} == expected
    assert "°C/decade" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["baseline"] == [1994, 2013]
    annual = pd.read_csv(out / "annual.csv")
    assert list(annual.columns) == ["year", "value", "anomaly"]


def test_cli_csv_input_without_plots(csv_file, tmp_path):
    out = tmp_path / "res"
    code = cli.main(
        ["analyze", str(csv_file), "-o", str(out), "--no-plots", "--forecast-months", "0"]
    )
    assert code == 0
    assert {p.name for p in out.iterdir()} == {"report.md", "summary.json", "annual.csv"}


def test_cli_errors(tmp_path, capsys):
    assert cli.main(["analyze", str(tmp_path / "missing.csv")]) == 2
    assert "not found" in capsys.readouterr().err
    assert cli.main(["analyze", "sample", "-c", "humidity", "-o", str(tmp_path)]) == 1
    assert "humidity" in capsys.readouterr().err


def test_cli_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_cli_fetch(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_fetch(lat, lon, start, end):
        captured.update(lat=lat, lon=lon, start=start, end=end)
        index = pd.date_range(start, end, freq="D", name="date")
        return pd.DataFrame({"tavg": range(len(index))}, index=index, dtype=float)

    monkeypatch.setattr(cli, "fetch_open_meteo", fake_fetch)
    out = tmp_path / "data" / "astana.csv"
    code = cli.main(["fetch", "--start", "2020-01-01", "--end", "2020-01-10", "-o", str(out)])
    assert code == 0
    assert captured["lat"] == pytest.approx(51.1694)
    assert "Astana: wrote 10 days" in capsys.readouterr().out
    assert len(pd.read_csv(out)) == 10

    code = cli.main(["fetch", "--coords", "43.2", "76.9", "--end", "2020-01-02", "-o", str(out)])
    assert code == 0
    assert captured["lat"] == 43.2


def test_cli_fetch_failure(monkeypatch, tmp_path, capsys):
    def failing(*args):
        raise DataSourceError("offline")

    monkeypatch.setattr(cli, "fetch_open_meteo", failing)
    assert cli.main(["fetch", "-o", str(tmp_path / "x.csv")]) == 1
    assert "offline" in capsys.readouterr().err
