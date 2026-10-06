"""Smoke tests for the Streamlit UI using Streamlit's headless AppTest runner."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_app_runs_on_sample_data(app):
    assert not app.exception
    assert app.title[0].value == "Climate Analyzer"
    labels = [m.label for m in app.metric]
    assert "Trend per decade" in labels
    assert "Forecast skill" in labels
    assert len(app.tabs) == 7


def test_app_switches_to_precipitation(app):
    variable = next(s for s in app.sidebar.selectbox if s.label == "Variable")
    variable.select("prcp").run()
    assert not app.exception
    units = next(t for t in app.sidebar.text_input if t.label == "Units")
    assert units.value == "mm"
