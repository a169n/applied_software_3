from __future__ import annotations

import io
import json
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from climate_analyzer.sources import (
    CITIES,
    DataSourceError,
    build_url,
    fetch_city,
    fetch_open_meteo,
    parse_response,
)

PAYLOAD = {
    "latitude": 51.125,
    "longitude": 71.5,
    "daily": {
        "time": ["2023-01-02", "2023-01-01"],
        "temperature_2m_mean": [-15.2, -14.0],
        "temperature_2m_min": [-20.1, None],
        "temperature_2m_max": [-10.3, -9.0],
        "precipitation_sum": [0.4, 0.0],
    },
}


def fake_opener(payload, calls=None):
    def opener(url, timeout):
        if calls is not None:
            calls.append(url)
        return io.BytesIO(json.dumps(payload).encode())

    return opener


def test_build_url():
    url = build_url(51.17, 71.45, date(2000, 1, 1), date(2000, 12, 31))
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert parsed.netloc == "archive-api.open-meteo.com"
    assert query["start_date"] == ["2000-01-01"]
    assert query["end_date"] == ["2000-12-31"]
    assert "temperature_2m_mean" in query["daily"][0]
    with pytest.raises(ValueError):
        build_url(51, 71, date(2001, 1, 1), date(2000, 1, 1))
    with pytest.raises(ValueError):
        build_url(91, 71, date(2000, 1, 1), date(2000, 1, 2))


def test_parse_response():
    df = parse_response(PAYLOAD)
    assert list(df.columns) == ["tavg", "tmin", "tmax", "prcp"]
    assert df.index.is_monotonic_increasing
    assert df.loc["2023-01-01", "tavg"] == -14.0
    assert df["tmin"].isna().sum() == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"error": True, "reason": "bad date"}, "bad date"),
        ({"hourly": {}}, "daily.time"),
        ({"daily": {"time": ["2023-01-01"]}}, "none of the requested"),
        ({"daily": {"time": ["2023-01-01"], "temperature_2m_mean": [1, 2]}}, "malformed"),
    ],
)
def test_parse_response_errors(payload, message):
    with pytest.raises(DataSourceError, match=message):
        parse_response(payload)


def test_fetch_uses_url_and_parses():
    calls: list[str] = []
    df = fetch_open_meteo(
        51.17, 71.45, date(2023, 1, 1), date(2023, 1, 2), opener=fake_opener(PAYLOAD, calls)
    )
    assert len(df) == 2
    assert "latitude=51.17" in calls[0]


def test_fetch_city_astana():
    calls: list[str] = []
    fetch_city("Astana", date(2023, 1, 1), date(2023, 1, 2), opener=fake_opener(PAYLOAD, calls))
    assert f"latitude={CITIES['astana'].latitude}" in calls[0]
    with pytest.raises(ValueError, match="unknown city"):
        fetch_city("atlantis", date(2023, 1, 1), date(2023, 1, 2))


def test_fetch_network_errors():
    def offline(url, timeout):
        raise URLError("no route to host")

    with pytest.raises(DataSourceError, match="could not download"):
        fetch_open_meteo(51, 71, date(2023, 1, 1), date(2023, 1, 2), opener=offline)


def test_fetch_http_error_reports_reason():
    def bad_request(url, timeout):
        body = io.BytesIO(json.dumps({"error": True, "reason": "end_date out of range"}).encode())
        raise HTTPError(url, 400, "Bad Request", {}, body)  # type: ignore[arg-type]

    with pytest.raises(DataSourceError, match=r"400.*out of range"):
        fetch_open_meteo(51, 71, date(2023, 1, 1), date(2023, 1, 2), opener=bad_request)


def test_fetch_invalid_json():
    def garbage(url, timeout):
        return io.BytesIO(b"<html>")

    with pytest.raises(DataSourceError, match="invalid JSON"):
        fetch_open_meteo(51, 71, date(2023, 1, 1), date(2023, 1, 2), opener=garbage)
