"""Downloading real observations from the Open-Meteo historical weather API.

Open-Meteo (https://open-meteo.com) serves ERA5 reanalysis data for any point
on Earth, free and without an API key.  Reanalysis combines observations with
a weather model on a ~25 km grid, so values describe the city area rather than
one particular weather station.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import IO, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

#: Open-Meteo daily variable -> column name used by this package.
DAILY_VARIABLES: dict[str, str] = {
    "temperature_2m_mean": "tavg",
    "temperature_2m_min": "tmin",
    "temperature_2m_max": "tmax",
    "precipitation_sum": "prcp",
}


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float


CITIES: dict[str, City] = {
    "astana": City("Astana", 51.1694, 71.4491),
    "almaty": City("Almaty", 43.2389, 76.8897),
    "shymkent": City("Shymkent", 42.3417, 69.5901),
}


class DataSourceError(RuntimeError):
    """Raised when data cannot be downloaded or the response is malformed."""


def build_url(latitude: float, longitude: float, start: date, end: date) -> str:
    """URL of the Open-Meteo archive request for daily data at one location."""
    if start > end:
        raise ValueError("start date must not be after end date")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("latitude/longitude out of range")
    query = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
    }
    return f"{OPEN_METEO_ARCHIVE_URL}?{urlencode(query)}"


def parse_response(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert an Open-Meteo JSON response into the package's tidy daily format."""
    if payload.get("error"):
        raise DataSourceError(f"Open-Meteo error: {payload.get('reason', 'unknown')}")
    daily = payload.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise DataSourceError("unexpected response: 'daily.time' is missing")

    columns = {ours: daily[theirs] for theirs, ours in DAILY_VARIABLES.items() if theirs in daily}
    if not columns:
        raise DataSourceError("response contains none of the requested variables")
    try:
        frame = pd.DataFrame(columns, index=pd.DatetimeIndex(daily["time"], name="date"))
    except ValueError as exc:
        raise DataSourceError(f"malformed daily data: {exc}") from exc
    return frame.astype(float).sort_index()


def fetch_open_meteo(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    timeout: float = 60.0,
    opener: Callable[..., IO[bytes]] = urlopen,
) -> pd.DataFrame:
    """Download daily mean/min/max temperature and precipitation for a location."""
    url = build_url(latitude, longitude, start, end)
    try:
        with opener(url, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        # Open-Meteo explains bad requests in a JSON body, e.g. a date out of range.
        try:
            reason = json.load(exc).get("reason", exc.reason)
        except (ValueError, AttributeError, OSError):
            reason = exc.reason
        raise DataSourceError(f"Open-Meteo request failed ({exc.code}): {reason}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise DataSourceError(f"could not download data from Open-Meteo: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DataSourceError("Open-Meteo returned invalid JSON") from exc
    return parse_response(payload)


def fetch_city(name: str, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
    """Like :func:`fetch_open_meteo` for one of the predefined :data:`CITIES`."""
    try:
        city = CITIES[name.lower()]
    except KeyError:
        known = ", ".join(sorted(CITIES))
        raise ValueError(f"unknown city {name!r}; choose one of {known}") from None
    return fetch_open_meteo(city.latitude, city.longitude, start, end, **kwargs)
