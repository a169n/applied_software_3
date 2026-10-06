from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_daily(
    start: str = "2000-01-01",
    years: int = 10,
    trend_per_decade: float = 0.0,
    amplitude: float = 15.0,
    noise: float = 0.0,
    seed: int = 0,
) -> pd.Series:
    """Daily temperature-like series with a known trend and annual cycle."""
    index = pd.date_range(start, periods=int(years * 365.25), freq="D")
    t = (index - index[0]).days.to_numpy() / 365.25
    rng = np.random.default_rng(seed)
    values = (
        5.0
        + trend_per_decade / 10 * t
        + amplitude * np.cos(2 * np.pi * (index.dayofyear.to_numpy() - 200) / 365.25)
        + rng.normal(0, noise, len(index))
    )
    return pd.Series(values, index=index, name="tavg")


@pytest.fixture
def daily() -> pd.Series:
    return make_daily(years=20, trend_per_decade=0.5, noise=2.0)


@pytest.fixture
def daily_frame(daily: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    prcp = pd.Series(rng.gamma(0.8, 4.0, len(daily)), index=daily.index)
    return pd.DataFrame({"tavg": daily, "prcp": prcp.where(rng.random(len(daily)) < 0.3, 0.0)})


@pytest.fixture
def csv_file(tmp_path, daily_frame: pd.DataFrame):
    path = tmp_path / "station.csv"
    daily_frame.rename_axis("date").to_csv(path)
    return path
