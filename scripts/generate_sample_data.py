"""Generate the bundled synthetic daily weather dataset.

The dataset imitates the climate of Astana, Kazakhstan (strongly continental:
January mean about -14 degC, July mean about +21 degC) for 1994-2023 with a
built-in warming trend of 0.35 degC per decade.  It intentionally contains the
kinds of defects real station data has, so that the cleaning step has
something to do:

* randomly missing days and one two-week station outage;
* a handful of sensor spikes (physically implausible values);
* the ``-999`` sentinel used by many archives for "no data".

The output is fully deterministic (fixed random seed), so the file in the
repository can always be regenerated::

    python scripts/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "1994-01-01", "2023-12-31"
ANNUAL_MEAN = 3.5  # degC
SEASONAL_AMPLITUDE = 17.5  # degC
WARMEST_DAY_OF_YEAR = 200  # mid July
TREND_PER_YEAR = 0.035  # degC / year
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "climate_analyzer"
    / "data"
    / "astana_daily_sample.csv"
)


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    doy = dates.dayofyear.to_numpy()
    years = (dates - dates[0]).days.to_numpy() / 365.25

    phase = 2 * np.pi * (doy - WARMEST_DAY_OF_YEAR) / 365.25
    seasonal = ANNUAL_MEAN + SEASONAL_AMPLITUDE * np.cos(phase)
    trend = TREND_PER_YEAR * years

    # AR(1) weather noise, more variable in winter than in summer.
    noise_std = 4.5 - 1.5 * np.cos(phase)
    noise = np.empty(n)
    noise[0] = 0.0
    eps = rng.standard_normal(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + np.sqrt(1 - 0.7**2) * eps[i]
    noise *= noise_std

    tavg = seasonal + trend + noise
    daily_range = 10.0 + 2.0 * np.cos(phase) + rng.normal(0, 1.5, n)
    tmin = tavg - daily_range / 2
    tmax = tavg + daily_range / 2

    wet_probability = 0.25 + 0.12 * np.cos(phase)
    wet = rng.random(n) < wet_probability
    prcp = np.where(wet, rng.gamma(shape=0.8, scale=4.0, size=n), 0.0)

    df = pd.DataFrame(
        {
            "date": dates,
            "tavg": tavg.round(1),
            "tmin": tmin.round(1),
            "tmax": tmax.round(1),
            "prcp": prcp.round(1),
        }
    )

    # --- data quality defects -------------------------------------------
    missing = rng.choice(n, size=n // 100, replace=False)
    df.loc[missing, ["tavg", "tmin", "tmax"]] = np.nan

    outage = (df["date"] >= "2009-03-01") & (df["date"] <= "2009-03-14")
    df.loc[outage, ["tavg", "tmin", "tmax", "prcp"]] = np.nan

    spikes = rng.choice(np.setdiff1d(np.arange(n), missing), size=8, replace=False)
    df.loc[spikes, "tavg"] = df.loc[spikes, "tavg"] + rng.choice([-1, 1], 8) * 40

    sentinels = rng.choice(np.setdiff1d(np.arange(n), missing), size=5, replace=False)
    df.loc[sentinels, "tavg"] = -999.0

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
