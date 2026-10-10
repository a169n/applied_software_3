# Requirements specification: climate-analyzer

## 1. Purpose and scope

climate-analyzer cleans daily weather records, estimates long-term climate
change and forecasts the coming months. The case study is Astana, Kazakhstan.
It is a research tool for one analyst working on one data series at a time. It
is not a numerical weather model, a real-time service or a multi-user system.

## 2. Users and use cases

| Actor | Goal | Interface |
|---|---|---|
| Researcher / analyst | Analyse own station CSV, get trend, anomalies, forecast and a report | CLI |
| Student / non-programmer | Explore results interactively and read plain-language explanations | Web UI |
| Developer | Reuse the pipeline inside another program | Python API |
| Maintainer | Check every change and publish releases without manual steps | CI/CD |

| ID | Use case |
|---|---|
| UC1 | Analyse a CSV file and obtain a report with figures |
| UC2 | Download real daily history for a city or coordinates and analyse it |
| UC3 | Try the tool offline on the bundled sample |
| UC4 | Explore the result interactively and download the report |
| UC5 | Repeat the study on fresh data every month without human action |

## 3. Functional requirements

| ID | Requirement | Verified by |
|---|---|---|
| F1 | Read a CSV file with a date column; detect the date column automatically, allow it to be set by name | `test_io` |
| F2 | Convert placeholder codes such as `-999` to missing values; sort rows and remove duplicate dates | `test_io` |
| F3 | Download daily history for a named city or for latitude/longitude from Open-Meteo | `test_sources` (offline, injected opener) |
| F4 | Detect outliers relative to the same calendar month (z-score or IQR, threshold configurable, or disabled) | `test_cleaning` |
| F5 | Interpolate gaps of up to 3 days (configurable); longer gaps stay missing | `test_cleaning` |
| F6 | Report how many values were removed and filled | `test_cleaning` |
| F7 | Aggregate to monthly and yearly values (mean for temperature, sum for precipitation); drop months/years with less than 80 % coverage | `test_analysis` |
| F8 | Estimate the annual trend by OLS and by Mann-Kendall with Sen's slope, with p-values | `test_analysis` |
| F9 | Compute anomalies against a configurable baseline period, monthly climatology and seasonal decomposition | `test_analysis` |
| F10 | Count warm and cold extreme days per year (above/below the 90th/10th percentile of the same month) | `test_analysis` |
| F11 | Fit trend + annual harmonics + AR(1) model and forecast N months with a 95 % prediction interval | `test_forecast` |
| F12 | Backtest on the last N months and compare MAE with the climatology baseline (skill score) | `test_forecast` |
| F13 | Write `report.md`, `summary.json`, CSV tables and PNG figures; plots can be skipped | `test_pipeline`, `test_plotting_cli` |
| F14 | If a stage cannot run (too little data), skip it with a warning and finish the rest | `test_pipeline` |
| F15 | Web UI: choose data source (sample, Open-Meteo, upload), set baseline and forecast options, show one interactive chart per analysis with an explanation, download the report | `test_app` (Streamlit AppTest) |
| F16 | Command line: `analyze` and `fetch` sub-commands with options for variable, units, output folder, cleaning, baseline, horizon | `test_plotting_cli` |

## 4. Non-functional requirements

| ID | Category | Requirement | How it is met |
|---|---|---|---|
| N1 | Correctness | The tool must recover a known trend from data with known defects | Synthetic dataset with +0.35 °C/decade and tests against it |
| N2 | Reproducibility | Same input gives the same output on any machine | Fixed random seed, display-free plotting, no network in tests |
| N3 | Stability | A library update must not change results silently | `FutureWarning` treated as error in tests |
| N4 | Honesty | Uncertainty is always reported; claims are not stronger than the evidence | Two trend tests, prediction intervals, backtest against baseline |
| N5 | Portability | Runs on Linux, Windows, macOS, Python 3.10-3.13 | CI build matrix |
| N6 | Installability | One command: `pip install -e ".[ui]"` | `pyproject.toml` with optional extras |
| N7 | Offline use | Core analysis and tests need no internet | Bundled sample; injectable network layer |
| N8 | Maintainability | Small single-purpose modules; typed code; consistent style | `src/` layout, mypy, Ruff |
| N9 | Test quality | At least 90 % statement coverage enforced in CI | Quality gate (actual: 99 %, 84 tests) |
| N10 | Performance | A 32-year daily series (about 11 700 rows) analyses in seconds on a laptop | Vectorised pandas/NumPy; no heavy models |
| N11 | Security | No TLS verification switched off; CI token read-only | `truststore` on the affected machine; least-privilege permissions |
| N12 | Licensing | Free and open source | MIT licence, `CITATION.cff` |

## 5. Data requirements

- Input: CSV with one row per day, one date column and at least one numeric
  column (`tavg`, `tmin`, `tmax`, `prcp`, or any other).
- At least a few full years are needed for a trend; less data skips the stage
  with a warning (F14).
- Real data source: Open-Meteo archive (ERA5/ERA5-Land, ECMWF IFS from 2017).
  It is a model grid cell, not a weather station.

## 6. Constraints and assumptions

- Python 3.10 or newer.
- The forecast is statistical and cannot predict individual weather events.
- Residual variance is assumed constant across seasons.
- The 2017 model change in Open-Meteo can create an artificial step in
  long series, especially for precipitation.
- The outlier filter is designed for sensor faults and may remove genuine
  extremes (e.g. the cold wave of 28 November 1998).

## 7. Acceptance criteria

1. `pip install -e ".[dev]"` and `pytest --cov` pass with coverage of at
   least 90 %.
2. `climate-analyzer analyze sample --baseline 1994 2013` produces the report,
   JSON, CSV tables and figures, and the recovered trend is close to
   +0.35 °C per decade.
3. `streamlit run app.py` starts and shows the analysis without errors.
4. All CI jobs pass on Ubuntu, Windows and macOS.
5. Pushing a `v*` tag publishes a GitHub Release with a wheel and sdist.
