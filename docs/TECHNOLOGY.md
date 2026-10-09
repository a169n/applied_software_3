# Choice of technologies

This document explains why each technology in the project was chosen and what
the alternatives were. The selection criteria were:

1. **Fitness for scientific work**: correct, well-tested numerical methods.
2. **Ecosystem and community**: libraries that are maintained and documented
   and that researchers already know.
3. **Reproducibility**: the same input gives the same output on any machine,
   including CI servers without a display or internet access.
4. **Low barrier to entry**: free and open source, cross-platform, easy to
   install with one command.
5. **Testability and automation**: the stack works with automated testing,
   linting and CI/CD.

## Programming language: Python 3.10+

| Criterion | Python | R | MATLAB | Julia |
|---|---|---|---|---|
| Scientific libraries | excellent (NumPy/SciPy/pandas) | excellent (statistics) | good | growing |
| General-purpose (CLI, web UI, packaging) | excellent | limited | limited | moderate |
| Licence cost | free | free | commercial | free |
| CI / tooling maturity | excellent | good | limited | moderate |
| Popularity in climate science | very high (xarray, MetPy) | high | declining | niche |

Python is the de facto language of data-driven climate research and offers
mature tooling for every step of the assignment, from analysis to web UI,
packaging, testing and CI. Python 3.10 is the oldest version still receiving
security fixes, so it is the minimum.

## Libraries

| Library | Purpose | Why this one |
|---|---|---|
| **pandas** | Time-indexed tables, resampling, rolling windows | Native support for date indexes, calendar-aware resampling (`MS`, `YS`) and missing values, which is exactly what daily weather series need. |
| **NumPy** | Vectorised numerics, least squares | Foundation of the whole stack; `numpy.linalg.lstsq` fits the forecast model. |
| **SciPy** | Statistical tests | Peer-reviewed implementations of `linregress`, `kendalltau` (Mann-Kendall) and `theilslopes` (Sen's slope). Writing these by hand would risk errors. |
| **Matplotlib** | Figures written by the command line | Publication-quality static images that work headless on CI (`Agg` backend). The object-oriented `Figure` API avoids global state. |
| **Streamlit** | Web UI | Turns one Python script into an interactive app without HTML or JavaScript; about half of `app.py` is explanatory text for non-specialist users. It ships `AppTest` for automated UI tests. Alternatives: Dash/Panel (more boilerplate), Flask + JS (much more code). |
| **Plotly** | Interactive charts in the web UI | Hover values, zoom, a range slider and legend toggling work out of the box, and `st.plotly_chart` renders the figures natively. It is an optional dependency (the `ui` extra), so the library and the command line do not need it. Alternatives: Altair (already installed with Streamlit, but hover on line charts and shaded intervals need extra selection code for every chart), Bokeh (needs its own Streamlit component), Matplotlib (static images only). |
| **urllib (stdlib)** | Download from Open-Meteo | One GET request does not need an extra dependency such as `requests`. |

**Deliberately not used**

- *statsmodels*: it offers seasonal decomposition and ARIMA, but it is a heavy
  dependency. The classical decomposition and the trend + harmonics + AR(1)
  model are ~150 lines here. They are transparent, documented and tested
  against known answers.
- *Deep learning (e.g. LSTM)*: 30 years of monthly data is about 360 points,
  which is far too few. An interpretable statistical model is the scientifically
  appropriate choice, and its skill is measured honestly against climatology.

## Data source: Open-Meteo historical API

| Option | Pros | Cons |
|---|---|---|
| **Open-Meteo (ERA5 / ECMWF IFS)** | free, no API key, any coordinates, continuous daily data from 1940 | model grid cell, not a single station; the underlying model changes in 2017 |
| Meteostat | real station data | gaps, bulk endpoint changes |
| NOAA GHCN-Daily | authoritative station archive | large files, station-specific formats |
| Kazhydromet | official national source | no open machine-readable API |

A synthetic Astana dataset is bundled for tests and offline demos. It is
generated deterministically by `scripts/generate_sample_data.py` with a known
trend, so the tests can check that the analysis recovers the true answer.

## Development tools

| Tool | Role | Why |
|---|---|---|
| **Git + GitHub** | Version control, issues, pull requests, releases | Industry standard. GitHub adds issue/PR templates, Dependabot and Actions in one place. |
| **GitHub Actions** | CI/CD | Built into GitHub, free for public repositories, runs Linux/Windows/macOS, configured as YAML in the repository. Alternatives: GitLab CI (would need a GitLab mirror), Jenkins (needs your own server). |
| **pytest + pytest-cov** | Automated tests and coverage | Concise tests, fixtures and parametrisation. CI fails if coverage drops below 90 %. |
| **Ruff** | Linter + formatter | Replaces flake8, isort and black with one very fast tool. |
| **mypy** | Static type checking | Finds type errors (e.g. a non-date index) before runtime. |
| **setuptools + pyproject.toml** | Packaging | Standard PEP 621 metadata; `pip install .` gives the `climate-analyzer` command. |
| **Dependabot** | Dependency updates | Opens PRs when actions or libraries have new versions. |
