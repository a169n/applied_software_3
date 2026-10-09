# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-09

### Added
- CSV loader with automatic date column detection and sentinel handling.
- Quality control: seasonal z-score / IQR outlier detection, short-gap interpolation.
- Analysis: coverage-aware resampling, OLS and Mann-Kendall/Sen trends, climatology,
  anomalies, classical seasonal decomposition, warm/cold extreme-day counts.
- Forecasting model (trend + annual harmonics + AR(1)) with hold-out backtest
  against climatology.
- Open-Meteo downloader for Astana and other cities (`climate-analyzer fetch`).
- Command line interface and Markdown/JSON reports.
- Streamlit web UI with interactive Plotly charts, plain-language labels and an
  explanation of every analysis.
- Bundled synthetic Astana dataset (1994-2023) for offline tests and demos.
- GitHub Actions: CI (lint, types, tests on 3 OSes), scheduled Astana forecast,
  tag-triggered release.
