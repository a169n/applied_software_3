"""climate-analyzer: clean weather station data and analyse long-term climate trends."""

from .analysis import (
    TrendResult,
    anomalies,
    climatology,
    extreme_days,
    linear_trend,
    moving_average,
    resample,
    seasonal_decompose,
    summary_statistics,
)
from .cleaning import CleaningReport, clean, detect_outliers, fill_gaps
from .forecast import BacktestResult, SeasonalTrendModel, backtest
from .io import DataValidationError, load_csv, load_sample
from .pipeline import AnalysisConfig, AnalysisResult, run_analysis

__version__ = "0.1.0"

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "BacktestResult",
    "CleaningReport",
    "DataValidationError",
    "SeasonalTrendModel",
    "TrendResult",
    "__version__",
    "anomalies",
    "backtest",
    "clean",
    "climatology",
    "detect_outliers",
    "extreme_days",
    "fill_gaps",
    "linear_trend",
    "load_csv",
    "load_sample",
    "moving_average",
    "resample",
    "run_analysis",
    "seasonal_decompose",
    "summary_statistics",
]
