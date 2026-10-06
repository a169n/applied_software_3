"""Command line interface: ``climate-analyzer analyze data.csv -o results/``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from . import __version__, plotting
from .io import DataValidationError, datetime_index, load_csv, load_sample
from .pipeline import AnalysisConfig, AnalysisResult, run_analysis
from .sources import CITIES, DataSourceError, fetch_open_meteo


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="climate-analyzer",
        description="Clean weather station data and analyse long-term climate trends.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("analyze", help="analyse a CSV file and write a report with figures")
    run.add_argument(
        "input",
        help="CSV file with a date column and numeric columns, or 'sample' for the bundled data",
    )
    run.add_argument("-c", "--column", default="tavg", help="variable to analyse (default: tavg)")
    run.add_argument("-u", "--units", default="°C", help="units used in labels (default: °C)")
    run.add_argument("-o", "--output", type=Path, default=Path("results"), help="output folder")
    run.add_argument("--date-column", help="name of the date column (auto-detected by default)")
    run.add_argument(
        "--outliers",
        choices=["zscore", "iqr", "none"],
        default="zscore",
        help="outlier detection method (default: zscore)",
    )
    run.add_argument("--threshold", type=float, help="outlier threshold (z-score or IQR factor)")
    run.add_argument(
        "--max-gap", type=int, default=3, help="longest gap (days) to interpolate (default: 3)"
    )
    run.add_argument(
        "--baseline",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="reference period for climatology and anomalies, e.g. 1994 2013",
    )
    run.add_argument(
        "--smoothing", type=int, default=5, help="moving average window in years (default: 5)"
    )
    run.add_argument(
        "--forecast-months",
        type=int,
        default=24,
        help="months to forecast after the end of the data, 0 to disable (default: 24)",
    )
    run.add_argument(
        "--backtest-months",
        type=int,
        default=36,
        help="months held out to validate the model, 0 to disable (default: 36)",
    )
    run.add_argument("--no-plots", action="store_true", help="skip writing figures")

    last_year = date.today().year - 1
    fetch = sub.add_parser(
        "fetch", help="download daily weather history from Open-Meteo (needs internet)"
    )
    where = fetch.add_mutually_exclusive_group()
    where.add_argument(
        "--city",
        choices=sorted(CITIES),
        default="astana",
        help="predefined location (default: astana)",
    )
    where.add_argument(
        "--coords", type=float, nargs=2, metavar=("LAT", "LON"), help="custom location"
    )
    fetch.add_argument(
        "--start", type=date.fromisoformat, default=date(1994, 1, 1), help="first day (ISO date)"
    )
    fetch.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date(last_year, 12, 31),
        help="last day (ISO date, default: end of last year)",
    )
    fetch.add_argument("-o", "--output", type=Path, required=True, help="CSV file to write")
    return parser


def write_outputs(result: AnalysisResult, output: Path, plots: bool = True) -> list[Path]:
    """Write ``report.md``, ``summary.json``, ``annual.csv`` and figures to ``output``."""
    output.mkdir(parents=True, exist_ok=True)
    written = []

    for name, content in (("report.md", result.to_markdown()), ("summary.json", result.to_json())):
        path = output / name
        path.write_text(content, encoding="utf-8")
        written.append(path)

    annual = result.annual.to_frame("value").assign(anomaly=result.annual_anomalies)
    annual.index = datetime_index(annual).year.rename("year")
    path = output / "annual.csv"
    annual.to_csv(path, float_format="%.3f")
    written.append(path)

    if result.forecast is not None:
        path = output / "forecast.csv"
        result.forecast.to_csv(path, float_format="%.3f")
        written.append(path)

    if plots:
        units = result.config.units
        figures = {
            "trend.png": plotting.plot_time_series(
                result.cleaned, result.annual, result.trend, result.smoothed, units
            ),
            "anomalies.png": plotting.plot_anomalies(result.annual_anomalies, units),
            "climatology.png": plotting.plot_climatology(result.climatology, units=units),
        }
        if result.decomposition is not None:
            figures["decomposition.png"] = plotting.plot_decomposition(result.decomposition, units)
        if result.extremes is not None:
            figures["extremes.png"] = plotting.plot_extremes(result.extremes)
        if result.forecast is not None:
            figures["forecast.png"] = plotting.plot_forecast(
                result.monthly, result.forecast, result.backtest, units=units
            )
        for name, fig in figures.items():
            path = output / name
            fig.savefig(path, dpi=120)
            written.append(path)

    return written


def _fetch(args: argparse.Namespace) -> int:
    if args.coords:
        latitude, longitude = args.coords
        label = f"{latitude}, {longitude}"
    else:
        city = CITIES[args.city]
        latitude, longitude, label = city.latitude, city.longitude, city.name
    try:
        df = fetch_open_meteo(latitude, longitude, args.start, args.end)
    except (DataSourceError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, float_format="%.1f")
    print(f"{label}: wrote {len(df)} days ({args.start} to {args.end}) to {args.output}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    try:
        df = (
            load_sample()
            if args.input == "sample"
            else load_csv(args.input, date_column=args.date_column)
        )
        config = AnalysisConfig(
            column=args.column,
            units=args.units,
            outlier_method=args.outliers,
            outlier_threshold=args.threshold,
            max_gap=args.max_gap,
            baseline=tuple(args.baseline) if args.baseline else None,
            smoothing_years=args.smoothing,
            forecast_months=args.forecast_months,
            backtest_months=args.backtest_months,
        )
        result = run_analysis(df, config)
    except FileNotFoundError:
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 2
    except (DataValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    written = write_outputs(result, args.output, plots=not args.no_plots)
    t = result.trend
    if t is None:
        print(f"{config.column}: not enough complete years to estimate a trend")
    else:
        print(
            f"{config.column}: trend {t.slope_per_decade:+.3f} {config.units}/decade "
            f"(Mann-Kendall p = {t.mann_kendall_p_value:.3g})"
        )
    for warning in result.warnings:
        print(f"  warning: {warning}")
    if result.backtest is not None:
        print(f"  model skill vs climatology: {result.backtest.skill:+.1%}")
    for path in written:
        print(f"  wrote {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "fetch":
        return _fetch(args)
    return _analyze(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
