"""Statistical model of a seasonal climate series and forecasts from it.

The model is a classic, fully transparent regression used in climatology::

    y(t) = b0 + b1*t + sum_k [a_k cos(2*pi*k*t) + c_k sin(2*pi*k*t)] + r(t)
    r(t) = phi * r(t-1) + e(t)

where ``t`` is time in decimal years, the harmonic terms describe the annual
cycle, ``b1`` is the long-term trend, and the residual ``r`` follows a
first-order autoregressive process (warm months tend to follow warm months).
Forecasts combine the deterministic part with the decaying AR(1) memory, and
prediction intervals widen with the horizon accordingly.

The model is evaluated honestly with a hold-out *backtest*: it is fitted on
the earlier part of the record and its predictions for the last years are
compared with what actually happened and with the naive *climatology*
forecast ("next January will be like an average January").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import decimal_year
from .io import datetime_index


def _design_matrix(index: pd.DatetimeIndex, harmonics: int, t0: float) -> np.ndarray:
    t = decimal_year(index)
    columns = [np.ones_like(t), t - t0]
    for k in range(1, harmonics + 1):
        columns += [np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]
    return np.column_stack(columns)


def _infer_freq(index: pd.DatetimeIndex) -> str:
    if index.freqstr:
        return index.freqstr
    freq = pd.infer_freq(index[: min(len(index), 50)])
    if freq is None:
        raise ValueError("cannot infer the frequency of the series; resample it first")
    return freq


@dataclass
class SeasonalTrendModel:
    """Linear trend + annual harmonics + AR(1) residuals, fitted by least squares.

    Parameters
    ----------
    harmonics:
        Number of sine/cosine pairs describing the annual cycle.  Two capture
        the asymmetric continental cycle of Astana well; more risk overfitting.
    non_negative:
        Clip predictions at zero (for precipitation).
    """

    harmonics: int = 2
    non_negative: bool = False

    def __post_init__(self) -> None:
        if self.harmonics < 0:
            raise ValueError("harmonics must be non-negative")
        self._coef: np.ndarray | None = None

    # -- fitting ---------------------------------------------------------
    def fit(self, series: pd.Series) -> SeasonalTrendModel:
        index = datetime_index(series)
        valid = series.dropna()
        valid_index = datetime_index(valid)
        n_params = 2 + 2 * self.harmonics
        if len(valid) < n_params + 2:
            raise ValueError(f"need at least {n_params + 2} observations to fit the model")

        self.freq_ = _infer_freq(index)
        self.t0_ = float(decimal_year(valid_index).mean())
        X = _design_matrix(valid_index, self.harmonics, self.t0_)
        coef, *_ = np.linalg.lstsq(X, valid.to_numpy(dtype=float), rcond=None)
        self._coef = coef

        residuals = series - self._deterministic(index)
        lagged = pd.concat([residuals, residuals.shift(1)], axis=1).dropna()
        if len(lagged) > 2 and lagged.iloc[:, 1].std() > 0:
            phi = float(np.corrcoef(lagged.iloc[:, 0], lagged.iloc[:, 1])[0, 1])
        else:
            phi = 0.0
        self.phi_ = float(np.clip(phi, -0.99, 0.99))
        self.sigma_ = float(residuals.std(ddof=n_params))
        self.last_index_: pd.Timestamp = index[-1]
        # The AR(1) memory only applies if the very last period was observed.
        last = residuals.iloc[-1]
        self.last_residual_: float = 0.0 if pd.isna(last) else float(last)
        self.fitted_ = self._deterministic(index)
        self.residuals_ = residuals
        return self

    def _check_fitted(self) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("the model is not fitted yet; call fit() first")
        return self._coef

    def _deterministic(self, index: pd.DatetimeIndex) -> pd.Series:
        coef = self._check_fitted()
        return pd.Series(_design_matrix(index, self.harmonics, self.t0_) @ coef, index=index)

    # -- inspection ------------------------------------------------------
    @property
    def trend_per_decade(self) -> float:
        """Slope of the linear trend term in units per decade."""
        return float(self._check_fitted()[1] * 10)

    @property
    def seasonal_amplitude(self) -> float:
        """Half the peak-to-peak range of the fitted annual cycle."""
        self._check_fitted()
        days = pd.date_range("2001-01-01", "2001-12-31", freq="D")
        cycle = self._deterministic(days)
        return float((cycle.max() - cycle.min()) / 2)

    def r_squared(self) -> float:
        """Share of variance explained by the deterministic part."""
        self._check_fitted()
        resid = self.residuals_.dropna()
        observed = resid + self.fitted_.reindex(resid.index)
        return float(1 - resid.var() / observed.var())

    # -- prediction ------------------------------------------------------
    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        """Deterministic model values (trend + seasonal cycle) at ``index``."""
        values = self._deterministic(index)
        return values.clip(lower=0) if self.non_negative else values

    def forecast(self, steps: int, level: float = 0.95) -> pd.DataFrame:
        """Forecast ``steps`` periods after the end of the training series.

        Returns a frame with ``forecast``, ``lower`` and ``upper`` columns; the
        bounds form a ``level`` prediction interval.
        """
        self._check_fitted()
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1")

        index = pd.date_range(self.last_index_, periods=steps + 1, freq=self.freq_)[1:]
        h = np.arange(1, steps + 1)
        memory = self.last_residual_ * self.phi_**h
        mean = self._deterministic(index).to_numpy() + memory
        # Variance of an AR(1) h steps ahead: sigma^2 * (1 - phi^(2h)), plus a
        # floor so the first step is never overconfident.
        std = self.sigma_ * np.sqrt(np.maximum(1 - self.phi_ ** (2 * h), 1 - self.phi_**2))
        z = stats.norm.ppf(0.5 + level / 2)
        frame = pd.DataFrame(
            {"forecast": mean, "lower": mean - z * std, "upper": mean + z * std}, index=index
        )
        if self.non_negative:
            frame = frame.clip(lower=0)
        frame.index.name = "date"
        return frame


@dataclass(frozen=True)
class BacktestResult:
    """Out-of-sample accuracy of the model compared with climatology."""

    train_end: pd.Timestamp
    n_test: int
    mae: float
    rmse: float
    bias: float
    climatology_mae: float
    coverage: float
    predictions: pd.DataFrame

    @property
    def skill(self) -> float:
        """Mean absolute error skill score: 1 is perfect, 0 is no better than climatology."""
        if self.climatology_mae == 0:
            return 0.0
        return 1 - self.mae / self.climatology_mae


def backtest(
    series: pd.Series,
    test_periods: int,
    harmonics: int = 2,
    non_negative: bool = False,
    level: float = 0.95,
) -> BacktestResult:
    """Fit on all but the last ``test_periods`` values and score the forecast for them."""
    if test_periods < 1:
        raise ValueError("test_periods must be at least 1")
    if test_periods >= len(series):
        raise ValueError("test_periods must be shorter than the series")

    train, test = series.iloc[:-test_periods], series.iloc[-test_periods:]
    model = SeasonalTrendModel(harmonics=harmonics, non_negative=non_negative).fit(train)
    fc = model.forecast(test_periods, level=level).set_axis(test.index)

    clim = train.groupby(datetime_index(train).month).mean()
    test_index = datetime_index(test)
    climatology = pd.Series(clim.reindex(test_index.month).to_numpy(), index=test_index)

    frame = pd.DataFrame(
        {
            "actual": test,
            "predicted": fc["forecast"],
            "lower": fc["lower"],
            "upper": fc["upper"],
            "climatology": climatology,
        }
    )
    scored = frame.dropna(subset=["actual"])
    if scored.empty:
        raise ValueError("the test period contains no observations")
    error = scored["predicted"] - scored["actual"]
    inside = scored["actual"].between(scored["lower"], scored["upper"])
    return BacktestResult(
        train_end=datetime_index(train)[-1],
        n_test=len(scored),
        mae=float(error.abs().mean()),
        rmse=float(np.sqrt((error**2).mean())),
        bias=float(error.mean()),
        climatology_mae=float((scored["climatology"] - scored["actual"]).abs().mean()),
        coverage=float(inside.mean()),
        predictions=frame,
    )
