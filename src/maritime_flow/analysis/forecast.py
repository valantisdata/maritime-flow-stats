"""Short-term forecasting with honest evaluation.

Three models of increasing complexity are compared on a rolling-origin
backtest (fit on data up to an origin, forecast the next ``horizon`` days,
move the origin forward, repeat). The seasonal naive model ("same as last
week") is the benchmark every other model has to beat; MASE < 1 means a model
beats it on average. A Diebold-Mariano test checks whether the difference is
larger than noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from maritime_flow.analysis.models import harmonic_design, newey_west_cov, ols


class SeasonalNaive:
    name = "seasonal_naive"

    def __init__(self, m: int = 7):
        self.m = m

    def fit(self, y: pd.Series) -> SeasonalNaive:
        self._last = y.to_numpy()[-self.m :]
        return self

    def predict(self, h: int) -> np.ndarray:
        return np.resize(self._last, h).astype(float)


class HarmonicRegression:
    """OLS on log1p(counts) with weekday dummies, annual Fourier terms and trend."""

    name = "harmonic_regression"

    def __init__(self, k_annual: int = 3, window: int = 730, trend: bool = True):
        self.k_annual, self.window, self.trend = k_annual, window, trend

    def fit(self, y: pd.Series) -> HarmonicRegression:
        y = y.iloc[-self.window :]
        self._origin = y.index[0]
        self._end = y.index[-1]
        X = harmonic_design(y.index, self.k_annual, trend=self.trend, origin=self._origin)
        self._fit = ols(np.log1p(y.to_numpy()), X)
        # smearing correction for the log back-transform (Duan, 1983)
        self._smear = float(np.mean(np.exp(self._fit.resid)))
        return self

    def predict(self, h: int) -> np.ndarray:
        idx = pd.date_range(self._end + pd.Timedelta(days=1), periods=h, freq="D")
        X = harmonic_design(idx, self.k_annual, trend=self.trend, origin=self._origin)
        return np.maximum(np.exp(X.to_numpy() @ self._fit.params.to_numpy()) * self._smear - 1, 0)


class HoltWinters:
    """Additive Holt-Winters with damped trend, parameters chosen by minimising one-step SSE."""

    name = "holt_winters"

    def __init__(self, m: int = 7, window: int = 365, phi: float = 0.98):
        self.m, self.window, self.phi = m, window, phi

    def _filter(self, params, y):
        a, b, g = params
        m, phi = self.m, self.phi
        level = y[:m].mean()
        trend = (y[m : 2 * m].mean() - y[:m].mean()) / m if len(y) >= 2 * m else 0.0
        season = list(y[:m] - level)
        sse = 0.0
        for t in range(m, len(y)):
            s_prev = season[t - m]
            pred = level + phi * trend + s_prev
            err = y[t] - pred
            sse += err * err
            new_level = a * (y[t] - s_prev) + (1 - a) * (level + phi * trend)
            trend = b * (new_level - level) + (1 - b) * phi * trend
            season.append(g * (y[t] - level - phi * trend) + (1 - g) * s_prev)
            level = new_level
        return sse, level, trend, season

    def fit(self, y: pd.Series) -> HoltWinters:
        yv = y.to_numpy(dtype=float)[-self.window :]
        res = optimize.minimize(
            lambda p: self._filter(p, yv)[0],
            x0=[0.2, 0.02, 0.1],
            bounds=[(1e-3, 0.999), (1e-4, 0.5), (1e-3, 0.999)],
            method="L-BFGS-B",
        )
        self.params_ = res.x
        _, self._level, self._trend, season = self._filter(res.x, yv)
        self._season = np.array(season[-self.m :])
        return self

    def predict(self, h: int) -> np.ndarray:
        steps = np.arange(1, h + 1)
        damp = np.cumsum(self.phi**steps)
        seas = self._season[(steps - 1) % self.m]
        return np.maximum(self._level + damp * self._trend + seas, 0)


DEFAULT_MODELS = {
    "seasonal_naive": SeasonalNaive,
    "harmonic_regression": HarmonicRegression,
    "holt_winters": HoltWinters,
}


@dataclass
class BacktestResult:
    errors: pd.DataFrame
    scores: pd.DataFrame
    dm_tests: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def best_model(self) -> str:
        return str(self.scores["mase"].idxmin())


def rolling_origin_backtest(
    y: pd.Series,
    models: dict | None = None,
    horizon: int = 28,
    n_origins: int = 10,
    min_train: int = 400,
) -> BacktestResult:
    """Non-overlapping rolling-origin evaluation over the last ``n_origins * horizon`` days."""
    models = models or DEFAULT_MODELS
    y = y.dropna().asfreq("D").interpolate(limit_area="inside").dropna()
    n = len(y)
    origins = [n - horizon * k for k in range(n_origins, 0, -1)]
    origins = [o for o in origins if o >= min_train]
    if not origins:
        raise ValueError(f"series too short for backtest ({n} days)")

    rows = []
    for o in origins:
        train, test = y.iloc[:o], y.iloc[o : o + horizon]
        # MASE scale: in-sample MAE of the weekly naive forecast over the last year
        recent = train.to_numpy()[-365:]
        scale = np.mean(np.abs(recent[7:] - recent[:-7]))
        for name, factory in models.items():
            pred = factory().fit(train).predict(len(test))
            for step, (date, actual, p) in enumerate(zip(test.index, test.to_numpy(), pred), start=1):
                rows.append((name, train.index[-1], date, step, actual, p, scale))
    err = pd.DataFrame(rows, columns=["model", "origin", "date", "step", "actual", "forecast", "scale"])
    err["error"] = err["actual"] - err["forecast"]

    scores = err.groupby("model").apply(
        lambda g: pd.Series(
            {
                "mae": g["error"].abs().mean(),
                "rmse": np.sqrt((g["error"] ** 2).mean()),
                "mase": (g["error"].abs() / g["scale"]).mean(),
                "bias": g["error"].mean(),
            }
        ),
        include_groups=False,
    ).sort_values("mase")

    dm_rows = []
    bench = "seasonal_naive"
    if bench in models:
        e_b = err[err.model == bench].sort_values(["origin", "step"])["error"].to_numpy()
        for name in models:
            if name == bench:
                continue
            e_m = err[err.model == name].sort_values(["origin", "step"])["error"].to_numpy()
            stat, p = diebold_mariano(e_m, e_b, h=horizon)
            dm_rows.append({"model": name, "vs": bench, "dm_stat": stat, "p_value": p})
    return BacktestResult(err, scores, pd.DataFrame(dm_rows))


def diebold_mariano(e1, e2, h: int = 1, power: int = 2) -> tuple[float, float]:
    """DM test of equal accuracy with the Harvey-Leybourne-Newbold correction.

    Negative statistic = the first model has lower loss.
    """
    d = np.abs(np.asarray(e1)) ** power - np.abs(np.asarray(e2)) ** power
    n = len(d)
    lags = max(h - 1, 0)
    # long-run variance of the mean loss differential (Bartlett kernel, h-1 lags)
    var_mean = newey_west_cov(np.ones((n, 1)), d - d.mean(), lags)[0, 0] * (n - 1) / n
    stat = d.mean() / np.sqrt(var_mean) if var_mean > 0 else 0.0
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= correction
    p = 2 * stats.t.sf(abs(stat), n - 1)
    return float(stat), float(p)


def forecast_with_intervals(
    y: pd.Series, model_name: str, backtest: BacktestResult, horizon: int = 28, levels=(0.8, 0.95)
) -> pd.DataFrame:
    """Point forecast plus empirical intervals from backtest errors.

    Errors are pooled by forecast week (days 1-7, 8-14, ...) so each quantile
    rests on ~n_origins * 7 errors instead of a handful, while still letting
    the intervals widen with the horizon.
    """
    y = y.dropna().asfreq("D").interpolate(limit_area="inside").dropna()
    model = DEFAULT_MODELS[model_name]().fit(y)
    pred = model.predict(horizon)
    idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    out = pd.DataFrame({"forecast": pred}, index=idx)
    e = backtest.errors[backtest.errors.model == model_name].copy()
    e["week"] = (e["step"] - 1) // 7
    weeks = (np.arange(horizon)) // 7
    for lvl in levels:
        a = (1 - lvl) / 2
        q = e.groupby("week")["error"].quantile([a, 1 - a]).unstack()
        q = q.reindex(range(weeks.max() + 1)).ffill().bfill()
        out[f"lower_{int(lvl * 100)}"] = np.maximum(pred + q.iloc[weeks, 0].to_numpy(), 0)
        out[f"upper_{int(lvl * 100)}"] = pred + q.iloc[weeks, 1].to_numpy()
    return out
