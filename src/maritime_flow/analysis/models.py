"""Small, dependency-light regression toolkit.

Implemented directly with numpy/scipy instead of pulling in statsmodels, both
to keep the install light and because the estimators used here are short:

* OLS with classical or Newey-West (HAC) standard errors
* Poisson GLM fitted by IRLS, with a quasi-Poisson (Pearson) dispersion
  correction for over-dispersed counts
* a harmonic (Fourier) design matrix for weekly + annual seasonality

The tests compare these against closed-form results and simulated truths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

DAYS_PER_YEAR = 365.25
EPOCH = pd.Timestamp("1970-01-01")


def harmonic_design(
    index: pd.DatetimeIndex,
    k_annual: int = 3,
    trend: bool = True,
    weekday: bool = True,
    origin: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Design matrix: intercept, optional linear trend, weekday dummies, annual Fourier terms.

    Fourier terms use absolute time so fitted and forecast rows share the same
    phase; the trend is measured in years from ``origin`` (default: first date).
    """
    index = pd.DatetimeIndex(index)
    t_abs = np.asarray((index - EPOCH) / pd.Timedelta(days=1), dtype=float)
    cols: dict[str, np.ndarray] = {"const": np.ones(len(index))}
    if trend:
        origin = index[0] if origin is None else pd.Timestamp(origin)
        cols["trend"] = np.asarray((index - origin) / pd.Timedelta(days=1), dtype=float) / DAYS_PER_YEAR
    if weekday:
        dow = index.dayofweek.to_numpy()
        for d, name in enumerate(["tue", "wed", "thu", "fri", "sat", "sun"], start=1):
            cols[f"dow_{name}"] = (dow == d).astype(float)
    for k in range(1, k_annual + 1):
        w = 2 * np.pi * k * t_abs / DAYS_PER_YEAR
        cols[f"sin{k}"] = np.sin(w)
        cols[f"cos{k}"] = np.cos(w)
    return pd.DataFrame(cols, index=index)


@dataclass
class FitResult:
    params: pd.Series
    cov: np.ndarray
    fitted: np.ndarray
    resid: np.ndarray
    df_resid: int
    dispersion: float = 1.0
    cov_type: str = "nonrobust"

    @property
    def bse(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.cov)), index=self.params.index)

    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    def pvalues(self) -> pd.Series:
        return pd.Series(2 * stats.t.sf(np.abs(self.tvalues()), self.df_resid), index=self.params.index)

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        q = stats.t.ppf(1 - alpha / 2, self.df_resid)
        return pd.DataFrame(
            {"lower": self.params - q * self.bse, "upper": self.params + q * self.bse}
        )

    def summary_frame(self, alpha: float = 0.05) -> pd.DataFrame:
        ci = self.conf_int(alpha)
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.bse,
                "t": self.tvalues(),
                "p_value": self.pvalues(),
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
            }
        )


def default_hac_lags(n: int) -> int:
    """Newey-West (1994) rule of thumb."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_cov(X: np.ndarray, resid: np.ndarray, maxlags: int) -> np.ndarray:
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    xu = X * resid[:, None]
    meat = xu.T @ xu
    for lag in range(1, maxlags + 1):
        w = 1 - lag / (maxlags + 1)  # Bartlett kernel
        gamma = xu[lag:].T @ xu[:-lag]
        meat += w * (gamma + gamma.T)
    return n / (n - k) * bread @ meat @ bread


def ols(y, X: pd.DataFrame, cov_type: str = "nonrobust", maxlags: int | None = None) -> FitResult:
    yv = np.asarray(y, dtype=float)
    Xv = np.asarray(X, dtype=float)
    n, k = Xv.shape
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    fitted = Xv @ beta
    resid = yv - fitted
    if cov_type == "nonrobust":
        sigma2 = resid @ resid / (n - k)
        cov = sigma2 * np.linalg.pinv(Xv.T @ Xv)
    elif cov_type == "hac":
        lags = default_hac_lags(n) if maxlags is None else maxlags
        cov = newey_west_cov(Xv, resid, lags)
    else:
        raise ValueError(f"unknown cov_type {cov_type!r}")
    return FitResult(pd.Series(beta, index=X.columns), cov, fitted, resid, n - k, cov_type=cov_type)


def poisson_glm(
    y,
    X: pd.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-10,
    quasi: bool = True,
) -> FitResult:
    """Poisson regression with log link via iteratively reweighted least squares.

    With ``quasi=True`` the covariance is scaled by the Pearson dispersion
    statistic, which keeps inference honest when counts are over-dispersed
    (variance > mean), as ship counts almost always are.
    """
    yv = np.asarray(y, dtype=float)
    Xv = np.asarray(X, dtype=float)
    n, k = Xv.shape
    mu = yv + 0.5 * max(yv.mean(), 1e-3)
    eta = np.log(mu)
    dev_old = np.inf
    for _ in range(max_iter):
        z = eta + (yv - mu) / mu
        sw = np.sqrt(mu)
        beta, *_ = np.linalg.lstsq(Xv * sw[:, None], z * sw, rcond=None)
        eta = np.clip(Xv @ beta, -30, 30)
        mu = np.exp(eta)
        dev = poisson_deviance(yv, mu)
        if abs(dev - dev_old) < tol * (abs(dev) + 0.1):
            break
        dev_old = dev
    pearson = (yv - mu) / np.sqrt(mu)
    dispersion = float(pearson @ pearson / (n - k)) if quasi else 1.0
    cov = dispersion * np.linalg.pinv((Xv * mu[:, None]).T @ Xv)
    return FitResult(
        pd.Series(beta, index=X.columns), cov, mu, yv - mu, n - k, dispersion, "quasi-poisson"
    )


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(term - (y - mu)))
