"""Unusual days, after accounting for weekday, season and the local level.

Residuals come from a harmonic regression on log counts. A centred rolling
median of those residuals absorbs slow level changes, so what remains is
day-specific. Scores are robust z-scores (median / MAD, Iglewicz & Hoaglin
1993); |z| > 3.5 is their recommended outlier cut-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maritime_flow.analysis.models import harmonic_design, ols


def detect_anomalies(
    series: pd.Series, k_annual: int = 3, window: int = 29, threshold: float = 3.5
) -> pd.DataFrame:
    s = series.dropna()
    y = np.log1p(s.to_numpy())
    X = harmonic_design(s.index, k_annual=k_annual, trend=False)
    fit = ols(y, X)
    resid = pd.Series(fit.resid, index=s.index)
    local_level = resid.rolling(window, center=True, min_periods=window // 2).median()
    local = resid - local_level
    mad = np.median(np.abs(local.dropna() - local.median()))
    z = 0.6745 * (local - local.median()) / (mad if mad > 0 else 1.0)
    expected = np.expm1(fit.fitted + local_level.to_numpy())
    out = pd.DataFrame({"observed": s, "expected": expected, "robust_z": z})
    out["anomaly"] = out["robust_z"].abs() > threshold
    out["direction"] = np.where(out["robust_z"] > 0, "high", "low")
    return out


def anomaly_table(anoms: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    a = anoms[anoms["anomaly"]].copy()
    a["abs_z"] = a["robust_z"].abs()
    return a.sort_values("abs_z", ascending=False).head(top).drop(columns="abs_z")
