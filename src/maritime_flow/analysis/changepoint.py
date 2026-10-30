"""Multiple change-point detection with PELT (Killick, Fearnhead & Eckley, 2012).

The cost is the Gaussian change-in-mean cost (within-segment sum of squares),
computed in O(1) per segment from cumulative sums. Pruning keeps the expected
run time close to linear. We run it on weekly means rather than raw daily
counts: weekly aggregation removes the weekday cycle and makes the Gaussian
cost a reasonable approximation for count data at these volumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from maritime_flow.analysis.models import harmonic_design, ols


@dataclass
class ChangepointResult:
    breakpoints: list[int]
    segments: pd.DataFrame
    penalty: float
    sigma: float


def robust_sigma(x: np.ndarray) -> float:
    """Noise scale from first differences via the MAD.

    Differencing removes level shifts, so the estimate is not inflated by the
    very changes we are trying to find.
    """
    d = np.diff(x)
    mad = np.median(np.abs(d - np.median(d)))
    return float(mad / 0.6745 / np.sqrt(2)) or float(np.std(x)) or 1.0


def pelt(x, penalty: float | None = None, min_size: int = 4) -> ChangepointResult:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_size:
        return ChangepointResult([], _segments(x, []), penalty or 0.0, 0.0)
    sigma = robust_sigma(x)
    if penalty is None:
        # BIC-style penalty expressed in sum-of-squares units
        penalty = 2.0 * sigma**2 * np.log(n)

    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def cost(s: np.ndarray, t: int) -> np.ndarray:
        m = t - s
        seg = cs[t] - cs[s]
        return (cs2[t] - cs2[s]) - seg * seg / m

    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    candidates = np.array([0])

    for t in range(min_size, n + 1):
        ok = candidates[t - candidates >= min_size]
        if ok.size:
            vals = F[ok] + cost(ok, t) + penalty
            j = int(np.argmin(vals))
            F[t], last[t] = vals[j], ok[j]
            keep_ok = ok[vals - penalty <= F[t]]
        else:
            keep_ok = ok
        young = candidates[t - candidates < min_size]
        candidates = np.concatenate([keep_ok, young, [t]])

    bps, t = [], n
    while t > 0:
        s = last[t]
        if s > 0:
            bps.append(int(s))
        t = s
    bps.sort()
    return ChangepointResult(bps, _segments(x, bps), float(penalty), sigma)


def _segments(x: np.ndarray, bps: list[int]) -> pd.DataFrame:
    edges = [0, *bps, len(x)]
    rows = [
        {"start": a, "end": b, "n": b - a, "mean": float(np.mean(x[a:b])), "std": float(np.std(x[a:b], ddof=1)) if b - a > 1 else np.nan}
        for a, b in zip(edges[:-1], edges[1:])
    ]
    return pd.DataFrame(rows)


def seasonal_adjust(
    series: pd.Series, k_annual: int = 3, breaks: list[pd.Timestamp] | None = None
) -> pd.Series:
    """Divide out weekday and annual seasonality estimated on log counts.

    Without this step PELT happily reports every seasonal peak and trough as a
    "change point". ``breaks`` adds a level dummy per known regime so that a
    large shift is not partly absorbed into the Fourier terms. The seasonal
    factors are normalised to average 1, so the adjusted series stays in
    ships/day.
    """
    s = series.dropna()
    X = harmonic_design(s.index, k_annual=k_annual, trend=False)
    for i, b in enumerate(breaks or []):
        X[f"regime_{i}"] = (s.index >= b).astype(float)
    fit = ols(np.log1p(s.to_numpy()), X)
    seas_cols = [c for c in X.columns if c != "const" and not c.startswith("regime_")]
    log_factor = X[seas_cols].to_numpy() @ fit.params[seas_cols].to_numpy()
    factor = np.exp(log_factor - log_factor.mean())
    return pd.Series(s.to_numpy() / factor, index=s.index, name=series.name)


def detect_level_shifts(
    series: pd.Series,
    freq: str = "W-MON",
    penalty_scale: float = 1.0,
    min_weeks: int = 4,
    adjust: bool = True,
) -> tuple[pd.Series, ChangepointResult, pd.DataFrame]:
    """Weekly change points on the seasonally adjusted series, with dated segments.

    Returns the weekly series, the raw result and a segment table with dates and
    the percent change in level versus the previous segment.
    """
    breaks: list[pd.Timestamp] = []
    # Two passes: estimate seasonality, find regimes, re-estimate seasonality
    # with those regimes held out, find regimes again.
    # PELT runs on log weekly means: count noise grows with the level, and the
    # log makes it roughly constant, which the Gaussian cost assumes.
    for _ in range(2 if adjust else 1):
        base = seasonal_adjust(series, breaks=breaks) if adjust else series
        weekly = base.resample(freq, label="left", closed="left").mean().dropna()
        weekly = weekly[weekly > 0]
        x = np.log(weekly.to_numpy())
        penalty = 2.0 * robust_sigma(x) ** 2 * np.log(len(x)) * penalty_scale
        res = pelt(x, penalty=penalty, min_size=min_weeks)
        breaks = [weekly.index[b] for b in res.breakpoints]

    edges = [0, *res.breakpoints, len(weekly)]
    seg = pd.DataFrame(
        [
            {
                "start_date": weekly.index[a],
                "end_date": weekly.index[b - 1],
                "n": b - a,
                "mean": weekly.iloc[a:b].mean(),
                "std": weekly.iloc[a:b].std(),
            }
            for a, b in zip(edges[:-1], edges[1:])
        ]
    )
    seg["change_vs_prev"] = seg["mean"].pct_change()
    return weekly, res, seg
