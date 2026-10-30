"""How much did an event change traffic? Three complementary answers.

1. ``bootstrap_mean_diff``: simple before/after comparison of daily means with a
   moving-block bootstrap. Blocks preserve autocorrelation; an ordinary
   bootstrap on daily data would give confidence intervals that are too narrow.
2. ``interrupted_time_series``: quasi-Poisson regression with seasonality,
   a pre-existing trend, a level change at the event and a post-event slope.
   The level change is reported as a rate ratio (e.g. 0.45 = traffic at 45% of
   what the pre-event model implies).
3. ``counterfactual_gap``: fit on the pre-event window only, project forward
   and count the "missing" transits, with simulation-based uncertainty that
   includes both parameter uncertainty and day-to-day noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from maritime_flow.analysis.models import FitResult, harmonic_design, poisson_glm


def moving_block_resample(x: np.ndarray, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return x[idx]


def bootstrap_mean_diff(
    pre, post, block: int = 14, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0
) -> dict:
    pre = np.asarray(pd.Series(pre).dropna(), dtype=float)
    post = np.asarray(pd.Series(post).dropna(), dtype=float)
    rng = np.random.default_rng(seed)
    m_pre = moving_block_resample(pre, block, n_boot, rng).mean(axis=1)
    m_post = moving_block_resample(post, block, n_boot, rng).mean(axis=1)
    diff = m_post - m_pre
    rel = m_post / m_pre - 1
    lo, hi = alpha / 2, 1 - alpha / 2
    mw = stats.mannwhitneyu(post, pre, alternative="two-sided")
    return {
        "pre_mean": pre.mean(),
        "post_mean": post.mean(),
        "diff": post.mean() - pre.mean(),
        "diff_ci": tuple(np.quantile(diff, [lo, hi])),
        "rel_change": post.mean() / pre.mean() - 1,
        "rel_change_ci": tuple(np.quantile(rel, [lo, hi])),
        "mann_whitney_p": float(mw.pvalue),
        "n_pre": len(pre),
        "n_post": len(post),
        "block": block,
    }


@dataclass
class ITSResult:
    rate_ratio: float
    rate_ratio_ci: tuple[float, float]
    p_value: float
    monthly_slope_ratio: float
    monthly_slope_ratio_ci: tuple[float, float]
    dispersion: float
    fit: FitResult
    fitted: pd.Series

    def as_dict(self) -> dict:
        return {
            "rate_ratio": self.rate_ratio,
            "rate_ratio_ci": self.rate_ratio_ci,
            "p_value": self.p_value,
            "monthly_slope_ratio": self.monthly_slope_ratio,
            "monthly_slope_ratio_ci": self.monthly_slope_ratio_ci,
            "dispersion": self.dispersion,
        }


def interrupted_time_series(
    series: pd.Series,
    event: str | pd.Timestamp,
    k_annual: int = 3,
    pre_years: float | None = 3.0,
    post_days: int | None = 365,
) -> ITSResult:
    event = pd.Timestamp(event)
    s = series.dropna()
    if pre_years is not None:
        s = s[s.index >= event - pd.Timedelta(days=int(pre_years * 365.25))]
    if post_days is not None:
        s = s[s.index < event + pd.Timedelta(days=post_days)]
    if (s.index < event).sum() < 60 or (s.index >= event).sum() < 14:
        raise ValueError("need at least 60 pre-event and 14 post-event days")

    X = harmonic_design(s.index, k_annual=k_annual, trend=True)
    post = (s.index >= event).astype(float)
    X["step"] = post
    X["post_slope"] = post * np.asarray((s.index - event) / pd.Timedelta(days=1), dtype=float) / 30.4375
    fit = poisson_glm(s.to_numpy(), X)
    ci = fit.conf_int()
    return ITSResult(
        rate_ratio=float(np.exp(fit.params["step"])),
        rate_ratio_ci=(float(np.exp(ci.loc["step", "lower"])), float(np.exp(ci.loc["step", "upper"]))),
        p_value=float(fit.pvalues()["step"]),
        monthly_slope_ratio=float(np.exp(fit.params["post_slope"])),
        monthly_slope_ratio_ci=(
            float(np.exp(ci.loc["post_slope", "lower"])),
            float(np.exp(ci.loc["post_slope", "upper"])),
        ),
        dispersion=fit.dispersion,
        fit=fit,
        fitted=pd.Series(fit.fitted, index=s.index, name="fitted"),
    )


def counterfactual_gap(
    series: pd.Series,
    event: str | pd.Timestamp,
    pre_days: int = 730,
    post_days: int | None = None,
    k_annual: int = 3,
    trend: bool = False,
    n_sims: int = 2000,
    alpha: float = 0.1,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Expected-vs-observed after an event, with simulation intervals.

    The default is *no* trend: extrapolating a two-year linear trend for many
    months is fragile and tends to dominate the answer. Seasonality is kept.
    """
    event = pd.Timestamp(event)
    s = series.dropna()
    pre = s[(s.index < event) & (s.index >= event - pd.Timedelta(days=pre_days))]
    post = s[s.index >= event]
    if post_days is not None:
        post = post[post.index < event + pd.Timedelta(days=post_days)]
    if len(pre) < 90 or len(post) == 0:
        raise ValueError("not enough data around the event")

    X_pre = harmonic_design(pre.index, k_annual=k_annual, trend=trend, origin=pre.index[0])
    fit = poisson_glm(pre.to_numpy(), X_pre)
    X_post = harmonic_design(post.index, k_annual=k_annual, trend=trend, origin=pre.index[0])

    rng = np.random.default_rng(seed)
    betas = rng.multivariate_normal(fit.params.to_numpy(), fit.cov, size=n_sims)
    mu = np.exp(np.clip(X_post.to_numpy() @ betas.T, -30, 30))  # (n_post, n_sims)
    phi = max(fit.dispersion, 1.0 + 1e-6)
    # negative binomial with mean mu and variance phi * mu
    r = mu / (phi - 1)
    sims = rng.negative_binomial(np.maximum(r, 1e-8), 1 / phi)

    lo, hi = alpha / 2, 1 - alpha / 2
    expected = np.exp(X_post.to_numpy() @ fit.params.to_numpy())
    out = pd.DataFrame(
        {
            "observed": post.to_numpy(),
            "expected": expected,
            "lower": np.quantile(sims, lo, axis=1),
            "upper": np.quantile(sims, hi, axis=1),
        },
        index=post.index,
    )
    # interval for the 7-day mean, matching how the chart smooths observed data
    sims_7d = pd.DataFrame(sims).rolling(7, min_periods=1).mean().to_numpy()
    out["lower_7d"] = np.quantile(sims_7d, lo, axis=1)
    out["upper_7d"] = np.quantile(sims_7d, hi, axis=1)
    gaps = sims.sum(axis=0) - post.sum()
    summary = {
        "days": len(post),
        "observed_total": float(post.sum()),
        "expected_total": float(expected.sum()),
        "missing_transits": float(expected.sum() - post.sum()),
        "missing_ci": tuple(float(v) for v in np.quantile(gaps, [lo, hi])),
        "observed_vs_expected": float(post.sum() / expected.sum() - 1),
        "pre_fit_dispersion": float(fit.dispersion),
        "interval_level": 1 - alpha,
    }
    return out, summary
