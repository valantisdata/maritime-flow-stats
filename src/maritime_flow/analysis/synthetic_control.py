"""Synthetic control (Abadie & Gardeazabal 2003; Abadie, Diamond & Hainmueller 2010).

Question: what would traffic at the treated unit (a chokepoint or a port) have
looked like without the event? The before/after and ITS methods answer that
from the unit's own history. Synthetic control answers it from *other* units
that were not affected: it finds non-negative weights, summing to one, so that
a weighted average of those donors tracks the treated unit closely before the
event. After the event, the gap between the treated unit and its synthetic
twin is the estimated effect.

Inference is by permutation (placebo-in-space): pretend each donor was the
treated one, rerun, and see where the real unit's post/pre RMSPE ratio ranks.
With J donors the smallest attainable p-value is 1 / (J + 1), so a small
donor pool cannot produce a "significant" result. That is a feature, not a bug.

Two robustness checks are included:
* in-time placebo: a fake event date before the real one should show no effect
* leave-one-out: dropping any single donor with real weight should not flip the result

The key assumption is that donors were *not* affected by the event. It cannot
be tested from the data and has to be argued (e.g. the Cape of Good Hope is a
bad donor for Suez during the Red Sea crisis: ships moved there).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize


def to_wide(panel: pd.DataFrame, column: str, names: list[str] | None = None, freq: str = "W-MON") -> pd.DataFrame:
    """Weekly means, one column per unit (portname), only weeks where every unit has data."""
    df = panel if names is None else panel[panel["portname"].isin(names)]
    wide = df.pivot_table(index="date", columns="portname", values=column, aggfunc="mean")
    weekly = wide.resample(freq, label="left", closed="left").mean()
    return weekly.dropna(how="any")


def fit_weights(y1: np.ndarray, Y0: np.ndarray) -> np.ndarray:
    """argmin ||y1 - Y0 w||^2  subject to  w >= 0, sum(w) = 1 (SLSQP)."""
    J = Y0.shape[1]
    if J == 1:
        return np.ones(1)
    scale = np.mean(y1**2) or 1.0
    res = optimize.minimize(
        lambda w: np.sum((y1 - Y0 @ w) ** 2) / scale,
        x0=np.full(J, 1 / J),
        jac=lambda w: -2 * Y0.T @ (y1 - Y0 @ w) / scale,
        bounds=[(0.0, 1.0)] * J,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(J)}],
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0, None)
    return w / w.sum()


@dataclass
class SCFit:
    treated: str
    weights: pd.Series
    path: pd.DataFrame  # treated, synthetic, gap (index points)
    pre_rmspe: float
    post_rmspe: float
    effect_pct: float

    @property
    def ratio(self) -> float:
        return self.post_rmspe / self.pre_rmspe if self.pre_rmspe > 0 else np.inf


@dataclass
class SCResult:
    fit: SCFit
    event: pd.Timestamp
    placebos: pd.DataFrame = field(default_factory=pd.DataFrame)
    placebo_paths: pd.DataFrame = field(default_factory=pd.DataFrame)
    p_value: float = np.nan
    in_time_placebo: dict = field(default_factory=dict)
    leave_one_out: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> dict:
        top = self.fit.weights[self.fit.weights > 0.01].sort_values(ascending=False)
        return {
            "treated": self.fit.treated,
            "event": str(self.event.date()),
            "effect_pct": self.fit.effect_pct,
            "pre_rmspe": self.fit.pre_rmspe,
            "post_rmspe": self.fit.post_rmspe,
            "rmspe_ratio": self.fit.ratio,
            "p_value": self.p_value,
            "n_donors": int(len(self.fit.weights)),
            "min_attainable_p": 1 / (len(self.fit.weights) + 1),
            "top_weights": {k: round(float(v), 3) for k, v in top.items()},
            "in_time_placebo": self.in_time_placebo,
        }


def _index(wide: pd.DataFrame, pre: np.ndarray) -> pd.DataFrame:
    """Rescale every unit so its pre-event mean is 100: units of very different size become comparable."""
    return wide / wide.loc[pre].mean() * 100


def _fit_one(idx: pd.DataFrame, treated: str, donors: list[str], pre: np.ndarray) -> SCFit:
    y1 = idx[treated].to_numpy()
    Y0 = idx[donors].to_numpy()
    w = fit_weights(y1[pre], Y0[pre])
    synth = Y0 @ w
    gap = y1 - synth
    post = ~pre
    path = pd.DataFrame({"treated": y1, "synthetic": synth, "gap": gap}, index=idx.index)
    return SCFit(
        treated=treated,
        weights=pd.Series(w, index=donors).sort_values(ascending=False),
        path=path,
        pre_rmspe=float(np.sqrt(np.mean(gap[pre] ** 2))),
        post_rmspe=float(np.sqrt(np.mean(gap[post] ** 2))) if post.any() else np.nan,
        effect_pct=float(gap[post].mean() / synth[post].mean()) if post.any() else np.nan,
    )


def synthetic_control(
    wide: pd.DataFrame,
    treated: str,
    event: str | pd.Timestamp,
    donors: list[str] | None = None,
    min_pre_weeks: int = 52,
    placebo: bool = True,
    in_time_offset_weeks: int = 52,
) -> SCResult:
    event = pd.Timestamp(event)
    donors = [c for c in (donors or wide.columns) if c != treated]
    if not donors:
        raise ValueError("synthetic control needs at least one donor unit")
    wide = wide[[treated, *donors]].dropna()
    pre = np.asarray(wide.index < event)
    if pre.sum() < min_pre_weeks or (~pre).sum() < 4:
        raise ValueError(f"need >= {min_pre_weeks} pre-event weeks and >= 4 post-event weeks")
    idx = _index(wide, pre)
    fit = _fit_one(idx, treated, donors, pre)
    res = SCResult(fit=fit, event=event)

    if placebo and len(donors) >= 2:
        rows, paths = [], {treated: fit.path["gap"]}
        for d in donors:
            others = [o for o in donors if o != d]
            pf = _fit_one(idx, d, others, pre)
            rows.append({"unit": d, "pre_rmspe": pf.pre_rmspe, "post_rmspe": pf.post_rmspe,
                         "ratio": pf.ratio, "effect_pct": pf.effect_pct})
            paths[d] = pf.path["gap"]
        rows.append({"unit": treated, "pre_rmspe": fit.pre_rmspe, "post_rmspe": fit.post_rmspe,
                     "ratio": fit.ratio, "effect_pct": fit.effect_pct})
        plac = pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)
        plac["rank"] = np.arange(1, len(plac) + 1)
        res.placebos = plac
        res.placebo_paths = pd.DataFrame(paths)
        res.p_value = float((plac["ratio"] >= fit.ratio).sum() / len(plac))

    # in-time placebo: only pre-event data, fake event a year earlier
    fake = event - pd.Timedelta(weeks=in_time_offset_weeks)
    early = wide[wide.index < event]
    early_pre = np.asarray(early.index < fake)
    if early_pre.sum() >= min_pre_weeks // 2 and (~early_pre).sum() >= 4:
        f2 = _fit_one(_index(early, early_pre), treated, donors, early_pre)
        res.in_time_placebo = {"fake_event": str(fake.date()), "effect_pct": f2.effect_pct}

    loo = []
    for d in fit.weights[fit.weights > 0.01].index:
        others = [o for o in donors if o != d]
        if others:
            loo.append({"dropped": d, "effect_pct": _fit_one(idx, treated, others, pre).effect_pct})
    res.leave_one_out = pd.DataFrame(loo)
    return res
