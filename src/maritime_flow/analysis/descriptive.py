from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def summary_table(panel: pd.DataFrame, column: str = "n_total", by: str = "portname") -> pd.DataFrame:
    """Distribution summary per chokepoint.

    ``dispersion_index`` is variance / mean. For a Poisson process it is ~1;
    values well above 1 justify the quasi-Poisson models used later.
    """

    def _summ(s: pd.Series) -> pd.Series:
        x = s.dropna()
        mean = x.mean()
        return pd.Series(
            {
                "days": len(s),
                "missing_share": s.isna().mean(),
                "mean": mean,
                "median": x.median(),
                "std": x.std(),
                "cv": x.std() / mean if mean else np.nan,
                "dispersion_index": x.var() / mean if mean else np.nan,
                "p05": x.quantile(0.05),
                "p95": x.quantile(0.95),
            }
        )

    return panel.groupby(by)[column].apply(_summ).unstack()


def weekday_profile(series: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Average weekday pattern on detrended data plus a Kruskal-Wallis test.

    Each day is divided by its centred 28-day rolling mean so that level shifts
    (e.g. a crisis) do not masquerade as weekday effects.
    """
    s = series.dropna()
    ratio = (s / s.rolling(28, center=True, min_periods=14).mean()).dropna()
    groups = [ratio[ratio.index.dayofweek == d].to_numpy() for d in range(7)]
    h, p = stats.kruskal(*groups)
    prof = pd.DataFrame(
        {
            "weekday": WEEKDAYS,
            "relative_level": [g.mean() for g in groups],
            "ci95": [1.96 * g.std(ddof=1) / np.sqrt(len(g)) for g in groups],
            "n": [len(g) for g in groups],
        }
    )
    return prof, {"kruskal_h": float(h), "p_value": float(p)}


def yearly_table(series: pd.Series) -> pd.DataFrame:
    s = series.dropna()
    g = s.groupby(s.index.year)
    out = pd.DataFrame({"mean_daily": g.mean(), "days": g.size()})
    out["yoy_change"] = out["mean_daily"].pct_change()
    out.index.name = "year"
    return out


def vessel_mix(panel: pd.DataFrame, key: str, freq: str = "QE", id_col: str = "portname") -> pd.DataFrame:
    """Share of each vessel type in total transits, by period."""
    cols = ["n_container", "n_dry_bulk", "n_general_cargo", "n_roro", "n_tanker"]
    sub = panel.loc[panel[id_col] == key].set_index("date")[cols]
    agg = sub.resample(freq).sum(min_count=1)
    shares = agg.div(agg.sum(axis=1), axis=0)
    shares.columns = [c.removeprefix("n_") for c in cols]
    return shares.dropna(how="all")


def trade_flows(panel: pd.DataFrame, key: str, freq: str = "W-MON", id_col: str = "portname") -> pd.DataFrame:
    """Weekly import and export volumes (metric tons) for one port."""
    cols = [c for c in ["import", "export", "portcalls"] if c in panel.columns]
    sub = panel.loc[panel[id_col] == key].set_index("date")[cols]
    weekly = sub.resample(freq, label="left", closed="left").sum(min_count=6).dropna()
    if {"import", "export"} <= set(weekly.columns):
        weekly["net_export"] = weekly["export"] - weekly["import"]
    return weekly


def trade_yearly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Annual totals (thousand tonnes) with year-over-year change."""
    cols = [c for c in ["import", "export"] if c in weekly.columns]
    yearly = weekly[cols].groupby(weekly.index.year).sum() / 1000
    for c in cols:
        yearly[f"{c}_yoy"] = yearly[c].pct_change()
    yearly.index.name = "year"
    return yearly
