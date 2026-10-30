"""Do ships that avoid one route show up on another?

Classic example: when the Suez / Bab el-Mandeb route becomes risky, some
traffic goes around the Cape of Good Hope instead. We look at (a) the share of
the alternative route in the combined weekly traffic and (b) correlations of
weekly *changes*. Correlating levels of two trending series produces spurious
correlation, so differences are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def weekly_pair(a: pd.Series, b: pd.Series, freq: str = "W-MON") -> pd.DataFrame:
    df = pd.concat({"a": a, "b": b}, axis=1)
    df = df.resample(freq, label="left", closed="left").sum(min_count=6).dropna()
    df["share_b"] = df["b"] / (df["a"] + df["b"])
    return df


def change_correlation(a: pd.Series, b: pd.Series, freq: str = "W-MON") -> dict:
    w = weekly_pair(a, b, freq)
    da, db = w["a"].diff().dropna(), w["b"].diff().dropna()
    pr = stats.pearsonr(da, db)
    sr = stats.spearmanr(da, db)
    return {
        "n_weeks": len(da),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def cross_correlation(a: pd.Series, b: pd.Series, max_lag: int = 8, freq: str = "W-MON") -> pd.DataFrame:
    """corr(diff a_t, diff b_{t+lag}); positive lag = b responds after a."""
    w = weekly_pair(a, b, freq)
    da = w["a"].diff().dropna().to_numpy()
    db = w["b"].diff().dropna().to_numpy()
    da = (da - da.mean()) / da.std()
    db = (db - db.mean()) / db.std()
    n = len(da)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = np.mean(da[: n - lag] * db[lag:])
        else:
            r = np.mean(da[-lag:] * db[: n + lag])
        rows.append((lag, r))
    out = pd.DataFrame(rows, columns=["lag_weeks", "ccf"])
    out["band"] = 1.96 / np.sqrt(n)
    return out


def share_shift(a: pd.Series, b: pd.Series, event: str | pd.Timestamp, freq: str = "W-MON") -> dict:
    """Mean share of route b before vs after the event, Welch t-test on weekly shares."""
    w = weekly_pair(a, b, freq)
    event = pd.Timestamp(event)
    pre, post = w.loc[w.index < event, "share_b"], w.loc[w.index >= event, "share_b"]
    t = stats.ttest_ind(post, pre, equal_var=False)
    return {
        "share_pre": float(pre.mean()),
        "share_post": float(post.mean()),
        "change_pp": float(100 * (post.mean() - pre.mean())),
        "welch_t": float(t.statistic),
        "p_value": float(t.pvalue),
        "weeks_pre": len(pre),
        "weeks_post": len(post),
    }
