"""Synthetic data in the PortWatch schemas (chokepoints and ports).

Used for tests, for the offline ``demo`` command, for the dashboard's demo mode
and for CI, so the whole project runs without network access. Counts are
negative binomial (over-dispersed) with weekday and annual seasonality.

Every unit also loads on a shared "world trade" factor with its own exposure.
That common movement is what makes synthetic control meaningful: untreated
units share the treated unit's ups and downs, so a weighted mix of them can
stand in for the treated unit's missing counterfactual.

All shocks are known in advance, so every estimator can be checked against
the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maritime_flow.config import CARGO_TYPES

WEEKDAY_EFFECT = np.array([1.04, 1.03, 1.01, 1.00, 0.99, 0.96, 0.97])
TYPE_SHARES = {"container": 0.30, "dry_bulk": 0.25, "general_cargo": 0.13, "roro": 0.05, "tanker": 0.27}
MEAN_CAPACITY = {"container": 80_000, "dry_bulk": 55_000, "general_cargo": 12_000, "roro": 15_000, "tanker": 60_000}

DEMO_START, DEMO_END, DEMO_EVENT = "2019-01-01", "2026-09-30", "2023-12-01"


def global_factor(dates: pd.DatetimeIndex, seed: int = 99) -> np.ndarray:
    """Smooth multiplicative world-trade factor around 1: slow drift, a 2020 dip, wiggles."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    weekly_steps = rng.normal(0, 0.006, n // 7 + 2).cumsum()
    walk = np.repeat(weekly_steps, 7)[:n]
    walk = pd.Series(walk).rolling(28, min_periods=1, center=True).mean().to_numpy()
    years = np.asarray((dates - dates[0]) / pd.Timedelta(days=365.25), dtype=float)
    covid = -0.10 * np.exp(-0.5 * ((dates - pd.Timestamp("2020-05-01")).days / 35.0) ** 2)
    return np.exp(0.01 * years + walk - walk.mean() + covid)


def simulate_counts(
    dates: pd.DatetimeIndex,
    base: float,
    shift_date: str | None = None,
    shift_ratio: float = 1.0,
    dispersion: float = 25.0,
    seed: int = 0,
    common: np.ndarray | None = None,
    exposure: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    doy = dates.dayofyear.to_numpy()
    annual = 1 + 0.06 * np.sin(2 * np.pi * doy / 365.25) + 0.03 * np.cos(4 * np.pi * doy / 365.25)
    mu = base * annual * WEEKDAY_EFFECT[dates.dayofweek.to_numpy()]
    if common is not None:
        mu = mu * common**exposure
    if shift_date is not None:
        mu = np.where(dates >= pd.Timestamp(shift_date), mu * shift_ratio, mu)
    # negative binomial with mean mu and variance mu + mu^2 / dispersion
    p = dispersion / (dispersion + mu)
    return rng.negative_binomial(dispersion, p)


def _epoch_ms(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((dates - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1), dtype="int64")


def make_chokepoint_frame(
    portid: str,
    portname: str,
    start: str = DEMO_START,
    end: str = DEMO_END,
    base: float = 60.0,
    shift_date: str | None = None,
    shift_ratio: float = 1.0,
    seed: int = 0,
    common: np.ndarray | None = None,
    exposure: float = 1.0,
) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    total = simulate_counts(dates, base, shift_date, shift_ratio, seed=seed, common=common, exposure=exposure)
    rng = np.random.default_rng(seed + 1)
    shares = np.array(list(TYPE_SHARES.values()))
    split = np.vstack([rng.multinomial(n, shares) for n in total])

    df = pd.DataFrame({"date": _epoch_ms(dates), "portid": portid, "portname": portname})
    df["year"], df["month"], df["day"] = dates.year, dates.month, dates.day
    for i, t in enumerate(TYPE_SHARES):
        df[f"n_{t}"] = split[:, i]
        df[f"capacity_{t}"] = split[:, i] * MEAN_CAPACITY[t] * rng.lognormal(0, 0.1, len(dates))
    df["n_cargo"] = df[[f"n_{t}" for t in CARGO_TYPES]].sum(axis=1)
    df["n_total"] = df["n_cargo"] + df["n_tanker"]
    df["capacity_cargo"] = df[[f"capacity_{t}" for t in CARGO_TYPES]].sum(axis=1)
    df["capacity"] = df["capacity_cargo"] + df["capacity_tanker"]
    df["ObjectId"] = np.arange(1, len(df) + 1)
    return df


def make_port_frame(
    portid: str,
    portname: str,
    country: str = "Demo Country",
    start: str = DEMO_START,
    end: str = DEMO_END,
    base_calls: float = 20.0,
    shift_date: str | None = None,
    shift_ratio: float = 1.0,
    seed: int = 0,
    common: np.ndarray | None = None,
    exposure: float = 1.0,
) -> pd.DataFrame:
    """Daily port calls and trade volumes (metric tons) in the Daily_Ports_Data layout."""
    dates = pd.date_range(start, end, freq="D")
    calls = simulate_counts(dates, base_calls, shift_date, shift_ratio, dispersion=40.0,
                            seed=seed, common=common, exposure=exposure)
    rng = np.random.default_rng(seed + 1)
    tanker = rng.binomial(calls, 0.22)
    cargo = calls - tanker
    df = pd.DataFrame({"date": _epoch_ms(dates), "portid": portid, "portname": portname, "country": country})
    df["portcalls_cargo"] = cargo
    df["portcalls_tanker"] = tanker
    df["portcalls"] = calls
    for flow, share in [("import", 0.55), ("export", 0.45)]:
        df[f"{flow}_cargo"] = cargo * share * 9_000 * rng.lognormal(0, 0.25, len(dates))
        df[f"{flow}_tanker"] = tanker * share * 30_000 * rng.lognormal(0, 0.25, len(dates))
        df[flow] = df[f"{flow}_cargo"] + df[f"{flow}_tanker"]
    df["ObjectId"] = np.arange(1, len(df) + 1)
    return df


# (portid, name, base level, exposure to the common factor, shift ratio at the event)
DEMO_CHOKEPOINTS = [
    ("chokepoint1", "Demo Canal", 70, 1.0, 0.45),
    ("chokepoint7", "Demo Cape", 45, 0.9, 1.60),
    ("chokepoint2", "Demo Strait A", 55, 0.7, 1.0),
    ("chokepoint3", "Demo Strait B", 90, 0.9, 1.0),
    ("chokepoint4", "Demo Strait C", 40, 1.2, 1.0),
    ("chokepoint5", "Demo Strait D", 65, 1.4, 1.0),
    ("chokepoint6", "Demo Strait E", 35, 0.8, 1.0),
    ("chokepoint8", "Demo Strait F", 80, 1.1, 1.0),
    ("chokepoint9", "Demo Strait G", 50, 1.3, 1.0),
    ("chokepoint10", "Demo Strait H", 60, 0.6, 1.0),
]
DEMO_PORTS = [
    ("port1", "Demo Piraeus", 22, 1.0, 0.80),
    ("port2", "Demo Port A", 30, 0.8, 1.0),
    ("port3", "Demo Port B", 15, 1.1, 1.0),
    ("port4", "Demo Port C", 25, 1.3, 1.0),
    ("port5", "Demo Port D", 18, 0.9, 1.0),
    ("port6", "Demo Port E", 28, 1.2, 1.0),
    ("port7", "Demo Port F", 12, 0.7, 1.0),
    ("port8", "Demo Port G", 35, 1.0, 1.0),
]


def _corrupt(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Drop 12 interior rows and duplicate 2, so the cleaning step has work to do."""
    rng = np.random.default_rng(seed)
    days = pd.to_datetime(df["date"], unit="ms")
    interior = np.flatnonzero((days > pd.Timestamp(DEMO_START) + pd.Timedelta(days=30))
                              & (days < pd.Timestamp(DEMO_END) - pd.Timedelta(days=30)))
    drop = rng.choice(interior, size=12, replace=False)
    df = df.drop(index=df.index[drop])
    return pd.concat([df, df.iloc[[5, 6]]], ignore_index=True)


def demo_dataset(seed: int = 7) -> pd.DataFrame:
    """Ten chokepoints: one hit by a shock, one alternative route that gains, eight unaffected."""
    common = global_factor(pd.date_range(DEMO_START, DEMO_END, freq="D"))
    frames = [
        make_chokepoint_frame(pid, name, base=base, shift_date=DEMO_EVENT if ratio != 1 else None,
                              shift_ratio=ratio, seed=seed + 10 * i, common=common, exposure=exp)
        for i, (pid, name, base, exp, ratio) in enumerate(DEMO_CHOKEPOINTS)
    ]
    return _corrupt(pd.concat(frames, ignore_index=True), seed)


def demo_ports(seed: int = 11) -> pd.DataFrame:
    """Eight ports: one loses 20% of calls at the event, seven do not."""
    common = global_factor(pd.date_range(DEMO_START, DEMO_END, freq="D"))
    frames = [
        make_port_frame(pid, name, base_calls=base, shift_date=DEMO_EVENT if ratio != 1 else None,
                        shift_ratio=ratio, seed=seed + 10 * i, common=common, exposure=exp)
        for i, (pid, name, base, exp, ratio) in enumerate(DEMO_PORTS)
    ]
    return _corrupt(pd.concat(frames, ignore_index=True), seed)
