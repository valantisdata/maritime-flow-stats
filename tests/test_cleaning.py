import numpy as np
import pandas as pd

from maritime_flow.data.cleaning import clean_daily, to_series, trim_incomplete_tail
from maritime_flow.data.synthetic import demo_dataset, demo_ports, make_chokepoint_frame


def test_dates_parsed_from_epoch_ms():
    raw = make_chokepoint_frame("cp", "X", start="2024-01-01", end="2024-01-10")
    panel, rep = clean_daily(raw)
    assert panel["date"].min() == pd.Timestamp("2024-01-01")
    assert len(panel) == 10
    assert rep.n_duplicates == 0


def test_demo_problems_are_reported():
    panel, rep = clean_daily(demo_dataset())
    assert rep.n_duplicates == 2
    assert all(v == 0 for v in rep.identity_mismatches.values())
    assert sum(rep.missing_days.values()) == 12  # rows dropped on purpose by the generator
    assert panel.groupby("portid")["date"].apply(lambda d: d.diff().dropna().eq(pd.Timedelta("1D")).all()).all()


def test_missing_days_are_not_zero_filled():
    raw = make_chokepoint_frame("cp", "X", start="2024-01-01", end="2024-01-31")
    raw = raw[raw["day"] != 15]
    panel, rep = clean_daily(raw)
    assert rep.missing_days["X"] == 1
    assert np.isnan(panel.loc[panel["date"] == "2024-01-15", "n_total"].iloc[0])


def test_consistency_and_negative_checks():
    raw = make_chokepoint_frame("cp", "X", start="2024-01-01", end="2024-01-31")
    raw.loc[3, "n_total"] += 5
    raw.loc[4, "n_tanker"] = -1
    _, rep = clean_daily(raw)
    assert rep.identity_mismatches["n_total != n_cargo + n_tanker"] >= 1
    assert rep.n_negative == 1


def test_to_series_interpolates_only_short_gaps():
    raw = make_chokepoint_frame("cp", "X", start="2024-01-01", end="2024-03-31")
    drop_short = raw["date"].isin(raw["date"].iloc[[10, 11]])
    drop_long = raw["date"].isin(raw["date"].iloc[40:50])
    panel, _ = clean_daily(raw[~drop_short & ~drop_long])
    s = to_series(panel, "X")
    assert s.iloc[10:12].notna().all()
    assert s.iloc[40:50].isna().sum() == 10


def test_trim_incomplete_tail():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    s = pd.Series(50.0, index=idx)
    s.iloc[-2:] = [10.0, 3.0]
    assert trim_incomplete_tail(s).index[-1] == idx[-3]


def test_port_layer_cleaning():
    raw = demo_ports()
    panel, rep = clean_daily(raw)
    assert panel["country"].notna().all()  # text columns survive re-indexing
    assert rep.identity_mismatches["portcalls != portcalls_cargo + portcalls_tanker"] == 0
    assert sum(rep.missing_days.values()) == 12
    s = to_series(panel, "Demo Piraeus", "portcalls")
    assert s.index.freq == "D" and s.mean() > 10
