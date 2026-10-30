import pandas as pd
import pytest

from maritime_flow.data.cleaning import clean_daily
from maritime_flow.data.synthetic import make_chokepoint_frame, make_port_frame
from maritime_flow.storage import Database, available_queries, load_query


@pytest.fixture()
def db():
    cp, _ = clean_daily(pd.concat([
        make_chokepoint_frame("c1", "Alpha", start="2022-01-01", end="2024-12-31", base=50,
                              shift_date="2024-01-01", shift_ratio=0.5, seed=1),
        make_chokepoint_frame("c2", "Beta", start="2022-01-01", end="2024-12-31", base=30, seed=2),
    ]))
    ports, _ = clean_daily(make_port_frame("p1", "Gamma", country="Greece", start="2023-01-01", end="2024-06-30"))
    with Database("sqlite://") as d:
        d.load_panel(cp, "chokepoint")
        d.load_panel(ports, "port")
        yield d


def test_locations_and_kinds(db):
    loc = db.locations()
    assert set(loc["portname"]) == {"Alpha", "Beta", "Gamma"}
    assert loc.set_index("portname").loc["Gamma", "country"] == "Greece"


def test_upsert_is_idempotent(db):
    before = db.query("SELECT COUNT(*) AS n FROM observations")["n"].iloc[0]
    panel = db.read_panel(["Alpha"])
    db.load_panel(panel, "chokepoint")
    after = db.query("SELECT COUNT(*) AS n FROM observations")["n"].iloc[0]
    assert before == after


def test_upsert_overwrites_revised_values(db):
    panel = db.read_panel(["Alpha"])
    panel.loc[panel["date"] == "2024-06-03", "n_total"] = 999
    db.load_panel(panel, "chokepoint")
    v = db.query("SELECT value FROM observations WHERE portid = 'c1' AND obs_date = '2024-06-03' "
                 "AND metric = 'n_total'")["value"].iloc[0]
    assert v == 999


def test_read_panel_roundtrip(db):
    panel = db.read_panel(["Alpha"])
    assert panel["date"].is_monotonic_increasing
    assert panel["n_total"].notna().all()
    assert (panel["n_total"] == panel["n_cargo"] + panel["n_tanker"]).all()


def test_weekly_totals_match_pandas(db):
    sql = db.query("weekly_totals", {"metric": "n_total"})
    a = sql[sql["portname"] == "Alpha"].set_index("week_start")["total"]
    panel = db.read_panel(["Alpha"]).set_index("date")["n_total"]
    pd_weekly = panel.resample("W-MON", label="left", closed="left").agg(["sum", "count"])
    pd_weekly = pd_weekly[pd_weekly["count"] == 7]["sum"]
    pd_weekly.index = pd_weekly.index.strftime("%Y-%m-%d")
    pd.testing.assert_series_equal(a, pd_weekly, check_names=False, check_index_type=False)


def test_rolling_average_window(db):
    r = db.query("rolling_averages", {"metric": "n_total", "portname": "Beta"})
    manual = r["value"].rolling(7, min_periods=1).mean()
    assert r["ma_7d"].to_numpy() == pytest.approx(manual.to_numpy())


def test_event_ranking_finds_shocked_unit(db):
    r = db.query("event_ranking", {"metric": "n_total", "pre_start": "2023-01-01",
                                   "event": "2024-01-01", "post_end": "2024-12-31"})
    top = r.iloc[0]
    assert top["portname"] == "Alpha" and top["pct_change"] == pytest.approx(-0.5, abs=0.05)


def test_monthly_yoy(db):
    r = db.query("monthly_yoy", {"metric": "n_total"})
    jun = r[(r["portname"] == "Alpha") & (r["month_start"] == "2024-06-01")].iloc[0]
    assert jun["month_ly"] == "2023-06-01"
    assert jun["yoy_change"] == pytest.approx(-0.5, abs=0.1)


def test_every_named_query_runs(db):
    params = {"metric": "n_total", "portname": "Alpha", "pre_start": "2023-01-01",
              "event": "2024-01-01", "post_end": "2024-12-31"}
    for name in available_queries():
        assert not db.query(name, params).empty, name


def test_only_select_allowed(db):
    with pytest.raises(ValueError, match="SELECT"):
        db.query("DELETE FROM observations")
    with pytest.raises(ValueError, match="parameters"):
        db.query("weekly_totals", {})
    with pytest.raises(KeyError):
        load_query("nope")


def test_postgres_param_translation():
    d = Database.__new__(Database)
    d.backend = "postgresql"
    assert d._sql("SELECT x::int FROM t WHERE a = :a AND b LIKE '5%'") == \
        "SELECT x::int FROM t WHERE a = %(a)s AND b LIKE '5%%'"
