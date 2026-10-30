import numpy as np
import pandas as pd
import pytest

from maritime_flow.analysis import synthetic_control as sc
from maritime_flow.data.cleaning import clean_daily
from maritime_flow.data.synthetic import DEMO_CHOKEPOINTS, DEMO_EVENT, DEMO_PORTS, demo_dataset, demo_ports

DONORS = [d[1] for d in DEMO_CHOKEPOINTS[2:]]


@pytest.fixture(scope="module")
def wide():
    panel, _ = clean_daily(demo_dataset())
    return sc.to_wide(panel, "n_total")


def test_weights_are_a_convex_combination():
    rng = np.random.default_rng(0)
    Y0 = rng.normal(100, 10, size=(80, 5))
    true_w = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    y1 = Y0 @ true_w
    w = sc.fit_weights(y1, Y0)
    assert w.min() >= 0 and w.sum() == pytest.approx(1)
    np.testing.assert_allclose(w, true_w, atol=1e-3)


def test_recovers_known_effect(wide):
    res = sc.synthetic_control(wide, "Demo Canal", DEMO_EVENT, donors=DONORS)
    assert res.fit.effect_pct == pytest.approx(-0.55, abs=0.04)
    # the treated unit has the most extreme post/pre ratio of all placebo runs
    assert res.placebos.iloc[0]["unit"] == "Demo Canal"
    assert res.p_value == pytest.approx(1 / (len(DONORS) + 1))


def test_in_time_placebo_is_near_zero(wide):
    res = sc.synthetic_control(wide, "Demo Canal", DEMO_EVENT, donors=DONORS)
    assert abs(res.in_time_placebo["effect_pct"]) < 0.05


def test_leave_one_out_is_stable(wide):
    res = sc.synthetic_control(wide, "Demo Canal", DEMO_EVENT, donors=DONORS)
    assert res.leave_one_out["effect_pct"].between(-0.62, -0.48).all()


def test_untreated_unit_shows_no_effect(wide):
    donors = [d for d in DONORS if d != "Demo Strait B"]
    res = sc.synthetic_control(wide, "Demo Strait B", DEMO_EVENT, donors=donors)
    assert abs(res.fit.effect_pct) < 0.05
    assert res.p_value > 0.2


def test_contaminated_donor_biases_estimate(wide):
    """Using the route traffic moved to as a donor inflates the effect: the reason donors must be unaffected."""
    clean = sc.synthetic_control(wide, "Demo Canal", DEMO_EVENT, donors=DONORS, placebo=False)
    dirty = sc.synthetic_control(wide, "Demo Canal", DEMO_EVENT, donors=DONORS + ["Demo Cape"], placebo=False)
    assert dirty.fit.effect_pct <= clean.fit.effect_pct


def test_port_effect():
    panel, _ = clean_daily(demo_ports())
    w = sc.to_wide(panel, "portcalls")
    res = sc.synthetic_control(w, "Demo Piraeus", DEMO_EVENT, donors=[d[1] for d in DEMO_PORTS[1:]])
    assert res.fit.effect_pct == pytest.approx(-0.20, abs=0.04)


def test_needs_enough_history(wide):
    with pytest.raises(ValueError, match="pre-event"):
        sc.synthetic_control(wide, "Demo Canal", "2019-03-01", donors=DONORS)
    with pytest.raises(ValueError, match="donor"):
        sc.synthetic_control(wide[["Demo Canal"]], "Demo Canal", DEMO_EVENT)


def test_to_wide_keeps_complete_weeks_only():
    idx = pd.date_range("2024-01-01", periods=28, freq="D")
    panel = pd.concat([
        pd.DataFrame({"date": idx, "portname": "a", "v": 1.0}),
        pd.DataFrame({"date": idx[7:], "portname": "b", "v": 2.0}),
    ])
    w = sc.to_wide(panel, "v")
    assert len(w) == 3 and list(w.columns) == ["a", "b"]
