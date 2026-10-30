import json

import pandas as pd
import pytest

from maritime_flow.app import data as D
from maritime_flow.data.synthetic import DEMO_EVENT


@pytest.fixture(scope="module")
def panel():
    return D.load_demo("chokepoint")


def test_metric_options(panel):
    opts = D.metric_options(panel)
    assert "n_total" in opts and "is_missing" not in opts


def test_line_spec_is_valid_json(panel):
    frame = D.overview_frame(panel, ["Demo Canal", "Demo Cape"], "n_total")
    spec = D.line_spec(frame, "date", "value", color="location", event=DEMO_EVENT, band=None)
    text = json.dumps(spec)  # must be serialisable for st.vega_lite_chart
    assert "Demo Cape" in text and spec["layer"][-1]["mark"]["type"] == "rule"


def test_impact_view(panel):
    y = D.series(panel, "Demo Canal", "n_total")
    view = D.run_impact(y, DEMO_EVENT)
    assert view.its["rate_ratio"] == pytest.approx(0.45, abs=0.06)
    assert {"observed", "expected without event"} == set(view.counterfactual["series"])
    json.dumps(D.line_spec(view.counterfactual, "date", "value", color="series", band=("lower", "upper")))


def test_synthetic_view(panel):
    donors = [n for n in panel["portname"].unique() if n.startswith("Demo Strait")]
    res, long, weights = D.run_synthetic(panel, "n_total", "Demo Canal", donors, DEMO_EVENT)
    assert set(long["series"]) == {"Demo Canal", "synthetic control"}
    assert weights["weight"].sum() == pytest.approx(1, abs=0.02)
    json.dumps(D.bar_spec(weights, "weight", "donor"))


def test_forecast_view(panel):
    y = D.series(panel, "Demo Strait A", "n_total")
    bt, fc, frame = D.run_forecast(y, horizon=14, n_origins=4)
    assert len(fc) == 14 and set(frame["series"]) == {"observed", "forecast"}


def test_port_demo_defaults():
    ports = D.load_demo("port")
    assert D.DEFAULT_COLUMN["port"] in D.metric_options(ports)
    assert ports["country"].notna().all()
    assert isinstance(D.series(ports, "Demo Piraeus", "portcalls"), pd.Series)
