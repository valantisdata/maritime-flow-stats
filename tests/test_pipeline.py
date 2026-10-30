import json

import pytest

from maritime_flow.cli import main


@pytest.fixture(scope="module")
def demo_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    assert main(["demo", "--out", str(out)]) == 0
    return out


def test_demo_chokepoint_report(demo_out):
    out = demo_out / "chokepoint"
    html = (out / "report.html").read_text()
    assert "IMF PortWatch" in html  # attribution must always be present
    assert "SYNTHETIC" in html
    assert "synthetic control" in html.lower()
    results = json.loads((out / "results.json").read_text())
    assert 0.35 < results["its"]["rate_ratio"] < 0.6
    assert -0.6 < results["synthetic_control"]["effect_pct"] < -0.5
    for f in ["forecast", "sc_path", "sc_placebos", "route_share"]:
        assert (out / "figures" / f"{f}.png").exists()
    assert (out / "tables" / "segments.csv").exists()


def test_demo_port_report(demo_out):
    out = demo_out / "port"
    results = json.loads((out / "results.json").read_text())
    assert results["kind"] == "port" and results["column"] == "portcalls"
    assert -0.3 < results["synthetic_control"]["effect_pct"] < -0.1
    assert (out / "figures" / "trade.png").exists()


def test_db_cli_roundtrip(tmp_path, capsys):
    url = f"sqlite:///{tmp_path / 'x.db'}"
    assert main(["--db-url", url, "db", "load", "--demo"]) == 0
    assert main(["--db-url", url, "db", "query", "event_ranking", "--metric", "portcalls",
                 "--event", "2023-12-01"]) == 0
    printed = capsys.readouterr().out
    assert "Demo Piraeus" in printed.splitlines()[-8]  # most affected port is listed first
    assert main(["--db-url", url, "analyze", "--from-db", "--kind", "port", "--target", "Demo Piraeus",
                 "--event", "2023-12-01", "--out", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r" / "report.html").exists()


def test_bad_event_date_is_rejected(tmp_path):
    with pytest.raises(Exception):
        main(["analyze", "--target", "x", "--panel", "missing.csv", "--event", "not-a-date",
              "--out", str(tmp_path)])
