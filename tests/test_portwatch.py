import pandas as pd
import pytest
import requests

from maritime_flow.data.portwatch import PortWatchClient, PortWatchError, resolve_names, where_in


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload or {}
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Serves ``n_rows`` records in pages, optionally failing first."""

    def __init__(self, n_rows=2500, fail_first=0, page=1000):
        self.n_rows, self.fail_first, self.page = n_rows, fail_first, page
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        if self.fail_first:
            self.fail_first -= 1
            return FakeResponse(status=503)
        off = params["resultOffset"]
        rows = [{"attributes": {"ObjectId": i, "portid": "chokepoint1"}}
                for i in range(off, min(off + self.page, self.n_rows))]
        more = off + self.page < self.n_rows
        return FakeResponse({"features": rows, "exceededTransferLimit": more})


def client(session, **kw):
    return PortWatchClient(session=session, cache_dir=None, sleep=lambda s: None, **kw)


def test_pagination_collects_all_rows():
    s = FakeSession(n_rows=2500)
    df = client(s).query("chokepoints_daily")
    assert len(df) == 2500
    assert [c["resultOffset"] for c in s.calls] == [0, 1000, 2000]
    assert df["ObjectId"].is_unique


def test_exact_multiple_of_page_size_terminates():
    s = FakeSession(n_rows=2000)
    assert len(client(s).query("chokepoints_daily")) == 2000


def test_retries_then_succeeds():
    s = FakeSession(n_rows=10, fail_first=2)
    assert len(client(s).query("chokepoints_daily")) == 10
    assert len(s.calls) == 3


def test_gives_up_after_max_retries():
    s = FakeSession(n_rows=10, fail_first=10)
    with pytest.raises(PortWatchError, match="giving up"):
        client(s, max_retries=2).query("chokepoints_daily")


def test_arcgis_error_payload_raises():
    class ErrSession(FakeSession):
        def get(self, url, params, timeout):
            return FakeResponse({"error": {"code": 400, "message": "Invalid query"}})

    with pytest.raises(PortWatchError, match="Invalid query"):
        client(ErrSession()).query("chokepoints_daily")


def test_cache_roundtrip(tmp_path):
    s = FakeSession(n_rows=5)
    c = PortWatchClient(session=s, cache_dir=tmp_path, sleep=lambda _: None)
    c.query("chokepoints_daily")
    c.query("chokepoints_daily")
    assert len(s.calls) == 1


def test_offline_without_cache_fails(tmp_path):
    c = PortWatchClient(session=FakeSession(), cache_dir=tmp_path, offline=True)
    with pytest.raises(PortWatchError, match="offline"):
        c.query("chokepoints_daily")


def test_where_in_escapes_quotes():
    assert where_in("portname", ["Bab el-Mandeb", "O'Hare"]) == "portname IN ('Bab el-Mandeb', 'O''Hare')"


def test_resolve_names():
    cat = pd.DataFrame({"portid": ["chokepoint1", "chokepoint7", "chokepoint8"],
                        "portname": ["Suez Canal", "Cape of Good Hope", "Strait of Good Luck"]})
    assert resolve_names(cat, ["suez"]) == {"Suez Canal": "chokepoint1"}
    assert resolve_names(cat, ["cape of good hope"]) == {"Cape of Good Hope": "chokepoint7"}
    with pytest.raises(KeyError, match="ambiguous"):
        resolve_names(cat, ["good"])
    with pytest.raises(KeyError, match="no match"):
        resolve_names(cat, ["panama"])
