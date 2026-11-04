"""Thin client for the public IMF PortWatch ArcGIS feature services.

The services return at most ~1000 records per request, so every query is paged
with ``resultOffset``. Responses are cached on disk to avoid hammering a free
public service while iterating on the analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from maritime_flow import __version__
from maritime_flow.config import DEFAULT_CACHE_DIR, LAYERS

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
# ArcGIS Online sometimes answers a valid request with an error inside a 200
# response while a hosted layer is being republished. After the IMF
# republished the daily layers in late October 2026 most requests failed this
# way for days, each time with a different message ("Invalid URL", "Object
# reference not set...", "Invalid query parameters"), so every error payload is
# retried, many times with short waits. The queries this client sends are fixed
# and tested, so a genuinely bad query only costs a few retries before failing.
MAX_WAIT_SECONDS = 8.0


class PortWatchError(RuntimeError):
    """Raised when the service returns an error or cannot be reached."""


class _Retryable(Exception):
    pass


def _quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def where_in(field_name: str, values: Iterable[str]) -> str:
    values = list(values)
    if not values:
        return "1=1"
    return f"{field_name} IN ({', '.join(_quote(v) for v in values)})"


@dataclass
class PortWatchClient:
    session: requests.Session | None = None
    page_size: int = 1000
    timeout: float = 30.0
    max_retries: int = 30
    backoff: float = 2.0
    cache_dir: Path | None = DEFAULT_CACHE_DIR
    cache_ttl_hours: float = 24.0
    offline: bool = False
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
            self.session.headers["User-Agent"] = (
                f"maritime-flow-stats/{__version__} (non-commercial research)"
            )
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)

    # ------------------------------------------------------------------ http
    def _get(self, url: str, params: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in RETRY_STATUS:
                    raise _Retryable(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                payload = resp.json()
                if "error" in payload:
                    err = payload["error"]
                    raise _Retryable(f"ArcGIS error {err.get('code')}: {err.get('message')}")
            except (requests.ConnectionError, requests.Timeout, _Retryable) as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                wait = min(self.backoff**attempt, MAX_WAIT_SECONDS)
                log.warning("request failed (%s), retrying in %.1fs", exc, wait)
                self.sleep(wait)
                continue
            except requests.HTTPError as exc:
                raise PortWatchError(str(exc)) from exc

            if "error" in payload:
                err = payload["error"]
                raise PortWatchError(f"ArcGIS error {err.get('code')}: {err.get('message')}")
            return payload
        raise PortWatchError(
            f"giving up after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    # ----------------------------------------------------------------- cache
    def _cache_path(self, url: str, params: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key[:24]}.csv.gz"

    def _cache_read(self, path: Path | None) -> pd.DataFrame | None:
        if path is None or not path.exists():
            return None
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h > self.cache_ttl_hours and not self.offline:
            return None
        log.info("cache hit %s (%.1fh old)", path.name, age_h)
        return pd.read_csv(path)

    def _cache_write(self, path: Path | None, df: pd.DataFrame) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    # ----------------------------------------------------------------- query
    def query(
        self,
        layer: str,
        where: str = "1=1",
        out_fields: str = "*",
        order_by: str = "ObjectId",
    ) -> pd.DataFrame:
        """Run a paged attribute query against a layer and return all rows."""
        url = LAYERS.get(layer, layer)
        base = {
            "where": where,
            "outFields": out_fields,
            "orderByFields": order_by,
            "returnGeometry": "false",
            "resultRecordCount": self.page_size,
            "f": "json",
        }
        path = self._cache_path(url, base)
        cached = self._cache_read(path)
        if cached is not None:
            return cached
        if self.offline:
            raise PortWatchError(f"offline mode and no cached copy of {layer} ({where})")

        rows: list[dict] = []
        offset = 0
        while True:
            payload = self._get(url, {**base, "resultOffset": offset})
            features = payload.get("features", [])
            rows.extend(f["attributes"] for f in features)
            more = payload.get("exceededTransferLimit") or len(features) == self.page_size
            if not features or not more:
                break
            offset += len(features)
        log.info("fetched %d rows from %s", len(rows), layer)

        df = pd.DataFrame(rows)
        self._cache_write(path, df)
        return df

    # ----------------------------------------------------------- convenience
    def chokepoint_catalog(self) -> pd.DataFrame:
        return self.query("chokepoints_meta")

    def port_catalog(self) -> pd.DataFrame:
        return self.query("ports_meta")

    def chokepoint_daily(self, portids: Iterable[str]) -> pd.DataFrame:
        return self.query("chokepoints_daily", where=where_in("portid", portids))

    def port_daily(self, portids: Iterable[str]) -> pd.DataFrame:
        return self.query("ports_daily", where=where_in("portid", portids))


def resolve_names(
    catalog: pd.DataFrame,
    queries: Iterable[str],
    name_col: str = "portname",
    id_col: str = "portid",
) -> dict[str, str]:
    """Map loose user input ("suez", "Hormuz") to PortWatch ids.

    Exact (case-insensitive) matches win; otherwise a unique substring match is
    accepted. Ambiguous or unknown names raise with the candidate list so the
    user can be more specific.
    """
    names = catalog[[name_col, id_col]].drop_duplicates()
    lowered = names[name_col].str.casefold()
    out: dict[str, str] = {}
    for q in queries:
        qn = q.strip().casefold()
        exact = names[lowered == qn]
        hits = exact if len(exact) else names[lowered.str.contains(qn, regex=False)]
        if len(hits) == 1:
            row = hits.iloc[0]
            out[row[name_col]] = row[id_col]
        elif len(hits) == 0:
            raise KeyError(f"no match for {q!r}; known: {sorted(names[name_col])}")
        else:
            raise KeyError(f"{q!r} is ambiguous: {sorted(hits[name_col])}")
    return out
