"""SQL storage: one schema, two engines (SQLite by default, PostgreSQL optional).

Why a database at all? The API is slow-ish and rate-sensitive, while the
analysis is re-run often. Loading once into a table keeps a local history,
makes the data queryable with plain SQL (window functions, joins) and lets the
dashboard start instantly.

Design:
* a long ("tidy") table ``observations(portid, obs_date, metric, value)`` holds
  both chokepoint and port metrics, so new metrics never need a schema change;
* calendar keys (week, month, year) are computed once at load time, which keeps
  the SQL identical on SQLite and PostgreSQL (their date functions differ);
* loads are idempotent upserts (``ON CONFLICT ... DO UPDATE``), so re-running a
  load after PortWatch revises recent days simply overwrites those rows;
* missing days are not inserted, consistent with the cleaning step.

Queries live in ``maritime_flow/sql/*.sql`` with ``:name`` parameters and are
translated to the driver's parameter style at run time.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import date, datetime, timezone
from importlib import resources
from pathlib import Path

import pandas as pd

DEFAULT_URL = os.environ.get("MARITIME_FLOW_DB_URL", "sqlite:///data/maritime.db")
NON_METRIC = {"date", "portid", "portname", "country", "ISO3", "continent", "fullname", "LOCODE",
              "countrynoaccents", "is_missing"}
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS locations (
        portid    TEXT PRIMARY KEY,
        portname  TEXT NOT NULL,
        kind      TEXT NOT NULL CHECK (kind IN ('chokepoint', 'port')),
        country   TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS observations (
        portid      TEXT NOT NULL REFERENCES locations (portid),
        obs_date    DATE NOT NULL,
        week_start  DATE NOT NULL,
        month_start DATE NOT NULL,
        obs_year    INTEGER NOT NULL,
        metric      TEXT NOT NULL,
        value       DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (portid, obs_date, metric)
    )""",
    "CREATE INDEX IF NOT EXISTS ix_obs_metric_week ON observations (metric, week_start)",
    "CREATE INDEX IF NOT EXISTS ix_obs_metric_date ON observations (metric, obs_date)",
    """CREATE TABLE IF NOT EXISTS load_log (
        loaded_at TEXT NOT NULL,
        kind      TEXT NOT NULL,
        n_rows    INTEGER NOT NULL,
        source    TEXT NOT NULL
    )""",
]

UPSERT_LOCATION = """
INSERT INTO locations (portid, portname, kind, country)
VALUES (:portid, :portname, :kind, :country)
ON CONFLICT (portid) DO UPDATE SET portname = excluded.portname, kind = excluded.kind, country = excluded.country
"""
UPSERT_OBSERVATION = """
INSERT INTO observations (portid, obs_date, week_start, month_start, obs_year, metric, value)
VALUES (:portid, :obs_date, :week_start, :month_start, :obs_year, :metric, :value)
ON CONFLICT (portid, obs_date, metric) DO UPDATE SET value = excluded.value
"""


def available_queries() -> dict[str, str]:
    """Named queries shipped with the package: name -> first comment line."""
    out = {}
    for f in sorted(resources.files("maritime_flow.sql").iterdir(), key=lambda p: p.name):
        if f.name.endswith(".sql"):
            first = f.read_text(encoding="utf-8").splitlines()[0]
            out[f.name.removesuffix(".sql")] = first.lstrip("- ").strip()
    return out


def load_query(name: str) -> str:
    path = resources.files("maritime_flow.sql") / f"{name}.sql"
    if not path.is_file():
        raise KeyError(f"unknown query {name!r}; available: {sorted(available_queries())}")
    return path.read_text(encoding="utf-8")


class Database:
    """Minimal DB-API wrapper. ``url``: ``sqlite:///path.db``, ``sqlite://`` (memory) or ``postgresql://...``."""

    def __init__(self, url: str = DEFAULT_URL):
        self.url = url
        if url.startswith("sqlite://"):
            self.backend = "sqlite"
            path = url.removeprefix("sqlite://").removeprefix("/") or ":memory:"
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path)
            self.conn.execute("PRAGMA foreign_keys = ON")
        elif url.startswith(("postgresql://", "postgres://")):
            try:
                import psycopg
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("PostgreSQL support needs: pip install 'maritime-flow-stats[db]'") from exc
            self.backend = "postgresql"
            self.conn = psycopg.connect(url)
        else:
            raise ValueError(f"unsupported database url {url!r}")

    # ------------------------------------------------------------ plumbing
    def _sql(self, sql: str) -> str:
        if self.backend == "sqlite":
            return sql
        sql = sql.replace("%", "%%")
        return re.sub(r"(?<![:\w]):([A-Za-z_]\w*)", r"%(\1)s", sql)

    def _params(self, params: dict) -> dict:
        if self.backend == "sqlite":
            return {k: (v.isoformat() if isinstance(v, date) else v) for k, v in params.items()}
        return {k: (date.fromisoformat(v) if isinstance(v, str) and ISO_DATE.match(v) else v)
                for k, v in params.items()}

    def execute(self, sql: str, params: dict | None = None) -> None:
        cur = self.conn.cursor()
        cur.execute(self._sql(sql), self._params(params or {}))
        self.conn.commit()

    def executemany(self, sql: str, rows: list[dict]) -> None:
        cur = self.conn.cursor()
        cur.executemany(self._sql(sql), [self._params(r) for r in rows])
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------- schema
    def init_schema(self) -> None:
        for stmt in SCHEMA:
            self.execute(stmt)

    # ---------------------------------------------------------------- load
    def load_panel(self, panel: pd.DataFrame, kind: str, source: str = "IMF PortWatch") -> int:
        """Upsert a cleaned panel (output of ``clean_daily``). Returns rows written."""
        self.init_schema()
        locs = panel.groupby("portid").agg(portname=("portname", "last")).reset_index()
        locs["country"] = panel.groupby("portid")["country"].last().to_numpy() if "country" in panel else None
        locs["kind"] = kind
        self.executemany(UPSERT_LOCATION, locs[["portid", "portname", "kind", "country"]]
                         .astype(object).where(locs.notna(), None).to_dict("records"))

        metrics = [c for c in panel.columns if c not in NON_METRIC and pd.api.types.is_numeric_dtype(panel[c])]
        long = panel.melt(id_vars=["portid", "date"], value_vars=metrics, var_name="metric").dropna(subset=["value"])
        d = pd.to_datetime(long["date"])
        long = long.assign(
            obs_date=d.dt.strftime("%Y-%m-%d"),
            week_start=(d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d"),
            month_start=d.dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d"),
            obs_year=d.dt.year.astype(int),
            value=long["value"].astype(float),
        )
        rows = long[["portid", "obs_date", "week_start", "month_start", "obs_year", "metric", "value"]]
        records = rows.to_dict("records")
        for i in range(0, len(records), 20_000):
            self.executemany(UPSERT_OBSERVATION, records[i : i + 20_000])
        self.execute(
            "INSERT INTO load_log (loaded_at, kind, n_rows, source) VALUES (:t, :k, :n, :s)",
            {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "k": kind, "n": len(records), "s": source},
        )
        return len(records)

    # --------------------------------------------------------------- read
    def query(self, sql_or_name: str, params: dict | None = None) -> pd.DataFrame:
        sql = sql_or_name if re.search(r"\s", sql_or_name.strip()) else load_query(sql_or_name)
        if not re.match(r"^\s*(--[^\n]*\n\s*)*(select|with)\b", sql, re.I):
            raise ValueError("only SELECT / WITH queries are allowed here")
        used = set(re.findall(r"(?<![:\w]):([A-Za-z_]\w*)", sql))
        params = {k: v for k, v in (params or {}).items() if k in used}
        missing = used - set(params)
        if missing:
            raise ValueError(f"query needs parameters: {sorted(missing)}")
        cur = self.conn.cursor()
        cur.execute(self._sql(sql), self._params(params))
        cols = [c[0] for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)

    def locations(self) -> pd.DataFrame:
        return self.query("SELECT portid, portname, kind, country FROM locations ORDER BY kind, portname")

    def read_panel(self, names: list[str] | None = None, kind: str | None = None) -> pd.DataFrame:
        """Rebuild a wide daily panel (same layout as ``clean_daily``) from the database."""
        sql = """
            SELECT o.portid, l.portname, l.country, o.obs_date, o.metric, o.value
            FROM observations o JOIN locations l ON l.portid = o.portid
            WHERE (:kind = '' OR l.kind = :kind)
        """
        long = self.query(sql, {"kind": kind or ""})
        if names:
            long = long[long["portname"].isin(names) | long["portid"].isin(names)]
        if long.empty:
            raise KeyError("no matching rows in the database; run 'maritime-flow db load' first")
        wide = long.pivot_table(index=["portid", "portname", "obs_date"], columns="metric",
                                values="value", aggfunc="last").reset_index()
        wide.columns.name = None
        wide = wide.rename(columns={"obs_date": "date"})
        wide["date"] = pd.to_datetime(wide["date"])
        countries = long.groupby("portid")["country"].last()
        frames = []
        for pid, g in wide.groupby("portid"):
            full = pd.date_range(g["date"].min(), g["date"].max(), freq="D", name="date")
            g = g.set_index("date").reindex(full)
            g["portid"], g["portname"] = pid, g["portname"].dropna().iloc[0]
            g["country"] = countries.get(pid)
            frames.append(g.reset_index())
        return pd.concat(frames, ignore_index=True)
