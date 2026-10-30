"""Turn raw PortWatch rows into a validated daily panel.

Choices worth knowing about:
* Dates arrive as epoch milliseconds (UTC). They are converted to naive dates.
* Missing days are *not* filled with zeros. A day with no record is unknown,
  not a day with no ships, so it stays NaN and is flagged.
* Accounting identities are checked (n_total = n_cargo + n_tanker,
  n_cargo = sum of cargo sub-types, portcalls = cargo + tanker calls) and
  reported rather than "fixed".
* The same code handles the chokepoint and the port layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from maritime_flow.config import CARGO_TYPES

DROP_COLUMNS = ["year", "month", "day", "ObjectId", "FID"]
TEXT_COLUMNS = ["portname", "country", "ISO3", "continent", "fullname", "LOCODE", "countrynoaccents"]

# accounting identities that hold in each PortWatch layer: total = sum(parts)
IDENTITIES = [
    ("n_total", ["n_cargo", "n_tanker"]),
    ("n_cargo", [f"n_{t}" for t in CARGO_TYPES]),
    ("portcalls", ["portcalls_cargo", "portcalls_tanker"]),
]


@dataclass
class QualityReport:
    n_rows_raw: int
    n_duplicates: int
    n_negative: int
    identity_mismatches: dict[str, int] = field(default_factory=dict)
    missing_days: dict[str, int] = field(default_factory=dict)
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None

    def as_frame(self) -> pd.DataFrame:
        rows = [
            ("raw rows", self.n_rows_raw),
            ("duplicate (id, date) rows dropped", self.n_duplicates),
            ("negative values set to NaN", self.n_negative),
        ]
        rows += [(f"rows where {k}", v) for k, v in self.identity_mismatches.items()]
        rows += [
            ("first date", self.date_min.date() if self.date_min is not None else None),
            ("last date", self.date_max.date() if self.date_max is not None else None),
        ]
        rows += [(f"missing days: {k}", v) for k, v in self.missing_days.items()]
        return pd.DataFrame(rows, columns=["check", "value"])


def _parse_dates(col: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(col):
        parsed = pd.to_datetime(col, unit="ms", utc=True)
    else:
        parsed = pd.to_datetime(col, utc=True)
    return parsed.dt.tz_convert(None).dt.normalize()


def clean_daily(raw: pd.DataFrame, id_col: str = "portid") -> tuple[pd.DataFrame, QualityReport]:
    """Validate and reshape raw daily rows (chokepoints or ports) into a complete daily panel."""
    if raw.empty:
        raise ValueError("no rows to clean")
    df = raw.drop(columns=[c for c in DROP_COLUMNS if c in raw.columns]).copy()
    df["date"] = _parse_dates(df["date"])

    text = [c for c in TEXT_COLUMNS if c in df.columns]
    numeric = [c for c in df.columns if c not in {"date", id_col, *text}]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_raw = len(df)
    df = df.sort_values([id_col, "date"]).drop_duplicates([id_col, "date"], keep="last")
    n_dup = n_raw - len(df)

    value_cols = numeric
    neg = df[value_cols] < 0
    n_neg = int(neg.to_numpy().sum())
    df[value_cols] = df[value_cols].mask(neg)

    mismatches = {}
    for total, parts in IDENTITIES:
        if {total, *parts} <= set(df.columns):
            diff = df[total] - df[parts].sum(axis=1, min_count=len(parts))
            mismatches[f"{total} != {' + '.join(parts)}"] = int((diff.abs() > 1e-9).sum())

    labels = df.groupby(id_col)[text].last() if text else None
    first_metric = next((c for c in ["n_total", "portcalls"] if c in value_cols), value_cols[0])
    frames, missing = [], {}
    for pid, g in df.groupby(id_col, sort=True):
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="D", name="date")
        g = g.set_index("date").reindex(full)
        g[id_col] = pid
        for c in text:
            g[c] = labels.loc[pid, c]
        g["is_missing"] = g[first_metric].isna()
        label = labels.loc[pid, "portname"] if labels is not None and "portname" in text else pid
        missing[str(label)] = int(g["is_missing"].sum())
        frames.append(g.reset_index())
    panel = pd.concat(frames, ignore_index=True)

    report = QualityReport(
        n_rows_raw=len(raw),
        n_duplicates=n_dup,
        n_negative=n_neg,
        identity_mismatches=mismatches,
        missing_days=missing,
        date_min=panel["date"].min(),
        date_max=panel["date"].max(),
    )
    return panel, report


def to_series(
    panel: pd.DataFrame,
    key: str,
    column: str = "n_total",
    id_col: str = "portid",
    max_gap: int = 3,
) -> pd.Series:
    """Extract one daily series by id or name.

    Short gaps (<= ``max_gap`` days) are filled by linear interpolation so that
    weekly models can run; longer gaps are left as NaN on purpose.
    """
    mask = panel[id_col] == key
    if not mask.any() and "portname" in panel:
        mask = panel["portname"].str.casefold() == key.casefold()
    if not mask.any():
        raise KeyError(f"{key!r} not found in panel")
    s = panel.loc[mask].set_index("date")[column].astype(float).asfreq("D")
    # pandas' ``limit`` would also fill the first days of a long gap, so fill
    # only whole runs of NaN that are short enough
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    s = s.where(~isna | (run_len > max_gap), filled)
    s.name = column
    return s


def trim_incomplete_tail(s: pd.Series, lookback: int = 28, ratio: float = 0.35) -> pd.Series:
    """Drop trailing days that look like unsettled, partially published data.

    PortWatch publishes with a lag and the newest days can be revised upward.
    A trailing value far below the recent median is treated as provisional.
    """
    s = s.dropna()
    if len(s) <= lookback:
        return s
    ref = np.nanmedian(s.iloc[-lookback:-7]) if len(s) > lookback else np.nanmedian(s)
    end = len(s)
    while end > len(s) - 7 and s.iloc[end - 1] < ratio * ref:
        end -= 1
    return s.iloc[:end]
