"""Everything the dashboard computes, kept free of Streamlit so it can be tested.

The Streamlit file only lays out widgets and calls these functions. Charts are
Vega-Lite specs (plain dicts) rendered with ``st.vega_lite_chart``, which keeps
the dashboard free of extra plotting dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from maritime_flow.analysis import changepoint, forecast, impact
from maritime_flow.analysis import synthetic_control as sc
from maritime_flow.data.cleaning import clean_daily, to_series, trim_incomplete_tail
from maritime_flow.data.portwatch import PortWatchClient, resolve_names
from maritime_flow.data.synthetic import demo_dataset, demo_ports

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DEFAULT_COLUMN = {"chokepoint": "n_total", "port": "portcalls"}


# ----------------------------------------------------------------- loading
def load_demo(kind: str) -> pd.DataFrame:
    raw = demo_dataset() if kind == "chokepoint" else demo_ports()
    return clean_daily(raw)[0]


def catalog_names(kind: str, client: PortWatchClient | None = None) -> pd.DataFrame:
    client = client or PortWatchClient()
    cat = client.chokepoint_catalog() if kind == "chokepoint" else client.port_catalog()
    cols = [c for c in ["portid", "portname", "country"] if c in cat.columns]
    return cat[cols].drop_duplicates().sort_values("portname")


def load_live(kind: str, names: list[str], client: PortWatchClient | None = None) -> pd.DataFrame:
    client = client or PortWatchClient()
    cat = client.chokepoint_catalog() if kind == "chokepoint" else client.port_catalog()
    ids = resolve_names(cat, names)
    raw = client.chokepoint_daily(ids.values()) if kind == "chokepoint" else client.port_daily(ids.values())
    return clean_daily(raw)[0]


def load_db(url: str, kind: str) -> pd.DataFrame:
    from maritime_flow.storage import Database

    with Database(url) as db:
        return db.read_panel(kind=kind)


def metric_options(panel: pd.DataFrame) -> list[str]:
    skip = {"date", "portid", "portname", "country", "is_missing"}
    return [c for c in panel.columns if c not in skip and pd.api.types.is_numeric_dtype(panel[c])]


def series(panel: pd.DataFrame, name: str, column: str) -> pd.Series:
    return trim_incomplete_tail(to_series(panel, name, column))


# ------------------------------------------------------------------ charts
def line_spec(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    y_title: str = "",
    event: str | None = None,
    dashed: list[str] | None = None,
    band: tuple[str, str] | None = None,
    height: int = 320,
) -> dict:
    """Vega-Lite line chart with optional colour split, uncertainty band and event rule."""
    data = df.copy()
    data[x] = pd.to_datetime(data[x]).dt.strftime("%Y-%m-%d")
    layers: list[dict] = []
    if band:
        layers.append({
            "mark": {"type": "area", "opacity": 0.25, "color": COLORS[0]},
            "encoding": {"x": {"field": x, "type": "temporal"},
                         "y": {"field": band[0], "type": "quantitative"},
                         "y2": {"field": band[1]}},
        })
    enc: dict = {
        "x": {"field": x, "type": "temporal", "title": None},
        "y": {"field": y, "type": "quantitative", "title": y_title},
        "tooltip": [{"field": x, "type": "temporal"}, {"field": y, "type": "quantitative", "format": ",.1f"}],
    }
    if color:
        names = list(dict.fromkeys(data[color]))
        enc["color"] = {"field": color, "type": "nominal", "title": None,
                        "scale": {"domain": names, "range": COLORS[: len(names)]}}
        enc["tooltip"].insert(0, {"field": color, "type": "nominal"})
        if dashed:
            enc["strokeDash"] = {"condition": {"test": {"field": color, "oneOf": dashed}, "value": [5, 3]},
                                 "value": [1, 0]}
    layers.append({"mark": {"type": "line", "strokeWidth": 1.6}, "encoding": enc})
    if event:
        layers.append({
            "data": {"values": [{"event": event}]},
            "mark": {"type": "rule", "strokeDash": [4, 4], "color": "#52514e"},
            "encoding": {"x": {"field": "event", "type": "temporal"}},
        })
    return {"data": {"values": data.replace({np.nan: None}).to_dict("records")},
            "height": height, "layer": layers}


def bar_spec(df: pd.DataFrame, x: str, y: str, x_title: str = "", fmt: str = ".0%") -> dict:
    return {
        "data": {"values": df.to_dict("records")},
        "mark": {"type": "bar", "color": COLORS[0], "cornerRadiusEnd": 3},
        "encoding": {
            "y": {"field": y, "type": "nominal", "sort": "-x", "title": None},
            "x": {"field": x, "type": "quantitative", "title": x_title, "axis": {"format": fmt}},
            "tooltip": [{"field": y}, {"field": x, "format": fmt}],
        },
        "height": max(120, 26 * len(df)),
    }


# ---------------------------------------------------------------- analyses
def overview_frame(panel: pd.DataFrame, names: list[str], column: str) -> pd.DataFrame:
    parts = []
    for n in names:
        s = series(panel, n, column).rolling(7, min_periods=5).mean().dropna()
        parts.append(pd.DataFrame({"date": s.index, "value": s.to_numpy(), "location": n}))
    return pd.concat(parts, ignore_index=True)


@dataclass
class ImpactView:
    bootstrap: dict
    its: dict
    counterfactual: pd.DataFrame
    cf_summary: dict
    segments: pd.DataFrame


def run_impact(y: pd.Series, event: str, window_days: int = 365) -> ImpactView:
    ev = pd.Timestamp(event)
    w = pd.Timedelta(days=window_days)
    boot = impact.bootstrap_mean_diff(y[(y.index >= ev - w) & (y.index < ev)],
                                      y[(y.index >= ev) & (y.index < ev + w)], n_boot=2000)
    its = impact.interrupted_time_series(y, ev, post_days=window_days).as_dict()
    cf, cf_sum = impact.counterfactual_gap(y, ev, post_days=window_days)
    hist = y[(y.index >= ev - pd.Timedelta(days=365)) & (y.index < ev + w)].rolling(7, min_periods=5).mean()
    roll = cf.rolling(7, min_periods=5).mean()
    frame = pd.concat([
        pd.DataFrame({"date": hist.index, "value": hist.to_numpy(), "series": "observed"}),
        pd.DataFrame({"date": roll.index, "value": roll["expected"].to_numpy(), "series": "expected without event",
                      "lower": cf["lower_7d"].to_numpy(), "upper": cf["upper_7d"].to_numpy()}),
    ], ignore_index=True)
    _, _, seg = changepoint.detect_level_shifts(y)
    return ImpactView(boot, its, frame, cf_sum, seg)


def run_synthetic(panel: pd.DataFrame, column: str, treated: str, donors: list[str], event: str):
    wide = sc.to_wide(panel, column, [treated, *donors])
    res = sc.synthetic_control(wide, treated, event, donors=donors)
    path = res.fit.path.reset_index().rename(columns={"index": "date"})
    path = path.rename(columns={path.columns[0]: "date"})
    long = path.melt(id_vars="date", value_vars=["treated", "synthetic"], var_name="series", value_name="index")
    long["series"] = long["series"].replace({"treated": treated, "synthetic": "synthetic control"})
    weights = res.fit.weights[res.fit.weights > 0.005].rename_axis("donor").reset_index(name="weight")
    return res, long, weights


def run_forecast(y: pd.Series, horizon: int = 28, n_origins: int = 8):
    bt = forecast.rolling_origin_backtest(y, horizon=horizon, n_origins=n_origins)
    fc = forecast.forecast_with_intervals(y, bt.best_model, bt, horizon=horizon)
    hist = y.iloc[-120:]
    frame = pd.concat([
        pd.DataFrame({"date": hist.index, "value": hist.to_numpy(), "series": "observed"}),
        pd.DataFrame({"date": fc.index, "value": fc["forecast"].to_numpy(), "series": "forecast",
                      "lower": fc["lower_80"].to_numpy(), "upper": fc["upper_80"].to_numpy()}),
    ], ignore_index=True)
    return bt, fc, frame
