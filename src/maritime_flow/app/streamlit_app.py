"""Interactive dashboard: `maritime-flow dashboard` locally, or deploy the repo to Streamlit Community Cloud.

All numbers come from ``maritime_flow.app.data`` (tested); this file is layout only.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from maritime_flow.app import data as D
from maritime_flow.config import ATTRIBUTION
from maritime_flow.data.synthetic import DEMO_EVENT
from maritime_flow.events import EVENTS

SOURCES = ["Demo (synthetic)", "Live: IMF PortWatch", "Database"]


@st.cache_data(show_spinner="Building demo data...")
def _demo(kind: str) -> pd.DataFrame:
    return D.load_demo(kind)


@st.cache_data(ttl=24 * 3600, show_spinner="Reading the PortWatch catalogue...")
def _catalog(kind: str) -> pd.DataFrame:
    return D.catalog_names(kind)


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading from IMF PortWatch...")
def _live(kind: str, names: tuple[str, ...]) -> pd.DataFrame:
    return D.load_live(kind, list(names))


@st.cache_data(ttl=600, show_spinner="Reading the database...")
def _db(url: str, kind: str) -> pd.DataFrame:
    return D.load_db(url, kind)


def _pct(v: float) -> str:
    return f"{v * 100:+.1f}%"


def sidebar() -> dict:
    sb = st.sidebar
    sb.header("Data")
    source = sb.radio("Source", SOURCES, index=0)
    kind = sb.radio("Type", ["chokepoint", "port"], index=0, format_func=str.capitalize, horizontal=True)

    if source == SOURCES[0]:
        panel = _demo(kind)
        names = sorted(panel["portname"].unique())
    elif source == SOURCES[1]:
        cat = _catalog(kind)
        if kind == "port" and "country" in cat:
            countries = sorted(cat["country"].dropna().unique())
            picked = sb.multiselect("Countries", countries, default=[c for c in ["Greece"] if c in countries])
            if picked:
                cat = cat[cat["country"].isin(picked)]
        names = cat["portname"].tolist()
        panel = None
    else:
        url = sb.text_input("Database URL", "sqlite:///data/maritime.db")
        panel = _db(url, kind)
        names = sorted(panel["portname"].unique())

    if not names:
        st.warning("Nothing to show for this selection.")
        st.stop()

    sb.header("Analysis")
    target = sb.selectbox("Location", names, index=0)
    others = [n for n in names if n != target]
    comp_options = ["-"] + others
    # in demo mode, start with the alternative route selected so it is kept out of the donor pool
    comp_default = comp_options.index("Demo Cape") if source == SOURCES[0] and "Demo Cape" in comp_options else 0
    comparator = sb.selectbox("Compare with (optional)", comp_options, index=comp_default)
    default_donors = [n for n in others if n != comparator] if source == SOURCES[0] else []
    donors = sb.multiselect("Synthetic-control donors (unaffected units)", others, default=default_donors)

    presets = {f"{e.label} ({e.date})": e.date for e in EVENTS.values()}
    if source == SOURCES[0]:
        presets = {f"Simulated shock ({DEMO_EVENT})": DEMO_EVENT}
    choice = sb.selectbox("Event", ["none", *presets, "custom date"], index=1)
    event = None
    if choice == "custom date":
        event = sb.date_input("Event date", dt.date(2024, 1, 1)).isoformat()
    elif choice != "none":
        event = presets[choice]

    if panel is None:  # live: download only what the user picked
        wanted = tuple(sorted({target, *donors, *([comparator] if comparator != "-" else [])}))
        panel = _live(kind, wanted)

    metrics = D.metric_options(panel)
    default_metric = D.DEFAULT_COLUMN[kind]
    column = sb.selectbox("Metric", metrics, index=metrics.index(default_metric) if default_metric in metrics else 0)
    return dict(source=source, kind=kind, panel=panel, target=target,
                comparator=None if comparator == "-" else comparator,
                donors=[d for d in donors if d in set(panel["portname"])], event=event, column=column)


def tab_overview(cfg: dict) -> None:
    names = [cfg["target"]] + ([cfg["comparator"]] if cfg["comparator"] else [])
    frame = D.overview_frame(cfg["panel"], names, cfg["column"])
    st.vega_lite_chart(D.line_spec(frame, "date", "value", color="location",
                                   y_title=f"{cfg['column']} per day (7-day mean)", event=cfg["event"]),
                       use_container_width=True)
    y = D.series(cfg["panel"], cfg["target"], cfg["column"])
    yearly = y.groupby(y.index.year).mean().rename("mean per day").to_frame()
    yearly["change vs previous year"] = yearly["mean per day"].pct_change()
    c1, c2, c3 = st.columns(3)
    c1.metric("Days of data", f"{len(y):,}")
    c2.metric("Mean per day", f"{y.mean():,.1f}")
    c3.metric("Dispersion index (var/mean)", f"{y.var() / y.mean():.1f}")
    yearly["mean per day"] = yearly["mean per day"].round(1)
    yearly["change vs previous year"] = (100 * yearly["change vs previous year"]).round(1).astype(str) + "%"
    st.dataframe(yearly, use_container_width=True)


def tab_impact(cfg: dict) -> None:
    if not cfg["event"]:
        st.info("Pick an event in the sidebar to estimate its impact.")
        return
    y = D.series(cfg["panel"], cfg["target"], cfg["column"])
    try:
        view = D.run_impact(y, cfg["event"])
    except ValueError as exc:
        st.warning(f"Not enough data around this date: {exc}")
        return
    b, its, c = view.bootstrap, view.its, view.cf_summary
    m1, m2, m3 = st.columns(3)
    m1.metric("Before/after (block bootstrap)", _pct(b["rel_change"]),
              help=f"95% CI {_pct(b['rel_change_ci'][0])} to {_pct(b['rel_change_ci'][1])}")
    m2.metric("Level change (ITS)", _pct(its["rate_ratio"] - 1),
              help=f"rate ratio {its['rate_ratio']:.3f}, 95% CI {its['rate_ratio_ci'][0]:.3f}-{its['rate_ratio_ci'][1]:.3f}")
    m3.metric("Below expectation", f"{c['missing_transits']:,.0f}",
              help=f"90% range {c['missing_ci'][0]:,.0f} to {c['missing_ci'][1]:,.0f} over {c['days']} days")
    st.vega_lite_chart(D.line_spec(view.counterfactual, "date", "value", color="series", event=cfg["event"],
                                   dashed=["expected without event"], band=("lower", "upper"),
                                   y_title="per day, 7-day mean"), use_container_width=True)
    st.caption("Expected = seasonal quasi-Poisson model fitted on the two years before the event. "
               "Shaded = 90% simulation range. This shows how unusual the change is, not what caused it.")
    st.subheader("Level shifts found by PELT")
    st.dataframe(view.segments, use_container_width=True, hide_index=True)


def tab_synthetic(cfg: dict) -> None:
    if not cfg["event"]:
        st.info("Pick an event in the sidebar.")
        return
    if len(cfg["donors"]) < 2:
        st.info("Pick at least two donors: comparable locations that the event did not affect.")
        return
    try:
        res, long, weights = D.run_synthetic(cfg["panel"], cfg["column"], cfg["target"], cfg["donors"], cfg["event"])
    except ValueError as exc:
        st.warning(str(exc))
        return
    s = res.summary()
    m1, m2, m3 = st.columns(3)
    m1.metric("Effect vs synthetic", _pct(s["effect_pct"]))
    m2.metric("Placebo p-value", f"{s['p_value']:.2f}", help=f"smallest possible: {s['min_attainable_p']:.2f}")
    itp = s.get("in_time_placebo") or {}
    m3.metric("In-time placebo", _pct(itp["effect_pct"]) if itp else "-", help="fake event a year earlier; should be near 0")
    st.vega_lite_chart(D.line_spec(long, "date", "index", color="series", dashed=["synthetic control"],
                                   event=cfg["event"], y_title="index, pre-event mean = 100"),
                       use_container_width=True)
    left, right = st.columns(2)
    left.markdown("**Donor weights**")
    left.vega_lite_chart(D.bar_spec(weights, "weight", "donor"), use_container_width=True)
    right.markdown("**Placebo runs** (post/pre RMSPE ratio)")
    right.dataframe(res.placebos[["unit", "ratio", "effect_pct", "rank"]], hide_index=True, use_container_width=True)
    st.caption("Key assumption: donors were not affected by the event. Units traffic moved to are not valid donors.")


def tab_forecast(cfg: dict) -> None:
    y = D.series(cfg["panel"], cfg["target"], cfg["column"])
    horizon = st.slider("Horizon (days)", 7, 56, 28, step=7)
    with st.spinner("Backtesting three models..."):
        bt, fc, frame = D.run_forecast(y, horizon=horizon)
    st.vega_lite_chart(D.line_spec(frame, "date", "value", color="series", band=("lower", "upper"),
                                   y_title="per day"), use_container_width=True)
    st.markdown(f"Best model in the rolling-origin backtest: **{bt.best_model.replace('_', ' ')}** "
                "(MASE below 1 beats repeating last week).")
    st.dataframe(bt.scores.round(3), use_container_width=True)
    st.download_button("Download forecast (CSV)", fc.to_csv().encode(), "forecast.csv", "text/csv")


def tab_sql(cfg: dict) -> None:
    from maritime_flow.storage import Database, available_queries, load_query

    st.markdown("The selected data is loaded into an in-memory SQLite database. Pick a ready-made query or "
                "write your own `SELECT`.")
    queries = available_queries()
    name = st.selectbox("Query", list(queries), format_func=lambda k: f"{k} - {queries[k]}")
    sql = st.text_area("SQL", load_query(name), height=260)
    ev = pd.Timestamp(cfg["event"] or "2024-01-01")
    params = {"metric": cfg["column"], "portname": cfg["target"],
              "event": ev.strftime("%Y-%m-%d"),
              "pre_start": (ev - pd.Timedelta(days=365)).strftime("%Y-%m-%d"),
              "post_end": (ev + pd.Timedelta(days=365)).strftime("%Y-%m-%d")}
    if st.button("Run query", type="primary"):
        try:
            with Database("sqlite://") as db:
                db.load_panel(cfg["panel"], cfg["kind"], cfg["source"])
                out = db.query(sql, params)
            st.dataframe(out, use_container_width=True, hide_index=True)
        except Exception as exc:  # show SQL errors to the user instead of crashing the app
            st.error(f"{type(exc).__name__}: {exc}")


def main() -> None:
    st.set_page_config(page_title="Maritime flow stats", layout="wide")
    st.title("Maritime chokepoint and port traffic")
    st.caption("Daily vessel counts from IMF PortWatch, with change-point detection, event impact, "
               "synthetic control and forecasting.")
    cfg = sidebar()
    if cfg["source"] == SOURCES[0]:
        st.warning("Demo mode: synthetic data with a known simulated shock. Switch the source for real data.")
    tabs = st.tabs(["Overview", "Event impact", "Synthetic control", "Forecast", "SQL"])
    with tabs[0]:
        tab_overview(cfg)
    with tabs[1]:
        tab_impact(cfg)
    with tabs[2]:
        tab_synthetic(cfg)
    with tabs[3]:
        tab_forecast(cfg)
    with tabs[4]:
        tab_sql(cfg)
    st.divider()
    st.caption(ATTRIBUTION)


if __name__ == "__main__":
    main()
