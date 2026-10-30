"""Command line interface.

    maritime-flow list [--kind port --country Greece]   # what PortWatch covers
    maritime-flow fetch "Suez" "Good Hope"               # download + clean, save a CSV panel
    maritime-flow analyze --target Suez --comparator "Good Hope" --event red_sea --donors Malacca Gibraltar
    maritime-flow analyze --kind port --target Piraeus --event red_sea --donors Valencia Algeciras
    maritime-flow db load --kind chokepoint Suez "Good Hope" Malacca
    maritime-flow db query event_ranking --metric n_total --event 2023-11-19
    maritime-flow dashboard                              # interactive app (needs the 'app' extra)
    maritime-flow demo                                   # everything on synthetic data, no network
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from maritime_flow import __version__
from maritime_flow.config import ATTRIBUTION
from maritime_flow.data.cleaning import clean_daily
from maritime_flow.data.portwatch import PortWatchClient, PortWatchError, resolve_names
from maritime_flow.events import EVENTS

log = logging.getLogger("maritime_flow")
KINDS = ("chokepoint", "port")


def _client(args) -> PortWatchClient:
    return PortWatchClient(offline=args.offline, cache_ttl_hours=args.cache_hours)


def _resolve_event(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if value in EVENTS:
        ev = EVENTS[value]
        return ev.date, ev.label
    pd.Timestamp(value)  # validates the date
    return value, None


def _catalog(client: PortWatchClient, kind: str) -> pd.DataFrame:
    return client.chokepoint_catalog() if kind == "chokepoint" else client.port_catalog()


def _download(args, kind: str, names: list[str]) -> pd.DataFrame:
    client = _client(args)
    ids = resolve_names(_catalog(client, kind), names)
    log.info("resolved %s", ids)
    return client.chokepoint_daily(ids.values()) if kind == "chokepoint" else client.port_daily(ids.values())


def _match(panel: pd.DataFrame, query: str) -> str:
    names = panel["portname"].dropna().unique()
    exact = [n for n in names if n.casefold() == query.casefold()]
    hits = exact or [n for n in names if query.casefold() in n.casefold()]
    if len(hits) != 1:
        raise SystemExit(f"could not match {query!r} to one of {sorted(names)}")
    return hits[0]


# ------------------------------------------------------------------ commands
def cmd_list(args) -> int:
    cat = _catalog(_client(args), args.kind)
    if args.country and "country" in cat:
        cat = cat[cat["country"].str.casefold() == args.country.casefold()]
    cols = [c for c in ["portid", "portname", "country", "vessel_count_total"] if c in cat.columns]
    print(cat[cols].sort_values("portname").to_string(index=False))
    return 0


def cmd_fetch(args) -> int:
    panel, report = clean_daily(_download(args, args.kind, args.names))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)
    print(report.as_frame().to_string(index=False))
    print(f"\nsaved {len(panel):,} rows to {out}\n{ATTRIBUTION}")
    return 0


def cmd_analyze(args) -> int:
    from maritime_flow.pipeline import AnalysisConfig, run_analysis

    wanted = [args.target, *([args.comparator] if args.comparator else []), *(args.donors or [])]
    quality = None
    if args.panel:
        panel = pd.read_csv(args.panel, parse_dates=["date"])
    elif args.from_db:
        from maritime_flow.storage import Database

        with Database(args.db_url) as db:
            panel = db.read_panel(kind=args.kind)
    else:
        panel, quality = clean_daily(_download(args, args.kind, wanted))
    ev, label = _resolve_event(args.event)
    cfg = AnalysisConfig(
        target=_match(panel, args.target),
        kind=args.kind,
        comparator=_match(panel, args.comparator) if args.comparator else None,
        donors=[_match(panel, d) for d in args.donors] if args.donors else None,
        event=ev,
        event_label=label,
        horizon=args.horizon,
        column=args.column,
    )
    res = run_analysis(panel, cfg, Path(args.out), quality)
    print(f"report written to {Path(args.out) / 'report.html'}")
    if "its" in res.numbers:
        its = res.numbers["its"]
        print(f"level change at event (ITS): rate ratio {its['rate_ratio']:.3f} "
              f"(95% CI {its['rate_ratio_ci'][0]:.3f}-{its['rate_ratio_ci'][1]:.3f})")
    if "synthetic_control" in res.numbers:
        s = res.numbers["synthetic_control"]
        print(f"synthetic control: effect {s['effect_pct']:+.1%}, placebo p = {s['p_value']:.2f}")
    return 0


def cmd_db_load(args) -> int:
    from maritime_flow.storage import Database

    if args.demo:
        from maritime_flow.data.synthetic import demo_dataset, demo_ports

        loads = [("chokepoint", demo_dataset(), "synthetic demo"), ("port", demo_ports(), "synthetic demo")]
    else:
        if not args.names:
            raise SystemExit("give at least one name, or --demo")
        loads = [(args.kind, _download(args, args.kind, args.names), "IMF PortWatch")]
    with Database(args.db_url) as db:
        for kind, raw, source in loads:
            panel, report = clean_daily(raw)
            n = db.load_panel(panel, kind, source)
            print(f"{kind}: {n:,} observations upserted ({panel['portname'].nunique()} locations)")
    print(f"database: {args.db_url}")
    return 0


def cmd_db_query(args) -> int:
    from maritime_flow.storage import Database, available_queries

    if args.name == "list":
        for k, v in available_queries().items():
            print(f"{k:20s} {v}")
        return 0
    params = {"metric": args.metric, "portname": args.portname}
    if args.event:
        ev = pd.Timestamp(_resolve_event(args.event)[0])
        w = pd.Timedelta(days=args.window)
        params.update(event=ev.strftime("%Y-%m-%d"), pre_start=(ev - w).strftime("%Y-%m-%d"),
                      post_end=(ev + w).strftime("%Y-%m-%d"))
    with Database(args.db_url) as db:
        df = db.query(args.name, {k: v for k, v in params.items() if v is not None})
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"saved {len(df):,} rows to {args.csv}")
    else:
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(df.tail(args.rows).to_string(index=False) if args.rows else df.to_string(index=False))
    return 0


def cmd_dashboard(args) -> int:
    app = Path(__file__).with_name("app") / "streamlit_app.py"
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit("the dashboard needs: pip install -e '.[app]'") from None
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def cmd_demo(args) -> int:
    from maritime_flow.data.synthetic import DEMO_CHOKEPOINTS, DEMO_EVENT, DEMO_PORTS, demo_dataset, demo_ports
    from maritime_flow.pipeline import AnalysisConfig, run_analysis

    note = "SYNTHETIC data generated by maritime_flow.data.synthetic - for testing only"
    out = Path(args.out)
    panel, quality = clean_daily(demo_dataset())
    cfg = AnalysisConfig(
        target="Demo Canal", comparator="Demo Cape",
        donors=[d[1] for d in DEMO_CHOKEPOINTS[2:]],
        event=DEMO_EVENT, event_label="simulated shock",
        title="Demo report: chokepoint (synthetic data)", data_note=note,
    )
    run_analysis(panel, cfg, out / "chokepoint", quality)
    ports, pquality = clean_daily(demo_ports())
    pcfg = AnalysisConfig(
        target="Demo Piraeus", kind="port", donors=[d[1] for d in DEMO_PORTS[1:]],
        event=DEMO_EVENT, event_label="simulated shock",
        title="Demo report: port (synthetic data)", data_note=note,
    )
    run_analysis(ports, pcfg, out / "port", pquality)
    print(f"demo reports written to {out / 'chokepoint' / 'report.html'} and {out / 'port' / 'report.html'}")
    return 0


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    from maritime_flow.storage import DEFAULT_URL

    p = argparse.ArgumentParser(prog="maritime-flow",
                                description="Statistics on maritime chokepoint and port traffic (IMF PortWatch).")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--offline", action="store_true", help="use cached downloads only")
    p.add_argument("--cache-hours", type=float, default=24.0, help="cache lifetime (default 24)")
    p.add_argument("--db-url", default=DEFAULT_URL, help=f"database (default {DEFAULT_URL})")
    sub = p.add_subparsers(dest="command", required=True)

    li = sub.add_parser("list", help="list chokepoints or ports")
    li.add_argument("--kind", choices=KINDS, default="chokepoint")
    li.add_argument("--country", help="ports only, e.g. Greece")
    li.set_defaults(func=cmd_list)

    f = sub.add_parser("fetch", help="download and clean daily data")
    f.add_argument("names", nargs="+", help="names (partial match is fine)")
    f.add_argument("--kind", choices=KINDS, default="chokepoint")
    f.add_argument("--out", default="data/processed/panel.csv")
    f.set_defaults(func=cmd_fetch)

    a = sub.add_parser("analyze", help="run the full analysis and build the report")
    a.add_argument("--target", required=True)
    a.add_argument("--kind", choices=KINDS, default="chokepoint")
    a.add_argument("--comparator", help="alternative route, e.g. 'Good Hope'")
    a.add_argument("--donors", nargs="+", help="unaffected units for synthetic control")
    a.add_argument("--event", help=f"YYYY-MM-DD or one of: {', '.join(EVENTS)}")
    src = a.add_mutually_exclusive_group()
    src.add_argument("--panel", help="use a CSV saved by 'fetch' instead of downloading")
    src.add_argument("--from-db", action="store_true", help="read data from the database")
    a.add_argument("--column", help="series to analyse (default: n_total for chokepoints, portcalls for ports)")
    a.add_argument("--horizon", type=int, default=28)
    a.add_argument("--out", default="reports/latest")
    a.set_defaults(func=cmd_analyze)

    db = sub.add_parser("db", help="SQL database: load data, run queries")
    dbs = db.add_subparsers(dest="db_command", required=True)
    dl = dbs.add_parser("load", help="download and upsert into the database")
    dl.add_argument("names", nargs="*")
    dl.add_argument("--kind", choices=KINDS, default="chokepoint")
    dl.add_argument("--demo", action="store_true", help="load the synthetic demo data instead")
    dl.set_defaults(func=cmd_db_load)
    dq = dbs.add_parser("query", help="run a named query ('list' shows them) or raw SELECT")
    dq.add_argument("name", help="query name, 'list', or a SELECT statement in quotes")
    dq.add_argument("--metric", default="n_total")
    dq.add_argument("--portname")
    dq.add_argument("--event", help="for event_ranking: date or event key")
    dq.add_argument("--window", type=int, default=365, help="days before/after the event")
    dq.add_argument("--rows", type=int, default=30, help="show last N rows (0 = all)")
    dq.add_argument("--csv", help="save the result instead of printing")
    dq.set_defaults(func=cmd_db_query)

    sub.add_parser("dashboard", help="start the interactive dashboard").set_defaults(func=cmd_dashboard)

    d = sub.add_parser("demo", help="run everything on synthetic data (no network)")
    d.add_argument("--out", default="reports/demo")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except (PortWatchError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
