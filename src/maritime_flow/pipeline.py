"""End-to-end analysis of one chokepoint or port.

Pure functions in ``analysis`` do the statistics; this module only wires them
together, saves tables/figures and hands everything to the report builder.
Keeping the two apart is what makes the statistics unit-testable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from maritime_flow.analysis import anomaly, changepoint, descriptive, diversion, forecast, impact
from maritime_flow.analysis import synthetic_control as sc
from maritime_flow.data.cleaning import QualityReport, to_series, trim_incomplete_tail
from maritime_flow.report.html import build_report
from maritime_flow.viz import plots

log = logging.getLogger(__name__)

KIND_DEFAULTS = {
    # kind: (default column, unit word for charts, data description)
    "chokepoint": ("n_total", "transits", "IMF PortWatch daily chokepoint transit calls"),
    "port": ("portcalls", "port calls", "IMF PortWatch daily port calls and trade estimates"),
}


@dataclass
class AnalysisConfig:
    target: str
    kind: str = "chokepoint"
    comparator: str | None = None
    donors: list[str] | None = None
    event: str | None = None
    event_label: str | None = None
    column: str | None = None
    horizon: int = 28
    n_origins: int = 10
    impact_window_days: int = 365
    title: str | None = None
    data_note: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KIND_DEFAULTS:
            raise ValueError(f"kind must be one of {list(KIND_DEFAULTS)}")
        col, _, note = KIND_DEFAULTS[self.kind]
        self.column = self.column or col
        self.data_note = self.data_note or note

    @property
    def unit(self) -> str:
        return KIND_DEFAULTS[self.kind][1] if self.column == KIND_DEFAULTS[self.kind][0] else self.column


@dataclass
class AnalysisResult:
    config: AnalysisConfig
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    numbers: dict[str, object] = field(default_factory=dict)


def _name_of(panel: pd.DataFrame, key: str) -> str:
    m = panel.loc[(panel["portid"] == key) | (panel["portname"].str.casefold() == key.casefold()), "portname"]
    return str(m.iloc[0]) if len(m) else key


def run_analysis(
    panel: pd.DataFrame, cfg: AnalysisConfig, outdir: Path, quality: QualityReport | None = None
) -> AnalysisResult:
    outdir = Path(outdir)
    figdir = outdir / "figures"
    tabdir = outdir / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    res = AnalysisResult(cfg)
    unit = cfg.unit

    def fig(key: str) -> Path:
        res.figures[key] = figdir / f"{key}.png"
        return res.figures[key]

    name = _name_of(panel, cfg.target)
    y = trim_incomplete_tail(to_series(panel, cfg.target, cfg.column))
    log.info("analysing %s: %d days (%s to %s)", name, len(y), y.index[0].date(), y.index[-1].date())
    res.numbers.update(target=name, kind=cfg.kind, column=cfg.column, unit=unit,
                       first=str(y.index[0].date()), last=str(y.index[-1].date()), days=len(y))

    comp_name, z = None, None
    if cfg.comparator:
        comp_name = _name_of(panel, cfg.comparator)
        z = trim_incomplete_tail(to_series(panel, cfg.comparator, cfg.column))
        res.numbers["comparator"] = comp_name

    # 1. descriptive
    names = [name] + ([comp_name] if comp_name else [])
    res.tables["summary"] = descriptive.summary_table(panel[panel["portname"].isin(names)], cfg.column)
    res.tables["yearly"] = descriptive.yearly_table(y)
    prof, kw = descriptive.weekday_profile(y)
    res.tables["weekday"] = prof
    res.numbers["weekday_test"] = kw
    if quality is not None:
        res.tables["quality"] = quality.as_frame()
    overview = {name: y} | ({comp_name: z} if z is not None else {})
    plots.plot_overview(overview, fig("overview"), cfg.event, cfg.event_label, unit=unit)
    plots.plot_weekday(prof, f"Weekday pattern - {name}", fig("weekday"))
    if cfg.kind == "chokepoint" and "n_container" in panel:
        mix = descriptive.vessel_mix(panel, name)
        res.tables["vessel_mix"] = mix
        plots.plot_vessel_mix(mix, f"Vessel mix by quarter - {name}", fig("vessel_mix"))
    if cfg.kind == "port" and {"import", "export"} <= set(panel.columns):
        trade = descriptive.trade_flows(panel, name)
        res.tables["trade"] = descriptive.trade_yearly(trade)
        plots.plot_trade(trade, f"Estimated trade volumes - {name}", cfg.event, fig("trade"))

    # 2. structural breaks
    weekly, _, seg = changepoint.detect_level_shifts(y)
    res.tables["segments"] = seg
    plots.plot_changepoints(weekly, seg, f"Level shifts - {name}", fig("changepoints"), unit=unit)

    # 3. event impact, from the unit's own history
    if cfg.event:
        ev = pd.Timestamp(cfg.event)
        w = pd.Timedelta(days=cfg.impact_window_days)
        pre = y[(y.index >= ev - w) & (y.index < ev)]
        post = y[(y.index >= ev) & (y.index < ev + w)]
        res.numbers["bootstrap"] = impact.bootstrap_mean_diff(pre, post)
        its = impact.interrupted_time_series(y, ev, post_days=cfg.impact_window_days)
        res.numbers["its"] = its.as_dict()
        res.tables["its_coefficients"] = its.fit.summary_frame().loc[["trend", "step", "post_slope"]]
        cf, cf_sum = impact.counterfactual_gap(y, ev, post_days=cfg.impact_window_days)
        res.numbers["counterfactual"] = cf_sum
        cf.to_csv(tabdir / "counterfactual.csv")
        plots.plot_counterfactual(y, cf, ev, f"Observed vs expected - {name}", fig("counterfactual"), unit=unit)

    # 4. event impact, from comparable units (synthetic control)
    if cfg.event and cfg.donors:
        donors = [_name_of(panel, d) for d in cfg.donors]
        wide = sc.to_wide(panel, cfg.column, [name, *donors])
        scr = sc.synthetic_control(wide, name, cfg.event, donors=donors)
        res.numbers["synthetic_control"] = scr.summary()
        res.tables["sc_weights"] = scr.fit.weights.rename("weight").to_frame()
        res.tables["sc_placebos"] = scr.placebos
        res.tables["sc_leave_one_out"] = scr.leave_one_out
        scr.fit.path.to_csv(tabdir / "sc_path.csv")
        plots.plot_synthetic_control(scr.fit.path, name, cfg.event, fig("sc_path"))
        plots.plot_weights(scr.fit.weights, fig("sc_weights"))
        if not scr.placebo_paths.empty:
            plots.plot_placebo_gaps(scr.placebo_paths, name, cfg.event, fig("sc_placebos"))

    # 5. alternative route
    if z is not None:
        wk = diversion.weekly_pair(y, z)
        res.numbers["change_correlation"] = diversion.change_correlation(y, z)
        ccf = diversion.cross_correlation(y, z)
        res.tables["ccf"] = ccf
        plots.plot_route_share(wk, name, comp_name, cfg.event, fig("route_share"))
        plots.plot_ccf(ccf, f"Cross-correlation of weekly changes: {name} vs {comp_name}", fig("ccf"))
        if cfg.event:
            res.numbers["share_shift"] = diversion.share_shift(y, z, cfg.event)

    # 6. anomalies
    an = anomaly.detect_anomalies(y)
    res.tables["anomalies"] = anomaly.anomaly_table(an)
    plots.plot_anomalies(an.iloc[-730:], f"Unusual days (last two years) - {name}", fig("anomalies"), unit=unit)

    # 7. forecast
    bt = forecast.rolling_origin_backtest(y, horizon=cfg.horizon, n_origins=cfg.n_origins)
    res.tables["backtest"] = bt.scores
    res.tables["dm_tests"] = bt.dm_tests
    res.numbers["best_model"] = bt.best_model
    fc = forecast.forecast_with_intervals(y, bt.best_model, bt, horizon=cfg.horizon)
    res.tables["forecast"] = fc
    plots.plot_forecast(y, fc, f"{cfg.horizon}-day forecast - {name} ({bt.best_model.replace('_', ' ')})",
                        fig("forecast"), unit=unit)

    for key, tab in res.tables.items():
        tab.to_csv(tabdir / f"{key}.csv")
    (outdir / "results.json").write_text(json.dumps(res.numbers, indent=2, default=_json_default))
    build_report(res, outdir / "report.html")
    return res


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)
