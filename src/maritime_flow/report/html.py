"""Self-contained HTML report (figures embedded, no external assets)."""

from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from maritime_flow import __version__
from maritime_flow.config import ATTRIBUTION

if TYPE_CHECKING:
    from maritime_flow.pipeline import AnalysisResult

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e6e5e1;--bg:#f9f9f7;--card:#fcfcfb;--accent:#2a78d6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 8px;padding-top:8px;border-top:1px solid var(--line)}
.sub{color:var(--ink2);margin:0 0 24px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font-size:24px;font-weight:600}.kpi .l{color:var(--ink2);font-size:13px}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:var(--card);margin:8px 0}
.tbl{overflow-x:auto}table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;margin:8px 0}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}p.note{color:var(--ink2);font-size:13.5px}
footer{margin-top:48px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:12px}
"""


def _img(path: Path | None) -> str:
    if path is None or not Path(path).exists():
        return ""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img alt="{html.escape(Path(path).stem)}" src="data:image/png;base64,{data}">'


def _table(df: pd.DataFrame | None, floatfmt: str = "{:,.3f}") -> str:
    if df is None or df.empty:
        return ""
    return '<div class="tbl">' + df.to_html(float_format=lambda v: floatfmt.format(v), border=0) + "</div>"


def _pct(v: float) -> str:
    return f"{v * 100:+.1f}%"


def _kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{html.escape(value)}</div><div class="l">{html.escape(label)}</div></div>'


def build_report(res: AnalysisResult, path: Path) -> Path:
    n, t, f, cfg = res.numbers, res.tables, res.figures, res.config
    title = cfg.title or f"Traffic analysis: {n['target']}"
    parts = [f"<h1>{html.escape(title)}</h1>",
             f'<p class="sub">{html.escape(cfg.data_note)} &middot; {n["first"]} to {n["last"]} '
             f'&middot; {n["days"]:,} days</p>']

    unit = n.get("unit", "transits")
    sec = iter(range(1, 20))

    def h2(text: str) -> str:
        return f"<h2>{next(sec)}. {html.escape(text)}</h2>"

    kpis = []
    if "its" in n:
        its = n["its"]
        lo, hi = its["rate_ratio_ci"]
        kpis.append(_kpi(_pct(its["rate_ratio"] - 1), f"level change at event (ITS, 95% CI {_pct(lo - 1)} to {_pct(hi - 1)})"))
    if "synthetic_control" in n:
        s = n["synthetic_control"]
        kpis.append(_kpi(_pct(s["effect_pct"]), f"effect vs synthetic control (placebo p = {s['p_value']:.2f})"))
    if "counterfactual" in n:
        c = n["counterfactual"]
        kpis.append(_kpi(f"{c['missing_transits']:,.0f}", f"{unit} below expectation over {c['days']} days"))
    if "share_shift" in n:
        kpis.append(_kpi(f"{n['share_shift']['change_pp']:+.1f} pp", f"change in {n['comparator']} share"))
    kpis.append(_kpi(n["best_model"].replace("_", " "), "most accurate forecast model"))
    parts.append(f'<div class="kpis">{"".join(kpis)}</div>')

    parts += [h2("Overview"), _img(f.get("overview")), _table(t.get("summary"), "{:,.2f}"),
              '<p class="note">dispersion_index = variance / mean. Values well above 1 mean the counts are '
              "over-dispersed, which is why the regression models use a quasi-Poisson correction.</p>",
              _table(t.get("yearly"), "{:,.3f}")]
    if "trade" in t:
        parts += [_img(f.get("trade")), _table(t["trade"], "{:,.2f}")]

    kw = n["weekday_test"]
    parts += [h2("Seasonality"), _img(f.get("weekday")),
              f'<p class="note">Kruskal-Wallis test across weekdays on detrended data: '
              f'H = {kw["kruskal_h"]:.1f}, p = {kw["p_value"]:.2g}.</p>', _img(f.get("vessel_mix"))]

    parts += [h2("Structural breaks"), _img(f.get("changepoints")),
              '<p class="note">PELT change-point search on the log of seasonally adjusted weekly means, '
              "BIC-type penalty. Each row is a regime with a stable average level.</p>",
              _table(t.get("segments"), "{:,.3f}")]

    if "its" in n:
        b, its, c = n["bootstrap"], n["its"], n["counterfactual"]
        parts += [
            h2(f"Impact of the event ({cfg.event}): own history"), _img(f.get("counterfactual")),
            '<p class="note">Three estimates that rely on different assumptions. If they agree, the conclusion '
            "does not hinge on one modelling choice.</p>",
            "<ul>",
            f"<li><b>Before/after means</b> ({b['n_pre']} vs {b['n_post']} days): {b['pre_mean']:.1f} &rarr; "
            f"{b['post_mean']:.1f} per day ({_pct(b['rel_change'])}, block-bootstrap 95% CI "
            f"{_pct(b['rel_change_ci'][0])} to {_pct(b['rel_change_ci'][1])}).</li>",
            f"<li><b>Interrupted time series</b> (quasi-Poisson, seasonality + trend): rate ratio "
            f"{its['rate_ratio']:.3f} (95% CI {its['rate_ratio_ci'][0]:.3f}-{its['rate_ratio_ci'][1]:.3f}), "
            f"p = {its['p_value']:.2g}; dispersion {its['dispersion']:.2f}.</li>",
            f"<li><b>Counterfactual</b>: {c['observed_total']:,.0f} {unit} observed vs {c['expected_total']:,.0f} "
            f"expected ({_pct(c['observed_vs_expected'])}); shortfall {c['missing_transits']:,.0f} "
            f"({int(c['interval_level'] * 100)}% range {c['missing_ci'][0]:,.0f} to {c['missing_ci'][1]:,.0f}).</li>",
            "</ul>", _table(t.get("its_coefficients"), "{:,.4f}"),
        ]

    if "synthetic_control" in n:
        s = n["synthetic_control"]
        itp = s.get("in_time_placebo") or {}
        parts += [
            h2("Impact of the event: synthetic control"), _img(f.get("sc_path")),
            f'<p class="note">The synthetic control is the weighted mix of {s["n_donors"]} unaffected units that '
            f"best tracks {html.escape(s['treated'])} before the event (pre-event RMSPE {s['pre_rmspe']:.1f} index "
            f"points). After the event the gap averages <b>{_pct(s['effect_pct'])}</b>.</p>",
            _img(f.get("sc_placebos")),
            f'<p class="note">Placebo test: each donor was treated as if it had been hit. The real unit\'s '
            f"post/pre RMSPE ratio is {s['rmspe_ratio']:.2f}, giving a permutation p-value of {s['p_value']:.2f} "
            f"(the smallest possible with {s['n_donors']} donors is {s['min_attainable_p']:.2f}).</p>",
        ]
        if itp:
            parts.append(f'<p class="note">In-time placebo: with a fake event on {itp["fake_event"]} the estimated '
                         f'effect is {_pct(itp["effect_pct"])}, which should be close to zero.</p>')
        parts += [_img(f.get("sc_weights")), _table(t.get("sc_placebos"), "{:,.3f}"),
                  '<p class="note">Leave-one-out: the effect re-estimated without each donor that has weight.</p>',
                  _table(t.get("sc_leave_one_out"), "{:,.3f}")]

    if "change_correlation" in n:
        cc = n["change_correlation"]
        parts += [h2(f"Alternative route: {n['comparator']}"), _img(f.get("route_share")),
                  _img(f.get("ccf")),
                  f'<p class="note">Correlation of weekly changes: Pearson r = {cc["pearson_r"]:.2f} '
                  f'(p = {cc["pearson_p"]:.2g}), Spearman &rho; = {cc["spearman_rho"]:.2f} '
                  f'(p = {cc["spearman_p"]:.2g}), {cc["n_weeks"]} weeks.</p>']
        if "share_shift" in n:
            s = n["share_shift"]
            parts.append(f'<p class="note">Share before vs after the event: {s["share_pre"]:.1%} &rarr; '
                         f'{s["share_post"]:.1%} (Welch t = {s["welch_t"]:.1f}, p = {s["p_value"]:.2g}).</p>')

    parts += [h2("Unusual days"), _img(f.get("anomalies")), _table(t.get("anomalies"), "{:,.2f}")]

    parts += [h2("Forecast"), _img(f.get("forecast")),
              '<p class="note">Rolling-origin backtest, non-overlapping 28-day windows. MASE &lt; 1 means the model '
              "beats repeating last week. Diebold-Mariano tests compare each model with that benchmark "
              "(negative statistic = more accurate).</p>",
              _table(t.get("backtest"), "{:,.3f}"), _table(t.get("dm_tests"), "{:,.4f}")]

    if "quality" in t:
        parts += ["<h2>Data quality</h2>", _table(t["quality"])]

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"<footer>{html.escape(ATTRIBUTION)}<br>Generated {stamp} by maritime-flow-stats "
                 f"v{__version__}. Estimates are statistical models of AIS-derived data and carry "
                 "uncertainty; they are not official statistics.</footer>")

    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body><main>{''.join(parts)}</main></body></html>")
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path
