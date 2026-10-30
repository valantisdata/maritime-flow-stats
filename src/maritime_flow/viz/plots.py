"""Static figures for the report.

One visual language for every chart: thin lines, recessive grid, a fixed
categorical colour order, direct labels instead of busy legends where possible,
and the data attribution printed on every figure (an IMF reuse condition).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from maritime_flow.config import ATTRIBUTION_SHORT  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BAND = "#b7d3f6"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 9.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _finish(fig, ax, path: Path | None, subtitle: str | None = None) -> plt.Figure:
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    fig.text(0.01, -0.07 if ax.get_xlabel() else -0.02, ATTRIBUTION_SHORT, color=MUTED, fontsize=7.5, ha="left")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return fig


def _event_line(ax, event, label: str | None = None) -> None:
    if event is None:
        return
    ev = pd.Timestamp(event)
    ax.axvline(ev, color=INK_2, lw=1, ls=(0, (3, 3)))
    if label:
        ax.text(ev, ax.get_ylim()[1], f" {label}", color=INK_2, fontsize=8, va="top", ha="left")


def plot_overview(series: dict[str, pd.Series], path: Path | None = None, event=None, event_label=None,
                  unit: str = "transits"):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (name, s) in enumerate(series.items()):
        smooth = s.rolling(7, min_periods=5).mean()
        ax.plot(smooth.index, smooth, color=SERIES[i], label=name)
        last = smooth.dropna()
        ax.text(last.index[-1], last.iloc[-1], f"  {name}", color=INK, fontsize=8.5, va="center")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"{unit} per day (7-day mean)")
    ax.set_title(f"Daily {unit}")
    if len(series) > 1:
        ax.legend(loc="upper left", ncols=len(series))
    _event_line(ax, event, event_label)
    return _finish(fig, ax, path)


def plot_changepoints(weekly: pd.Series, segments: pd.DataFrame, title: str, path: Path | None = None,
                      unit: str = "transits"):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(weekly.index, weekly, color=SERIES[0], lw=1.2, label="weekly mean (seasonally adjusted)")
    for _, r in segments.iterrows():
        ax.hlines(r["mean"], r["start_date"], r["end_date"] + pd.Timedelta(days=6), color=SERIES[1], lw=2.4)
    for d in segments["start_date"].iloc[1:]:
        ax.axvline(d, color=INK_2, lw=1, ls=(0, (3, 3)))
    ax.plot([], [], color=SERIES[1], lw=2.4, label="segment level (PELT)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"{unit} per day")
    ax.set_title(title)
    ax.legend(loc="lower left")
    return _finish(fig, ax, path, f"{len(segments) - 1} level shift(s) detected")


def plot_counterfactual(history: pd.Series, cf: pd.DataFrame, event, title: str, path: Path | None = None,
                        unit: str = "transits"):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    start = pd.Timestamp(event) - pd.Timedelta(days=365)
    h = history[history.index >= start].rolling(7, min_periods=5).mean()
    roll = cf.rolling(7, min_periods=5).mean()
    ax.fill_between(cf.index, cf["lower_7d"], cf["upper_7d"], color=BAND, lw=0, label="expected range (90%)")
    ax.plot(roll.index, roll["expected"], color=SERIES[0], ls=(0, (4, 2)), label="expected without the event")
    ax.plot(h.index, h, color=INK, lw=1.4, label="observed")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"{unit} per day (7-day mean)")
    ax.set_title(title)
    ax.legend(loc="lower left")
    _event_line(ax, event, "event")
    return _finish(fig, ax, path)


def plot_weekday(profile: pd.DataFrame, title: str, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(profile))
    ax.bar(x, profile["relative_level"] - 1, bottom=1, color=SERIES[0], width=0.6)
    ax.errorbar(x, profile["relative_level"], yerr=profile["ci95"], fmt="none", ecolor=INK_2, lw=1, capsize=3)
    ax.axhline(1, color=INK_2, lw=0.8)
    ax.set_xticks(x, profile["weekday"])
    ax.set_ylabel("relative to 28-day mean")
    ax.set_title(title)
    return _finish(fig, ax, path, "bars = mean ratio, whiskers = 95% CI")


def plot_forecast(history: pd.Series, fc: pd.DataFrame, title: str, path: Path | None = None, days: int = 120,
                  unit: str = "transits"):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    h = history.iloc[-days:]
    ax.plot(h.index, h, color=INK, lw=1.1, label="observed")
    ax.fill_between(fc.index, fc["lower_95"], fc["upper_95"], color=BAND, alpha=0.55, lw=0, label="95% interval")
    ax.fill_between(fc.index, fc["lower_80"], fc["upper_80"], color=BAND, lw=0, label="80% interval")
    ax.plot(fc.index, fc["forecast"], color=SERIES[0], lw=2, label="forecast")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"{unit} per day")
    ax.set_title(title)
    ax.legend(loc="lower left", ncols=4)
    return _finish(fig, ax, path)


def plot_anomalies(anoms: pd.DataFrame, title: str, path: Path | None = None, unit: str = "transits"):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(anoms.index, anoms["observed"], color=MUTED, lw=0.7, label="observed")
    ax.plot(anoms.index, anoms["expected"], color=SERIES[0], lw=1.3, label="expected (season + local level)")
    flagged = anoms[anoms["anomaly"]]
    for direction, color in [("high", SERIES[1]), ("low", SERIES[6])]:
        f = flagged[flagged["direction"] == direction]
        ax.scatter(f.index, f["observed"], s=28, color=color, edgecolor=SURFACE, lw=1.5, zorder=3,
                   label=f"unusually {direction}")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"{unit} per day")
    ax.set_title(title)
    ax.legend(loc="upper left", ncols=4)
    return _finish(fig, ax, path, f"{len(flagged)} days with |robust z| > 3.5")


def plot_route_share(weekly: pd.DataFrame, name_a: str, name_b: str, event=None, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(weekly.index, 100 * weekly["share_b"], color=SERIES[2])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of combined weekly traffic")
    ax.set_title(f"Share of {name_b} in {name_a} + {name_b} traffic")
    _event_line(ax, event, "event")
    return _finish(fig, ax, path)


def plot_ccf(ccf: pd.DataFrame, title: str, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.axhspan(-ccf["band"].iloc[0], ccf["band"].iloc[0], color=GRID, lw=0)
    ax.vlines(ccf["lag_weeks"], 0, ccf["ccf"], color=SERIES[0], lw=2)
    ax.scatter(ccf["lag_weeks"], ccf["ccf"], color=SERIES[0], s=24, zorder=3)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("lag (weeks)")
    ax.set_ylabel("correlation")
    ax.set_title(title)
    return _finish(fig, ax, path, "weekly changes; grey band = approx. 95% null range")


def plot_vessel_mix(shares: pd.DataFrame, title: str, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    bottom = np.zeros(len(shares))
    width = 70
    for i, col in enumerate(shares.columns):
        ax.bar(shares.index, shares[col], bottom=bottom, width=width, color=SERIES[i],
               edgecolor=SURFACE, lw=1, label=col.replace("_", " "))
        bottom += shares[col].fillna(0).to_numpy()
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _finish(fig, ax, path)


def plot_synthetic_control(path_df: pd.DataFrame, treated: str, event, path: Path | None = None):
    """Treated unit vs its synthetic twin, both indexed to pre-event mean = 100."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(path_df.index, path_df["synthetic"], color=SERIES[0], ls=(0, (4, 2)), label="synthetic control")
    ax.plot(path_df.index, path_df["treated"], color=INK, lw=1.4, label=treated)
    ax.axhline(100, color=GRID, lw=1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("index, pre-event mean = 100")
    ax.set_title(f"{treated} vs its synthetic control")
    ax.legend(loc="lower left")
    _event_line(ax, event, "event")
    return _finish(fig, ax, path, "weekly means; synthetic = weighted mix of unaffected units")


def plot_placebo_gaps(paths: pd.DataFrame, treated: str, event, path: Path | None = None):
    """Gap (actual - synthetic) for the treated unit and every placebo run."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for col in paths.columns:
        if col != treated:
            ax.plot(paths.index, paths[col], color=MUTED, lw=0.8, alpha=0.6)
    ax.plot(paths.index, paths[treated], color=SERIES[1], lw=2, label=treated)
    ax.plot([], [], color=MUTED, lw=0.8, label="placebo runs (each donor as if treated)")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("gap, index points")
    ax.set_title("Placebo test: is the treated gap unusual?")
    ax.legend(loc="lower left")
    _event_line(ax, event, "event")
    return _finish(fig, ax, path)


def plot_weights(weights: pd.Series, path: Path | None = None):
    w = weights[weights > 0.005].sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(w) + 1.2))
    ax.barh(w.index, w.to_numpy(), color=SERIES[0], height=0.6)
    for y, v in enumerate(w.to_numpy()):
        ax.text(v, y, f" {v:.0%}", va="center", color=INK, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(0, max(w.max() * 1.25, 0.1))
    ax.set_title("Synthetic control weights")
    return _finish(fig, ax, path)


def plot_trade(weekly: pd.DataFrame, title: str, event=None, path: Path | None = None):
    """Weekly import and export volumes (thousand metric tons)."""
    fig, ax = plt.subplots(figsize=(10, 4.0))
    for i, col in enumerate(["import", "export"]):
        if col in weekly:
            sm = weekly[col].rolling(4, min_periods=2).mean() / 1000
            ax.plot(sm.index, sm, color=SERIES[i], label=f"{col}s (4-week mean)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("thousand tonnes per week")
    ax.set_title(title)
    ax.legend(loc="lower left")
    _event_line(ax, event, "event")
    return _finish(fig, ax, path, "IMF PortWatch trade estimates (modelled from AIS data)")
