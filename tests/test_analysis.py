import numpy as np
import pandas as pd
import pytest

from maritime_flow.analysis import anomaly, changepoint, descriptive, diversion, forecast, impact
from maritime_flow.data.cleaning import to_series

EVENT = "2023-12-01"


# ---------------------------------------------------------------- changepoints
def test_pelt_finds_known_breaks():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 80), rng.normal(1, 1, 120)])
    res = changepoint.pelt(x)
    assert len(res.breakpoints) == 2
    assert abs(res.breakpoints[0] - 100) <= 2
    assert abs(res.breakpoints[1] - 180) <= 2


def test_pelt_no_break_in_noise():
    rng = np.random.default_rng(1)
    assert changepoint.pelt(rng.normal(size=400)).breakpoints == []


def test_level_shift_dated_correctly(shocked_series):
    _, res, seg = changepoint.detect_level_shifts(shocked_series)
    assert len(seg) == 2
    assert abs((seg["start_date"].iloc[1] - pd.Timestamp(EVENT)).days) <= 7
    assert seg["change_vs_prev"].iloc[1] == pytest.approx(-0.5, abs=0.05)


def test_no_level_shift_on_flat_seasonal_series(flat_series):
    _, res, _ = changepoint.detect_level_shifts(flat_series)
    assert res.breakpoints == []


# ---------------------------------------------------------------------- impact
def test_its_recovers_rate_ratio(shocked_series):
    its = impact.interrupted_time_series(shocked_series, EVENT)
    lo, hi = its.rate_ratio_ci
    assert lo < 0.5 < hi
    assert its.p_value < 1e-6
    assert its.dispersion > 1  # simulated counts are over-dispersed


def test_its_no_effect_when_none(flat_series):
    its = impact.interrupted_time_series(flat_series, "2024-03-01")
    assert its.rate_ratio_ci[0] < 1 < its.rate_ratio_ci[1]


def test_counterfactual_gap(shocked_series):
    cf, s = impact.counterfactual_gap(shocked_series, EVENT, post_days=180)
    assert s["observed_vs_expected"] == pytest.approx(-0.5, abs=0.06)
    assert s["missing_ci"][0] < s["missing_transits"] < s["missing_ci"][1]
    assert (cf["lower"] <= cf["upper"]).all()


def test_block_bootstrap_ci_covers_truth(shocked_series):
    ev = pd.Timestamp(EVENT)
    pre = shocked_series[ev - pd.Timedelta(days=365): ev - pd.Timedelta(days=1)]
    post = shocked_series[ev: ev + pd.Timedelta(days=364)]
    b = impact.bootstrap_mean_diff(pre, post, n_boot=1000)
    assert b["rel_change_ci"][0] < -0.5 < b["rel_change_ci"][1]


def test_block_resample_shape():
    out = impact.moving_block_resample(np.arange(50.0), 7, 20, np.random.default_rng(0))
    assert out.shape == (20, 50)


# -------------------------------------------------------------------- forecast
def test_seasonal_naive_is_exact_on_periodic_series():
    idx = pd.date_range("2024-01-01", periods=70, freq="D")
    y = pd.Series(np.tile([5, 6, 7, 8, 9, 3, 2], 10), index=idx, dtype=float)
    pred = forecast.SeasonalNaive().fit(y).predict(14)
    np.testing.assert_array_equal(pred, y.iloc[:14].to_numpy())


def test_backtest_and_intervals(flat_series):
    bt = forecast.rolling_origin_backtest(flat_series, n_origins=4, horizon=14)
    assert set(bt.scores.index) == set(forecast.DEFAULT_MODELS)
    assert bt.scores.loc["harmonic_regression", "mase"] < bt.scores.loc["seasonal_naive", "mase"]
    fc = forecast.forecast_with_intervals(flat_series, bt.best_model, bt, horizon=14)
    assert len(fc) == 14
    assert (fc["lower_95"] <= fc["lower_80"]).all() and (fc["upper_80"] <= fc["upper_95"]).all()


def test_holt_winters_tracks_level():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-01", periods=400, freq="D")
    y = pd.Series(100 + rng.normal(0, 3, 400) + 5 * np.sin(2 * np.pi * np.arange(400) / 7), index=idx)
    pred = forecast.HoltWinters().fit(y).predict(7)
    assert np.abs(pred.mean() - 100) < 3


def test_diebold_mariano_signs():
    rng = np.random.default_rng(6)
    good, bad = rng.normal(0, 1, 500), rng.normal(0, 2, 500)
    stat, p = forecast.diebold_mariano(good, bad)
    assert stat < 0 and p < 0.01
    _, p_same = forecast.diebold_mariano(rng.normal(size=500), rng.normal(size=500))
    assert p_same > 0.01


# --------------------------------------------------------- anomalies, routes, ...
def test_injected_anomaly_is_flagged(flat_series):
    s = flat_series.copy()
    s.loc["2024-05-15"] = 2
    a = anomaly.detect_anomalies(s)
    assert a.loc["2024-05-15", "anomaly"]
    assert a["anomaly"].mean() < 0.02


def test_route_share_shift(demo_panel):
    a, b = to_series(demo_panel, "chokepoint1"), to_series(demo_panel, "chokepoint7")
    s = diversion.share_shift(a, b, EVENT)
    assert s["change_pp"] > 20 and s["p_value"] < 1e-6
    ccf = diversion.cross_correlation(a, b, max_lag=4)
    assert len(ccf) == 9 and ccf["ccf"].abs().max() <= 1


def test_weekday_profile_detects_pattern(flat_series):
    prof, test = descriptive.weekday_profile(flat_series)
    assert test["p_value"] < 0.05
    assert prof.loc[0, "relative_level"] > prof.loc[6, "relative_level"]


def test_summary_table(demo_panel):
    t = descriptive.summary_table(demo_panel)
    assert {"mean", "dispersion_index"} <= set(t.columns)
    assert (t["dispersion_index"] > 1).all()
