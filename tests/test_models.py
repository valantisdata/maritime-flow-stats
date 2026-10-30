import numpy as np
import pandas as pd
import pytest
from scipy import stats

from maritime_flow.analysis.models import harmonic_design, newey_west_cov, ols, poisson_glm


def test_ols_matches_scipy_linregress():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 1.5 + 2.0 * x + rng.normal(size=200)
    X = pd.DataFrame({"const": 1.0, "x": x})
    fit = ols(y, X)
    ref = stats.linregress(x, y)
    assert fit.params["x"] == pytest.approx(ref.slope)
    assert fit.bse["x"] == pytest.approx(ref.stderr)


def test_hac_with_zero_lags_is_hc0_like():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(300), rng.normal(size=300)])
    u = rng.normal(size=300) * (1 + np.abs(X[:, 1]))
    cov = newey_west_cov(X, u, maxlags=0)
    bread = np.linalg.inv(X.T @ X)
    hc0 = bread @ (X.T * u**2) @ X @ bread
    np.testing.assert_allclose(cov, 300 / 298 * hc0)


def test_hac_widens_se_under_autocorrelation():
    rng = np.random.default_rng(2)
    n = 1000
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + rng.normal()
    X = pd.DataFrame({"const": 1.0, "trend": np.arange(n) / n})
    y = 1 + e
    assert ols(y, X, "hac").bse["const"] > 1.5 * ols(y, X).bse["const"]


def test_poisson_glm_recovers_coefficients():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5000)
    y = rng.poisson(np.exp(1.0 + 0.4 * x))
    fit = poisson_glm(y, pd.DataFrame({"const": 1.0, "x": x}))
    assert fit.params["const"] == pytest.approx(1.0, abs=0.03)
    assert fit.params["x"] == pytest.approx(0.4, abs=0.03)
    assert fit.dispersion == pytest.approx(1.0, abs=0.08)


def test_quasi_poisson_detects_overdispersion():
    rng = np.random.default_rng(4)
    mu = 50
    y = rng.negative_binomial(5, 5 / (5 + mu), size=3000)  # var = mu + mu^2/5 = 550
    fit = poisson_glm(y, pd.DataFrame({"const": np.ones(3000)}))
    assert fit.dispersion == pytest.approx(11, rel=0.15)


def test_harmonic_design_shapes_and_phase():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    X = harmonic_design(idx, k_annual=2)
    assert list(X.columns[:2]) == ["const", "trend"]
    assert X.filter(like="dow_").sum(axis=1).max() == 1
    # Fourier terms depend on absolute date, not on where the sample starts
    X2 = harmonic_design(idx[5:], k_annual=2)
    np.testing.assert_allclose(X.loc[idx[5:], "sin1"], X2["sin1"])
