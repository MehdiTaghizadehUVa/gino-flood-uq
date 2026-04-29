import numpy as np

from neuralop.flood.eval.render import _safe_linear_fit_and_corr


def test_safe_linear_fit_skips_constant_or_nonfinite_spread_skill_samples():
    assert _safe_linear_fit_and_corr(np.ones(4), np.arange(4, dtype=float)) is None
    assert _safe_linear_fit_and_corr(
        np.array([0.0, np.nan, 1.0]),
        np.array([0.0, 1.0, np.inf]),
    ) is None


def test_safe_linear_fit_returns_finite_fit_for_valid_samples():
    fit = _safe_linear_fit_and_corr(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([1.0, 3.0, 5.0, 7.0]),
    )
    assert fit is not None
    corr, slope, intercept = fit
    assert np.isfinite(corr)
    assert np.isclose(slope, 2.0)
    assert np.isclose(intercept, 1.0)
