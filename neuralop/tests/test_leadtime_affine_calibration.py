import numpy as np
import pytest

from neuralop.training.leadtime_affine_calibration import (
    apply_leadtime_affine_to_ensemble,
    fit_leadtime_affine_calibration,
    validate_split_leakage_guard,
)


def test_affine_fit_recovers_synthetic_coefficients():
    rng = np.random.default_rng(123)
    n_steps = 4
    a_true = np.array([0.01, -0.02, 0.03, 0.00], dtype=np.float64)
    b_true = np.array([1.10, 0.90, 1.05, 1.00], dtype=np.float64)
    c_true = np.array([1.20, 0.80, 1.50, 1.00], dtype=np.float64)

    mu_pred = []
    sigma_pred = []
    mu_ref = []
    sigma_ref = []
    for t in range(n_steps):
        x = rng.uniform(0.02, 1.25, size=4000)
        sp = rng.uniform(0.05, 0.25, size=4000)
        y = a_true[t] + b_true[t] * x
        sr = c_true[t] * sp
        mu_pred.append(x)
        sigma_pred.append(sp)
        mu_ref.append(y)
        sigma_ref.append(sr)

    coeffs = fit_leadtime_affine_calibration(
        mu_pred,
        sigma_pred,
        mu_ref,
        sigma_ref,
        fit_wet_threshold=0.01,
        min_pred_std=1e-6,
        c_clip_min=0.1,
        c_clip_max=4.0,
        smooth_window=1,
    )

    assert np.allclose(coeffs["a"], a_true, atol=1e-6)
    assert np.allclose(coeffs["b"], b_true, atol=1e-6)
    assert np.allclose(coeffs["c"], c_true, atol=1e-6)


def test_wet_mask_excludes_dry_dominated_cells():
    # Dry region encodes conflicting mapping, wet region follows desired mapping.
    wet_thr = 0.01
    n = 500
    x_dry = np.full(n, 0.002, dtype=np.float64)
    y_dry = np.full(n, 0.003, dtype=np.float64)
    sp_dry = np.full(n, 0.02, dtype=np.float64)
    sr_dry = np.full(n, 0.02, dtype=np.float64)

    x_wet = np.linspace(0.05, 1.0, n)
    y_wet = 0.1 + 1.1 * x_wet
    sp_wet = np.full(n, 0.10, dtype=np.float64)
    sr_wet = np.full(n, 0.15, dtype=np.float64)

    mu_pred = [np.concatenate([x_dry, x_wet])]
    mu_ref = [np.concatenate([y_dry, y_wet])]
    sigma_pred = [np.concatenate([sp_dry, sp_wet])]
    sigma_ref = [np.concatenate([sr_dry, sr_wet])]

    coeffs = fit_leadtime_affine_calibration(
        mu_pred,
        sigma_pred,
        mu_ref,
        sigma_ref,
        fit_wet_threshold=wet_thr,
        min_pred_std=1e-6,
        c_clip_min=0.1,
        c_clip_max=4.0,
        smooth_window=1,
    )

    # Dry-dominated subset stays below threshold in both pred/ref means, so the wet subset drives fit.
    assert coeffs["a"][0] == pytest.approx(0.1, abs=1e-6)
    assert coeffs["b"][0] == pytest.approx(1.1, abs=1e-6)
    assert coeffs["c"][0] == pytest.approx(1.5, abs=1e-6)


def test_apply_affine_to_ensemble_matches_formula():
    pred = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float64,
    )
    out = apply_leadtime_affine_to_ensemble(pred, a_t=0.5, b_t=2.0, c_t=0.5)
    mu = pred.mean(axis=0, keepdims=True)
    expected = 0.5 + 2.0 * mu + 0.5 * (pred - mu)
    assert np.allclose(out, expected)


def test_leakage_guard_rejects_same_split_by_default():
    with pytest.raises(ValueError):
        validate_split_leakage_guard("test.txt", "test.txt", allow_same_split=False)

    # Explicit override should allow the same split path.
    validate_split_leakage_guard("test.txt", "test.txt", allow_same_split=True)
