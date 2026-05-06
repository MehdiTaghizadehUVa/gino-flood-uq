from types import SimpleNamespace

import numpy as np
import pytest

from neuralop.flood.eval.scientific_calibration import (
    CalibrationBins,
    apply_crps_mbm_to_wd_members,
    apply_member_by_member_transform,
    build_calibration_comparison,
    empirical_crps_mean,
    fit_crps_member_by_member_from_artifacts,
    isotonic_predict,
    pava_isotonic_fit,
    save_forecast_artifact,
    validate_reference_split_no_leakage,
    validate_scientific_calibration_config,
    _fit_one_bin,
)


def _manual_crps(forecast, reference):
    forecast = np.asarray(forecast, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    term1 = np.mean(np.abs(forecast[:, None, :] - reference[None, :, :]), axis=(0, 1))
    term2 = 0.5 * np.mean(np.abs(forecast[:, None, :] - forecast[None, :, :]), axis=(0, 1))
    return float(np.mean(term1 - term2))


def test_empirical_crps_matches_manual_small_array():
    forecast = np.array([[0.0, 2.0], [1.0, 3.0]])
    reference = np.array([[0.0, 4.0], [2.0, 2.0]])
    assert empirical_crps_mean(forecast, reference) == pytest.approx(_manual_crps(forecast, reference))


def test_member_by_member_transform_preserves_anomaly_ordering():
    pred = np.array([[0.2, 1.0], [0.4, 1.4], [0.6, 1.8]], dtype=np.float64)
    out = apply_member_by_member_transform(pred, a_m=0.1, beta=1.2, gamma=0.5)
    mu = pred.mean(axis=0, keepdims=True)
    expected = 0.1 + 1.2 * mu + 0.5 * (pred - mu)
    assert np.allclose(out, expected)
    assert np.all(np.diff(out[:, 0]) > 0)


def test_optimizer_improves_synthetic_crps_transform():
    rng = np.random.default_rng(7)
    pred = rng.uniform(0.05, 1.0, size=(6, 80))
    ref = apply_member_by_member_transform(pred, a_m=0.08, beta=1.15, gamma=0.65)
    ref = ref + rng.normal(0.0, 0.005, size=ref.shape)
    raw = empirical_crps_mean(pred, ref)
    fit = _fit_one_bin(
        pred,
        ref,
        bounds={"a_m": [-0.5, 0.5], "beta": [0.5, 1.8], "gamma": [0.2, 1.5]},
    )
    cal = apply_member_by_member_transform(
        pred, a_m=fit["a_m"], beta=fit["beta"], gamma=fit["gamma"]
    )
    assert empirical_crps_mean(cal, ref) < raw
    assert fit["a_m"] == pytest.approx(0.08, abs=0.02)
    assert fit["beta"] == pytest.approx(1.15, abs=0.05)
    assert fit["gamma"] == pytest.approx(0.65, abs=0.05)
    assert fit["n_points"] == 80


def test_apply_crps_mbm_zeroes_structural_dry_cells():
    model = {
        "coefficients": np.array([[[0.0, 1.0, 1.0], [0.1, 1.2, 0.8]]]),
        "lead_time_hours": np.array([0.0, np.inf]),
        "wet_frequency_edges": np.array([0.0, 0.5, 1.0]),
        "wet_threshold_m": 0.01,
        "wet_frequency_by_cell": np.array([0.2, 0.8, 0.9]),
    }
    pred = np.array([[0.2, 0.3, 9.0], [0.4, 0.5, 8.0]])
    out = apply_crps_mbm_to_wd_members(
        pred, lead_time_hour=1.0, calibration_model=model, wettable_mask=np.array([True, True, False])
    )
    assert np.all(out[:, 2] == 0.0)
    assert np.all(out[:, :2] >= 0.0)


def test_pava_isotonic_returns_monotone_predictions():
    raw = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    obs = np.array([0.05, 0.45, 0.30, 0.70, 0.65])
    model = pava_isotonic_fit(raw, obs)
    pred = isotonic_predict(model, raw)
    assert np.all(np.diff(pred) >= -1e-12)
    assert np.all((pred >= 0.0) & (pred <= 1.0))


def test_scientific_config_rejects_legacy_affine_keys():
    cfg = SimpleNamespace(
        rollout_calibration=SimpleNamespace(enabled=True, calib_txt="calib.txt", fit_wet_threshold=0.01)
    )
    with pytest.raises(ValueError, match="Legacy lead-time affine"):
        validate_scientific_calibration_config(cfg)


def test_reference_leakage_guard_rejects_same_family(tmp_path):
    calib_root = tmp_path / "calib"
    test_root = tmp_path / "test"
    calib_root.mkdir()
    test_root.mkdir()
    (calib_root / "calib.txt").write_text("TE000001_sim00\n", encoding="utf-8")
    (test_root / "test.txt").write_text("TE000001_sim99\n", encoding="utf-8")
    cfg = SimpleNamespace(
        rollout_calibration=SimpleNamespace(
            enabled=True,
            method="crps_member_by_member",
            reference=SimpleNamespace(
                calibration_root=str(calib_root),
                calibration_txt="calib.txt",
                test_root=str(test_root),
                test_txt="test.txt",
            ),
            forecast_artifacts=SimpleNamespace(format="hdf5_per_family"),
        )
    )
    validate_scientific_calibration_config(cfg)
    with pytest.raises(ValueError, match="overlapping hydrograph"):
        validate_reference_split_no_leakage(cfg)


def test_build_calibration_comparison_reports_delta():
    comp = build_calibration_comparison({"crps": 2.0, "label": "raw"}, {"crps": 1.5, "label": "cal"})
    assert comp["metrics"]["crps"]["delta"] == pytest.approx(-0.5)
    assert comp["metrics"]["crps"]["percent_change"] == pytest.approx(-25.0)


def test_artifact_fit_path_smoke(tmp_path):
    h5py = pytest.importorskip("h5py")
    del h5py
    rng = np.random.default_rng(11)
    pred = rng.uniform(0.0, 0.5, size=(4, 3, 6))
    ref = apply_member_by_member_transform(pred.reshape(4, -1), a_m=0.02, beta=1.1, gamma=0.8).reshape(4, 3, 6)
    art = save_forecast_artifact(
        tmp_path / "TE000001.calibration_artifact.h5",
        hydrograph_id="TE000001",
        pred_members_wd=pred,
        ref_members_wd=ref,
        wettable_mask=np.array([True, True, True, False, True, True]),
        structural_dry_mask=np.array([False, False, False, True, False, False]),
        time_hours=[1.0, 2.0, 3.0],
    )
    bins = CalibrationBins(
        lead_time_hours=(0.0, 2.0, np.inf),
        wet_frequency_edges=(0.0, 0.05, 0.5, 1.0),
        wet_threshold_m=0.01,
    )
    model = fit_crps_member_by_member_from_artifacts(
        [art], bins=bins, max_fit_points_per_bin=1000, min_fit_points_per_bin=2, seed=5
    )
    assert model["coefficients"].shape == (2, 3, 3)
    assert model["wet_frequency_by_cell"].shape == (6,)
