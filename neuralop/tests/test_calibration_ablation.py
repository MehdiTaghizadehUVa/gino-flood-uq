import json

import numpy as np
import pytest

from neuralop.flood.eval.calibration_ablation import run_calibration_ablation
from neuralop.flood.eval.scientific_calibration import (
    COEFFICIENTS_JSON,
    load_crps_mbm_coefficients,
    save_forecast_artifact,
)


def _write_tiny_artifact(path, hydrograph_id, offset):
    pred = np.array(
        [
            [[0.00, 0.08, 0.18, 0.34], [0.02, 0.11, 0.22, 0.42]],
            [[0.01, 0.10, 0.20, 0.38], [0.03, 0.13, 0.26, 0.48]],
            [[0.02, 0.12, 0.24, 0.44], [0.05, 0.16, 0.30, 0.55]],
        ],
        dtype=np.float64,
    )
    pred = pred + float(offset)
    ref = 0.02 + 0.92 * pred
    geometry = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    return save_forecast_artifact(
        path,
        hydrograph_id=hydrograph_id,
        pred_members_wd=pred,
        ref_members_wd=ref,
        wettable_mask=np.ones(4, dtype=bool),
        structural_dry_mask=np.zeros(4, dtype=bool),
        geometry_raw=geometry,
        time_hours=[1.0, 2.0],
    )


def test_offline_calibration_ablation_writes_scoreboard_and_coefficients(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("scipy")
    calibration_artifact = _write_tiny_artifact(
        tmp_path / "calibration" / "TE000001.calibration_artifact.h5",
        "TE000001",
        0.0,
    )
    heldout_artifact = _write_tiny_artifact(
        tmp_path / "heldout" / "TE000101.calibration_artifact.h5",
        "TE000101",
        0.01,
    )
    out_dir = tmp_path / "ablation"
    config = {
        "rollout_calibration": {
            "bins": {
                "lead_time_hours": [0.0, 3.0],
                "wet_frequency_edges": [0.0, 1.0],
                "wet_threshold_m": 0.01,
            },
            "optimizer": {
                "min_fit_points_per_bin": 1,
                "max_fit_points_per_bin": 16,
                "multistart": False,
                "seed": 3,
                "mean_rmse_weight": 0.5,
                "spread_ratio_weight": 0.5,
                "target_spread_ratio": 1.0,
            },
            "exceedance": {"min_fit_points_per_bin": 1},
        },
        "rollout": {
            "impact_metrics": {
                "enabled": True,
                "inundation_threshold_m": 0.10,
                "pooled_radii_m": [1.5],
            }
        },
    }

    result = run_calibration_ablation(
        [calibration_artifact],
        [heldout_artifact],
        out_dir,
        config=config,
        thresholds_m=[0.05, 0.10, 0.30, 0.50],
        include_impact_metrics=True,
    )

    scoreboard = json.loads((out_dir / "calibration_ablation_scoreboard.json").read_text())
    variants = {row["variant"] for row in scoreboard}
    assert {
        "E0_raw",
        "E1_empirical_crps_mbm",
        "E2_tail_weighted_crps_mbm",
        "E3_mean_regularized_crps_mbm",
        "E4_spread_regularized_crps_mbm",
        "E5_combined_regularized_crps_mbm",
    }.issubset(variants)
    assert any(row["variant"].startswith("E6_") for row in scoreboard)
    assert (out_dir / "calibration_ablation_scoreboard.csv").exists()
    assert (out_dir / "selected_calibration_variant.json").exists()
    coeff_path = out_dir / "E1_empirical_crps_mbm" / COEFFICIENTS_JSON
    coeffs = load_crps_mbm_coefficients(coeff_path)
    assert coeffs["coefficients"].shape == (1, 1, 3)
    impact = json.loads((out_dir / "E1_empirical_crps_mbm" / "impact_metrics.json").read_text())
    assert "crps_total_inundated_area_fraction_wd_mean" in impact
    assert result["selected"]["coefficients_json"]
