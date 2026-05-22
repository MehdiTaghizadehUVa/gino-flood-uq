import json

import numpy as np
import pytest

from neuralop.flood.eval import artifact_visualization as av
from neuralop.flood.eval.scientific_calibration import (
    load_forecast_artifact,
    save_forecast_artifact,
)


def _write_identityish_coefficients(path, n_cells, *, beta=2.0):
    path.write_text(
        json.dumps(
            {
                "coefficients": [[[0.0, beta, 1.0]]],
                "lead_time_hours": [0.0, 9999.0],
                "wet_frequency_edges": [0.0, 1.0],
                "wet_threshold_m": 0.01,
                "wet_frequency_by_cell": [1.0] * n_cells,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_artifact_visual_fields_apply_calibration_before_rendering():
    pred = np.array(
        [
            [[0.1, 0.2, 0.0], [0.2, 0.4, 0.0]],
            [[0.3, 0.4, 0.0], [0.4, 0.6, 0.0]],
        ],
        dtype=np.float64,
    )
    ref = pred + 0.05
    model = {
        "coefficients": np.array([[[0.0, 2.0, 1.0]]]),
        "lead_time_hours": np.array([0.0, 9999.0]),
        "wet_frequency_edges": np.array([0.0, 1.0]),
        "wet_threshold_m": 0.01,
        "wet_frequency_by_cell": np.ones(3),
    }

    fields = av.build_visual_fields_from_artifact(
        {
            "pred_members_wd": pred,
            "ref_members_wd": ref,
            "wettable_mask": np.array([True, True, False]),
            "time_hours": np.array([1.0, 2.0]),
        },
        calibration_model=model,
    )

    raw_mean = pred.mean(axis=0)
    calibrated_mean = fields["pred_mean_by_channel"]["wd"]
    assert np.allclose(calibrated_mean[:, :2], 2.0 * raw_mean[:, :2])
    assert np.all(calibrated_mean[:, 2] == 0.0)
    assert fields["crps_map_wd"].shape == (2, 3)
    assert fields["relative_l2_by_channel"]["wd"].shape == (2,)


def test_render_calibrated_artifacts_uses_saved_hdf_and_writes_manifest(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    del h5py
    pred = np.array(
        [
            [[0.1, 0.2, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0]],
            [[0.3, 0.4, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    ref = pred + 0.02
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    save_forecast_artifact(
        artifact_root / "TE000001.calibration_artifact.h5",
        hydrograph_id="TE000001",
        pred_members_wd=pred,
        ref_members_wd=ref,
        wettable_mask=np.array([True, True, True, False]),
        structural_dry_mask=np.array([False, False, False, True]),
        boundary_series_raw=np.zeros((4, 2)),
        boundary_ensemble_series_raw=np.stack(
            [
                np.column_stack([np.arange(4), np.arange(4) * 0.1]),
                np.column_stack([np.arange(4) + 10.0, np.arange(4) * 0.2]),
            ],
            axis=0,
        ),
        boundary_channel_names=["stage", "precipitation"],
        time_hours=[1.0, 2.0],
        geometry_raw=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        elevation_raw=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    coeff_path = _write_identityish_coefficients(tmp_path / "coefficients.json", n_cells=4, beta=2.0)
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(av, "_save_hydrograph_uq_figures_and_animation", fake_render)
    manifest = av.render_calibrated_artifact_visuals(
        artifact_root=artifact_root,
        coefficient_path=coeff_path,
        out_dir=tmp_path / "visuals",
        visualization_config={"map": {"enabled": False}, "output": {"write_gif": False, "write_mp4": False}},
    )

    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["n_artifacts"] == 1
    assert "overall_metrics" in payload
    assert (tmp_path / "visuals" / "artifact_raw_uq_overall_metrics.json").exists()
    assert (tmp_path / "visuals" / "artifact_calibrated_uq_overall_metrics.json").exists()
    assert (tmp_path / "visuals" / "artifact_calibration_comparison.json").exists()
    assert payload["hydrographs"][0]["calibration_applied"] is True
    assert captured["hydrograph_id"] == "TE000001"
    assert captured["target_variables"] == ["wd"]
    assert captured["boundary_ensemble_series_raw"].shape == (2, 4, 2)
    assert np.allclose(captured["elevation_raw"], [1.0, 2.0, 3.0, 4.0])
    assert captured["crps_map_wd"].shape == (2, 4)
    assert np.allclose(captured["pred_mean_by_channel"]["wd"][:, :3], 2.0 * pred.mean(axis=0)[:, :3])


def test_artifact_visualization_reconstructs_boundary_ensemble_from_config(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    del h5py
    boundary_root = tmp_path / "boundary"
    boundary_root.mkdir()
    split_path = tmp_path / "test.txt"
    split_path.write_text(
        "Flood_coastal_TE000001_sim00\nFlood_coastal_TE000001_sim01\n",
        encoding="utf-8",
    )
    (boundary_root / "Stage_Hydrographs_Test_Clean.txt").write_text(
        "Flood_coastal_TE000001\n0\n0\n0\n0\n",
        encoding="utf-8",
    )
    (boundary_root / "Stage_Hydrographs_Test.txt").write_text(
        "Flood_coastal_TE000001_sim00\tFlood_coastal_TE000001_sim01\n"
        "1\t10\n2\t20\n3\t30\n4\t40\n",
        encoding="utf-8",
    )
    artifact = save_forecast_artifact(
        tmp_path / "Flood_coastal_TE000001.calibration_artifact.h5",
        hydrograph_id="Flood_coastal_TE000001",
        pred_members_wd=np.zeros((2, 2, 4)),
        ref_members_wd=np.zeros((2, 2, 4)),
        wettable_mask=np.ones(4, dtype=bool),
        boundary_series_raw=np.zeros((4, 1), dtype=np.float32),
        boundary_channel_names=["stage"],
        time_hours=[1.0, 2.0],
        geometry_raw=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        metadata={"artifact_role": "heldout_test_raw", "test_txt": str(split_path)},
    )
    coeff_path = _write_identityish_coefficients(tmp_path / "coefficients.json", n_cells=4, beta=1.0)
    model = json.loads(coeff_path.read_text(encoding="utf-8"))
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(av, "_save_hydrograph_uq_figures_and_animation", fake_render)
    result = av.render_artifact_visualization(
        artifact,
        out_dir=tmp_path / "visuals",
        calibration_model=model,
        visualization_config={"map": {"enabled": False}, "output": {"write_gif": False, "write_mp4": False}},
        eval_config={
            "flood": {
                "rollout_calibration": {
                    "boundary": {
                        "channels": [
                            {
                                "name": "stage",
                                "mode": "clean_family",
                                "clean_boundary_root": str(boundary_root),
                                "clean_boundary_file": "Stage_Hydrographs_Test_Clean.txt",
                            }
                        ]
                    },
                    "reference": {"test_txt": str(split_path)},
                }
            }
        },
    )

    assert result["boundary_ensemble_source"] == "reconstructed"
    assert captured["boundary_ensemble_series_raw"].shape == (2, 4, 1)
    assert np.allclose(captured["boundary_ensemble_series_raw"][:, :, 0], [[1, 2, 3, 4], [10, 20, 30, 40]])


def test_forecast_artifact_round_trips_optional_elevation(tmp_path):
    h5py = pytest.importorskip("h5py")
    del h5py
    artifact = save_forecast_artifact(
        tmp_path / "with_elevation.calibration_artifact.h5",
        hydrograph_id="with_elevation",
        pred_members_wd=np.zeros((2, 1, 3)),
        ref_members_wd=np.zeros((2, 1, 3)),
        geometry_raw=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        elevation_raw=np.array([4.0, 5.0, 6.0]),
        time_hours=[1.0],
    )
    loaded = load_forecast_artifact(artifact, load_members=False)
    assert np.allclose(loaded["elevation_raw"], [4.0, 5.0, 6.0])


def test_forecast_artifact_round_trips_boundary_ensemble(tmp_path):
    h5py = pytest.importorskip("h5py")
    del h5py
    boundary_ensemble = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    artifact = save_forecast_artifact(
        tmp_path / "with_boundary_ensemble.calibration_artifact.h5",
        hydrograph_id="with_boundary_ensemble",
        pred_members_wd=np.zeros((2, 1, 3)),
        ref_members_wd=np.zeros((3, 1, 3)),
        boundary_series_raw=np.zeros((4, 2), dtype=np.float32),
        boundary_ensemble_series_raw=boundary_ensemble,
        time_hours=[1.0],
    )

    loaded = load_forecast_artifact(artifact, load_members=False)

    assert np.allclose(loaded["boundary_ensemble_series_raw"], boundary_ensemble)


def test_load_elevation_values_aligns_hecras_full_cells_to_cell_points(tmp_path):
    h5py = pytest.importorskip("h5py")
    hdf_path = tmp_path / "hecras.hdf"
    elevation_dataset = "Geometry/2D Flow Areas/Portsmouth/Cells Minimum Elevation"
    with h5py.File(hdf_path, "w") as handle:
        handle.create_dataset(
            "Geometry/2D Flow Areas/Cell Points",
            data=np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        )
        handle.create_dataset(
            "Geometry/2D Flow Areas/Portsmouth/Cells Center Coordinate",
            data=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64),
        )
        handle.create_dataset(
            elevation_dataset,
            data=np.array([10.0, 11.0, 12.0], dtype=np.float32),
        )

    aligned = av.load_elevation_values(hdf_path, dataset=elevation_dataset)

    assert np.allclose(aligned, [12.0, 10.0])
