from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neuralop.flood.serving.forcing import parse_forcing_csv
from neuralop.flood.serving.initial_conditions import (
    DryInitialConditionProvider,
    ForcingConditionedBaselineProvider,
    build_initial_condition_features,
)
from neuralop.flood.serving.model_bundle import load_model_bundle


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def _library(path: Path, *, n_refs: int = 6, n_cells: int = 4) -> None:
    stage = np.stack(
        [np.linspace(1.0 + 0.05 * i, 1.2 + 0.05 * i, 15, dtype=np.float32) for i in range(n_refs)],
        axis=0,
    )
    precip = np.stack(
        [np.linspace(0.0, 1.0 + 0.2 * i, 15, dtype=np.float32) for i in range(n_refs)],
        axis=0,
    )
    features = np.stack(
        [build_initial_condition_features(stage[i], precip[i], history_rows=15) for i in range(n_refs)],
        axis=0,
    )
    wd = np.zeros((n_refs, 3, n_cells, 1), dtype=np.float32)
    for i in range(n_refs):
        wd[i, :, :, 0] = (i + 1) * 0.01
    median = np.median(features, axis=0).astype(np.float32)
    iqr = (np.percentile(features, 75, axis=0) - np.percentile(features, 25, axis=0)).astype(np.float32)
    np.savez_compressed(
        path,
        reference_ids=np.asarray([f"TR{i:06d}" for i in range(n_refs)]),
        features=features.astype(np.float32),
        feature_median=median,
        feature_iqr=iqr,
        feature_names=np.asarray([f"f{i}" for i in range(features.shape[1])]),
        wd_history_m=wd,
        reference_distance_p95=np.asarray(1.0, dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "library_id": "ic-test-library",
                    "reference_scope": "train_calibration",
                    "history_rows": 15,
                    "n_history": 3,
                    "skip_before_timestep": 12,
                    "dt_seconds": 900,
                }
            )
        ),
    )


def _bundle(tmp_path: Path, *, with_library: bool = True):
    checkpoint_dirs = []
    for idx in range(3):
        d = tmp_path / f"ckpt_{idx}"
        _touch(d / "best_model_state_dict.pt")
        checkpoint_dirs.append(str(d))
    normalizer_path = tmp_path / "normalizers.pt"
    _touch(normalizer_path)
    coeff = tmp_path / "crps_mbm.json"
    coeff.write_text(json.dumps({"method": "identity"}), encoding="utf-8")
    iso = tmp_path / "isotonic.json"
    iso.write_text(json.dumps({"method": "identity"}), encoding="utf-8")
    static_files = []
    for idx in range(5):
        f = tmp_path / f"static_{idx}.txt"
        _touch(f)
        static_files.append(str(f))
    library_path = tmp_path / "initial_conditions" / "forcing_conditioned_initial_wd.npz"
    if with_library:
        library_path.parent.mkdir(parents=True, exist_ok=True)
        _library(library_path)
    manifest = {
        "bundle_id": "coastal-fgn-test",
        "domain_name": "coastal",
        "git_commit": "test",
        "checkpoint_dirs": checkpoint_dirs,
        "checkpoint_alias": "best_model",
        "normalizer_path": str(normalizer_path),
        "static_files": static_files,
        "calibration_coefficients_path": str(coeff),
        "isotonic_curves_path": str(iso),
        "boundary_channels": ["stage", "precipitation"],
        "dt_seconds": 900,
        "n_history": 3,
        "skip_before_timestep": 12,
        "max_forecast_steps": 6,
        "fgn_noise_dim": 32,
        "members_per_checkpoint": 20,
        "query_res": [4, 4],
        "initial_condition": {
            "default_mode": "forcing_conditioned_baseline",
            "library_path": "initial_conditions/forcing_conditioned_initial_wd.npz",
            "reference_scope": "train_calibration",
            "k_neighbors": 5,
        },
    }
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_model_bundle(manifest_path)


def _forcing_csv(*, stage_start: float = 1.0, precip_end: float = 1.0, n_rows: int = 24) -> str:
    lines = ["time_seconds,stage,precipitation"]
    for i in range(n_rows):
        frac = i / 14.0 if i < 15 else 1.0
        stage = stage_start + 0.2 * frac
        precip = precip_end * frac
        lines.append(f"{i * 900},{stage:.8f},{precip:.8f}")
    return "\n".join(lines)


def test_model_bundle_validates_initial_condition_library(tmp_path):
    bundle = _bundle(tmp_path, with_library=True)

    assert bundle.initial_condition.default_mode == "forcing_conditioned_baseline"
    assert bundle.initial_condition.library_path is not None


def test_model_bundle_rejects_missing_required_initial_condition_library(tmp_path):
    try:
        _bundle(tmp_path, with_library=False)
    except Exception as exc:
        assert "initial-condition library" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("Expected missing initial-condition library to fail validation.")


def test_forcing_conditioned_provider_exact_match_is_deterministic(tmp_path):
    bundle = _bundle(tmp_path)
    forcing = parse_forcing_csv(_forcing_csv(), bundle=bundle, requested_forecast_steps=4)
    provider = ForcingConditionedBaselineProvider.from_bundle(bundle)

    first = provider.resolve(forcing, bundle=bundle, n_cells=4)
    second = provider.resolve(forcing, bundle=bundle, n_cells=4)

    assert first.wd_history_m.shape == (3, 4, 1)
    assert np.all(first.wd_history_m >= 0.0)
    assert np.allclose(first.wd_history_m, second.wd_history_m)
    assert first.selection["selected_reference_ids"] == ["TR000000"]
    assert first.selection["weights"] == [1.0]


def test_forcing_conditioned_provider_blends_five_neighbors(tmp_path):
    bundle = _bundle(tmp_path)
    forcing = parse_forcing_csv(_forcing_csv(stage_start=1.075, precip_end=1.3), bundle=bundle, requested_forecast_steps=4)
    result = ForcingConditionedBaselineProvider.from_bundle(bundle).resolve(forcing, bundle=bundle, n_cells=4)

    assert len(result.selection["selected_reference_ids"]) == 5
    assert np.isclose(sum(result.selection["weights"]), 1.0)
    assert np.all(np.isfinite(result.wd_history_m))


def test_forcing_conditioned_provider_far_outside_reference_records_low_confidence(tmp_path):
    bundle = _bundle(tmp_path)
    forcing = parse_forcing_csv(_forcing_csv(stage_start=4.0, precip_end=100.0), bundle=bundle, requested_forecast_steps=4)
    result = ForcingConditionedBaselineProvider.from_bundle(bundle).resolve(forcing, bundle=bundle, n_cells=4)

    assert result.selection["low_confidence"] is True
    assert result.selection["confidence_label"] == "low"
    assert result.wd_history_m.shape == (3, 4, 1)


def test_dry_initial_condition_provider_returns_zero_history(tmp_path):
    bundle = _bundle(tmp_path)
    forcing = parse_forcing_csv(_forcing_csv(), bundle=bundle, requested_forecast_steps=4)
    result = DryInitialConditionProvider().resolve(forcing, bundle=bundle, n_cells=4)

    assert result.wd_history_m.shape == (3, 4, 1)
    assert np.all(result.wd_history_m == 0.0)
    assert result.selection["mode"] == "dry"
