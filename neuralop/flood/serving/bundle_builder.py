"""Build validated coastal FGN serving bundles from evaluated artifacts.

This CLI bridges research batch outputs and the serving scientific contract. It
exports fixed-domain tensors once, records mesh provenance, and writes a
versioned manifest consumed by the web worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neuralop.flood.data.datasets_impl import _align_static_text_to_reference_cells
from neuralop.flood.data.hec_ras import (
    HDF_PATHS,
    build_cell_point_index,
    get_hec_ras_hdf_shape,
    h5py,
    read_hec_ras_hdf_static,
)
from neuralop.flood.serving.initial_conditions import (
    DEFAULT_INITIAL_CONDITION_LIBRARY,
    build_initial_condition_features,
    initial_condition_feature_names,
)
from neuralop.flood.serving.model_bundle import load_model_bundle
from neuralop.flood.serving.monitoring_build_bundle import load_clean_boundary_table_forcings


DEFAULT_STATIC_FILES = [
    "Coastal_Slope.txt",
    "Coastal_Aspect.txt",
    "Coastal_FlowDirection.txt",
    "Coastal_Curvature.txt",
    "Coastal_FlowAccumulation.txt",
]


def _read_mapping(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            return json.load(handle)
        from ruamel.yaml import YAML

        loaded = YAML(typ="safe").load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Expected mapping in {path}, got {type(loaded).__name__}.")
    return loaded


def _get(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _merged_hdf_paths(section: Mapping[str, Any]) -> dict[str, str]:
    paths = dict(HDF_PATHS)
    override = section.get("hdf_paths") or {}
    if not isinstance(override, Mapping):
        raise ValueError("hdf_paths must be a mapping.")
    for key, value in override.items():
        paths[str(key)] = str(value)
    return paths


def _split_path(root: Path, split_txt: str | Path) -> Path:
    split = Path(str(split_txt)).expanduser()
    return split if split.is_absolute() else root / split


def _read_run_ids(root: Path, split_txt: str | Path) -> list[str]:
    path = _split_path(root, split_txt)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) == 1 and "," in lines[0]:
        return [token.strip() for token in lines[0].split(",") if token.strip()]
    return lines


def _first_existing_hdf(root: Path, run_ids: Sequence[str], suffix: str = ".hdf") -> Path:
    for run_id in run_ids:
        path = root / f"{run_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No HDF files found under {root} for the provided split.")


def _read_static_text(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(str(path), delimiter="\t", dtype=np.float32)
    except ValueError:
        arr = np.loadtxt(str(path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Static text file {path} must be 1D or 2D, got {arr.shape}.")
    return np.asarray(arr, dtype=np.float32)


def _stable_arrays_hash(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        contiguous = np.ascontiguousarray(arr)
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _copy_file(src: str | Path, dst_dir: Path) -> str:
    src_path = Path(src).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src_path.name
    if src_path != dst:
        shutil.copy2(src_path, dst)
    return str(dst.relative_to(dst_dir.parent))


def _discover_checkpoint_dirs(checkpoint_root: Path, expected: int = 3) -> list[Path]:
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")
    dirs = sorted(p for p in checkpoint_root.iterdir() if p.is_dir())
    valid = [p for p in dirs if (p / "best_model_state_dict.pt").exists()]
    if len(valid) != expected:
        names = ", ".join(str(p) for p in valid)
        raise ValueError(
            f"Expected {expected} best-model checkpoint dirs under {checkpoint_root}, "
            f"found {len(valid)}: {names}"
        )
    return valid


def _export_domain_assets(
    *,
    data_root: Path,
    split_txt: str | Path,
    static_text_files: Sequence[str],
    hdf_paths: Mapping[str, str],
    output_dir: Path,
    skip_before_timestep: int,
    n_history: int,
) -> dict[str, Any]:
    if h5py is None:
        raise ImportError("h5py is required to build a serving model bundle.")
    run_ids = _read_run_ids(data_root, split_txt)
    hdf_path = _first_existing_hdf(data_root, run_ids)
    cell_point_index = build_cell_point_index(hdf_path, dict(hdf_paths))
    n_cells, n_time = get_hec_ras_hdf_shape(hdf_path, dict(hdf_paths))
    with h5py.File(hdf_path, "r") as handle:
        geometry = np.asarray(handle[hdf_paths["geometry"]][:], dtype=np.float32)
    if int(geometry.shape[0]) != int(n_cells):
        raise ValueError(f"Geometry row count {geometry.shape[0]} does not match HDF cell-point count {n_cells}.")
    elev_full, area_full = read_hec_ras_hdf_static(hdf_path, dict(hdf_paths), cell_index=None)
    full_cell_count = int(elev_full.shape[0])
    static_parts = [elev_full[cell_point_index].reshape(-1, 1), area_full[cell_point_index].reshape(-1, 1)]
    copied_static_files: list[str] = []
    for name in static_text_files:
        source = Path(str(name)).expanduser()
        source = source if source.is_absolute() else data_root / source
        if not source.exists():
            raise FileNotFoundError(f"Static feature file not found: {source}")
        arr = _read_static_text(source)
        static_parts.append(
            _align_static_text_to_reference_cells(
                arr,
                reference_cell_count=len(cell_point_index),
                cell_point_index=cell_point_index,
                full_cell_count=full_cell_count,
                source=str(source),
            )
        )
        copied_static_files.append(_copy_file(source, output_dir / "source_static"))
    static = np.concatenate(static_parts, axis=1).astype(np.float32, copy=False)
    if static.shape[0] != geometry.shape[0]:
        raise ValueError(f"Static shape {static.shape} does not align with geometry shape {geometry.shape}.")
    if static.shape[1] != 2 + len(static_text_files):
        raise ValueError(f"Expected {2 + len(static_text_files)} static channels, got {static.shape[1]}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = output_dir / "domain" / "geometry.npy"
    static_path = output_dir / "domain" / "static.npy"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(geometry_path, geometry.astype(np.float32, copy=False))
    np.save(static_path, static)
    max_forecast_steps = int(n_time) - int(skip_before_timestep) - int(n_history)
    if max_forecast_steps < 1:
        raise ValueError(
            f"Reference HDF has n_time={n_time}; cannot support "
            f"skip={skip_before_timestep}, n_history={n_history}."
        )
    return {
        "geometry_path": str(geometry_path.relative_to(output_dir)),
        "static_tensor_path": str(static_path.relative_to(output_dir)),
        "static_files": copied_static_files,
        "mesh_hash": _stable_arrays_hash([geometry, static]),
        "n_cells": int(geometry.shape[0]),
        "n_static": int(static.shape[1]),
        "reference_hdf": str(hdf_path),
        "reference_run_id": str(hdf_path.stem),
        "max_forecast_steps": max_forecast_steps,
        "n_time": int(n_time),
        "cell_point_index_hash": _stable_arrays_hash([cell_point_index.astype(np.int64)]),
    }


def _clean_boundary_table_paths(boundary_spec: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    by_name = {str(item.get("name")): item for item in boundary_spec if isinstance(item, Mapping)}
    stage = by_name.get("stage")
    precip = by_name.get("precipitation")
    if not stage or not precip:
        raise ValueError("Initial-condition library requires clean-family stage and precipitation boundary channels.")
    paths = []
    for channel in (stage, precip):
        if channel.get("mode") != "clean_family":
            raise ValueError(
                "Initial-condition library currently supports clean_family boundary channels only. "
                f"Got {channel.get('name')!r} mode={channel.get('mode')!r}."
            )
        root = Path(str(channel["clean_boundary_root"])).expanduser()
        paths.append(root / str(channel["clean_boundary_file"]))
    return paths[0], paths[1]


def _reference_distance_p95(features: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> float:
    safe_iqr = np.where(np.abs(iqr) > 1.0e-6, iqr, 1.0).astype(np.float32)
    scaled = (features.astype(np.float32) - median.astype(np.float32)) / safe_iqr
    if scaled.shape[0] < 2:
        return 1.0
    nearest = []
    for idx in range(scaled.shape[0]):
        diff = scaled - scaled[idx : idx + 1]
        distances = np.linalg.norm(diff, axis=1) / max(1.0, np.sqrt(scaled.shape[1]))
        distances[idx] = np.inf
        nearest.append(float(np.min(distances)))
    value = float(np.percentile(np.asarray(nearest, dtype=np.float32), 95))
    return value if value > 0.0 and np.isfinite(value) else 1.0


def _build_initial_condition_library(
    *,
    reference_root: Path,
    reference_split_txt: str | Path,
    boundary_spec: Sequence[Mapping[str, Any]],
    hdf_paths: Mapping[str, str],
    output_dir: Path,
    dt_seconds: int,
    skip_before_timestep: int,
    n_history: int,
    reference_scope: str,
    bundle_id: str,
) -> dict[str, Any]:
    if h5py is None:
        raise ImportError("h5py is required to build an initial-condition library.")
    run_ids = _read_run_ids(reference_root, reference_split_txt)
    if not run_ids:
        raise ValueError("Initial-condition reference split is empty.")
    stage_table, precipitation_table = _clean_boundary_table_paths(boundary_spec)
    forcings = load_clean_boundary_table_forcings(
        stage_table=stage_table,
        precipitation_table=precipitation_table,
        dt_seconds=int(dt_seconds),
        skip_before_timestep=int(skip_before_timestep),
        n_history=int(n_history),
        max_forecast_steps=1,
    )
    forcing_by_id = {item.source_name.split(".")[0]: item for item in forcings}
    first_hdf = _first_existing_hdf(reference_root, run_ids)
    cell_point_index = build_cell_point_index(first_hdf, dict(hdf_paths))
    history_rows = int(skip_before_timestep) + int(n_history)
    features: list[np.ndarray] = []
    wd_histories: list[np.ndarray] = []
    selected_ids: list[str] = []
    for run_id in run_ids:
        forcing = forcing_by_id.get(run_id)
        if forcing is None:
            raise ValueError(f"Boundary forcing table is missing reference event {run_id}.")
        hdf_path = reference_root / f"{run_id}.hdf"
        if not hdf_path.exists():
            raise FileNotFoundError(f"Initial-condition reference HDF not found: {hdf_path}")
        if forcing.n_rows < history_rows:
            raise ValueError(f"Reference forcing {run_id} has too few rows for initial history.")
        with h5py.File(hdf_path, "r") as handle:
            wd_full = np.asarray(
                handle[hdf_paths["wd"]][
                    int(skip_before_timestep) : int(skip_before_timestep) + int(n_history),
                    :,
                ],
                dtype=np.float32,
            )
        wd = np.clip(wd_full[:, cell_point_index], 0.0, None).astype(np.float32, copy=False)
        wd_histories.append(wd[..., None])
        features.append(
            build_initial_condition_features(
                forcing.stage[:history_rows],
                forcing.precipitation[:history_rows],
                history_rows=history_rows,
            )
        )
        selected_ids.append(str(run_id))
    feature_matrix = np.stack(features, axis=0).astype(np.float32, copy=False)
    wd_history = np.stack(wd_histories, axis=0).astype(np.float32, copy=False)
    median = np.median(feature_matrix, axis=0).astype(np.float32)
    iqr = (np.percentile(feature_matrix, 75, axis=0) - np.percentile(feature_matrix, 25, axis=0)).astype(np.float32)
    relative_path = Path(DEFAULT_INITIAL_CONDITION_LIBRARY)
    out_path = output_dir / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "library_id": f"{bundle_id}-initial-condition-{reference_scope}",
        "bundle_id": bundle_id,
        "reference_scope": reference_scope,
        "source_root": str(reference_root),
        "source_split_txt": str(_split_path(reference_root, reference_split_txt)),
        "stage_table": str(stage_table),
        "precipitation_table": str(precipitation_table),
        "dt_seconds": int(dt_seconds),
        "skip_before_timestep": int(skip_before_timestep),
        "n_history": int(n_history),
        "history_rows": history_rows,
        "n_reference": int(len(selected_ids)),
        "mesh_hash": _stable_arrays_hash([cell_point_index.astype(np.int64)]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    np.savez_compressed(
        out_path,
        reference_ids=np.asarray(selected_ids),
        features=feature_matrix,
        feature_median=median,
        feature_iqr=iqr,
        feature_names=np.asarray(initial_condition_feature_names(history_rows=history_rows)),
        wd_history_m=wd_history,
        reference_distance_p95=np.asarray(_reference_distance_p95(feature_matrix, median, iqr), dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "library_path": str(relative_path),
        "reference_scope": reference_scope,
        "n_reference": int(len(selected_ids)),
        "metadata": metadata,
    }


def build_bundle(args: argparse.Namespace) -> Path:
    config_path = Path(args.config_path).expanduser().resolve()
    config = _read_mapping(config_path)
    flood = _get(config, ["flood"], config)
    data = _get(flood, ["rollout_data"], _get(flood, ["data"], {}))
    train_data = _get(flood, ["data"], {})
    gino = _get(flood, ["gino"], {})
    checkpoint = _get(flood, ["checkpoint"], {})
    rollout = _get(flood, ["rollout"], {})
    data_root = Path(args.data_root or data.get("root") or train_data.get("root")).expanduser().resolve()
    split_txt = args.split_txt or data.get("test_txt") or train_data.get("train_txt") or "test.txt"
    hdf_paths = _merged_hdf_paths(data if isinstance(data, Mapping) else {})
    static_text_files = (
        args.static_text_files
        or data.get("static_text_files")
        or train_data.get("static_text_files")
        or DEFAULT_STATIC_FILES
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_before_timestep = int(
        args.skip_before_timestep if args.skip_before_timestep is not None else train_data.get("skip_before_timestep", 12)
    )
    n_history = int(args.n_history if args.n_history is not None else train_data.get("n_history", 3))
    domain = _export_domain_assets(
        data_root=data_root,
        split_txt=split_txt,
        static_text_files=[str(x) for x in static_text_files],
        hdf_paths=hdf_paths,
        output_dir=output_dir,
        skip_before_timestep=skip_before_timestep,
        n_history=n_history,
    )
    checkpoint_root = Path(args.checkpoint_root or checkpoint.get("save_dir") or checkpoint.get("resume_from_dir")).expanduser().resolve()
    checkpoint_dirs = _discover_checkpoint_dirs(checkpoint_root, expected=int(args.expected_checkpoints))
    normalizer_path = _copy_file(args.normalizer_path or train_data.get("normalizer_path"), output_dir / "normalizers")
    coeff_path = _copy_file(args.calibration_coefficients_path, output_dir / "calibration")
    isotonic_path = _copy_file(args.isotonic_curves_path, output_dir / "calibration")
    config_copy = _copy_file(config_path, output_dir / "source_config")
    max_steps = int(args.max_forecast_steps or domain["max_forecast_steps"])
    if max_steps > int(domain["max_forecast_steps"]):
        raise ValueError(f"Requested max_forecast_steps={max_steps} exceeds domain support {domain['max_forecast_steps']}.")
    dt_seconds = int(args.dt_seconds or train_data.get("dt", 900))
    initial_condition_library = None
    if args.build_initial_condition_library:
        reference_root = Path(
            args.initial_condition_root
            or _get(flood, ["rollout_calibration", "reference", "calibration_root"])
            or data_root
        ).expanduser().resolve()
        reference_split_txt = (
            args.initial_condition_split_txt
            or _get(flood, ["rollout_calibration", "reference", "calibration_txt"])
            or split_txt
        )
        boundary_spec = (
            _get(flood, ["rollout_data", "boundary"], None)
            or _get(flood, ["data", "boundary"], None)
            or data.get("boundary")
            or train_data.get("boundary")
        )
        if not isinstance(boundary_spec, Sequence):
            raise ValueError("Could not find a clean-family boundary spec for initial-condition library build.")
        initial_condition_library = _build_initial_condition_library(
            reference_root=reference_root,
            reference_split_txt=reference_split_txt,
            boundary_spec=boundary_spec,
            hdf_paths=hdf_paths,
            output_dir=output_dir,
            dt_seconds=dt_seconds,
            skip_before_timestep=skip_before_timestep,
            n_history=n_history,
            reference_scope=args.initial_condition_reference_scope,
            bundle_id=args.bundle_id,
        )
    manifest = {
        "bundle_id": args.bundle_id,
        "domain_name": "coastal",
        "git_commit": args.git_commit,
        "checkpoint_dirs": [str(p) for p in checkpoint_dirs],
        "checkpoint_alias": "best_model",
        "normalizer_path": normalizer_path,
        "static_files": domain["static_files"],
        "calibration_coefficients_path": coeff_path,
        "isotonic_curves_path": isotonic_path,
        "boundary_channels": ["stage", "precipitation"],
        "dt_seconds": dt_seconds,
        "n_history": n_history,
        "skip_before_timestep": skip_before_timestep,
        "max_forecast_steps": max_steps,
        "fgn_noise_dim": int(args.fgn_noise_dim or gino.get("fgn_noise_dim", 32)),
        "members_per_checkpoint": int(args.members_per_checkpoint or rollout.get("n_ensemble_samples_per_model", 20)),
        "crs": args.crs,
        "mesh_hash": domain["mesh_hash"],
        "expected_mesh_hash": domain["mesh_hash"],
        "geometry_path": domain["geometry_path"],
        "static_tensor_path": domain["static_tensor_path"],
        "query_res": [int(x) for x in (args.query_res or train_data.get("query_res") or [48, 48])],
        "research_disclaimer": "Model outputs include documented validation scope, provenance, and governance requirements.",
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_config": config_copy,
            "source_data_root": str(data_root),
            "source_split_txt": str(_split_path(data_root, split_txt)),
            "source_checkpoint_root": str(checkpoint_root),
            "source_normalizer_path": str(args.normalizer_path or train_data.get("normalizer_path")),
            "source_calibration_coefficients_path": str(args.calibration_coefficients_path),
            "source_isotonic_curves_path": str(args.isotonic_curves_path),
            "hdf_paths": dict(hdf_paths),
            "n_cells": domain["n_cells"],
            "n_static": domain["n_static"],
            "reference_hdf": domain["reference_hdf"],
            "reference_run_id": domain["reference_run_id"],
            "reference_n_time": domain["n_time"],
            "cell_point_index_hash": domain["cell_point_index_hash"],
            "structural_dry_policy": _get(flood, ["structural_dry", "policy"], "unknown"),
        },
    }
    if initial_condition_library is not None:
        manifest["initial_condition"] = {
            "default_mode": "forcing_conditioned_baseline",
            "library_path": initial_condition_library["library_path"],
            "reference_scope": initial_condition_library["reference_scope"],
            "k_neighbors": int(args.initial_condition_k_neighbors),
            "metadata": {
                "n_reference": int(initial_condition_library["n_reference"]),
                "library_id": initial_condition_library["metadata"]["library_id"],
            },
        }
    if args.structural_dry_mask_path:
        manifest["structural_dry_mask_path"] = str(Path(args.structural_dry_mask_path).expanduser().resolve())
    manifest_path = output_dir / "coastal_fgn_bundle.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    load_model_bundle(manifest_path, validate_paths=True)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a deployable coastal FGN serving bundle.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--calibration-coefficients-path", required=True)
    parser.add_argument("--isotonic-curves-path", required=True)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--normalizer-path")
    parser.add_argument("--data-root")
    parser.add_argument("--split-txt")
    parser.add_argument("--static-text-files", nargs="+")
    parser.add_argument("--structural-dry-mask-path")
    parser.add_argument("--max-forecast-steps", type=int)
    parser.add_argument("--skip-before-timestep", type=int)
    parser.add_argument("--n-history", type=int)
    parser.add_argument("--dt-seconds", type=int)
    parser.add_argument("--fgn-noise-dim", type=int)
    parser.add_argument("--members-per-checkpoint", type=int)
    parser.add_argument("--expected-checkpoints", type=int, default=3)
    parser.add_argument("--query-res", nargs=2, type=int)
    parser.add_argument("--crs", default="EPSG:32618")
    parser.add_argument(
        "--build-initial-condition-library",
        action="store_true",
        help="Build a forcing-conditioned initial WD library and make it the bundle default.",
    )
    parser.add_argument("--initial-condition-root")
    parser.add_argument("--initial-condition-split-txt")
    parser.add_argument("--initial-condition-reference-scope", default="train_calibration")
    parser.add_argument("--initial-condition-k-neighbors", type=int, default=5)
    args = parser.parse_args(argv)
    print(str(build_bundle(args)))


if __name__ == "__main__":  # pragma: no cover
    main()
