"""Render calibrated rollout visualizations directly from forecast artifacts.

This module intentionally avoids model execution. It renders post-calibration
figures/animations from the HDF5 artifacts that are also used for calibrated
UQ metrics, so visual diagnostics and reported scores share the exact same
forecast-member realization.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from neuralop.flood.eval.datasets import _boundary_ensemble_series_from_reference_members
from neuralop.flood.eval.render import _save_hydrograph_uq_figures_and_animation
from neuralop.flood.eval.runtime import UQ_EXCEEDANCE_THRESHOLD
from neuralop.flood.eval.scientific_calibration import (
    apply_crps_mbm_to_wd_members,
    build_calibration_comparison,
    compute_artifact_uq_metrics,
    empirical_crps_per_location,
    list_forecast_artifacts,
    load_crps_mbm_coefficients,
    load_forecast_artifact,
    save_metrics_json,
)

LOGGER = logging.getLogger("flood_artifact_visualization")
MIN_EPS = 1e-12


def _load_json_or_yaml(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without PyYAML
        raise RuntimeError(
            f"Cannot read YAML config {path}; install PyYAML or pass JSON."
        ) from exc
    data = yaml.safe_load(text)
    return data if isinstance(data, Mapping) else {}


def load_visualization_config(path: str | Path | None) -> Dict[str, Any]:
    """Load a visualization block from a full eval config or standalone file."""
    if path is None:
        return {}
    payload = dict(_load_json_or_yaml(path))
    viz = payload.get("visualization", payload)
    if not isinstance(viz, Mapping):
        raise ValueError(f"Visualization config at {path} must be a mapping.")
    return dict(viz)


def load_eval_config(path: str | Path | None) -> Dict[str, Any]:
    """Load the full evaluation config when available for artifact reconstruction."""
    if path is None:
        return {}
    return dict(_load_json_or_yaml(path))


def _config_root(eval_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(eval_config, Mapping):
        return {}
    flood = eval_config.get("flood")
    return flood if isinstance(flood, Mapping) else eval_config


def _mapping_get(obj: Mapping[str, Any] | None, key: str) -> Any:
    return obj.get(key) if isinstance(obj, Mapping) else None


def _boundary_spec_from_eval_config(eval_config: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    root = _config_root(eval_config)
    for section_name in ("rollout_calibration", "rollout_data", "data"):
        section = _mapping_get(root, section_name)
        boundary = _mapping_get(section, "boundary")
        channels = _mapping_get(boundary, "channels")
        if isinstance(channels, Sequence) and not isinstance(channels, (str, bytes)):
            return [dict(ch) for ch in channels if isinstance(ch, Mapping)]
    return []


def _resolve_config_path(value: Any, *, base: Any = None) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base not in (None, ""):
        path = Path(str(base)).expanduser() / path
    return path


def _split_candidate_paths(artifact: Mapping[str, Any], eval_config: Mapping[str, Any] | None) -> List[Path]:
    """Return likely reference split files for old artifacts lacking member ids."""
    meta = artifact.get("metadata", {})
    meta = meta if isinstance(meta, Mapping) else {}
    root = _config_root(eval_config)
    cal = _mapping_get(root, "rollout_calibration")
    ref = _mapping_get(cal, "reference")
    role = str(meta.get("artifact_role", "")).strip().lower()

    candidates: List[Optional[Path]] = []
    if role == "calibration_fit":
        candidates.append(_resolve_config_path(meta.get("calibration_txt"), base=meta.get("calibration_root")))
        candidates.append(_resolve_config_path(_mapping_get(ref, "calibration_txt"), base=_mapping_get(ref, "calibration_root")))
    else:
        candidates.append(_resolve_config_path(meta.get("test_txt"), base=meta.get("test_root")))
        candidates.append(_resolve_config_path(_mapping_get(ref, "test_txt"), base=_mapping_get(ref, "test_root")))
    candidates.extend(
        [
            _resolve_config_path(meta.get("test_txt"), base=meta.get("test_root")),
            _resolve_config_path(meta.get("calibration_txt"), base=meta.get("calibration_root")),
            _resolve_config_path(_mapping_get(ref, "test_txt"), base=_mapping_get(ref, "test_root")),
            _resolve_config_path(_mapping_get(ref, "calibration_txt"), base=_mapping_get(ref, "calibration_root")),
        ]
    )

    out: List[Path] = []
    seen = set()
    for path in candidates:
        if path is None:
            continue
        resolved = path.resolve(strict=False)
        key = str(resolved)
        if key not in seen:
            out.append(resolved)
            seen.add(key)
    return out


def _read_split_run_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) == 1 and "," in lines[0]:
        return [part.strip() for part in lines[0].split(",") if part.strip()]
    return lines


def _family_id(run_id: str) -> str:
    text = str(run_id).strip()
    return text.rsplit("_sim", 1)[0] if "_sim" in text else text


def _family_aliases(label: str) -> set[str]:
    text = str(label).strip()
    if text.endswith(".calibration_artifact"):
        text = text[: -len(".calibration_artifact")]
    aliases = {text.casefold()} if text else set()
    parts = [part for part in text.split("_") if part]
    for start in range(1, len(parts)):
        aliases.add("_".join(parts[start:]).casefold())
    return aliases


def _explicit_reference_run_ids(artifact: Mapping[str, Any]) -> List[str]:
    value = artifact.get("reference_run_ids")
    if value is None:
        meta = artifact.get("metadata", {})
        if isinstance(meta, Mapping):
            value = meta.get("reference_run_ids")
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _reference_run_ids_for_artifact(
    artifact_path: str | Path,
    artifact: Mapping[str, Any],
    eval_config: Mapping[str, Any] | None,
) -> List[str]:
    explicit = _explicit_reference_run_ids(artifact)
    if explicit:
        return explicit
    hydrograph_id = _artifact_label(artifact_path, artifact)
    aliases = _family_aliases(hydrograph_id)
    if not aliases:
        return []
    for split_path in _split_candidate_paths(artifact, eval_config):
        run_ids = _read_split_run_ids(split_path)
        matched = [run_id for run_id in run_ids if _family_id(run_id).casefold() in aliases]
        if matched:
            LOGGER.info(
                "Recovered %d reference run ids for %s from %s",
                len(matched),
                hydrograph_id,
                split_path,
            )
            return matched
    return []


def _boundary_ensemble_from_artifact_or_config(
    artifact_path: str | Path,
    artifact: Mapping[str, Any],
    eval_config: Mapping[str, Any] | None,
) -> Tuple[Optional[np.ndarray], str]:
    saved = artifact.get("boundary_ensemble_series_raw")
    if saved is not None:
        return np.asarray(saved, dtype=np.float32), "artifact"
    fallback = artifact.get("boundary_series_raw")
    if fallback is None:
        return None, "missing_boundary_series"
    boundary_spec = _boundary_spec_from_eval_config(eval_config)
    if not boundary_spec:
        return None, "missing_boundary_config"
    reference_run_ids = _reference_run_ids_for_artifact(artifact_path, artifact, eval_config)
    if not reference_run_ids:
        return None, "missing_reference_run_ids"

    fallback_arr = np.asarray(fallback, dtype=np.float32)
    if fallback_arr.ndim == 2:
        fallback_arr = np.repeat(fallback_arr[None, :, :], len(reference_run_ids), axis=0)
    elif fallback_arr.ndim == 3:
        if fallback_arr.shape[0] == 1 and len(reference_run_ids) > 1:
            fallback_arr = np.repeat(fallback_arr, len(reference_run_ids), axis=0)
    if fallback_arr.ndim != 3 or fallback_arr.shape[0] != len(reference_run_ids):
        LOGGER.warning(
            "Cannot reconstruct boundary forcing ensemble for %s: fallback shape=%s reference_run_ids=%d",
            artifact_path,
            tuple(fallback_arr.shape),
            len(reference_run_ids),
        )
        return None, "fallback_shape_mismatch"

    ensemble = _boundary_ensemble_series_from_reference_members(
        boundary_spec,
        reference_run_ids,
        fallback_arr,
        logger=LOGGER,
    )
    if ensemble is None:
        return None, "reconstruction_failed"
    if hasattr(ensemble, "detach"):
        ensemble_arr = ensemble.detach().cpu().numpy()
    else:
        ensemble_arr = np.asarray(ensemble)
    LOGGER.info(
        "Reconstructed boundary forcing ensemble for %s with shape %s",
        _artifact_label(artifact_path, artifact),
        tuple(ensemble_arr.shape),
    )
    return np.asarray(ensemble_arr, dtype=np.float32), "reconstructed"


def _nested_dict(base: MutableMapping[str, Any], key: str) -> MutableMapping[str, Any]:
    value = base.get(key)
    if not isinstance(value, MutableMapping):
        value = {}
        base[key] = value
    return value


def apply_visualization_overrides(
    visualization_config: Mapping[str, Any] | None,
    *,
    write_gif: Optional[bool] = None,
    write_mp4: Optional[bool] = None,
    map_enabled: Optional[bool] = None,
    map_mode: Optional[str] = None,
    map_provider: Optional[str] = None,
    show_wet_edge: Optional[bool] = None,
) -> Dict[str, Any]:
    """Apply CLI overrides without mutating the caller's config."""
    config: Dict[str, Any] = json.loads(json.dumps(dict(visualization_config or {})))
    if write_gif is not None or write_mp4 is not None:
        output = _nested_dict(config, "output")
        if write_gif is not None:
            output["write_gif"] = bool(write_gif)
        if write_mp4 is not None:
            output["write_mp4"] = bool(write_mp4)
    if map_enabled is not None or map_mode is not None or map_provider is not None:
        map_cfg = _nested_dict(config, "map")
        if map_enabled is not None:
            map_cfg["enabled"] = bool(map_enabled)
        if map_mode is not None:
            map_cfg["mode"] = str(map_mode)
        if map_provider is not None:
            map_cfg["provider"] = str(map_provider)
    if show_wet_edge is not None:
        wd_cfg = _nested_dict(config, "wd")
        wd_cfg["show_wet_edge"] = bool(show_wet_edge)
    return config


def _artifact_label(artifact_path: str | Path, art: Mapping[str, Any]) -> str:
    return str(art.get("hydrograph_id") or Path(artifact_path).stem)


def select_artifacts(
    artifact_root: str | Path,
    *,
    hydrograph_ids: Sequence[str] | None = None,
    max_artifacts: int | None = None,
) -> List[Path]:
    """List artifacts, optionally filtering by artifact hydrograph id or file stem."""
    paths = list_forecast_artifacts(artifact_root)
    wanted = {str(x) for x in (hydrograph_ids or [])}
    if wanted:
        selected: List[Path] = []
        found = set()
        for path in paths:
            meta = load_forecast_artifact(path, load_members=False)
            labels = {str(meta.get("hydrograph_id", "")), path.stem, path.name}
            matched = labels & wanted
            if matched:
                selected.append(path)
                found.update(matched)
        missing = sorted(wanted - found)
        if missing:
            raise FileNotFoundError(f"Requested hydrograph ids not found in {artifact_root}: {missing}")
        paths = selected
    if max_artifacts is not None and max_artifacts > 0:
        paths = paths[: int(max_artifacts)]
    if not paths:
        raise FileNotFoundError(f"No forecast artifacts selected from {artifact_root}")
    return paths


def _load_hdf_dataset(path: Path, dataset: str | None) -> np.ndarray:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency controlled by runtime
        raise RuntimeError("Loading elevation from HDF5 requires h5py.") from exc
    with h5py.File(path, "r") as handle:
        if dataset:
            if dataset not in handle:
                raise KeyError(f"Dataset {dataset!r} not found in {path}")
            return np.asarray(handle[dataset][...])
        keys: List[str] = []
        handle.visit(lambda name: keys.append(name) if hasattr(handle[name], "shape") else None)
        candidates = [k for k in keys if "elev" in k.lower()]
        if not candidates:
            candidates = keys[:1]
        if not candidates:
            raise ValueError(f"No array datasets found in {path}")
        return np.asarray(handle[candidates[0]][...])


def _infer_hec_ras_cell_center_dataset(dataset: str | None) -> str | None:
    if not dataset or "/" not in dataset:
        return None
    parent, _ = dataset.rsplit("/", 1)
    return f"{parent}/Cells Center Coordinate"


def _cell_point_index_from_coordinates(cell_points: np.ndarray, cell_centers: np.ndarray) -> np.ndarray:
    """Match HEC-RAS Cell Points to full Cells Center Coordinate rows."""
    points = np.asarray(cell_points, dtype=np.float64)
    centers = np.asarray(cell_centers, dtype=np.float64)
    if points.ndim != 2 or centers.ndim != 2 or points.shape[1] != 2 or centers.shape[1] != 2:
        raise ValueError(
            "HEC-RAS cell-point alignment requires [n, 2] Cell Points and Cells Center Coordinate arrays."
        )
    try:
        from scipy.spatial import cKDTree  # type: ignore

        distances, indices = cKDTree(centers).query(points, k=1, distance_upper_bound=0.001)
        if np.any(indices >= centers.shape[0]) or np.any(~np.isfinite(distances)):
            raise ValueError("Some Cell Points have no matching Cells Center Coordinate row.")
        return np.asarray(indices, dtype=np.intp)
    except ImportError:
        lookup: Dict[Tuple[float, float], int] = {
            tuple(np.round(row, decimals=3)): idx for idx, row in enumerate(centers)
        }
        indices = []
        for row in points:
            key = tuple(np.round(row, decimals=3))
            if key not in lookup:
                raise ValueError("Some Cell Points have no rounded-coordinate match in Cells Center Coordinate.")
            indices.append(lookup[key])
        return np.asarray(indices, dtype=np.intp)


def _align_hec_ras_full_cell_values_to_cell_points(
    path: Path,
    values: np.ndarray,
    dataset: str | None,
) -> np.ndarray:
    """Align full HEC-RAS cell arrays to the model Cell Points subset when possible."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or dataset is None:
        return arr
    cell_center_dataset = _infer_hec_ras_cell_center_dataset(dataset)
    if cell_center_dataset is None:
        return arr
    cell_points_dataset = "Geometry/2D Flow Areas/Cell Points"
    try:
        import h5py  # type: ignore
    except Exception:
        return arr
    try:
        with h5py.File(path, "r") as handle:
            if cell_points_dataset not in handle or cell_center_dataset not in handle:
                return arr
            cell_points = np.asarray(handle[cell_points_dataset][...], dtype=np.float64)
            cell_centers = np.asarray(handle[cell_center_dataset][...], dtype=np.float64)
    except Exception:
        return arr
    if arr.size == cell_points.shape[0]:
        return arr
    if arr.size != cell_centers.shape[0]:
        return arr
    index = _cell_point_index_from_coordinates(cell_points, cell_centers)
    LOGGER.info(
        "Aligned HEC-RAS elevation dataset %s from full cell count %d to Cell Points count %d",
        dataset,
        arr.size,
        index.size,
    )
    return arr[index]


def load_elevation_values(path: str | Path | None, *, dataset: str | None = None) -> Optional[np.ndarray]:
    """Load optional raw elevation values for DEM-backed artifact rendering."""
    if path is None:
        return None
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        npz = np.load(path)
        key = dataset or next(iter(npz.files))
        arr = npz[key]
    elif suffix in {".h5", ".hdf5", ".hdf"}:
        arr = _load_hdf_dataset(path, dataset)
        arr = _align_hec_ras_full_cell_values_to_cell_points(path, arr, dataset)
    else:
        arr = np.loadtxt(path, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"Elevation values must be one-dimensional after loading; got shape {arr.shape} from {path}")
    return arr


def _elevation_from_artifact_or_override(
    art: Mapping[str, Any],
    override: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if override is not None:
        return np.asarray(override, dtype=np.float64).reshape(-1)
    if "elevation_raw" in art:
        return np.asarray(art["elevation_raw"], dtype=np.float64).reshape(-1)
    geom = art.get("geometry_raw")
    if geom is not None:
        geom_arr = np.asarray(geom, dtype=np.float64)
        if geom_arr.ndim == 2 and geom_arr.shape[1] >= 3:
            return geom_arr[:, 2]
    return None


def _relative_l2_series(pred_mean: np.ndarray, ref_mean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values: List[float] = []
    for t in range(pred_mean.shape[0]):
        pred_t = np.asarray(pred_mean[t], dtype=np.float64)[mask]
        ref_t = np.asarray(ref_mean[t], dtype=np.float64)[mask]
        denom = max(float(np.linalg.norm(ref_t)), MIN_EPS)
        values.append(float(np.linalg.norm(pred_t - ref_t) / denom))
    return np.asarray(values, dtype=np.float64)


def load_isotonic_calibration(path: str | Path | None) -> Optional[Dict[str, Any]]:
    """Load optional exceedance isotonic calibration curves."""
    if path is None or str(path).strip() == "":
        return None
    path = Path(path)
    if not path.exists():
        LOGGER.warning("Isotonic calibration path does not exist; skipping: %s", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_metric(metrics: Mapping[str, Any], key: str) -> float:
    try:
        value = float(metrics[key])
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _threshold_key(prefix: str, threshold_m: float) -> str:
    return f"{prefix}_{threshold_m:.2f}m_overall_mean".replace(".", "p")


def save_artifact_overall_metric_diagnostics(
    *,
    artifact_paths: Sequence[str | Path],
    calibration_model: Mapping[str, Any],
    out_dir: str | Path,
    isotonic_model: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Save overall raw-vs-calibrated metric JSON and a compact comparison figure."""
    out_dir = Path(out_dir)
    raw_metrics = compute_artifact_uq_metrics(artifact_paths)
    calibrated_metrics = compute_artifact_uq_metrics(
        artifact_paths,
        calibration_model=calibration_model,
        isotonic_model=isotonic_model,
        apply_isotonic=isotonic_model is not None,
    )
    comparison = build_calibration_comparison(raw_metrics, calibrated_metrics)

    raw_path = save_metrics_json(raw_metrics, out_dir / "artifact_raw_uq_overall_metrics.json")
    calibrated_path = save_metrics_json(
        calibrated_metrics,
        out_dir / "artifact_calibrated_uq_overall_metrics.json",
    )
    comparison_path = save_metrics_json(
        comparison,
        out_dir / "artifact_calibration_comparison.json",
    )
    figure_path = _save_artifact_metric_summary_figure(
        raw_metrics,
        calibrated_metrics,
        out_dir / "artifact_calibration_summary.png",
    )
    outputs = {
        "artifact_raw_metrics_json": str(raw_path),
        "artifact_calibrated_metrics_json": str(calibrated_path),
        "artifact_calibration_comparison_json": str(comparison_path),
    }
    if figure_path is not None:
        outputs["artifact_calibration_summary_png"] = str(figure_path)
    return outputs


def _save_artifact_metric_summary_figure(
    raw_metrics: Mapping[str, Any],
    calibrated_metrics: Mapping[str, Any],
    path: str | Path,
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is runtime optional
        LOGGER.warning("Could not create calibration summary figure: %s", exc)
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=180)

    core = [
        ("crps_wd_overall_mean", "CRPS"),
        ("rmse_wd_overall_mean", "RMSE"),
        ("spread_pred_wd_overall_mean", "Pred spread"),
        ("spread_ratio_wd_overall_mean", "Spread ratio"),
    ]
    x = np.arange(len(core))
    raw_vals = [_finite_metric(raw_metrics, key) for key, _ in core]
    cal_vals = [_finite_metric(calibrated_metrics, key) for key, _ in core]
    axes[0, 0].bar(x - 0.18, raw_vals, width=0.36, label="Raw", color="#8da0cb")
    axes[0, 0].bar(x + 0.18, cal_vals, width=0.36, label="Calibrated", color="#fc8d62")
    axes[0, 0].set_xticks(x, [label for _, label in core], rotation=20, ha="right")
    axes[0, 0].set_title("Core water-depth metrics")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)

    coverages = [50, 80, 90, 95]
    nominal = [p / 100.0 for p in coverages]
    raw_cov = [_finite_metric(raw_metrics, f"coverage_wd_{p}_overall_mean") for p in coverages]
    cal_cov = [_finite_metric(calibrated_metrics, f"coverage_wd_{p}_overall_mean") for p in coverages]
    x = np.arange(len(coverages))
    axes[0, 1].plot(x, nominal, "--", color="0.35", label="Nominal")
    axes[0, 1].plot(x, raw_cov, "o-", label="Raw", color="#8da0cb")
    axes[0, 1].plot(x, cal_cov, "o-", label="Calibrated", color="#fc8d62")
    axes[0, 1].set_xticks(x, [f"{p}%" for p in coverages])
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_title("Interval coverage")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False)

    thresholds = [0.01, 0.05, 0.10, 0.30, 0.50]
    x = np.arange(len(thresholds))
    raw_brier = [_finite_metric(raw_metrics, _threshold_key("brier_wd_exceed", t)) for t in thresholds]
    cal_brier = [_finite_metric(calibrated_metrics, _threshold_key("brier_wd_exceed", t)) for t in thresholds]
    iso_brier = [_finite_metric(calibrated_metrics, _threshold_key("brier_isotonic_wd_exceed", t)) for t in thresholds]
    axes[1, 0].plot(x, raw_brier, "o-", label="Raw", color="#8da0cb")
    axes[1, 0].plot(x, cal_brier, "o-", label="Calibrated ens.", color="#fc8d62")
    if np.any(np.isfinite(iso_brier)):
        axes[1, 0].plot(x, iso_brier, "o-", label="Calibrated isotonic", color="#66c2a5")
    axes[1, 0].set_xticks(x, [f"{t:g} m" for t in thresholds], rotation=20, ha="right")
    axes[1, 0].set_title("Exceedance Brier score")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False)

    raw_rel = [_finite_metric(raw_metrics, _threshold_key("reliability_abs_wd_exceed", t)) for t in thresholds]
    cal_rel = [_finite_metric(calibrated_metrics, _threshold_key("reliability_abs_wd_exceed", t)) for t in thresholds]
    axes[1, 1].plot(x, raw_rel, "o-", label="Raw", color="#8da0cb")
    axes[1, 1].plot(x, cal_rel, "o-", label="Calibrated", color="#fc8d62")
    axes[1, 1].set_xticks(x, [f"{t:g} m" for t in thresholds], rotation=20, ha="right")
    axes[1, 1].set_title("Mean exceedance reliability error")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Artifact-level held-out calibration diagnostics", y=0.995)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def build_visual_fields_from_artifact(
    artifact: Mapping[str, Any],
    *,
    calibration_model: Optional[Mapping[str, Any]] = None,
    exceedance_threshold_m: float = UQ_EXCEEDANCE_THRESHOLD,
) -> Dict[str, Any]:
    """Convert one forecast artifact into renderer-ready calibrated fields."""
    pred = np.asarray(artifact["pred_members_wd"], dtype=np.float64)
    ref = np.asarray(artifact["ref_members_wd"], dtype=np.float64)
    if pred.ndim != 3 or ref.ndim != 3:
        raise ValueError("Artifact members must have shape [members, time, cells].")
    if pred.shape[1:] != ref.shape[1:]:
        raise ValueError(f"Forecast/reference shape mismatch: pred={pred.shape} ref={ref.shape}")
    wettable = np.asarray(artifact.get("wettable_mask", np.ones(pred.shape[2], dtype=bool)), dtype=bool).reshape(-1)
    if wettable.size != pred.shape[2]:
        raise ValueError("wettable_mask length does not match artifact cell count.")
    time_hours = artifact.get("time_hours")
    if time_hours is None:
        time_hours = np.arange(1, pred.shape[1] + 1, dtype=np.float64)
    time_hours = np.asarray(time_hours, dtype=np.float64).reshape(-1)
    if time_hours.size != pred.shape[1]:
        raise ValueError("time_hours length does not match artifact rollout length.")

    pred_for_render = np.empty_like(pred, dtype=np.float64)
    for t in range(pred.shape[1]):
        pred_t = pred[:, t, :]
        if calibration_model is not None:
            pred_t = apply_crps_mbm_to_wd_members(
                pred_t,
                lead_time_hour=float(time_hours[t]),
                calibration_model=calibration_model,
                wettable_mask=wettable,
            )
        pred_for_render[:, t, :] = pred_t

    pred_mean = np.mean(pred_for_render, axis=0)
    pred_std = np.std(pred_for_render, axis=0)
    ref_mean = np.mean(ref, axis=0)
    ref_std = np.std(ref, axis=0)
    crps = np.stack(
        [empirical_crps_per_location(pred_for_render[:, t, :], ref[:, t, :]) for t in range(pred.shape[1])],
        axis=0,
    )
    pred_prob = np.mean(pred_for_render >= float(exceedance_threshold_m), axis=0)
    ref_prob = np.mean(ref >= float(exceedance_threshold_m), axis=0)
    rel_l2 = _relative_l2_series(pred_mean, ref_mean, wettable)

    return {
        "pred_members_wd": pred_for_render,
        "pred_mean_by_channel": {"wd": pred_mean},
        "pred_std_by_channel": {"wd": pred_std},
        "gt_mean_by_channel": {"wd": ref_mean},
        "gt_std_by_channel": {"wd": ref_std},
        "pred_prob_wd": pred_prob,
        "gt_prob_wd": ref_prob,
        "crps_map_wd": crps,
        "relative_l2_by_channel": {"wd": rel_l2},
    }


def render_artifact_visualization(
    artifact_path: str | Path,
    *,
    out_dir: str | Path,
    calibration_model: Optional[Mapping[str, Any]],
    visualization_config: Optional[Mapping[str, Any]] = None,
    eval_config: Optional[Mapping[str, Any]] = None,
    elevation_raw: Optional[np.ndarray] = None,
    dt_seconds: Optional[float] = None,
    exceedance_threshold_m: float = UQ_EXCEEDANCE_THRESHOLD,
) -> Dict[str, Any]:
    """Render one calibrated artifact and return a compact manifest entry."""
    art = load_forecast_artifact(artifact_path, load_members=True)
    geometry = art.get("geometry_raw")
    if geometry is None:
        raise ValueError(f"Artifact {artifact_path} does not contain geometry_raw; cannot render spatial maps.")
    fields = build_visual_fields_from_artifact(
        art,
        calibration_model=calibration_model,
        exceedance_threshold_m=exceedance_threshold_m,
    )
    time_hours_raw = art.get("time_hours")
    if time_hours_raw is None:
        time_hours = np.arange(1, fields["pred_mean_by_channel"]["wd"].shape[0] + 1, dtype=np.float64)
    else:
        time_hours = np.asarray(time_hours_raw, dtype=np.float64).reshape(-1)
    if dt_seconds is None:
        if time_hours.size >= 2:
            dt_seconds = float(np.nanmedian(np.diff(time_hours)) * 3600.0)
        elif time_hours.size == 1:
            dt_seconds = float(time_hours[0] * 3600.0)
        else:
            dt_seconds = 3600.0
    meta = dict(art.get("metadata", {}) or {})
    rollout_start_index = int(meta.get("rollout_start_index", 0) or 0)
    hydrograph_id = _artifact_label(artifact_path, art)
    elevation = _elevation_from_artifact_or_override(art, elevation_raw)
    if elevation is not None and elevation.size != np.asarray(geometry).shape[0]:
        raise ValueError(
            f"Elevation length {elevation.size} does not match geometry cell count {np.asarray(geometry).shape[0]} for {artifact_path}."
        )
    boundary_ensemble_raw, boundary_ensemble_source = _boundary_ensemble_from_artifact_or_config(
        artifact_path,
        art,
        eval_config,
    )

    _save_hydrograph_uq_figures_and_animation(
        geometry=geometry,
        pred_mean_by_channel=fields["pred_mean_by_channel"],
        pred_std_by_channel=fields["pred_std_by_channel"],
        gt_mean_by_channel=fields["gt_mean_by_channel"],
        gt_std_by_channel=fields["gt_std_by_channel"],
        target_variables=["wd"],
        out_dir=str(out_dir),
        hydrograph_id=hydrograph_id,
        dt_seconds=float(dt_seconds),
        n_ref_sims=int(art.get("n_reference_members", np.asarray(art["ref_members_wd"]).shape[0])),
        n_ens=int(art.get("n_forecast_members", np.asarray(art["pred_members_wd"]).shape[0])),
        pred_prob_wd=fields["pred_prob_wd"],
        gt_prob_wd=fields["gt_prob_wd"],
        crps_map_wd=fields["crps_map_wd"],
        boundary_series_raw=art.get("boundary_series_raw"),
        boundary_ensemble_series_raw=boundary_ensemble_raw,
        boundary_channel_names=list(art.get("boundary_channel_names", [])),
        relative_l2_by_channel=fields["relative_l2_by_channel"],
        rollout_start_index=rollout_start_index,
        elevation_raw=elevation,
        visualization_config=visualization_config,
    )
    return {
        "artifact_path": str(artifact_path),
        "hydrograph_id": hydrograph_id,
        "n_forecast_members": int(art.get("n_forecast_members", fields["pred_members_wd"].shape[0])),
        "n_reference_members": int(art.get("n_reference_members", np.asarray(art["ref_members_wd"]).shape[0])),
        "n_time": int(fields["pred_mean_by_channel"]["wd"].shape[0]),
        "n_cells": int(fields["pred_mean_by_channel"]["wd"].shape[1]),
        "calibration_applied": calibration_model is not None,
        "boundary_ensemble_source": boundary_ensemble_source,
    }


def render_calibrated_artifact_visuals(
    *,
    artifact_root: str | Path,
    coefficient_path: str | Path,
    out_dir: str | Path,
    hydrograph_ids: Sequence[str] | None = None,
    max_artifacts: int | None = None,
    visualization_config: Optional[Mapping[str, Any]] = None,
    eval_config: Optional[Mapping[str, Any]] = None,
    elevation_raw: Optional[np.ndarray] = None,
    isotonic_model: Optional[Mapping[str, Any]] = None,
    write_overall_metrics: bool = True,
    dt_seconds: Optional[float] = None,
    exceedance_threshold_m: float = UQ_EXCEEDANCE_THRESHOLD,
) -> Path:
    """Render calibrated visualizations for selected HDF5 forecast artifacts."""
    artifacts = select_artifacts(
        artifact_root,
        hydrograph_ids=hydrograph_ids,
        max_artifacts=max_artifacts,
    )
    calibration_model = load_crps_mbm_coefficients(coefficient_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "artifact_root": str(artifact_root),
        "coefficient_path": str(coefficient_path),
        "out_dir": str(out_dir),
        "n_artifacts": len(artifacts),
        "calibration_method": "crps_member_by_member",
        "hydrographs": [],
    }
    for index, artifact_path in enumerate(artifacts, start=1):
        LOGGER.info("Rendering calibrated artifact %d/%d: %s", index, len(artifacts), artifact_path)
        manifest["hydrographs"].append(
            render_artifact_visualization(
                artifact_path,
                out_dir=out_dir,
                calibration_model=calibration_model,
                visualization_config=visualization_config,
                eval_config=eval_config,
                elevation_raw=elevation_raw,
                dt_seconds=dt_seconds,
                exceedance_threshold_m=exceedance_threshold_m,
            )
        )
        manifest_path = out_dir / "artifact_calibrated_visualization_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if write_overall_metrics:
        LOGGER.info("Computing overall raw-vs-calibrated artifact metrics for %d artifacts", len(artifacts))
        manifest["overall_metrics"] = save_artifact_overall_metric_diagnostics(
            artifact_paths=artifacts,
            calibration_model=calibration_model,
            isotonic_model=isotonic_model,
            out_dir=out_dir,
        )
    manifest_path = out_dir / "artifact_calibrated_visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Saved calibrated artifact visualization manifest to %s", manifest_path)
    return manifest_path


def _parse_hydrograph_ids(values: Sequence[str] | None) -> List[str]:
    ids: List[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                ids.append(part)
    return ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render calibrated rollout GIF/MP4 diagnostics directly from saved forecast artifacts."
    )
    parser.add_argument("--artifact_root", required=True, help="Directory containing held-out test forecast HDF5 artifacts.")
    parser.add_argument("--coefficient_path", required=True, help="CRPS-MBM coefficient JSON from calibration fit.")
    parser.add_argument("--isotonic_path", default=None, help="Optional exceedance isotonic calibration JSON.")
    parser.add_argument("--out_dir", required=True, help="Output directory for calibrated artifact visualizations.")
    parser.add_argument("--hydrograph_id", action="append", default=None, help="Hydrograph id, file stem, or filename to render. Repeat or comma-separate.")
    parser.add_argument("--max_artifacts", type=int, default=None, help="Render only the first N selected artifacts.")
    parser.add_argument("--visualization_config_path", default=None, help="YAML/JSON config containing a visualization block, or a standalone visualization config.")
    parser.add_argument("--elevation_path", default=None, help="Optional 1D elevation vector (.txt/.npy/.npz/.h5) matching artifact cell order.")
    parser.add_argument("--elevation_dataset", default=None, help="Dataset/key to read from --elevation_path when needed.")
    parser.add_argument("--dt_seconds", type=float, default=None, help="Override time step seconds; default inferred from artifact time_hours.")
    parser.add_argument("--exceedance_threshold_m", type=float, default=UQ_EXCEEDANCE_THRESHOLD)
    parser.add_argument("--write-gif", dest="write_gif", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write-mp4", dest="write_mp4", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--map-enabled", dest="map_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--map-mode", default=None, help="Override visualization.map.mode, e.g. dem_elevation, 3dep_hillshade, imagery, topo.")
    parser.add_argument("--map-provider", default=None, help="Override visualization.map.provider.")
    parser.add_argument("--show-wet-edge", dest="show_wet_edge", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overall-metrics", dest="overall_metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    eval_config = load_eval_config(args.visualization_config_path)
    visualization_config = load_visualization_config(args.visualization_config_path)
    visualization_config = apply_visualization_overrides(
        visualization_config,
        write_gif=args.write_gif,
        write_mp4=args.write_mp4,
        map_enabled=args.map_enabled,
        map_mode=args.map_mode,
        map_provider=args.map_provider,
        show_wet_edge=args.show_wet_edge,
    )
    elevation = load_elevation_values(args.elevation_path, dataset=args.elevation_dataset)
    isotonic_model = load_isotonic_calibration(args.isotonic_path)
    manifest_path = render_calibrated_artifact_visuals(
        artifact_root=args.artifact_root,
        coefficient_path=args.coefficient_path,
        out_dir=args.out_dir,
        hydrograph_ids=_parse_hydrograph_ids(args.hydrograph_id),
        max_artifacts=args.max_artifacts,
        visualization_config=visualization_config,
        eval_config=eval_config,
        elevation_raw=elevation,
        isotonic_model=isotonic_model,
        write_overall_metrics=bool(args.overall_metrics),
        dt_seconds=args.dt_seconds,
        exceedance_threshold_m=float(args.exceedance_threshold_m),
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
