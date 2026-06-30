"""Scientific probabilistic calibration for flood UQ rollout artifacts.

This module replaces the legacy lead-time affine calibration with a
leakage-safe, CRPS-optimized member-by-member (MBM) calibration layer for
physical-space water-depth ensembles.  The implementation is deliberately
artifact-first: FGN, diffusion, and future stochastic baselines can all write the
same per-family forecast/reference HDF5 artifact and use the same fitter.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Optional on login nodes; required only when reading/writing artifacts.
    import h5py  # type: ignore
except Exception:  # pragma: no cover - exercised in environments without h5py
    h5py = None

try:  # SciPy gives robust bounded optimization; fallback keeps tests/use usable.
    from scipy.optimize import minimize  # type: ignore
except Exception:  # pragma: no cover
    minimize = None

SCIENTIFIC_CALIBRATION_METHOD = "crps_member_by_member"
ARTIFACT_FORMAT = "hdf5_per_family"
ARTIFACT_FILE_SUFFIX = ".calibration_artifact.h5"
COEFFICIENTS_JSON = "crps_mbm_coefficients.json"
ISOTONIC_JSON = "exceedance_isotonic_curves.json"
CALIBRATION_COMPARISON_JSON = "calibration_comparison.json"
FIT_DIAGNOSTICS_JSON = "crps_mbm_fit_diagnostics.json"
RAW_METRICS_SUBDIR = "raw"
CALIBRATED_METRICS_SUBDIR = "calibrated"

LEGACY_AFFINE_KEYS = {
    "calib_txt",
    "fit_wet_threshold",
    "min_pred_std",
    "c_clip_min",
    "c_clip_max",
    "smooth_window",
    "allow_same_split",
}
DEFAULT_THRESHOLDS_M = (0.01, 0.05, 0.10, 0.30, 0.50)
DEFAULT_LEAD_BINS_H = (0.0, 2.0, 6.0, 12.0, 24.0, math.inf)
DEFAULT_WET_FREQ_EDGES = (0.0, 0.05, 0.50, 1.0)
SUPPORTED_MBM_OBJECTIVES = {
    "empirical_crps",
    "crps",
    "tail_weighted_crps",
    "twcrps",
    "mean_regularized_crps",
    "spread_regularized_crps",
    "combined_regularized_crps",
}
MIN_EPS = 1e-12


def _cfg_get(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if cur is None:
            return default
        try:
            cur = getattr(cur, key)
            continue
        except (AttributeError, KeyError, TypeError):
            pass
        if isinstance(cur, Mapping):
            if key not in cur:
                return default
            cur = cur[key]
            continue
        try:
            cur = cur[key]
        except Exception:
            return default
    return cur


def _section_keys(section: Any) -> set[str]:
    if section is None:
        return set()
    if isinstance(section, Mapping):
        return {str(k) for k in section.keys()}
    if hasattr(section, "keys"):
        try:
            return {str(k) for k in section.keys()}
        except Exception:
            pass
    return {k for k in vars(section).keys() if not k.startswith("_")}


def _as_float_sequence(value: Any, default: Sequence[float]) -> List[float]:
    if value is None:
        return [float(v) for v in default]
    out: List[float] = []
    for item in value:
        if item is None:
            out.append(math.inf)
        else:
            out.append(float(item))
    return out


def _as_path(value: Any, *, base: Path | None = None) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError("Expected a non-empty path value.")
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve(strict=False)


def validate_scientific_calibration_config(config: Any) -> None:
    """Validate the new calibration interface and reject legacy affine configs."""
    section = _cfg_get(config, "rollout_calibration", default=None)
    enabled = bool(_cfg_get(config, "rollout_calibration", "enabled", default=False))
    if not enabled:
        return
    method = str(_cfg_get(config, "rollout_calibration", "method", default="")).strip()
    if method != SCIENTIFIC_CALIBRATION_METHOD:
        if method:
            raise ValueError(
                "rollout_calibration.method must be crps_member_by_member. "
                f"Got {method!r}. Legacy lead-time affine calibration has been removed."
            )
        legacy_present = sorted(_section_keys(section).intersection(LEGACY_AFFINE_KEYS))
        if legacy_present:
            raise ValueError(
                "Legacy lead-time affine calibration keys are no longer supported: "
                f"{legacy_present}. Migrate to rollout_calibration.method="
                "crps_member_by_member with reference/forecast_artifacts/bins/optimizer/exceedance blocks."
            )
        raise ValueError(
            "rollout_calibration.enabled=true requires rollout_calibration.method="
            "crps_member_by_member. Legacy lead-time affine calibration is no longer available."
        )
    legacy_present = sorted(_section_keys(section).intersection(LEGACY_AFFINE_KEYS))
    if legacy_present:
        raise ValueError(
            "Do not mix legacy affine calibration keys with crps_member_by_member: "
            f"{legacy_present}."
        )
    ref = _cfg_get(config, "rollout_calibration", "reference", default=None)
    missing = [
        key
        for key in ("calibration_root", "calibration_txt", "test_root", "test_txt")
        if _cfg_get(ref, key, default=None) in (None, "")
    ]
    if missing:
        raise ValueError(
            "rollout_calibration.reference is missing required leakage-safe fields: "
            f"{missing}."
        )
    fmt = str(
        _cfg_get(
            config,
            "rollout_calibration",
            "forecast_artifacts",
            "format",
            default=ARTIFACT_FORMAT,
        )
    ).strip()
    if fmt != ARTIFACT_FORMAT:
        raise ValueError(
            "rollout_calibration.forecast_artifacts.format must be "
            f"{ARTIFACT_FORMAT!r}; got {fmt!r}."
        )
    optimizer = _cfg_get(config, "rollout_calibration", "optimizer", default=None)
    objective = str(_cfg_get(optimizer, "objective", default="empirical_crps")).strip().lower()
    if objective not in SUPPORTED_MBM_OBJECTIVES:
        allowed = ", ".join(sorted(SUPPORTED_MBM_OBJECTIVES))
        raise ValueError(
            "Unsupported rollout_calibration.optimizer.objective "
            f"{objective!r}; expected one of: {allowed}."
        )
    numeric_constraints = {
        "mean_rmse_weight": (0.0, math.inf),
        "spread_ratio_weight": (0.0, math.inf),
        "target_spread_ratio": (MIN_EPS, math.inf),
        "tail_threshold_m": (0.0, math.inf),
        "tail_weight": (0.0, math.inf),
    }
    for key, (lo, hi) in numeric_constraints.items():
        raw_value = _cfg_get(optimizer, key, default=None)
        if raw_value is None:
            continue
        value = float(raw_value)
        if not (lo <= value <= hi):
            raise ValueError(
                f"rollout_calibration.optimizer.{key} must be in [{lo}, {hi}]; got {value}."
            )


def _load_split_ids(root: Any, txt: Any) -> set[str]:
    path = _as_path(txt, base=_as_path(root))
    if not path.exists():
        raise FileNotFoundError(f"Split file not found for calibration leakage guard: {path}")
    with open(path, "r", encoding="utf-8-sig") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) == 1 and "," in lines[0]:
        ids = [token.strip() for token in lines[0].split(",") if token.strip()]
    else:
        ids = lines
    return {item.casefold() for item in ids}


def _family_id(run_id: str) -> str:
    text = str(run_id)
    if "_sim" in text:
        return text.rsplit("_sim", 1)[0].casefold()
    return text.casefold()


def validate_reference_split_no_leakage(config: Any) -> None:
    """Reject overlapping calibration/test families and same physical split paths."""
    ref = _cfg_get(config, "rollout_calibration", "reference", default=None)
    calib_root = _cfg_get(ref, "calibration_root")
    calib_txt = _cfg_get(ref, "calibration_txt")
    test_root = _cfg_get(ref, "test_root")
    test_txt = _cfg_get(ref, "test_txt")
    calib_path = _as_path(calib_txt, base=_as_path(calib_root))
    test_path = _as_path(test_txt, base=_as_path(test_root))
    if os.path.normcase(str(calib_path)) == os.path.normcase(str(test_path)):
        raise ValueError(
            "Calibration and final test split files are identical. Publishable calibration "
            "requires a separate calibration package and held-out test package."
        )
    calib_ids = _load_split_ids(calib_root, calib_txt)
    test_ids = _load_split_ids(test_root, test_txt)
    overlap_runs = sorted(calib_ids.intersection(test_ids))
    if overlap_runs:
        raise ValueError(
            "Calibration and final test split contain overlapping run IDs. "
            f"count={len(overlap_runs)} examples={overlap_runs[:5]}"
        )
    calib_families = {_family_id(rid) for rid in calib_ids}
    test_families = {_family_id(rid) for rid in test_ids}
    overlap_families = sorted(calib_families.intersection(test_families))
    if overlap_families:
        raise ValueError(
            "Calibration and final test split contain overlapping hydrograph/family IDs. "
            f"count={len(overlap_families)} examples={overlap_families[:5]}"
        )


@dataclass(frozen=True)
class CalibrationBins:
    lead_time_hours: Tuple[float, ...]
    wet_frequency_edges: Tuple[float, ...]
    wet_threshold_m: float

    @classmethod
    def from_config(cls, config: Any) -> "CalibrationBins":
        return cls(
            lead_time_hours=tuple(
                _as_float_sequence(
                    _cfg_get(config, "rollout_calibration", "bins", "lead_time_hours", default=None),
                    DEFAULT_LEAD_BINS_H,
                )
            ),
            wet_frequency_edges=tuple(
                _as_float_sequence(
                    _cfg_get(config, "rollout_calibration", "bins", "wet_frequency_edges", default=None),
                    DEFAULT_WET_FREQ_EDGES,
                )
            ),
            wet_threshold_m=float(
                _cfg_get(config, "rollout_calibration", "bins", "wet_threshold_m", default=0.01)
            ),
        )

    def validate(self) -> None:
        if len(self.lead_time_hours) < 2:
            raise ValueError("At least two lead-time bin edges are required.")
        if any(b <= a for a, b in zip(self.lead_time_hours[:-1], self.lead_time_hours[1:])):
            raise ValueError("lead_time_hours must be strictly increasing.")
        if abs(float(self.lead_time_hours[0])) > 1e-12:
            raise ValueError("lead_time_hours must start at 0.0 for unambiguous lead-time binning.")
        if len(self.wet_frequency_edges) < 2:
            raise ValueError("At least two wet-frequency bin edges are required.")
        if self.wet_frequency_edges[0] > 0.0 or self.wet_frequency_edges[-1] < 1.0:
            raise ValueError("wet_frequency_edges must cover [0, 1].")
        if any(b <= a for a, b in zip(self.wet_frequency_edges[:-1], self.wet_frequency_edges[1:])):
            raise ValueError("wet_frequency_edges must be strictly increasing.")
        if self.wet_threshold_m < 0.0:
            raise ValueError("wet_threshold_m must be nonnegative.")

    @property
    def n_lead_bins(self) -> int:
        return len(self.lead_time_hours) - 1

    @property
    def n_wet_bins(self) -> int:
        return len(self.wet_frequency_edges) - 1

    def lead_index(self, lead_hour: float) -> int:
        idx = int(np.searchsorted(np.asarray(self.lead_time_hours[1:]), lead_hour, side="right"))
        return int(np.clip(idx, 0, self.n_lead_bins - 1))

    def wet_index(self, wet_frequency: np.ndarray) -> np.ndarray:
        edges = np.asarray(self.wet_frequency_edges, dtype=np.float64)
        idx = np.searchsorted(edges[1:], wet_frequency, side="right")
        return np.clip(idx, 0, self.n_wet_bins - 1).astype(np.int64, copy=False)


def _require_h5py() -> Any:
    if h5py is None:
        raise ImportError(
            "h5py is required for calibration forecast artifacts. Install project dependencies "
            "inside the eval environment before running calibrated evaluation."
        )
    return h5py


def _stable_array_hash(value: Any) -> str | None:
    if value is None:
        return None
    arr = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def _write_string_dataset(group: Any, name: str, values: Sequence[Any]) -> None:
    h5 = _require_h5py()
    dtype = h5.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray([str(v) for v in values], dtype=object), dtype=dtype)


def save_forecast_artifact(
    path: str | Path,
    *,
    hydrograph_id: str,
    pred_members_wd: np.ndarray,
    ref_members_wd: np.ndarray,
    wettable_mask: np.ndarray | None = None,
    structural_dry_mask: np.ndarray | None = None,
    boundary_series_raw: np.ndarray | None = None,
    boundary_ensemble_series_raw: np.ndarray | None = None,
    boundary_channel_names: Sequence[str] | None = None,
    member_model_id: Sequence[Any] | None = None,
    member_sample_id: Sequence[Any] | None = None,
    time_hours: Sequence[float] | None = None,
    geometry_raw: np.ndarray | None = None,
    elevation_raw: np.ndarray | None = None,
    cell_hash: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save one per-family forecast/reference artifact in the common schema."""
    h5 = _require_h5py()
    pred = np.asarray(pred_members_wd, dtype=np.float32)
    ref = np.asarray(ref_members_wd, dtype=np.float32)
    if pred.ndim != 3:
        raise ValueError("pred_members_wd must have shape [n_forecast_members, n_time, n_cells].")
    if ref.ndim != 3:
        raise ValueError("ref_members_wd must have shape [n_reference_members, n_time, n_cells].")
    if pred.shape[1:] != ref.shape[1:]:
        raise ValueError(
            "Forecast and reference artifacts must share [n_time, n_cells]; "
            f"got pred={pred.shape} ref={ref.shape}."
        )
    n_members, n_time, n_cells = pred.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5.File(path, "w") as f:
        f.attrs["schema"] = "flood_uq_calibration_artifact_v1"
        f.attrs["format"] = ARTIFACT_FORMAT
        f.attrs["hydrograph_id"] = str(hydrograph_id)
        f.attrs["n_forecast_members"] = int(n_members)
        f.attrs["n_reference_members"] = int(ref.shape[0])
        f.attrs["n_time"] = int(n_time)
        f.attrs["n_cells"] = int(n_cells)
        resolved_cell_hash = str(cell_hash) if cell_hash else _stable_array_hash(geometry_raw)
        if resolved_cell_hash:
            f.attrs["cell_hash"] = resolved_cell_hash
        f.create_dataset("pred_members_wd", data=pred, compression="gzip", compression_opts=4)
        f.create_dataset("ref_members_wd", data=ref, compression="gzip", compression_opts=4)
        if wettable_mask is None:
            wettable_mask = np.ones(n_cells, dtype=bool)
        if structural_dry_mask is None:
            structural_dry_mask = ~np.asarray(wettable_mask, dtype=bool)
        f.create_dataset("wettable_mask", data=np.asarray(wettable_mask, dtype=bool).reshape(n_cells))
        f.create_dataset("structural_dry_mask", data=np.asarray(structural_dry_mask, dtype=bool).reshape(n_cells))
        if boundary_series_raw is not None:
            f.create_dataset("boundary_series_raw", data=np.asarray(boundary_series_raw, dtype=np.float32))
        if boundary_ensemble_series_raw is not None:
            boundary_ens = np.asarray(boundary_ensemble_series_raw, dtype=np.float32)
            if boundary_ens.ndim != 3:
                raise ValueError(
                    "boundary_ensemble_series_raw must have shape "
                    "[n_reference_members, n_time, n_boundary_channels]."
                )
            if boundary_ens.shape[0] != ref.shape[0]:
                raise ValueError(
                    "boundary_ensemble_series_raw member count must match ref_members_wd; "
                    f"got {boundary_ens.shape[0]} and {ref.shape[0]}."
                )
            f.create_dataset(
                "boundary_ensemble_series_raw",
                data=boundary_ens,
                compression="gzip",
                compression_opts=4,
            )
        if geometry_raw is not None:
            f.create_dataset("geometry_raw", data=np.asarray(geometry_raw, dtype=np.float32), compression="gzip", compression_opts=4)
        if elevation_raw is not None:
            f.create_dataset("elevation_raw", data=np.asarray(elevation_raw, dtype=np.float32).reshape(n_cells), compression="gzip", compression_opts=4)
        _write_string_dataset(f, "boundary_channel_names", boundary_channel_names or [])
        _write_string_dataset(f, "member_model_id", member_model_id or list(range(n_members)))
        _write_string_dataset(f, "member_sample_id", member_sample_id or list(range(n_members)))
        if time_hours is None:
            time_hours = np.arange(1, n_time + 1, dtype=np.float64)
        f.create_dataset("time_hours", data=np.asarray(time_hours, dtype=np.float64).reshape(n_time))
        meta_group = f.create_group("metadata")
        for key, value in dict(metadata or {}).items():
            try:
                meta_group.attrs[str(key)] = json.dumps(value, sort_keys=True)
            except TypeError:
                meta_group.attrs[str(key)] = str(value)
    return path


def load_forecast_artifact(path: str | Path, *, load_members: bool = True) -> Dict[str, Any]:
    h5 = _require_h5py()
    path = Path(path)
    with h5.File(path, "r") as f:
        out: Dict[str, Any] = {
            "path": str(path),
            "hydrograph_id": str(f.attrs.get("hydrograph_id", path.stem)),
            "schema": str(f.attrs.get("schema", "")),
            "n_forecast_members": int(f.attrs.get("n_forecast_members", f["pred_members_wd"].shape[0])),
            "n_reference_members": int(f.attrs.get("n_reference_members", f["ref_members_wd"].shape[0])),
            "n_time": int(f.attrs.get("n_time", f["pred_members_wd"].shape[1])),
            "n_cells": int(f.attrs.get("n_cells", f["pred_members_wd"].shape[2])),
            "wettable_mask": f["wettable_mask"][...].astype(bool),
            "structural_dry_mask": f["structural_dry_mask"][...].astype(bool),
            "time_hours": f["time_hours"][...] if "time_hours" in f else None,
            "boundary_channel_names": [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f.get("boundary_channel_names", [])[:]],
            "member_model_id": [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f.get("member_model_id", [])[:]],
            "member_sample_id": [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f.get("member_sample_id", [])[:]],
            "metadata": {},
            "cell_hash": str(f.attrs.get("cell_hash", "")),
        }
        if "metadata" in f:
            for key, value in f["metadata"].attrs.items():
                try:
                    out["metadata"][key] = json.loads(value)
                except Exception:
                    out["metadata"][key] = value
        if "boundary_series_raw" in f:
            out["boundary_series_raw"] = f["boundary_series_raw"][...]
        if "boundary_ensemble_series_raw" in f:
            out["boundary_ensemble_series_raw"] = f["boundary_ensemble_series_raw"][...]
        if "geometry_raw" in f:
            out["geometry_raw"] = f["geometry_raw"][...]
        if "elevation_raw" in f:
            out["elevation_raw"] = f["elevation_raw"][...]
        if load_members:
            out["pred_members_wd"] = f["pred_members_wd"][...].astype(np.float64)
            out["ref_members_wd"] = f["ref_members_wd"][...].astype(np.float64)
    return out


def list_forecast_artifacts(root: str | Path) -> List[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Forecast artifact root does not exist: {root}")
    paths = sorted(root.glob(f"*{ARTIFACT_FILE_SUFFIX}"))
    if not paths:
        paths = sorted(root.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No calibration HDF5 artifacts found under {root}")
    return paths


def empirical_crps_per_location(forecast_ens: np.ndarray, reference_ens: np.ndarray) -> np.ndarray:
    """Empirical fair CRPS per location for forecast/reference ensembles.

    forecast_ens shape [K, N], reference_ens shape [R, N]. The reference ensemble
    is treated as samples from the verifying distribution, while the forecast
    self-distance term uses the finite-ensemble fair correction K*(K-1), not K^2.
    The implementation avoids materializing [K, R, N] tensors so 100x100-member
    ensembles remain usable.
    """
    x = np.asarray(forecast_ens, dtype=np.float64)
    y = np.asarray(reference_ens, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("forecast_ens and reference_ens must be [members, locations].")
    if x.shape[1] != y.shape[1]:
        raise ValueError("forecast/reference location dimensions differ.")
    k, n = x.shape
    r = y.shape[0]
    if k < 1 or r < 1:
        raise ValueError("CRPS requires at least one forecast and one reference member.")
    y_sorted = np.sort(y, axis=0)
    y_prefix = np.cumsum(y_sorted, axis=0)
    y_sum = y_prefix[-1]
    loc_idx = np.arange(n)
    cross = np.zeros(n, dtype=np.float64)
    for member in range(k):
        xm = x[member]
        counts = np.sum(y_sorted <= xm[None, :], axis=0).astype(np.int64, copy=False)
        left = np.zeros(n, dtype=np.float64)
        has_left = counts > 0
        left[has_left] = y_prefix[counts[has_left] - 1, loc_idx[has_left]]
        right = y_sum - left
        cross += xm * counts - left + right - xm * (r - counts)
    cross /= float(k * r)

    if k < 2:
        # A finite-ensemble fair spread correction is undefined for one member;
        # the degenerate predictive distribution reduces to MAE against the
        # reference ensemble.
        return cross
    x_sorted = np.sort(x, axis=0)
    coeff = (2.0 * np.arange(k, dtype=np.float64) - float(k) + 1.0).reshape(k, 1)
    ordered_pair_sum = 2.0 * np.sum(coeff * x_sorted, axis=0)
    fair_within = ordered_pair_sum / float(k * (k - 1))
    return cross - 0.5 * fair_within


def empirical_crps_mean(forecast_ens: np.ndarray, reference_ens: np.ndarray) -> float:
    vals = empirical_crps_per_location(forecast_ens, reference_ens)
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


def weighted_empirical_crps_mean(
    forecast_ens: np.ndarray,
    reference_ens: np.ndarray,
    *,
    threshold_m: float,
    tail_weight: float,
) -> float:
    vals = empirical_crps_per_location(forecast_ens, reference_ens)
    if vals.size == 0:
        return 0.0
    ref = np.asarray(reference_ens, dtype=np.float64)
    weights = 1.0 + float(tail_weight) * np.mean(ref > float(threshold_m), axis=0)
    return float(np.sum(vals * weights) / max(float(np.sum(weights)), MIN_EPS))


def ensemble_mean_rmse(forecast_ens: np.ndarray, reference_ens: np.ndarray) -> float:
    """RMSE between forecast and reference ensemble means over locations."""
    forecast = np.asarray(forecast_ens, dtype=np.float64)
    reference = np.asarray(reference_ens, dtype=np.float64)
    if forecast.ndim != 2 or reference.ndim != 2 or forecast.shape[1] != reference.shape[1]:
        raise ValueError("ensemble_mean_rmse expects forecast/reference [members, locations].")
    if forecast.shape[1] == 0:
        return 0.0
    diff = np.mean(forecast, axis=0) - np.mean(reference, axis=0)
    return float(np.sqrt(np.mean(diff ** 2)))


def spread_ratio_penalty(
    forecast_ens: np.ndarray,
    reference_ens: np.ndarray,
    *,
    target_spread_ratio: float,
) -> float:
    """Return a CRPS-scale penalty for forecast/reference spread mismatch."""
    forecast = np.asarray(forecast_ens, dtype=np.float64)
    reference = np.asarray(reference_ens, dtype=np.float64)
    if forecast.ndim != 2 or reference.ndim != 2 or forecast.shape[1] != reference.shape[1]:
        raise ValueError("spread_ratio_penalty expects forecast/reference [members, locations].")
    if forecast.shape[1] == 0:
        return 0.0
    pred_spread = float(np.mean(np.std(forecast, axis=0)))
    ref_spread = float(np.mean(np.std(reference, axis=0)))
    if ref_spread <= MIN_EPS:
        return abs(pred_spread - max(float(target_spread_ratio), 0.0) * ref_spread)
    ratio = pred_spread / ref_spread
    # Multiplying by reference spread keeps the penalty in water-depth units.
    return float(abs(ratio - float(target_spread_ratio)) * ref_spread)


def _objective_value(
    forecast_ens: np.ndarray,
    reference_ens: np.ndarray,
    *,
    objective: str,
    tail_threshold_m: float,
    tail_weight: float,
    mean_rmse_weight: float = 0.0,
    spread_ratio_weight: float = 0.0,
    target_spread_ratio: float = 1.0,
) -> float:
    objective_name = str(objective or "empirical_crps").strip().lower()
    if objective_name in {"empirical_crps", "crps"}:
        base = empirical_crps_mean(forecast_ens, reference_ens)
    elif objective_name in {"tail_weighted_crps", "twcrps"}:
        base = weighted_empirical_crps_mean(
            forecast_ens,
            reference_ens,
            threshold_m=float(tail_threshold_m),
            tail_weight=float(tail_weight),
        )
    elif objective_name in {
        "mean_regularized_crps",
        "spread_regularized_crps",
        "combined_regularized_crps",
    }:
        base = empirical_crps_mean(forecast_ens, reference_ens)
    else:
        allowed = ", ".join(sorted(SUPPORTED_MBM_OBJECTIVES))
        raise ValueError(
            "Unsupported rollout_calibration.optimizer.objective "
            f"{objective!r}; expected one of: {allowed}."
        )
    value = float(base)
    if objective_name in {"mean_regularized_crps", "combined_regularized_crps"}:
        value += float(mean_rmse_weight) * ensemble_mean_rmse(forecast_ens, reference_ens)
    if objective_name in {"spread_regularized_crps", "combined_regularized_crps"}:
        value += float(spread_ratio_weight) * spread_ratio_penalty(
            forecast_ens,
            reference_ens,
            target_spread_ratio=float(target_spread_ratio),
        )
    return float(value)


def apply_member_by_member_transform(
    pred_members: np.ndarray,
    *,
    a_m: float,
    beta: float,
    gamma: float,
    clip_nonnegative: bool = True,
) -> np.ndarray:
    """Apply MBM location/scale transform to [members, locations] predictions."""
    x = np.asarray(pred_members, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("pred_members must be [members, locations].")
    mu = np.mean(x, axis=0, keepdims=True)
    out = float(a_m) + float(beta) * mu + float(gamma) * (x - mu)
    if clip_nonnegative:
        out = np.maximum(out, 0.0)
    return out.astype(np.float64, copy=False)


def _fit_one_bin(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    bounds: Mapping[str, Sequence[float]],
    objective: str = "empirical_crps",
    tail_threshold_m: float = 0.30,
    tail_weight: float = 4.0,
    mean_rmse_weight: float = 0.0,
    spread_ratio_weight: float = 0.0,
    target_spread_ratio: float = 1.0,
    multistart: bool = True,
    seed: int = 123,
    maxiter: int = 600,
    progress_callback: Callable[[str], None] | None = None,
    progress_interval: int = 50,
) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.ndim != 2 or ref.ndim != 2 or pred.shape[1] != ref.shape[1]:
        raise ValueError("Bin fit expects pred/ref [members, points] with matching points.")
    if pred.shape[1] == 0:
        return {"a_m": 0.0, "beta": 1.0, "gamma": 1.0, "objective": math.nan, "n_points": 0}
    bnds = [
        tuple(float(v) for v in bounds.get("a_m", (-2.0, 2.0))),
        tuple(float(v) for v in bounds.get("beta", (0.2, 2.5))),
        tuple(float(v) for v in bounds.get("gamma", (0.05, 3.0))),
    ]

    score_calls = 0

    def score(theta: Sequence[float]) -> float:
        nonlocal score_calls
        score_calls += 1
        if (
            progress_callback is not None
            and progress_interval > 0
            and score_calls % int(progress_interval) == 0
        ):
            progress_callback(f"score_calls={score_calls} n_points={pred.shape[1]}")
        cal = apply_member_by_member_transform(
            pred,
            a_m=float(theta[0]),
            beta=float(theta[1]),
            gamma=float(theta[2]),
            clip_nonnegative=True,
        )
        return _objective_value(
            cal,
            ref,
            objective=objective,
            tail_threshold_m=tail_threshold_m,
            tail_weight=tail_weight,
            mean_rmse_weight=mean_rmse_weight,
            spread_ratio_weight=spread_ratio_weight,
            target_spread_ratio=target_spread_ratio,
        )

    def clip_theta(theta: Sequence[float]) -> np.ndarray:
        return np.asarray([np.clip(float(theta[i]), bnds[i][0], bnds[i][1]) for i in range(3)], dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    starts = [np.asarray([0.0, 1.0, 1.0], dtype=np.float64)]
    if multistart:
        starts.extend(
            np.asarray(
                [rng.uniform(*bnds[0]), rng.uniform(*bnds[1]), rng.uniform(*bnds[2])],
                dtype=np.float64,
            )
            for _ in range(4)
        )
    best_x = clip_theta(starts[0])
    best_val = score(best_x)
    methods: List[str] = []
    for start in starts:
        start = clip_theta(start)
        if minimize is not None:
            res = minimize(
                score,
                x0=start,
                method="Powell",
                bounds=bnds,
                options={"maxiter": int(maxiter), "xtol": 1e-3, "ftol": 1e-4},
            )
            methods.append("Powell")
            candidate = clip_theta(getattr(res, "x", start))
            val = score(candidate)
        else:
            val = score(start)
            candidate = start
            for a in np.linspace(bnds[0][0], bnds[0][1], 9):
                for beta in np.linspace(max(0.5, bnds[1][0]), min(1.5, bnds[1][1]), 9):
                    for gamma in np.linspace(max(0.5, bnds[2][0]), min(1.8, bnds[2][1]), 9):
                        grid_val = score((a, beta, gamma))
                        if grid_val < val:
                            val = grid_val
                            candidate = np.asarray([a, beta, gamma], dtype=np.float64)
            methods.append("grid_search")
        if np.isfinite(val) and val <= best_val:
            best_x = candidate
            best_val = float(val)
    return {
        "a_m": float(best_x[0]),
        "beta": float(best_x[1]),
        "gamma": float(best_x[2]),
        "objective": float(best_val),
        "n_points": int(pred.shape[1]),
        "optimizer_methods": sorted(set(methods)),
        "objective_name": str(objective),
        "mean_rmse_weight": float(mean_rmse_weight),
        "spread_ratio_weight": float(spread_ratio_weight),
        "target_spread_ratio": float(target_spread_ratio),
        "score_calls": int(score_calls),
    }

def _concat_member_points(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0, 0), dtype=np.float64)
    return np.concatenate(chunks, axis=1).astype(np.float64, copy=False)


def _validate_artifact_compatibility(artifact_paths: Sequence[str | Path]) -> Dict[str, Any]:
    metas = [load_forecast_artifact(path, load_members=False) for path in artifact_paths]
    if not metas:
        raise ValueError("No forecast artifacts supplied.")
    first = metas[0]
    keys = ("n_forecast_members", "n_reference_members", "n_time", "n_cells")
    for meta in metas[1:]:
        for key in keys:
            if int(meta[key]) != int(first[key]):
                raise ValueError(
                    "Calibration artifacts must have uniform member/time/cell dimensions; "
                    f"{meta['path']} has {key}={meta[key]} but expected {first[key]}."
                )
        first_hash = str(first.get("cell_hash", ""))
        meta_hash = str(meta.get("cell_hash", ""))
        if first_hash and meta_hash and first_hash != meta_hash:
            raise ValueError(
                "Calibration artifact cell/geometry hash mismatch; refusing to apply cell-index calibration "
                f"across incompatible meshes ({first['path']} vs {meta['path']})."
            )
    return {k: first.get(k) for k in (*keys, "cell_hash")}


def _calibration_cell_hash(calibration_model: Mapping[str, Any]) -> str:
    direct = str(calibration_model.get("cell_hash", "") or "").strip()
    if direct:
        return direct
    compatibility = calibration_model.get("artifact_compatibility", {}) or {}
    if isinstance(compatibility, Mapping):
        return str(compatibility.get("cell_hash", "") or "").strip()
    return ""


def _validate_calibration_cell_hash(
    calibration_model: Mapping[str, Any],
    *,
    cell_hash: str | None,
    context: str,
) -> None:
    expected = _calibration_cell_hash(calibration_model)
    if not expected:
        return
    observed = str(cell_hash or "").strip()
    if not observed:
        raise ValueError(
            f"{context} artifact is missing cell_hash; refusing to apply cell-index calibration "
            "without mesh compatibility metadata."
        )
    if observed != expected:
        raise ValueError(
            f"{context} cell_hash mismatch; refusing to apply calibration fitted on a different mesh "
            f"({observed} != {expected})."
        )


def compute_calibration_wet_frequency(
    artifact_paths: Sequence[str | Path],
    *,
    wet_threshold_m: float,
) -> np.ndarray:
    """Compute per-cell wet frequency from calibration GT only."""
    wet_counts: np.ndarray | None = None
    total = 0
    _validate_artifact_compatibility(artifact_paths)
    for path in artifact_paths:
        art = load_forecast_artifact(path, load_members=True)
        ref = np.asarray(art["ref_members_wd"], dtype=np.float64)
        if wet_counts is None:
            wet_counts = np.zeros(ref.shape[2], dtype=np.float64)
        if ref.shape[2] != wet_counts.size:
            raise ValueError("All calibration artifacts must share n_cells for wet-frequency bins.")
        wet_counts += np.sum(ref > float(wet_threshold_m), axis=(0, 1))
        total += int(ref.shape[0] * ref.shape[1])
    if wet_counts is None or total <= 0:
        raise ValueError("No calibration artifacts available for wet-frequency computation.")
    return np.clip(wet_counts / float(total), 0.0, 1.0)


def fit_crps_member_by_member_from_artifacts(
    artifact_paths: Sequence[str | Path],
    *,
    bins: CalibrationBins,
    max_fit_points_per_bin: int = 1_000_000,
    min_fit_points_per_bin: int = 1000,
    seed: int = 123,
    bounds: Mapping[str, Sequence[float]] | None = None,
    objective: str = "empirical_crps",
    tail_threshold_m: float = 0.30,
    tail_weight: float = 4.0,
    mean_rmse_weight: float = 0.0,
    spread_ratio_weight: float = 0.0,
    target_spread_ratio: float = 1.0,
    multistart: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    progress_interval: int = 50,
) -> Dict[str, Any]:
    """Fit CRPS-optimized MBM coefficients per lead-time x wet-frequency bin."""
    bins.validate()
    paths = [Path(p) for p in artifact_paths]
    if not paths:
        raise ValueError("No forecast artifacts supplied for calibration fitting.")
    rng = np.random.default_rng(int(seed))
    artifact_compatibility = _validate_artifact_compatibility(paths)
    wet_frequency = compute_calibration_wet_frequency(paths, wet_threshold_m=bins.wet_threshold_m)
    wet_bin_by_cell = bins.wet_index(wet_frequency)
    n_l, n_w = bins.n_lead_bins, bins.n_wet_bins
    pred_chunks: List[List[List[np.ndarray]]] = [[[] for _ in range(n_w)] for _ in range(n_l)]
    ref_chunks: List[List[List[np.ndarray]]] = [[[] for _ in range(n_w)] for _ in range(n_l)]
    lead_pred: List[List[np.ndarray]] = [[] for _ in range(n_l)]
    lead_ref: List[List[np.ndarray]] = [[] for _ in range(n_l)]
    global_pred: List[np.ndarray] = []
    global_ref: List[np.ndarray] = []
    counts = np.zeros((n_l, n_w), dtype=np.int64)

    for path in paths:
        art = load_forecast_artifact(path, load_members=True)
        pred = np.asarray(art["pred_members_wd"], dtype=np.float64)
        ref = np.asarray(art["ref_members_wd"], dtype=np.float64)
        wettable = np.asarray(art.get("wettable_mask", np.ones(pred.shape[2], dtype=bool)), dtype=bool)
        if pred.shape[2] != wet_frequency.size:
            raise ValueError(f"Artifact {path} n_cells differs from calibration wet-frequency map.")
        time_hours = art.get("time_hours")
        if time_hours is None:
            time_hours = np.arange(1, pred.shape[1] + 1, dtype=np.float64)
        for t in range(pred.shape[1]):
            lead_idx = bins.lead_index(float(time_hours[t]))
            for wet_idx in range(n_w):
                cell_mask = wettable & (wet_bin_by_cell == wet_idx)
                n_cells = int(np.sum(cell_mask))
                if n_cells <= 0:
                    continue
                remaining = max(0, int(max_fit_points_per_bin) - int(counts[lead_idx, wet_idx]))
                if remaining <= 0:
                    continue
                chosen = np.flatnonzero(cell_mask)
                if chosen.size > remaining:
                    chosen = rng.choice(chosen, size=remaining, replace=False)
                pred_sel = pred[:, t, chosen]
                ref_sel = ref[:, t, chosen]
                pred_chunks[lead_idx][wet_idx].append(pred_sel)
                ref_chunks[lead_idx][wet_idx].append(ref_sel)
                lead_pred[lead_idx].append(pred_sel)
                lead_ref[lead_idx].append(ref_sel)
                global_pred.append(pred_sel)
                global_ref.append(ref_sel)
                counts[lead_idx, wet_idx] += int(chosen.size)

    objective_name = str(objective)
    bounds = dict(bounds or {"a_m": (-2.0, 2.0), "beta": (0.2, 2.5), "gamma": (0.05, 3.0)})
    min_points = int(min_fit_points_per_bin)
    lead_point_counts = np.sum(counts, axis=1)
    needs_lead_fit = np.any(counts < min_points, axis=1) & (lead_point_counts >= min_points)
    needs_global_fit = bool(np.any((counts < min_points) & (lead_point_counts[:, None] < min_points)))
    global_fit: Dict[str, Any] | None = None
    if needs_global_fit:
        if progress_callback is not None:
            progress_callback("fit global fallback start")
        global_fit = _fit_one_bin(
            _concat_member_points(global_pred),
            _concat_member_points(global_ref),
            bounds=bounds,
            objective=objective_name,
            tail_threshold_m=tail_threshold_m,
            tail_weight=tail_weight,
            mean_rmse_weight=mean_rmse_weight,
            spread_ratio_weight=spread_ratio_weight,
            target_spread_ratio=target_spread_ratio,
            multistart=multistart,
            seed=seed,
            progress_callback=(
                (lambda message: progress_callback(f"global {message}"))
                if progress_callback is not None
                else None
            ),
            progress_interval=progress_interval,
        )
        if progress_callback is not None:
            progress_callback("fit global fallback done")
    lead_fits: List[Dict[str, Any] | None] = [None for _ in range(n_l)]
    for lead_idx in range(n_l):
        if not bool(needs_lead_fit[lead_idx]):
            continue
        pred_l = _concat_member_points(lead_pred[lead_idx])
        ref_l = _concat_member_points(lead_ref[lead_idx])
        if progress_callback is not None:
            progress_callback(f"fit lead fallback {lead_idx}/{n_l - 1} start n_points={pred_l.shape[1]}")
        lead_fits[lead_idx] = _fit_one_bin(
            pred_l,
            ref_l,
            bounds=bounds,
            objective=objective_name,
            tail_threshold_m=tail_threshold_m,
            tail_weight=tail_weight,
            mean_rmse_weight=mean_rmse_weight,
            spread_ratio_weight=spread_ratio_weight,
            target_spread_ratio=target_spread_ratio,
            multistart=multistart,
            seed=seed + lead_idx + 1,
            progress_callback=(
                (lambda message, idx=lead_idx: progress_callback(f"lead {idx} {message}"))
                if progress_callback is not None
                else None
            ),
            progress_interval=progress_interval,
        )
        if progress_callback is not None:
            progress_callback(f"fit lead fallback {lead_idx}/{n_l - 1} done")

    coeff = np.zeros((n_l, n_w, 3), dtype=np.float64)
    objective_values = np.full((n_l, n_w), np.nan, dtype=np.float64)
    n_points = np.zeros((n_l, n_w), dtype=np.int64)
    fallback = [["direct" for _ in range(n_w)] for _ in range(n_l)]
    for lead_idx in range(n_l):
        for wet_idx in range(n_w):
            pred_b = _concat_member_points(pred_chunks[lead_idx][wet_idx])
            ref_b = _concat_member_points(ref_chunks[lead_idx][wet_idx])
            if pred_b.shape[1] >= int(min_fit_points_per_bin):
                if progress_callback is not None:
                    progress_callback(
                        f"fit lead bin {lead_idx}/{n_l - 1} wet bin {wet_idx}/{n_w - 1} start n_points={pred_b.shape[1]}"
                    )
                fit = _fit_one_bin(
                    pred_b,
                    ref_b,
                    bounds=bounds,
                    objective=objective_name,
                    tail_threshold_m=tail_threshold_m,
                    tail_weight=tail_weight,
                    mean_rmse_weight=mean_rmse_weight,
                    spread_ratio_weight=spread_ratio_weight,
                    target_spread_ratio=target_spread_ratio,
                    multistart=multistart,
                    seed=seed + 1000 * lead_idx + wet_idx,
                    progress_callback=(
                        (
                            lambda message, lidx=lead_idx, widx=wet_idx: progress_callback(
                                f"lead {lidx} wet {widx} {message}"
                            )
                        )
                        if progress_callback is not None
                        else None
                    ),
                    progress_interval=progress_interval,
                )
                if progress_callback is not None:
                    progress_callback(
                        f"fit lead bin {lead_idx}/{n_l - 1} wet bin {wet_idx}/{n_w - 1} done"
                    )
            elif lead_fits[lead_idx] is not None and lead_fits[lead_idx].get("n_points", 0) >= min_points:
                fit = lead_fits[lead_idx]
                fallback[lead_idx][wet_idx] = "lead_all_wet_frequency"
                if progress_callback is not None:
                    progress_callback(
                        f"fit lead bin {lead_idx}/{n_l - 1} wet bin {wet_idx}/{n_w - 1} fallback=lead_all_wet_frequency n_points={pred_b.shape[1]}"
                    )
            else:
                if global_fit is None:
                    if progress_callback is not None:
                        progress_callback("fit global fallback start")
                    global_fit = _fit_one_bin(
                        _concat_member_points(global_pred),
                        _concat_member_points(global_ref),
                        bounds=bounds,
                        objective=objective_name,
                        tail_threshold_m=tail_threshold_m,
                        tail_weight=tail_weight,
                        mean_rmse_weight=mean_rmse_weight,
                        spread_ratio_weight=spread_ratio_weight,
                        target_spread_ratio=target_spread_ratio,
                        multistart=multistart,
                        seed=seed,
                        progress_callback=(
                            (lambda message: progress_callback(f"global {message}"))
                            if progress_callback is not None
                            else None
                        ),
                        progress_interval=progress_interval,
                    )
                    if progress_callback is not None:
                        progress_callback("fit global fallback done")
                fit = global_fit
                fallback[lead_idx][wet_idx] = "global"
                if progress_callback is not None:
                    progress_callback(
                        f"fit lead bin {lead_idx}/{n_l - 1} wet bin {wet_idx}/{n_w - 1} fallback=global n_points={pred_b.shape[1]}"
                    )
            coeff[lead_idx, wet_idx, :] = [fit["a_m"], fit["beta"], fit["gamma"]]
            objective_values[lead_idx, wet_idx] = float(fit.get("objective", np.nan))
            n_points[lead_idx, wet_idx] = int(pred_b.shape[1])
    return {
        "method": SCIENTIFIC_CALIBRATION_METHOD,
        "coefficients": coeff,
        "coefficient_names": ["a_m", "beta", "gamma"],
        "objective_name": objective_name,
        "objective_values": objective_values,
        "objective": objective_values,
        "n_points": n_points,
        "fallback": fallback,
        "lead_time_hours": np.asarray(bins.lead_time_hours, dtype=np.float64),
        "wet_frequency_edges": np.asarray(bins.wet_frequency_edges, dtype=np.float64),
        "wet_threshold_m": float(bins.wet_threshold_m),
        "wet_frequency_by_cell": wet_frequency.astype(np.float64, copy=False),
        "bounds": {k: [float(v[0]), float(v[1])] for k, v in bounds.items()},
        "optimizer": {
            "objective": objective_name,
            "tail_threshold_m": float(tail_threshold_m),
            "tail_weight": float(tail_weight),
            "mean_rmse_weight": float(mean_rmse_weight),
            "spread_ratio_weight": float(spread_ratio_weight),
            "target_spread_ratio": float(target_spread_ratio),
            "multistart": bool(multistart),
            "method": "Powell" if minimize is not None else "grid_search",
            "min_fit_points_per_bin": int(min_fit_points_per_bin),
        },
        "diagnostics": build_fit_diagnostics({
            "coefficients": coeff,
            "coefficient_names": ["a_m", "beta", "gamma"],
            "n_points": n_points,
            "fallback": fallback,
            "bounds": bounds,
        }),
        "cell_hash": str(artifact_compatibility.get("cell_hash", "") or ""),
        "artifact_compatibility": artifact_compatibility,
        "seed": int(seed),
        "artifact_paths": [str(p) for p in paths],
    }


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def build_fit_diagnostics(model: Mapping[str, Any]) -> Dict[str, Any]:
    fallback = np.asarray(model.get("fallback", []), dtype=object)
    n_points = np.asarray(model.get("n_points", []), dtype=np.int64)
    coeff = np.asarray(model.get("coefficients", []), dtype=np.float64)
    bounds = model.get("bounds", {}) or {}
    total_bins = int(fallback.size) if fallback.size else 0
    fallback_counts: Dict[str, int] = {}
    for value in fallback.reshape(-1).tolist() if fallback.size else []:
        fallback_counts[str(value)] = fallback_counts.get(str(value), 0) + 1
    bound_hits: Dict[str, int] = {}
    names = list(model.get("coefficient_names", ["a_m", "beta", "gamma"]))
    if coeff.size:
        for idx, name in enumerate(names):
            lo_hi = bounds.get(name)
            if lo_hi is None:
                continue
            lo, hi = float(lo_hi[0]), float(lo_hi[1])
            vals = coeff[..., idx]
            bound_hits[f"{name}_lower"] = int(np.sum(np.isclose(vals, lo, rtol=0.0, atol=1e-8)))
            bound_hits[f"{name}_upper"] = int(np.sum(np.isclose(vals, hi, rtol=0.0, atol=1e-8)))
    global_fallback_rate = float(fallback_counts.get("global", 0)) / float(total_bins) if total_bins else 0.0
    any_bound_hits = int(sum(bound_hits.values()))
    warnings: List[str] = []
    if global_fallback_rate > 0.25:
        warnings.append("More than 25% of calibration bins fell back to global coefficients.")
    if total_bins and any_bound_hits / float(total_bins * max(len(names), 1)) > 0.10:
        warnings.append("More than 10% of fitted coefficients are saturated at configured bounds.")
    return {
        "total_bins": total_bins,
        "fallback_counts": fallback_counts,
        "global_fallback_rate": global_fallback_rate,
        "n_points_min": int(np.min(n_points)) if n_points.size else 0,
        "n_points_max": int(np.max(n_points)) if n_points.size else 0,
        "n_points_by_bin": n_points,
        "bound_hits": bound_hits,
        "warnings": warnings,
    }


def save_fit_diagnostics(model: Mapping[str, Any], out_dir: str | Path) -> Path:
    path = Path(out_dir) / FIT_DIAGNOSTICS_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = dict(model.get("diagnostics") or build_fit_diagnostics(model))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonify(diagnostics), handle, indent=2, sort_keys=True)
    return path


def save_crps_mbm_coefficients(model: Mapping[str, Any], out_dir: str | Path) -> Path:
    path = Path(out_dir) / COEFFICIENTS_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonify(dict(model)), handle, indent=2, sort_keys=True)
    return path


def load_crps_mbm_coefficients(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for key in ("coefficients", "lead_time_hours", "wet_frequency_edges", "wet_frequency_by_cell"):
        if key in payload:
            payload[key] = np.asarray(payload[key], dtype=np.float64)
    if "n_points" in payload:
        payload["n_points"] = np.asarray(payload["n_points"], dtype=np.int64)
    return payload


def apply_crps_mbm_to_wd_members(
    pred_members_wd: np.ndarray,
    *,
    lead_time_hour: float,
    calibration_model: Mapping[str, Any],
    wettable_mask: np.ndarray | None = None,
    cell_hash: str | None = None,
) -> np.ndarray:
    """Apply fitted MBM calibration to one lead-time WD ensemble [members, cells]."""
    pred = np.asarray(pred_members_wd, dtype=np.float64)
    if pred.ndim != 2:
        raise ValueError("pred_members_wd must have shape [members, cells].")
    if cell_hash is not None:
        _validate_calibration_cell_hash(
            calibration_model,
            cell_hash=cell_hash,
            context="CRPS-MBM application",
        )
    coeff = np.asarray(calibration_model["coefficients"], dtype=np.float64)
    bins = CalibrationBins(
        lead_time_hours=tuple(np.asarray(calibration_model["lead_time_hours"], dtype=np.float64).tolist()),
        wet_frequency_edges=tuple(np.asarray(calibration_model["wet_frequency_edges"], dtype=np.float64).tolist()),
        wet_threshold_m=float(calibration_model.get("wet_threshold_m", 0.01)),
    )
    wet_frequency = np.asarray(calibration_model["wet_frequency_by_cell"], dtype=np.float64).reshape(-1)
    if wet_frequency.size != pred.shape[1]:
        raise ValueError(
            "Calibration wet_frequency_by_cell length does not match prediction cells: "
            f"{wet_frequency.size} vs {pred.shape[1]}."
        )
    lead_idx = bins.lead_index(float(lead_time_hour))
    wet_idx_by_cell = bins.wet_index(wet_frequency)
    out = np.zeros_like(pred, dtype=np.float64)
    if wettable_mask is None:
        wettable = np.ones(pred.shape[1], dtype=bool)
    else:
        wettable = np.asarray(wettable_mask, dtype=bool).reshape(-1)
        if wettable.size != pred.shape[1]:
            raise ValueError("wettable_mask length does not match prediction cells.")
    for wet_idx in range(bins.n_wet_bins):
        mask = wettable & (wet_idx_by_cell == wet_idx)
        if not np.any(mask):
            continue
        a_m, beta, gamma = coeff[lead_idx, wet_idx]
        out[:, mask] = apply_member_by_member_transform(
            pred[:, mask], a_m=float(a_m), beta=float(beta), gamma=float(gamma)
        )
    out[:, ~wettable] = 0.0
    return out


def pava_isotonic_fit(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Dict[str, Any]:
    """Fit monotone nondecreasing isotonic regression with PAVA."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have the same size for isotonic fit.")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if sample_weight is None:
        w = np.ones_like(y, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)[finite]
        w = np.maximum(w, 0.0)
    if x.size == 0:
        return {"x": [], "y": [], "n": 0}
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    w = w[order]
    blocks: List[Dict[str, float]] = []
    for xi, yi, wi in zip(x, y, w):
        if wi <= 0:
            continue
        blocks.append({"x_min": float(xi), "x_max": float(xi), "sum_w": float(wi), "sum_y": float(wi * yi)})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_val = left["sum_y"] / max(left["sum_w"], MIN_EPS)
            right_val = right["sum_y"] / max(right["sum_w"], MIN_EPS)
            if left_val <= right_val:
                break
            merged = {
                "x_min": left["x_min"],
                "x_max": right["x_max"],
                "sum_w": left["sum_w"] + right["sum_w"],
                "sum_y": left["sum_y"] + right["sum_y"],
            }
            blocks[-2:] = [merged]
    xs = [block["x_max"] for block in blocks]
    ys = [float(np.clip(block["sum_y"] / max(block["sum_w"], MIN_EPS), 0.0, 1.0)) for block in blocks]
    return {"x": xs, "y": ys, "n": int(x.size)}


def isotonic_predict(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    xs = np.asarray(model.get("x", []), dtype=np.float64)
    ys = np.asarray(model.get("y", []), dtype=np.float64)
    values = np.asarray(x, dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return np.clip(values, 0.0, 1.0)
    idx = np.searchsorted(xs, values, side="left")
    idx = np.clip(idx, 0, ys.size - 1)
    return ys[idx]


def fit_exceedance_isotonic_from_artifacts(
    artifact_paths: Sequence[str | Path],
    *,
    bins: CalibrationBins,
    wet_frequency_by_cell: np.ndarray,
    thresholds_m: Sequence[float] = DEFAULT_THRESHOLDS_M,
    min_fit_points_per_bin: int = 128,
    calibration_model: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Fit isotonic calibration for exceedance probabilities by threshold/lead/wet bin."""
    bins.validate()
    wet_frequency = np.asarray(wet_frequency_by_cell, dtype=np.float64).reshape(-1)
    artifact_compatibility = _validate_artifact_compatibility(artifact_paths)
    if calibration_model is not None:
        model_hash = _calibration_cell_hash(calibration_model)
        artifact_hash = str(artifact_compatibility.get("cell_hash", "") or "")
        if model_hash and artifact_hash and model_hash != artifact_hash:
            raise ValueError(
                "Isotonic calibration artifacts do not match CRPS-MBM calibration cell_hash "
                f"({artifact_hash} != {model_hash})."
            )
    wet_bin_by_cell = bins.wet_index(wet_frequency)
    payload: Dict[str, Any] = {
        "method": "isotonic_pava",
        "probability_input": "crps_mbm_calibrated_members" if calibration_model is not None else "raw_members",
        "thresholds_m": [float(t) for t in thresholds_m],
        "lead_time_hours": list(bins.lead_time_hours),
        "wet_frequency_edges": list(bins.wet_frequency_edges),
        "cell_hash": str(artifact_compatibility.get("cell_hash", "") or ""),
        "curves": {},
    }
    for threshold in thresholds_m:
        key_t = f"{float(threshold):.6g}"
        payload["curves"][key_t] = {}
        raw_by_bin: Dict[Tuple[int, int], List[np.ndarray]] = {}
        obs_by_bin: Dict[Tuple[int, int], List[np.ndarray]] = {}
        raw_by_lead: Dict[int, List[np.ndarray]] = {}
        obs_by_lead: Dict[int, List[np.ndarray]] = {}
        for path in artifact_paths:
            art = load_forecast_artifact(path, load_members=True)
            pred = np.asarray(art["pred_members_wd"], dtype=np.float64)
            ref = np.asarray(art["ref_members_wd"], dtype=np.float64)
            if pred.shape[2] != wet_frequency.size:
                raise ValueError(
                    f"Artifact {path} n_cells differs from isotonic wet-frequency map."
                )
            wettable = np.asarray(art.get("wettable_mask", np.ones(pred.shape[2], dtype=bool)), dtype=bool)
            time_hours = art.get("time_hours")
            if time_hours is None:
                time_hours = np.arange(1, pred.shape[1] + 1, dtype=np.float64)
            for t in range(pred.shape[1]):
                lead_idx = bins.lead_index(float(time_hours[t]))
                pred_t = pred[:, t, :]
                if calibration_model is not None:
                    pred_t = apply_crps_mbm_to_wd_members(
                        pred_t,
                        lead_time_hour=float(time_hours[t]),
                        calibration_model=calibration_model,
                        wettable_mask=wettable,
                        cell_hash=str(art.get("cell_hash", "")),
                    )
                raw_prob = np.mean(pred_t > float(threshold), axis=0)
                obs_prob = np.mean(ref[:, t, :] > float(threshold), axis=0)
                for wet_idx in range(bins.n_wet_bins):
                    mask = wettable & (wet_bin_by_cell == wet_idx)
                    if not np.any(mask):
                        continue
                    raw_by_bin.setdefault((lead_idx, wet_idx), []).append(raw_prob[mask])
                    obs_by_bin.setdefault((lead_idx, wet_idx), []).append(obs_prob[mask])
                    raw_by_lead.setdefault(lead_idx, []).append(raw_prob[mask])
                    obs_by_lead.setdefault(lead_idx, []).append(obs_prob[mask])
        lead_models = {}
        for lead_idx in range(bins.n_lead_bins):
            raw = np.concatenate(raw_by_lead.get(lead_idx, [np.empty(0)]))
            obs = np.concatenate(obs_by_lead.get(lead_idx, [np.empty(0)]))
            lead_models[lead_idx] = pava_isotonic_fit(raw, obs) if raw.size else {"x": [], "y": [], "n": 0}
        for lead_idx in range(bins.n_lead_bins):
            for wet_idx in range(bins.n_wet_bins):
                raw = np.concatenate(raw_by_bin.get((lead_idx, wet_idx), [np.empty(0)]))
                obs = np.concatenate(obs_by_bin.get((lead_idx, wet_idx), [np.empty(0)]))
                if raw.size >= int(min_fit_points_per_bin):
                    curve = pava_isotonic_fit(raw, obs)
                    fallback = "direct"
                elif lead_models[lead_idx].get("n", 0) >= int(min_fit_points_per_bin):
                    curve = lead_models[lead_idx]
                    fallback = "lead_all_wet_frequency"
                else:
                    all_raw = np.concatenate([np.concatenate(v) for v in raw_by_lead.values()]) if raw_by_lead else np.empty(0)
                    all_obs = np.concatenate([np.concatenate(v) for v in obs_by_lead.values()]) if obs_by_lead else np.empty(0)
                    curve = pava_isotonic_fit(all_raw, all_obs) if all_raw.size else {"x": [], "y": [], "n": 0}
                    fallback = "global"
                payload["curves"][key_t][f"lead_{lead_idx}_wet_{wet_idx}"] = {**curve, "fallback": fallback}
    return payload


def apply_isotonic_exceedance_probability(
    raw_probability: np.ndarray,
    *,
    threshold_m: float,
    lead_time_hour: float,
    wet_frequency_by_cell: np.ndarray,
    isotonic_model: Mapping[str, Any],
) -> np.ndarray:
    values = np.asarray(raw_probability, dtype=np.float64)
    bins = CalibrationBins(
        lead_time_hours=tuple(np.asarray(isotonic_model["lead_time_hours"], dtype=np.float64).tolist()),
        wet_frequency_edges=tuple(np.asarray(isotonic_model["wet_frequency_edges"], dtype=np.float64).tolist()),
        wet_threshold_m=0.01,
    )
    lead_idx = bins.lead_index(float(lead_time_hour))
    wet_idx_by_cell = bins.wet_index(np.asarray(wet_frequency_by_cell, dtype=np.float64))
    threshold_key = f"{float(threshold_m):.6g}"
    curves = isotonic_model.get("curves", {}).get(threshold_key, {})
    out = np.clip(values.copy(), 0.0, 1.0)
    for wet_idx in range(bins.n_wet_bins):
        mask = wet_idx_by_cell == wet_idx
        if not np.any(mask):
            continue
        curve = curves.get(f"lead_{lead_idx}_wet_{wet_idx}")
        if curve is not None:
            out[mask] = isotonic_predict(curve, out[mask])
    return np.clip(out, 0.0, 1.0)


def _validate_isotonic_probability_input(
    *,
    isotonic_model: Mapping[str, Any],
    calibration_model: Mapping[str, Any] | None,
) -> None:
    expected = "crps_mbm_calibrated_members" if calibration_model is not None else "raw_members"
    actual = str(isotonic_model.get("probability_input", "") or "").strip()
    if not actual:
        raise ValueError(
            "Isotonic calibration model is missing probability_input metadata; refit isotonic curves "
            "with the current calibration code to avoid mixing raw and calibrated probability scales."
        )
    if actual != expected:
        raise ValueError(
            "Isotonic calibration probability_input mismatch: "
            f"model expects {actual!r}, but this metric path is using {expected!r}."
        )


def save_exceedance_isotonic(model: Mapping[str, Any], out_dir: str | Path) -> Path:
    path = Path(out_dir) / ISOTONIC_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonify(dict(model)), handle, indent=2, sort_keys=True)
    return path


def _coverage_and_width(pred: np.ndarray, ref: np.ndarray, alpha: float) -> Tuple[float, float]:
    lo_q = 0.5 * (1.0 - alpha)
    hi_q = 1.0 - lo_q
    lo = np.quantile(pred, lo_q, axis=0)
    hi = np.quantile(pred, hi_q, axis=0)
    cover = np.mean((ref >= lo[None, :]) & (ref <= hi[None, :]))
    width = np.mean(hi - lo)
    return float(cover), float(width)


def compute_artifact_uq_metrics(
    artifact_paths: Sequence[str | Path],
    *,
    calibration_model: Mapping[str, Any] | None = None,
    isotonic_model: Mapping[str, Any] | None = None,
    apply_isotonic: bool = False,
    thresholds_m: Sequence[float] = DEFAULT_THRESHOLDS_M,
) -> Dict[str, float]:
    """Compute compact raw/calibrated UQ metrics from saved artifacts."""
    crps_vals: List[float] = []
    rmse_vals: List[float] = []
    spread_vals: List[float] = []
    ref_spread_vals: List[float] = []
    brier: Dict[float, List[float]] = {float(t): [] for t in thresholds_m}
    brier_iso: Dict[float, List[float]] = {float(t): [] for t in thresholds_m}
    reliability_abs: Dict[float, List[float]] = {float(t): [] for t in thresholds_m}
    rank_l1_vals: List[float] = []
    tail_crps_030_vals: List[float] = []
    coverage: Dict[float, List[float]] = {0.5: [], 0.8: [], 0.9: [], 0.95: []}
    width: Dict[float, List[float]] = {0.5: [], 0.8: [], 0.9: [], 0.95: []}
    _validate_artifact_compatibility(artifact_paths)
    if apply_isotonic and isotonic_model is not None:
        _validate_isotonic_probability_input(
            isotonic_model=isotonic_model,
            calibration_model=calibration_model,
        )
    for path in artifact_paths:
        art = load_forecast_artifact(path, load_members=True)
        pred = np.asarray(art["pred_members_wd"], dtype=np.float64)
        ref = np.asarray(art["ref_members_wd"], dtype=np.float64)
        wettable = np.asarray(art.get("wettable_mask", np.ones(pred.shape[2], dtype=bool)), dtype=bool)
        time_hours = art.get("time_hours")
        if time_hours is None:
            time_hours = np.arange(1, pred.shape[1] + 1, dtype=np.float64)
        for t in range(pred.shape[1]):
            pred_t = pred[:, t, :]
            ref_t = ref[:, t, :]
            if calibration_model is not None:
                pred_t = apply_crps_mbm_to_wd_members(
                    pred_t,
                    lead_time_hour=float(time_hours[t]),
                    calibration_model=calibration_model,
                    wettable_mask=wettable,
                    cell_hash=str(art.get("cell_hash", "")),
                )
            pred_sel = pred_t[:, wettable]
            ref_sel = ref_t[:, wettable]
            if pred_sel.size == 0 or ref_sel.size == 0:
                continue
            crps_vals.append(empirical_crps_mean(pred_sel, ref_sel))
            tail_crps_030_vals.append(weighted_empirical_crps_mean(pred_sel, ref_sel, threshold_m=0.30, tail_weight=4.0))
            rmse_vals.append(float(np.sqrt(np.mean((np.mean(pred_sel, axis=0) - np.mean(ref_sel, axis=0)) ** 2))))
            ranks = np.sum(pred_sel[:, None, :] < ref_sel[None, :, :], axis=0).reshape(-1)
            hist = np.bincount(np.clip(ranks, 0, pred_sel.shape[0]), minlength=pred_sel.shape[0] + 1).astype(np.float64)
            hist /= max(float(np.sum(hist)), MIN_EPS)
            rank_l1_vals.append(float(np.mean(np.abs(hist - 1.0 / hist.size))))
            spread_vals.append(float(np.mean(np.std(pred_sel, axis=0))))
            ref_spread_vals.append(float(np.mean(np.std(ref_sel, axis=0))))
            for alpha in coverage:
                cov, wid = _coverage_and_width(pred_sel, ref_sel, alpha)
                coverage[alpha].append(cov)
                width[alpha].append(wid)
            for threshold in thresholds_m:
                raw_prob = np.mean(pred_sel > float(threshold), axis=0)
                obs_prob = np.mean(ref_sel > float(threshold), axis=0)
                brier[float(threshold)].append(float(np.mean((raw_prob - obs_prob) ** 2)))
                reliability_abs[float(threshold)].append(float(abs(np.mean(raw_prob) - np.mean(obs_prob))))
                if apply_isotonic and isotonic_model is not None and calibration_model is not None:
                    wet_freq = np.asarray(calibration_model["wet_frequency_by_cell"], dtype=np.float64)[wettable]
                    iso_prob = apply_isotonic_exceedance_probability(
                        raw_prob,
                        threshold_m=float(threshold),
                        lead_time_hour=float(time_hours[t]),
                        wet_frequency_by_cell=wet_freq,
                        isotonic_model=isotonic_model,
                    )
                    brier_iso[float(threshold)].append(float(np.mean((iso_prob - obs_prob) ** 2)))
    out = {
        "crps_wd_overall_mean": float(np.mean(crps_vals)) if crps_vals else math.nan,
        "crps_tail_weighted_wd_exceed_0p30m_overall_mean": float(np.mean(tail_crps_030_vals)) if tail_crps_030_vals else math.nan,
        "rmse_ens_mean_wd_overall_mean": float(np.mean(rmse_vals)) if rmse_vals else math.nan,
        "rmse_wd_overall_mean": float(np.mean(rmse_vals)) if rmse_vals else math.nan,
        "rank_histogram_l1_wd_overall_mean": float(np.mean(rank_l1_vals)) if rank_l1_vals else math.nan,
        "spread_pred_wd_overall_mean": float(np.mean(spread_vals)) if spread_vals else math.nan,
        "spread_gt_wd_overall_mean": float(np.mean(ref_spread_vals)) if ref_spread_vals else math.nan,
        "spread_ratio_wd_overall_mean": float(np.mean(spread_vals) / max(np.mean(ref_spread_vals), MIN_EPS)) if spread_vals and ref_spread_vals else math.nan,
    }
    for alpha, values in coverage.items():
        pct = int(round(alpha * 100))
        out[f"coverage_wd_{pct}_overall_mean"] = float(np.mean(values)) if values else math.nan
        out[f"width_wd_{pct}_overall_mean"] = float(np.mean(width[alpha])) if width[alpha] else math.nan
    for threshold, values in brier.items():
        key = f"brier_wd_exceed_{threshold:.2f}m_overall_mean".replace(".", "p")
        out[key] = float(np.mean(values)) if values else math.nan
        rel_key = f"reliability_abs_wd_exceed_{threshold:.2f}m_overall_mean".replace(".", "p")
        out[rel_key] = float(np.mean(reliability_abs[float(threshold)])) if reliability_abs[float(threshold)] else math.nan
        iso_key = f"brier_isotonic_wd_exceed_{threshold:.2f}m_overall_mean".replace(".", "p")
        out[iso_key] = float(np.mean(brier_iso[float(threshold)])) if brier_iso[float(threshold)] else math.nan
    return out


def save_metrics_json(metrics: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonify(dict(metrics)), handle, indent=2, sort_keys=True)
    return path


def build_calibration_comparison(raw: Mapping[str, Any], calibrated: Mapping[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for key in sorted(set(raw.keys()).intersection(calibrated.keys())):
        rv = raw[key]
        cv = calibrated[key]
        if isinstance(rv, bool) or isinstance(cv, bool):
            continue
        try:
            rv_f = float(rv)
            cv_f = float(cv)
        except Exception:
            continue
        if not (np.isfinite(rv_f) and np.isfinite(cv_f)):
            continue
        delta = cv_f - rv_f
        metrics[key] = {
            "raw": rv_f,
            "calibrated": cv_f,
            "delta": float(delta),
            "percent_change": None if abs(rv_f) < MIN_EPS else float(100.0 * delta / abs(rv_f)),
        }
    return {"n_metrics_compared": len(metrics), "metrics": metrics}
