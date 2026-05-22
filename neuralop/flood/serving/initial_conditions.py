"""Initial water-depth history providers for FGN serving.

The serving rollout needs three dynamic water-depth frames before the first
forecast step. This module keeps that policy behind a small seam so tests can
use a dry diagnostic start while production can use a reproducible
forcing-conditioned baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

import numpy as np

if TYPE_CHECKING:
    from neuralop.flood.serving.forcing import ForcingInput


INITIAL_CONDITION_DRY = "dry"
INITIAL_CONDITION_FORCING_BASELINE = "forcing_conditioned_baseline"
DEFAULT_INITIAL_CONDITION_LIBRARY = "initial_conditions/forcing_conditioned_initial_wd.npz"


@dataclass(frozen=True)
class InitialConditionConfig:
    """Model-bundle initial-condition policy."""

    default_mode: str = INITIAL_CONDITION_DRY
    library_path: Path | None = None
    reference_scope: str = "train_calibration"
    k_neighbors: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_metadata(self) -> dict[str, Any]:
        return {
            "default_mode": self.default_mode,
            "reference_scope": self.reference_scope,
            "k_neighbors": int(self.k_neighbors),
            "has_library": self.library_path is not None,
            "library_name": self.library_path.name if self.library_path is not None else None,
        }


@dataclass(frozen=True)
class InitialConditionResult:
    """Resolved initial WD history plus user-facing provenance."""

    wd_history_m: np.ndarray
    selection: dict[str, Any]


class InitialConditionProvider(Protocol):
    def resolve(
        self,
        forcing_input: "ForcingInput",
        *,
        bundle: Any,
        n_cells: int,
    ) -> InitialConditionResult: ...


def initial_condition_provider_from_bundle(bundle: Any) -> InitialConditionProvider:
    config = getattr(bundle, "initial_condition", InitialConditionConfig())
    if config.default_mode == INITIAL_CONDITION_DRY:
        return DryInitialConditionProvider()
    if config.default_mode == INITIAL_CONDITION_FORCING_BASELINE:
        return ForcingConditionedBaselineProvider.from_bundle(bundle)
    raise ValueError(f"Unsupported initial-condition mode: {config.default_mode!r}.")


class DryInitialConditionProvider:
    """Diagnostic provider that reproduces the original dry-start behavior."""

    def resolve(
        self,
        forcing_input: "ForcingInput",
        *,
        bundle: Any,
        n_cells: int,
    ) -> InitialConditionResult:
        wd_history = np.zeros((int(bundle.n_history), int(n_cells), 1), dtype=np.float32)
        return InitialConditionResult(
            wd_history_m=wd_history,
            selection={
                "mode": INITIAL_CONDITION_DRY,
                "library_id": None,
                "reference_scope": None,
                "history_rows": int(bundle.skip_before_timestep) + int(bundle.n_history),
                "selected_reference_ids": [],
                "weights": [],
                "distances": [],
                "low_confidence": False,
                "confidence_label": "diagnostic_dry",
                "note": "Dry zero water-depth history used for diagnostic parity.",
            },
        )


class ForcingConditionedBaselineProvider:
    """Deterministic weighted nearest-neighbor initial WD provider."""

    def __init__(
        self,
        *,
        library_path: Path,
        k_neighbors: int = 5,
    ) -> None:
        self.library_path = Path(library_path).expanduser().resolve()
        self.k_neighbors = max(1, int(k_neighbors))
        self._library = _load_initial_condition_library(self.library_path)

    @classmethod
    def from_bundle(cls, bundle: Any) -> "ForcingConditionedBaselineProvider":
        config = getattr(bundle, "initial_condition", None)
        if config is None or config.library_path is None:
            raise ValueError("Forcing-conditioned baseline requires an initial-condition library path.")
        return cls(library_path=Path(config.library_path), k_neighbors=int(config.k_neighbors))

    def resolve(
        self,
        forcing_input: "ForcingInput",
        *,
        bundle: Any,
        n_cells: int,
    ) -> InitialConditionResult:
        history_rows = int(bundle.skip_before_timestep) + int(bundle.n_history)
        boundary = forcing_input.as_boundary_matrix()
        if boundary.shape[0] < history_rows:
            raise ValueError(
                f"Forcing input has {boundary.shape[0]} rows but initial-condition lookup needs {history_rows}."
            )
        feature = build_initial_condition_features(
            boundary[:history_rows, 0],
            boundary[:history_rows, 1],
            history_rows=history_rows,
        )
        library = self._library
        if feature.shape[0] != library.features.shape[1]:
            raise ValueError(
                "Initial-condition feature length mismatch: "
                f"forcing={feature.shape[0]}, library={library.features.shape[1]}."
            )
        if library.wd_history_m.shape[1] != int(bundle.n_history):
            raise ValueError(
                "Initial-condition library n_history mismatch: "
                f"library={library.wd_history_m.shape[1]}, bundle={bundle.n_history}."
            )
        if library.wd_history_m.shape[2] != int(n_cells):
            raise ValueError(
                "Initial-condition library cell count mismatch: "
                f"library={library.wd_history_m.shape[2]}, domain={n_cells}."
            )
        scaled = _standardize(feature, library.feature_median, library.feature_iqr)
        scaled_ref = _standardize(library.features, library.feature_median, library.feature_iqr)
        distances = np.linalg.norm(scaled_ref - scaled[None, :], axis=1) / max(1.0, np.sqrt(feature.shape[0]))
        order = np.argsort(distances, kind="mergesort")
        nearest = order[: min(self.k_neighbors, order.size)]
        nearest_dist = distances[nearest]
        exact = nearest_dist.size > 0 and float(nearest_dist[0]) <= 1.0e-5
        if exact:
            nearest = nearest[:1]
            nearest_dist = nearest_dist[:1]
            weights = np.ones((1,), dtype=np.float64)
        else:
            kernel_scale = max(float(nearest_dist[-1]) if nearest_dist.size else 1.0, 1.0e-6)
            weights = np.exp(-0.5 * np.square(nearest_dist.astype(np.float64) / kernel_scale))
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                weights = np.ones_like(weights, dtype=np.float64) / max(1, weights.size)
            else:
                weights = weights / weight_sum
        wd_history = np.tensordot(weights.astype(np.float32), library.wd_history_m[nearest], axes=(0, 0))
        wd_history = np.clip(wd_history.astype(np.float32, copy=False), 0.0, None)
        low_confidence = bool(float(nearest_dist[0]) > library.reference_distance_p95) if nearest_dist.size else True
        confidence_label = "low" if low_confidence else ("exact" if exact else "nominal")
        selection = {
            "mode": INITIAL_CONDITION_FORCING_BASELINE,
            "library_id": library.library_id,
            "library_path": self.library_path.name,
            "reference_scope": library.reference_scope,
            "history_rows": history_rows,
            "k_neighbors": int(nearest.size),
            "selected_reference_ids": [str(library.reference_ids[i]) for i in nearest],
            "weights": [round(float(x), 8) for x in weights],
            "distances": [round(float(x), 8) for x in nearest_dist],
            "nearest_distance": round(float(nearest_dist[0]), 8) if nearest_dist.size else None,
            "reference_distance_p95": round(float(library.reference_distance_p95), 8),
            "low_confidence": low_confidence,
            "confidence_label": confidence_label,
            "feature_space": "first_history_forcing_rows_plus_descriptors",
        }
        return InitialConditionResult(wd_history_m=wd_history, selection=selection)


@dataclass(frozen=True)
class _InitialConditionLibrary:
    reference_ids: np.ndarray
    features: np.ndarray
    feature_median: np.ndarray
    feature_iqr: np.ndarray
    feature_names: np.ndarray
    wd_history_m: np.ndarray
    reference_distance_p95: float
    metadata: Mapping[str, Any]

    @property
    def library_id(self) -> str:
        return str(self.metadata.get("library_id") or self.metadata.get("bundle_id") or "initial-condition-library")

    @property
    def reference_scope(self) -> str:
        return str(self.metadata.get("reference_scope") or "train_calibration")


def build_initial_condition_features(
    stage: np.ndarray,
    precipitation: np.ndarray,
    *,
    history_rows: int,
) -> np.ndarray:
    """Build the deterministic feature vector used for nearest-neighbor lookup."""
    stage = np.asarray(stage, dtype=np.float32).reshape(-1)
    precipitation = np.asarray(precipitation, dtype=np.float32).reshape(-1)
    if stage.size < history_rows or precipitation.size < history_rows:
        raise ValueError(f"Expected at least {history_rows} forcing rows for initial-condition features.")
    stage = stage[:history_rows]
    precipitation = precipitation[:history_rows]
    precip_log = np.log1p(np.clip(precipitation, 0.0, None))
    time = np.arange(history_rows, dtype=np.float32)
    stage_slope = 0.0
    if history_rows > 1:
        centered = time - float(np.mean(time))
        denom = float(np.sum(centered**2))
        if denom > 0.0:
            stage_slope = float(np.sum(centered * (stage - float(np.mean(stage)))) / denom)
    descriptors = np.asarray(
        [
            float(np.mean(stage)),
            float(np.min(stage)),
            float(np.max(stage)),
            float(stage[-1] - stage[0]),
            stage_slope,
            float(np.sum(precip_log)),
            float(np.max(precip_log)),
            float(np.count_nonzero(precipitation > 0.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([stage.astype(np.float32), precip_log.astype(np.float32), descriptors], axis=0)


def initial_condition_feature_names(*, history_rows: int) -> list[str]:
    names = [f"stage_t{i:02d}" for i in range(history_rows)]
    names.extend(f"precip_log1p_t{i:02d}" for i in range(history_rows))
    names.extend(
        [
            "stage_mean",
            "stage_min",
            "stage_max",
            "stage_delta",
            "stage_slope",
            "precip_log1p_sum",
            "precip_log1p_max",
            "precip_active_rows",
        ]
    )
    return names


def _load_initial_condition_library(path: Path) -> _InitialConditionLibrary:
    if not path.exists():
        raise FileNotFoundError(f"Missing initial-condition library: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "reference_ids",
            "features",
            "feature_median",
            "feature_iqr",
            "feature_names",
            "wd_history_m",
            "metadata_json",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Malformed initial-condition library {path}: missing {missing}.")
        features = np.asarray(data["features"], dtype=np.float32)
        wd_history = np.asarray(data["wd_history_m"], dtype=np.float32)
        if wd_history.ndim == 3:
            wd_history = wd_history[..., None]
        if features.ndim != 2:
            raise ValueError(f"Initial-condition features must have shape [n_reference,n_features], got {features.shape}.")
        if wd_history.ndim != 4:
            raise ValueError(
                "Initial-condition WD history must have shape [n_reference,n_history,n_cells,1], "
                f"got {wd_history.shape}."
            )
        if wd_history.shape[0] != features.shape[0]:
            raise ValueError(
                "Initial-condition reference count mismatch: "
                f"features={features.shape[0]}, wd_history={wd_history.shape[0]}."
            )
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        reference_distance_p95 = float(
            np.asarray(data["reference_distance_p95"]).item() if "reference_distance_p95" in data.files else 0.0
        )
        if reference_distance_p95 <= 0.0 or not np.isfinite(reference_distance_p95):
            reference_distance_p95 = _reference_distance_p95(features, data["feature_median"], data["feature_iqr"])
        return _InitialConditionLibrary(
            reference_ids=np.asarray(data["reference_ids"]).astype(str),
            features=features,
            feature_median=np.asarray(data["feature_median"], dtype=np.float32),
            feature_iqr=np.asarray(data["feature_iqr"], dtype=np.float32),
            feature_names=np.asarray(data["feature_names"]).astype(str),
            wd_history_m=wd_history,
            reference_distance_p95=reference_distance_p95,
            metadata=metadata,
        )


def validate_initial_condition_library(
    path: Path,
    *,
    n_history: int,
    expected_n_cells: int | None = None,
) -> None:
    library = _load_initial_condition_library(Path(path))
    if library.wd_history_m.shape[1] != int(n_history):
        raise ValueError(
            f"Initial-condition library n_history={library.wd_history_m.shape[1]} does not match bundle n_history={n_history}."
        )
    if expected_n_cells is not None and library.wd_history_m.shape[2] != int(expected_n_cells):
        raise ValueError(
            "Initial-condition library cell count mismatch: "
            f"{library.wd_history_m.shape[2]} != {expected_n_cells}."
        )
    if np.any(~np.isfinite(library.wd_history_m)) or np.any(library.wd_history_m < 0.0):
        raise ValueError("Initial-condition library contains invalid WD values.")


def _standardize(values: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    median = np.asarray(median, dtype=np.float32)
    iqr = np.asarray(iqr, dtype=np.float32)
    safe_iqr = np.where(np.abs(iqr) > 1.0e-6, iqr, 1.0).astype(np.float32)
    return (values - median) / safe_iqr


def _reference_distance_p95(features: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> float:
    scaled = _standardize(features, median, iqr)
    if scaled.shape[0] < 2:
        return 1.0
    distances = []
    for idx in range(scaled.shape[0]):
        diff = scaled - scaled[idx : idx + 1]
        d = np.linalg.norm(diff, axis=1) / max(1.0, np.sqrt(scaled.shape[1]))
        d[idx] = np.inf
        distances.append(float(np.min(d)))
    value = float(np.percentile(np.asarray(distances, dtype=np.float32), 95))
    return value if value > 0.0 and np.isfinite(value) else 1.0
