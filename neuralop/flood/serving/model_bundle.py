"""Versioned deployment model-bundle contract for coastal FGN serving.

The bundle is the scientific contract for the web deployment. Route handlers and
workers should depend on this module instead of passing mutable training/eval
configs through the web stack.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from neuralop.flood.serving.initial_conditions import (
    INITIAL_CONDITION_DRY,
    INITIAL_CONDITION_FORCING_BASELINE,
    InitialConditionConfig,
    validate_initial_condition_library,
)


COASTAL_FGN_DT_SECONDS = 900


class ModelBundleError(ValueError):
    """Raised when a deployment bundle is missing or scientifically incompatible."""


def _as_path(value: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def stable_file_hash(paths: Iterable[str | Path]) -> str:
    """Return a deterministic SHA256 digest over one or more files."""
    digest = hashlib.sha256()
    for raw in sorted(str(Path(p)) for p in paths):
        path = Path(raw)
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _pair(value: Sequence[int] | None, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if len(value) != 2:
        raise ModelBundleError(f"Expected a length-2 query_res, got {value!r}.")
    return (int(value[0]), int(value[1]))


@dataclass(frozen=True)
class FGNModelBundle:
    """Immutable deployment contract for one fixed-domain FGN model bundle."""

    bundle_id: str
    domain_name: str
    git_commit: str
    checkpoint_dirs: List[Path]
    checkpoint_alias: str
    normalizer_path: Path
    static_files: List[Path]
    calibration_coefficients_path: Path
    isotonic_curves_path: Path
    boundary_channels: List[str]
    dt_seconds: int
    n_history: int
    skip_before_timestep: int
    max_forecast_steps: int
    fgn_noise_dim: int
    members_per_checkpoint: int
    crs: str = "EPSG:32618"
    mesh_hash: Optional[str] = None
    expected_mesh_hash: Optional[str] = None
    geometry_path: Optional[Path] = None
    static_tensor_path: Optional[Path] = None
    structural_dry_mask_path: Optional[Path] = None
    model_config_path: Optional[Path] = None
    query_res: tuple[int, int] = (48, 48)
    initial_condition: InitialConditionConfig = field(default_factory=InitialConditionConfig)
    research_disclaimer: str = "Research only; not for emergency or operational decision use."
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_checkpoints(self) -> int:
        return len(self.checkpoint_dirs)

    @property
    def total_members(self) -> int:
        return self.n_checkpoints * int(self.members_per_checkpoint)

    @property
    def min_required_forcing_rows(self) -> int:
        return int(self.skip_before_timestep) + int(self.n_history) + 1

    def public_metadata(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "domain_name": self.domain_name,
            "git_commit": self.git_commit,
            "checkpoint_alias": self.checkpoint_alias,
            "n_checkpoints": self.n_checkpoints,
            "members_per_checkpoint": self.members_per_checkpoint,
            "total_members": self.total_members,
            "member_budget": {
                "max_ensembles": self.n_checkpoints,
                "max_members_per_ensemble": self.members_per_checkpoint,
                "default_ensembles": self.n_checkpoints,
                "default_members_per_ensemble": self.members_per_checkpoint,
                "default_total_members": self.total_members,
            },
            "boundary_channels": list(self.boundary_channels),
            "dt_seconds": self.dt_seconds,
            "n_history": self.n_history,
            "skip_before_timestep": self.skip_before_timestep,
            "max_forecast_steps": self.max_forecast_steps,
            "fgn_noise_dim": self.fgn_noise_dim,
            "query_res": list(self.query_res),
            "crs": self.crs,
            "mesh_hash": self.mesh_hash,
            "research_disclaimer": self.research_disclaimer,
            "has_domain_assets": self.geometry_path is not None and self.static_tensor_path is not None,
            "has_structural_dry_mask": self.structural_dry_mask_path is not None,
            "initial_condition": self.initial_condition.public_metadata(),
            "input_contract": {
                "format": "csv",
                "required_columns": ["time_seconds", "stage", "precipitation"],
                "accepted_stage_aliases": ["stage", "water_stage", "surge_stage"],
                "accepted_precipitation_aliases": ["precipitation", "precip", "rainfall", "rain"],
                "min_rows": self.min_required_forcing_rows,
            },
            "supported_outputs": [
                "raw_summary_json",
                "calibrated_summary_json",
                "comparison_summary_json",
                "map_pngs",
                "calibrated_mean_wd_animation_gif",
                "optional_full_hdf5",
            ],
        }

    def validate(self, *, validate_paths: bool = True) -> "FGNModelBundle":
        if not self.bundle_id:
            raise ModelBundleError("Model bundle requires a non-empty bundle_id.")
        if not self.domain_name:
            raise ModelBundleError("Model bundle requires a non-empty domain_name.")
        if self.boundary_channels != ["stage", "precipitation"]:
            raise ModelBundleError(
                "Coastal FGN serving v1 requires boundary_channels exactly "
                "[stage, precipitation]."
            )
        if self.n_checkpoints < 1:
            raise ModelBundleError("Model bundle requires at least one checkpoint directory.")
        if self.members_per_checkpoint < 1:
            raise ModelBundleError("members_per_checkpoint must be >= 1.")
        if self.total_members != 60:
            raise ModelBundleError(
                "Coastal FGN serving v1 is pinned to 60 members. "
                f"Got {self.n_checkpoints} checkpoints x {self.members_per_checkpoint} = {self.total_members}."
            )
        if self.dt_seconds != COASTAL_FGN_DT_SECONDS:
            raise ModelBundleError(
                f"Expected dt_seconds={COASTAL_FGN_DT_SECONDS} for coastal FGN, got {self.dt_seconds}."
            )
        if self.n_history != 3:
            raise ModelBundleError(f"Expected n_history=3 for coastal FGN, got {self.n_history}.")
        if self.skip_before_timestep < 0:
            raise ModelBundleError("skip_before_timestep must be non-negative.")
        if self.max_forecast_steps < 1:
            raise ModelBundleError("max_forecast_steps must be >= 1.")
        if self.fgn_noise_dim != 32:
            raise ModelBundleError(f"Expected fgn_noise_dim=32 for current coastal FGN, got {self.fgn_noise_dim}.")
        if min(self.query_res) < 2:
            raise ModelBundleError(f"query_res must be at least [2,2], got {self.query_res}.")
        if self.initial_condition.default_mode not in {
            INITIAL_CONDITION_DRY,
            INITIAL_CONDITION_FORCING_BASELINE,
        }:
            raise ModelBundleError(
                f"Unsupported initial-condition mode: {self.initial_condition.default_mode!r}."
            )
        if self.initial_condition.k_neighbors < 1:
            raise ModelBundleError("initial_condition.k_neighbors must be >= 1.")
        if self.mesh_hash and self.expected_mesh_hash and self.mesh_hash != self.expected_mesh_hash:
            raise ModelBundleError(
                "Model bundle mesh_hash does not match expected_mesh_hash: "
                f"{self.mesh_hash} != {self.expected_mesh_hash}."
            )
        if not validate_paths:
            return self
        missing: List[str] = []
        required_paths = [
            self.normalizer_path,
            self.calibration_coefficients_path,
            self.isotonic_curves_path,
            *self.static_files,
        ]
        for optional in (
            self.geometry_path,
            self.static_tensor_path,
            self.structural_dry_mask_path,
            self.model_config_path,
        ):
            if optional is not None:
                required_paths.append(optional)
        for path in required_paths:
            if path is None or not Path(path).exists():
                missing.append(str(path))
        if self.initial_condition.default_mode == INITIAL_CONDITION_FORCING_BASELINE:
            if self.initial_condition.library_path is None:
                missing.append("initial-condition library path")
            elif not Path(self.initial_condition.library_path).exists():
                missing.append(f"initial-condition library: {self.initial_condition.library_path}")
        checkpoint_file = "best_model_state_dict.pt" if self.checkpoint_alias == "best_model" else "model_state_dict.pt"
        for checkpoint_dir in self.checkpoint_dirs:
            if not checkpoint_dir.exists():
                missing.append(str(checkpoint_dir))
            elif not (checkpoint_dir / checkpoint_file).exists():
                missing.append(str(checkpoint_dir / checkpoint_file))
        if missing:
            raise ModelBundleError("Model bundle is missing required files: " + "; ".join(missing))
        if self.initial_condition.default_mode == INITIAL_CONDITION_FORCING_BASELINE:
            try:
                validate_initial_condition_library(
                    Path(self.initial_condition.library_path),
                    n_history=int(self.n_history),
                )
            except Exception as exc:
                raise ModelBundleError(f"Malformed initial-condition library: {exc}") from exc
        return self


def _load_mapping(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            return json.load(handle)
        try:
            from ruamel.yaml import YAML
        except Exception as exc:  # pragma: no cover
            raise ModelBundleError(
                f"YAML bundle {path} requires ruamel.yaml or use JSON instead."
            ) from exc
        loaded = YAML(typ="safe").load(handle)
        if loaded is None:
            return {}
        if not isinstance(loaded, Mapping):
            raise ModelBundleError(f"Expected mapping in model bundle {path}, got {type(loaded).__name__}.")
        return loaded


def _initial_condition_config(raw: Mapping[str, Any], *, base_dir: Path) -> InitialConditionConfig:
    payload = dict(raw.get("initial_condition") or {})
    library_path = _as_path(payload.get("library_path"), base_dir=base_dir)
    return InitialConditionConfig(
        default_mode=str(payload.get("default_mode", INITIAL_CONDITION_DRY)),
        library_path=library_path,
        reference_scope=str(payload.get("reference_scope", "train_calibration")),
        k_neighbors=int(payload.get("k_neighbors", 5)),
        metadata=dict(payload.get("metadata", {})),
    )


def load_model_bundle(path: str | Path, *, validate_paths: bool = True) -> FGNModelBundle:
    """Load and validate a model bundle manifest from JSON/YAML."""
    manifest_path = Path(path).expanduser().resolve()
    base_dir = manifest_path.parent
    raw = dict(_load_mapping(manifest_path))
    try:
        bundle = FGNModelBundle(
            bundle_id=str(raw["bundle_id"]),
            domain_name=str(raw.get("domain_name", "coastal")),
            git_commit=str(raw.get("git_commit", "unknown")),
            checkpoint_dirs=[Path(_as_path(p, base_dir=base_dir)) for p in raw["checkpoint_dirs"]],
            checkpoint_alias=str(raw.get("checkpoint_alias", "best_model")),
            normalizer_path=Path(_as_path(raw["normalizer_path"], base_dir=base_dir)),
            static_files=[Path(_as_path(p, base_dir=base_dir)) for p in raw.get("static_files", [])],
            calibration_coefficients_path=Path(_as_path(raw["calibration_coefficients_path"], base_dir=base_dir)),
            isotonic_curves_path=Path(_as_path(raw["isotonic_curves_path"], base_dir=base_dir)),
            boundary_channels=[str(x) for x in raw.get("boundary_channels", [])],
            dt_seconds=int(raw.get("dt_seconds", COASTAL_FGN_DT_SECONDS)),
            n_history=int(raw.get("n_history", 3)),
            skip_before_timestep=int(raw.get("skip_before_timestep", 12)),
            max_forecast_steps=int(raw["max_forecast_steps"]),
            fgn_noise_dim=int(raw.get("fgn_noise_dim", 32)),
            members_per_checkpoint=int(raw.get("members_per_checkpoint", 20)),
            crs=str(raw.get("crs", "EPSG:32618")),
            mesh_hash=raw.get("mesh_hash"),
            expected_mesh_hash=raw.get("expected_mesh_hash"),
            geometry_path=_as_path(raw.get("geometry_path"), base_dir=base_dir),
            static_tensor_path=_as_path(raw.get("static_tensor_path"), base_dir=base_dir),
            structural_dry_mask_path=_as_path(raw.get("structural_dry_mask_path"), base_dir=base_dir),
            model_config_path=_as_path(raw.get("model_config_path"), base_dir=base_dir),
            query_res=_pair(raw.get("query_res"), (48, 48)),
            initial_condition=_initial_condition_config(raw, base_dir=base_dir),
            research_disclaimer=str(
                raw.get("research_disclaimer", "Research only; not for emergency or operational decision use.")
            ),
            metadata=dict(raw.get("metadata", {})),
        )
    except KeyError as exc:
        raise ModelBundleError(f"Missing required model bundle field: {exc.args[0]}") from exc
    return bundle.validate(validate_paths=validate_paths)
