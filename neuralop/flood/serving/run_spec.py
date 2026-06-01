"""Run specification and state-machine primitives for flood serving."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Sequence
from uuid import uuid4


class RunStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    VALIDATING = "VALIDATING"
    WAITING_FOR_CACHE = "WAITING_FOR_CACHE"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    POSTPROCESSING = "POSTPROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    DELETED = "DELETED"


TERMINAL_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELED,
    RunStatus.EXPIRED,
    RunStatus.DELETED,
}

DEFAULT_EXCEEDANCE_THRESHOLDS_M = (0.01, 0.05, 0.10, 0.30, 0.50)
ALLOWED_EXCEEDANCE_THRESHOLDS_M = DEFAULT_EXCEEDANCE_THRESHOLDS_M
ALLOWED_OUTPUT_DETAILS = {"standard", "full"}

_ALLOWED_TRANSITIONS = {
    RunStatus.SUBMITTED: {RunStatus.VALIDATING, RunStatus.CANCELED},
    RunStatus.VALIDATING: {
        RunStatus.WAITING_FOR_CACHE,
        RunStatus.QUEUED,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELED,
    },
    RunStatus.WAITING_FOR_CACHE: {RunStatus.QUEUED, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED},
    RunStatus.QUEUED: {RunStatus.RUNNING, RunStatus.CANCELED, RunStatus.FAILED},
    RunStatus.RUNNING: {RunStatus.POSTPROCESSING, RunStatus.FAILED, RunStatus.CANCELED},
    RunStatus.POSTPROCESSING: {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED},
    RunStatus.COMPLETED: {RunStatus.EXPIRED, RunStatus.DELETED},
    RunStatus.FAILED: {RunStatus.EXPIRED, RunStatus.DELETED},
    RunStatus.CANCELED: {RunStatus.EXPIRED, RunStatus.DELETED},
    RunStatus.EXPIRED: {RunStatus.DELETED},
    RunStatus.DELETED: set(),
}


class RunStateError(ValueError):
    """Raised for invalid run lifecycle transitions."""


def ensure_transition(current: RunStatus, target: RunStatus) -> None:
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise RunStateError(f"Invalid run transition {current.value} -> {target.value}.")


@dataclass(frozen=True)
class RunSpec:
    """Immutable normalized description of one inference run."""

    run_id: str
    user_id: str
    bundle_id: str
    input_hash: str
    forecast_steps: int
    output_detail: str = "standard"
    request_full_hdf5: bool = True
    request_animation: bool = False
    ensemble_count: int = 3
    members_per_ensemble: int = 20
    calibration_mode: str = "calibrated_default"
    exceedance_thresholds_m: Sequence[float] = (0.01, 0.05, 0.10, 0.30, 0.50)
    seed: int = 123
    label: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def new(
        *,
        user_id: str,
        bundle_id: str,
        input_hash: str,
        forecast_steps: int,
        label: Optional[str] = None,
        output_detail: str = "standard",
        request_full_hdf5: bool = True,
        request_animation: bool = False,
        ensemble_count: int = 3,
        members_per_ensemble: int = 20,
        exceedance_thresholds_m: Sequence[float] | None = None,
        seed: int = 123,
    ) -> "RunSpec":
        output_detail = str(output_detail or "standard").strip().lower()
        if output_detail not in ALLOWED_OUTPUT_DETAILS:
            raise ValueError(
                f"output_detail must be one of {sorted(ALLOWED_OUTPUT_DETAILS)}, got {output_detail!r}."
            )
        thresholds = tuple(float(x) for x in (exceedance_thresholds_m or DEFAULT_EXCEEDANCE_THRESHOLDS_M))
        allowed = {round(float(x), 6) for x in ALLOWED_EXCEEDANCE_THRESHOLDS_M}
        invalid = [x for x in thresholds if round(float(x), 6) not in allowed]
        if invalid:
            raise ValueError(
                "exceedance_thresholds_m may only contain configured thresholds "
                f"{list(ALLOWED_EXCEEDANCE_THRESHOLDS_M)}; got {invalid}."
            )
        ensemble_count = int(ensemble_count)
        members_per_ensemble = int(members_per_ensemble)
        if ensemble_count < 1:
            raise ValueError("ensemble_count must be >= 1.")
        if members_per_ensemble < 1:
            raise ValueError("members_per_ensemble must be >= 1.")
        return RunSpec(
            run_id=uuid4().hex,
            user_id=user_id,
            bundle_id=bundle_id,
            input_hash=input_hash,
            forecast_steps=int(forecast_steps),
            label=label,
            output_detail=output_detail,
            request_full_hdf5=bool(request_full_hdf5),
            request_animation=bool(request_animation),
            ensemble_count=ensemble_count,
            members_per_ensemble=members_per_ensemble,
            exceedance_thresholds_m=thresholds,
            seed=int(seed),
        )

    def manifest(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "user_id": self.user_id,
            "bundle_id": self.bundle_id,
            "input_hash": self.input_hash,
            "forecast_steps": self.forecast_steps,
            "output_detail": self.output_detail,
            "request_full_hdf5": self.request_full_hdf5,
            "request_animation": self.request_animation,
            "ensemble_count": self.ensemble_count,
            "members_per_ensemble": self.members_per_ensemble,
            "calibration_mode": self.calibration_mode,
            "exceedance_thresholds_m": list(self.exceedance_thresholds_m),
            "seed": self.seed,
            "label": self.label,
            "created_at": self.created_at.isoformat(),
        }
