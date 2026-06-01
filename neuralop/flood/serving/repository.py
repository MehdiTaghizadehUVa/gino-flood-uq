"""Run metadata repository seam for flood serving."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import RLock
from typing import Dict, Iterable, Optional, Protocol

from neuralop.flood.serving.run_spec import RunSpec, RunStatus, ensure_transition


DEFAULT_PROGRESS_BY_STATUS = {
    RunStatus.SUBMITTED: 0.05,
    RunStatus.VALIDATING: 0.15,
    RunStatus.WAITING_FOR_CACHE: 0.30,
    RunStatus.QUEUED: 0.25,
    RunStatus.RUNNING: 0.40,
    RunStatus.POSTPROCESSING: 0.82,
    RunStatus.COMPLETED: 1.0,
    RunStatus.FAILED: 1.0,
    RunStatus.CANCELED: 1.0,
    RunStatus.EXPIRED: 1.0,
    RunStatus.DELETED: 1.0,
}

DEFAULT_PROGRESS_LABEL_BY_STATUS = {
    RunStatus.SUBMITTED: "Submission received",
    RunStatus.VALIDATING: "Validating forcing input",
    RunStatus.WAITING_FOR_CACHE: "Waiting for matching cached result",
    RunStatus.QUEUED: "Waiting for the GPU worker",
    RunStatus.RUNNING: "GPU inference is running",
    RunStatus.POSTPROCESSING: "Post-processing forecast products",
    RunStatus.COMPLETED: "Completed",
    RunStatus.FAILED: "Failed",
    RunStatus.CANCELED: "Canceled",
    RunStatus.EXPIRED: "Expired",
    RunStatus.DELETED: "Deleted",
}


def default_progress_for_status(status: RunStatus) -> float:
    return DEFAULT_PROGRESS_BY_STATUS[status]


def default_progress_label_for_status(status: RunStatus) -> str:
    return DEFAULT_PROGRESS_LABEL_BY_STATUS[status]


def _clamp_progress(progress: float) -> float:
    return max(0.0, min(1.0, float(progress)))


def _runtime_seconds(started_at: datetime | None, completed_at: datetime | None) -> float | None:
    if started_at is None or completed_at is None:
        return None
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    if completed_at.tzinfo is None:
        completed_at = completed_at.replace(tzinfo=timezone.utc)
    return max(0.0, (completed_at - started_at).total_seconds())


@dataclass(frozen=True)
class RunRecord:
    spec: RunSpec
    status: RunStatus = RunStatus.SUBMITTED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pinned: bool = False
    failure_reason: Optional[str] = None
    internal_log_ref: Optional[str] = None
    progress: Optional[float] = None
    progress_label: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    runtime_seconds: Optional[float] = None


class RunRepository(Protocol):
    def create(self, spec: RunSpec) -> RunRecord: ...
    def get(self, run_id: str) -> RunRecord: ...
    def list_for_user(self, user_id: str) -> Iterable[RunRecord]: ...
    def list_all(self) -> Iterable[RunRecord]: ...
    def transition(self, run_id: str, status: RunStatus, *, failure_reason: str | None = None) -> RunRecord: ...
    def update_progress(self, run_id: str, progress: float, *, label: str | None = None) -> RunRecord: ...
    def set_pinned(self, run_id: str, pinned: bool) -> RunRecord: ...


class InMemoryRunRepository:
    """Thread-safe test/development repository adapter."""

    def __init__(self) -> None:
        self._runs: Dict[str, RunRecord] = {}
        self._lock = RLock()

    def create(self, spec: RunSpec) -> RunRecord:
        with self._lock:
            if spec.run_id in self._runs:
                raise KeyError(f"Run already exists: {spec.run_id}")
            record = RunRecord(spec=spec)
            self._runs[spec.run_id] = record
            return record

    def get(self, run_id: str) -> RunRecord:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Unknown run_id: {run_id}")
            return self._runs[run_id]

    def list_for_user(self, user_id: str) -> Iterable[RunRecord]:
        with self._lock:
            return [r for r in self._runs.values() if r.spec.user_id == user_id]

    def list_all(self) -> Iterable[RunRecord]:
        with self._lock:
            return list(self._runs.values())

    def transition(self, run_id: str, status: RunStatus, *, failure_reason: str | None = None) -> RunRecord:
        with self._lock:
            record = self.get(run_id)
            ensure_transition(record.status, status)
            now = datetime.now(timezone.utc)
            started_at = record.started_at or (now if status == RunStatus.RUNNING else None)
            completed_at = record.completed_at
            if status == RunStatus.COMPLETED and record.status == RunStatus.WAITING_FOR_CACHE:
                started_at = started_at or now
                completed_at = completed_at or now
            if status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED} and started_at is not None:
                completed_at = completed_at or now
            updated = replace(
                record,
                status=status,
                updated_at=now,
                failure_reason=failure_reason if status == RunStatus.FAILED else record.failure_reason,
                progress=default_progress_for_status(status),
                progress_label=default_progress_label_for_status(status),
                started_at=started_at,
                completed_at=completed_at,
                runtime_seconds=record.runtime_seconds or _runtime_seconds(started_at, completed_at),
            )
            self._runs[run_id] = updated
            return updated

    def update_progress(self, run_id: str, progress: float, *, label: str | None = None) -> RunRecord:
        with self._lock:
            record = self.get(run_id)
            updated = replace(
                record,
                progress=_clamp_progress(progress),
                progress_label=str(label) if label is not None else record.progress_label,
                updated_at=datetime.now(timezone.utc),
            )
            self._runs[run_id] = updated
            return updated

    def set_pinned(self, run_id: str, pinned: bool) -> RunRecord:
        with self._lock:
            record = self.get(run_id)
            updated = replace(record, pinned=bool(pinned), updated_at=datetime.now(timezone.utc))
            self._runs[run_id] = updated
            return updated
