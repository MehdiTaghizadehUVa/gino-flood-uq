"""Run metadata repository seam for flood serving."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import RLock
from typing import Dict, Iterable, Optional, Protocol

from neuralop.flood.serving.run_spec import RunSpec, RunStatus, ensure_transition


@dataclass(frozen=True)
class RunRecord:
    spec: RunSpec
    status: RunStatus = RunStatus.SUBMITTED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pinned: bool = False
    failure_reason: Optional[str] = None
    internal_log_ref: Optional[str] = None


class RunRepository(Protocol):
    def create(self, spec: RunSpec) -> RunRecord: ...
    def get(self, run_id: str) -> RunRecord: ...
    def list_for_user(self, user_id: str) -> Iterable[RunRecord]: ...
    def list_all(self) -> Iterable[RunRecord]: ...
    def transition(self, run_id: str, status: RunStatus, *, failure_reason: str | None = None) -> RunRecord: ...
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
            updated = replace(
                record,
                status=status,
                updated_at=datetime.now(timezone.utc),
                failure_reason=failure_reason if status == RunStatus.FAILED else record.failure_reason,
            )
            self._runs[run_id] = updated
            return updated

    def set_pinned(self, run_id: str, pinned: bool) -> RunRecord:
        with self._lock:
            record = self.get(run_id)
            updated = replace(record, pinned=bool(pinned), updated_at=datetime.now(timezone.utc))
            self._runs[run_id] = updated
            return updated
