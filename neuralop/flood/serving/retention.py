"""Retention cleanup for FGN serving artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from neuralop.flood.serving.repository import RunRecord, RunRepository
from neuralop.flood.serving.run_spec import RunStatus, TERMINAL_STATUSES
from neuralop.flood.serving.storage import ArtifactStore


@dataclass(frozen=True)
class ExpirationResult:
    expired_run_ids: tuple[str, ...]
    skipped_run_ids: tuple[str, ...]


class RetentionManager:
    """Find unpinned terminal runs due for artifact expiration."""

    def __init__(self, *, repository: RunRepository, artifact_store: ArtifactStore, retention_days: int = 30) -> None:
        if int(retention_days) < 1:
            raise ValueError("retention_days must be >= 1.")
        self.repository = repository
        self.artifact_store = artifact_store
        self.retention_days = int(retention_days)

    def expire_due_runs(self, *, now: datetime | None = None) -> ExpirationResult:
        expired: list[str] = []
        skipped: list[str] = []
        due = {record.spec.run_id for record in self.due_records(now=now)}
        list_all = getattr(self.repository, "list_all", None)
        if not callable(list_all):
            raise RuntimeError("RunRepository adapter must implement list_all for retention cleanup.")
        for record in list_all():
            if record.spec.run_id not in due:
                skipped.append(record.spec.run_id)
                continue
            self.artifact_store.delete_run_artifacts(record.spec.run_id)
            self.repository.transition(record.spec.run_id, RunStatus.EXPIRED)
            expired.append(record.spec.run_id)
        return ExpirationResult(tuple(expired), tuple(skipped))

    def due_records(self, *, now: datetime | None = None) -> list[RunRecord]:
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.retention_days)
        list_all = getattr(self.repository, "list_all", None)
        if not callable(list_all):
            raise RuntimeError("RunRepository adapter must implement list_all for retention cleanup.")
        return [record for record in list_all() if self._should_expire(record, cutoff=cutoff)]

    @staticmethod
    def _should_expire(record: RunRecord, *, cutoff: datetime) -> bool:
        if record.pinned:
            return False
        if record.status not in TERMINAL_STATUSES - {RunStatus.EXPIRED, RunStatus.DELETED}:
            return False
        updated_at = record.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return updated_at < cutoff
