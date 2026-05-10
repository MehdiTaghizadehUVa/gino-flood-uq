"""Queue and concurrency policy for serving runs."""

from __future__ import annotations

from dataclasses import dataclass

from neuralop.flood.serving.repository import RunRepository
from neuralop.flood.serving.run_spec import RunStatus


class QuotaError(ValueError):
    """Raised when a user exceeds serving queue limits."""


@dataclass(frozen=True)
class QuotaPolicy:
    max_user_queued: int = 3
    max_user_running: int = 1
    max_global_running: int = 1

    def validate_submit(self, repository: RunRepository, user_id: str) -> None:
        records = list(repository.list_for_user(user_id))
        queued = sum(1 for r in records if r.status in {RunStatus.SUBMITTED, RunStatus.VALIDATING, RunStatus.QUEUED})
        running = sum(1 for r in records if r.status in {RunStatus.RUNNING, RunStatus.POSTPROCESSING})
        if queued >= int(self.max_user_queued):
            raise QuotaError(f"User already has {queued} queued jobs; limit is {self.max_user_queued}.")
        if running >= int(self.max_user_running):
            raise QuotaError(f"User already has {running} active jobs; limit is {self.max_user_running}.")

    def can_start(self, repository: RunRepository, run_id: str) -> bool:
        list_all = getattr(repository, "list_all", None)
        if not callable(list_all):
            return True
        active = [
            r for r in list_all()
            if r.spec.run_id != run_id and r.status in {RunStatus.RUNNING, RunStatus.POSTPROCESSING}
        ]
        return len(active) < int(self.max_global_running)
