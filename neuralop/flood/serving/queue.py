"""Job-queue seam for flood serving."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Protocol


@dataclass(frozen=True)
class QueuedJob:
    run_id: str


class JobQueue(Protocol):
    def enqueue(self, run_id: str) -> QueuedJob: ...


class InMemoryJobQueue:
    """Small queue adapter for tests and local development."""

    def __init__(self) -> None:
        self.jobs: Deque[QueuedJob] = deque()

    def enqueue(self, run_id: str) -> QueuedJob:
        job = QueuedJob(run_id=run_id)
        self.jobs.append(job)
        return job

    def drain(self, worker: Callable[[str], None]) -> None:
        while self.jobs:
            worker(self.jobs.popleft().run_id)
