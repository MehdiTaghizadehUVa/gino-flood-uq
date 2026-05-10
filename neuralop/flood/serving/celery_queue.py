"""Celery job-queue adapter for serving runs."""

from __future__ import annotations

from dataclasses import dataclass

from neuralop.flood.serving.queue import QueuedJob


@dataclass
class CeleryJobQueue:
    broker_url: str
    task_name: str = "neuralop.flood.serving.execute_run"

    def enqueue(self, run_id: str) -> QueuedJob:
        try:
            from celery import Celery
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("CeleryJobQueue requires Celery; install neuraloperator[serve].") from exc
        app = Celery("fgn_serving_client", broker=self.broker_url)
        app.send_task(self.task_name, args=[run_id])
        return QueuedJob(run_id=run_id)
