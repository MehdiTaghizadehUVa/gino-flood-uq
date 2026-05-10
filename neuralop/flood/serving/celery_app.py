"""Celery worker app for FGN serving."""

from __future__ import annotations

import os

try:
    from celery import Celery
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError("Celery serving worker requires installing neuraloperator[serve].") from exc

from neuralop.flood.serving.factory import build_orchestrator

_broker = os.environ.get("CELERY_BROKER_URL") or os.environ.get("REDIS_URL", "redis://redis:6379/0")
_backend = os.environ.get("CELERY_RESULT_BACKEND", _broker)
app = Celery("fgn_serving", broker=_broker, backend=_backend)
app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=False,
)

_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        preload = os.environ.get("FGN_PRELOAD_MODELS", "1").strip().lower() not in {"0", "false", "no"}
        _orchestrator = build_orchestrator(queue_override=None, preload_models=preload)
    return _orchestrator


@app.task(name="neuralop.flood.serving.execute_run")
def execute_run(run_id: str) -> str:
    get_orchestrator().execute(str(run_id))
    return str(run_id)
