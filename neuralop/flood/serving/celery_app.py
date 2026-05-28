"""Celery worker app for FGN serving."""

from __future__ import annotations

import os

try:
    from celery import Celery
    from celery.signals import worker_process_init
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError("Celery serving worker requires installing neuraloperator[serve].") from exc

from neuralop.flood.serving.factory import build_orchestrator

_broker = os.environ.get("CELERY_BROKER_URL") or os.environ.get("REDIS_URL", "redis://redis:6379/0")
_backend = os.environ.get("CELERY_RESULT_BACKEND", _broker)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# Recycle the prefork child after a bounded number of tasks so any
# matplotlib / CUDA / cartopy global-state drift (notably the RcParams
# corruption that produced the ``unhashable type: 'RcParams'`` failure in
# render.py::_plot_spatial_field) cannot accumulate indefinitely. The
# ``worker_process_init`` handler below reloads model bundles after each
# recycle so the cost is one model-load (~30s) per ~20 inferences.
_max_tasks_per_child = _int_env("FGN_WORKER_MAX_TASKS_PER_CHILD", 20)

app = Celery("fgn_serving", broker=_broker, backend=_backend)
app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=False,
    worker_max_tasks_per_child=_max_tasks_per_child,
)

_orchestrator = None


def _preload_models_enabled() -> bool:
    return os.environ.get("FGN_PRELOAD_MODELS", "1").strip().lower() not in {"0", "false", "no", "off"}


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator(queue_override=None, preload_models=_preload_models_enabled())
    return _orchestrator


@worker_process_init.connect
def _preload_orchestrator_on_worker_start(**_kwargs) -> None:  # pragma: no cover - exercised in deployment
    if _preload_models_enabled():
        get_orchestrator()


@app.task(name="neuralop.flood.serving.execute_run")
def execute_run(run_id: str) -> str:
    get_orchestrator().execute(str(run_id))
    return str(run_id)
