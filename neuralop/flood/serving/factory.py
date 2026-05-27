"""Environment-backed serving object factory."""

from __future__ import annotations

import os
from pathlib import Path

from neuralop.flood.serving.access import AccessPolicy
from neuralop.flood.serving.calibration import CalibrationAdapter
from neuralop.flood.serving.inference import FakeFGNInferenceService, ProductionFGNInferenceService
from neuralop.flood.serving.model_bundle import load_model_bundle
from neuralop.flood.serving.monitoring import InMemoryMonitoringRepository, MonitoringBundle, MonitoringOrchestrator
from neuralop.flood.serving.orchestrator import RunOrchestrator
from neuralop.flood.serving.products import ForecastProductBuilder
from neuralop.flood.serving.queue import InMemoryJobQueue
from neuralop.flood.serving.storage import LocalArtifactStore


def split_env(name: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_repository():
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        from neuralop.flood.serving.sql_repository import SqlRunRepository

        return SqlRunRepository(database_url)
    from neuralop.flood.serving.repository import InMemoryRunRepository

    return InMemoryRunRepository()


def build_access_repository():
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        from neuralop.flood.serving.sql_access import SqlAccessRepository

        return SqlAccessRepository(database_url)
    from neuralop.flood.serving.access import InMemoryAccessRepository

    return InMemoryAccessRepository()


def build_queue():
    broker_url = os.environ.get("CELERY_BROKER_URL") or os.environ.get("REDIS_URL")
    if broker_url:
        from neuralop.flood.serving.celery_queue import CeleryJobQueue

        return CeleryJobQueue(broker_url=broker_url)
    return InMemoryJobQueue()


def build_monitoring_repository():
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        from neuralop.flood.serving.sql_monitoring import SqlMonitoringRepository

        return SqlMonitoringRepository(database_url)
    return InMemoryMonitoringRepository()


def build_orchestrator(*, queue_override=None, preload_models: bool = False) -> RunOrchestrator:
    bundle_path = os.environ.get("FGN_MODEL_BUNDLE_PATH")
    if not bundle_path:
        raise RuntimeError("FGN_MODEL_BUNDLE_PATH is not configured.")
    bundle = load_model_bundle(bundle_path)
    artifact_root = Path(os.environ.get("FGN_ARTIFACT_ROOT", "/tmp/fgn-serving-artifacts"))
    inference_mode = os.environ.get("FGN_INFERENCE_MODE", "production").strip().lower()
    if inference_mode == "fake":
        inference = FakeFGNInferenceService(bundle)
    elif inference_mode == "production":
        inference = ProductionFGNInferenceService(
            bundle,
            device=os.environ.get("FGN_DEVICE", "cuda:0"),
            member_chunk_size=os.environ.get("FGN_MEMBER_CHUNK_SIZE", "4"),
            inference_dtype=os.environ.get("FGN_INFERENCE_DTYPE", "fp32"),
        )
        if preload_models:
            inference._ensure_loaded()
    else:
        raise RuntimeError(f"Unknown FGN_INFERENCE_MODE={inference_mode!r}.")
    run_artifact_store = LocalArtifactStore(artifact_root)
    monitoring_orchestrator = None
    monitoring_bundle_path = os.environ.get("FGN_MONITORING_BUNDLE_PATH")
    if monitoring_bundle_path:
        candidate_root = Path(
            os.environ.get("FGN_CANDIDATE_ROOT", str(artifact_root / "retraining_candidates"))
        )
        monitoring_orchestrator = MonitoringOrchestrator(
            bundle=MonitoringBundle.load(monitoring_bundle_path),
            repository=build_monitoring_repository(),
            run_artifact_store=run_artifact_store,
            candidate_artifact_store=LocalArtifactStore(candidate_root),
        )
    return RunOrchestrator(
        bundle=bundle,
        repository=build_repository(),
        queue=queue_override or build_queue(),
        artifact_store=run_artifact_store,
        access_policy=AccessPolicy(
            allowed_emails=split_env("FGN_ALLOWED_EMAILS"),
            admin_emails=split_env("FGN_ADMIN_EMAILS"),
            repository=build_access_repository(),
            open_authenticated_access=env_flag("FGN_OPEN_AUTHENTICATED_ACCESS"),
        ),
        inference_service=inference,
        calibration_adapter=CalibrationAdapter.from_files(bundle.calibration_coefficients_path, bundle.isotonic_curves_path),
        product_builder=ForecastProductBuilder(),
        monitoring_orchestrator=monitoring_orchestrator,
    )
