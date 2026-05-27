"""Default FastAPI app entrypoint for Docker/uvicorn."""

from __future__ import annotations

import os
import shutil

try:
    from fastapi import FastAPI, HTTPException, Request
except Exception as exc:  # pragma: no cover - optional dependency import guard
    raise RuntimeError("Install neuraloperator[serve] to run the serving API.") from exc

from neuralop.flood.serving.api import create_app
from neuralop.flood.serving.auth import TrustedHeaderAuth
from neuralop.flood.serving.factory import build_orchestrator, env_flag, split_env
from neuralop.flood.serving.model_bundle import ModelBundleError, load_model_bundle


try:
    _orchestrator = build_orchestrator(preload_models=False) if os.environ.get("FGN_MODEL_BUNDLE_PATH") else None
except Exception as exc:  # Keep health endpoint alive with explicit failure.
    _startup_error = exc
    _orchestrator = None
else:
    _startup_error = None

_open_authenticated_access = env_flag("FGN_OPEN_AUTHENTICATED_ACCESS")
_auth = TrustedHeaderAuth.from_lists(
    allowed_emails=() if _open_authenticated_access else split_env("FGN_ALLOWED_EMAILS"),
    admin_emails=split_env("FGN_ADMIN_EMAILS"),
)


def _current_user(request: Request):
    return _auth.user_from_headers(request.headers)


if _orchestrator is None:
    app = FastAPI(title="Coastal FGN UQ Serving", version="0.1.0")
else:
    app = create_app(_orchestrator, current_user=_current_user)


@app.get("/api/health")
def health():
    artifact_root = os.environ.get("FGN_ARTIFACT_ROOT")
    disk = None
    if artifact_root:
        try:
            usage = shutil.disk_usage(artifact_root)
            disk = {
                "artifact_root": artifact_root,
                "total_bytes": usage.total,
                "free_bytes": usage.free,
            }
        except Exception as exc:
            disk = {"artifact_root": artifact_root, "error": str(exc)}
    return {
        "status": "error" if _startup_error is not None else "ok",
        "model_bundle_configured": bool(os.environ.get("FGN_MODEL_BUNDLE_PATH")),
        "production_inference_ready": _orchestrator is not None and os.environ.get("FGN_INFERENCE_MODE", "production") == "production",
        "artifact_storage": disk,
        "startup_error": str(_startup_error) if _startup_error is not None else None,
    }


@app.get("/api/admin/system-health")
def admin_system_health(request: Request):
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail=str(_startup_error) if _startup_error else "Orchestrator is not configured.")
    user = _current_user(request)
    try:
        _orchestrator.access_policy.require_admin(user)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    artifact_usage = shutil.disk_usage(os.environ.get("FGN_ARTIFACT_ROOT", "/tmp"))
    gpu = {"available": False}
    try:
        import torch

        gpu = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        gpu = {"available": False, "error": str(exc)}
    list_all = getattr(_orchestrator.repository, "list_all", None)
    records = list(list_all()) if callable(list_all) else []
    return {
        "bundle": _orchestrator.bundle.public_metadata(),
        "gpu": gpu,
        "queue_depth": len(getattr(_orchestrator.queue, "jobs", [])),
        "run_count": len(records),
        "artifact_storage": {
            "root": os.environ.get("FGN_ARTIFACT_ROOT"),
            "total_bytes": artifact_usage.total,
            "free_bytes": artifact_usage.free,
        },
    }


@app.get("/api/model-bundle-health")
def model_bundle_health():
    bundle_path = os.environ.get("FGN_MODEL_BUNDLE_PATH")
    if not bundle_path:
        raise HTTPException(status_code=503, detail="FGN_MODEL_BUNDLE_PATH is not configured.")
    try:
        return load_model_bundle(bundle_path).public_metadata()
    except ModelBundleError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
