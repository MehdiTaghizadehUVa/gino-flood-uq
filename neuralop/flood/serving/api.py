"""FastAPI adapter for the flood serving application."""

from __future__ import annotations

from typing import Callable

from neuralop.flood.serving.access import User
from neuralop.flood.serving.forcing import build_forcing_template_csv, parse_forcing_csv
from neuralop.flood.serving.orchestrator import RunOrchestrator
from neuralop.flood.serving.run_spec import (
    ALLOWED_EXCEEDANCE_THRESHOLDS_M,
    RunStatus,
    TERMINAL_STATUSES,
)


def create_app(orchestrator: RunOrchestrator, *, current_user: Callable[[], User] | None = None):
    try:
        from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
        from fastapi.responses import Response
    except Exception as exc:  # pragma: no cover - depends on optional serving deps
        raise RuntimeError("FastAPI serving requires installing neuraloperator[serve].") from exc
    globals().update({"Request": Request, "UploadFile": UploadFile})

    app = FastAPI(title="Coastal FGN UQ Serving", version="0.1.0")

    def _current_user(request: Request) -> User:
        if current_user is None:
            raise HTTPException(status_code=401, detail="Authentication adapter is not configured.")
        try:
            return current_user(request)
        except TypeError:
            return current_user()
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def _require_allowed(user: User) -> User:
        try:
            return orchestrator.access_policy.require_allowed(user)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    def _require_run_read(user: User, record) -> None:
        try:
            orchestrator.access_policy.require_run_read(user, record)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    def _require_admin(user: User) -> None:
        try:
            orchestrator.access_policy.require_admin(user)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    _PROGRESS = {
        RunStatus.SUBMITTED: 0.05,
        RunStatus.VALIDATING: 0.15,
        RunStatus.QUEUED: 0.25,
        RunStatus.RUNNING: 0.60,
        RunStatus.POSTPROCESSING: 0.85,
        RunStatus.COMPLETED: 1.0,
        RunStatus.FAILED: 1.0,
        RunStatus.CANCELED: 1.0,
        RunStatus.EXPIRED: 1.0,
    }

    def _artifact_availability(run_id: str) -> dict[str, bool]:
        try:
            ids = {ref.artifact_id for ref in orchestrator.artifact_store.list(run_id)}
        except Exception:
            ids = set()
        return {
            "forcing_csv": "forcing.csv" in ids,
            "raw_summary": "raw_summary.json" in ids,
            "calibrated_summary": "calibrated_summary.json" in ids,
            "comparison_summary": "comparison_summary.json" in ids,
            "map_pngs": any(artifact_id.endswith(".png") for artifact_id in ids),
            "animation_gif": "calibrated_mean_wd_animation.gif" in ids,
            "full_hdf5": "forecast_members.h5" in ids,
        }

    def _run_payload(record):
        return {
            "run_id": record.spec.run_id,
            "label": record.spec.label,
            "status": record.status.value,
            "progress": _PROGRESS[record.status],
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "pinned": record.pinned,
            "failure_reason": record.failure_reason,
            "validation_messages": [],
            "result_availability": _artifact_availability(record.spec.run_id),
            "spec": record.spec.manifest(),
        }

    def _parse_thresholds(raw: str | None):
        if raw is None or not str(raw).strip():
            return None
        values = tuple(float(x.strip()) for x in str(raw).split(",") if x.strip())
        allowed = {round(float(x), 6) for x in ALLOWED_EXCEEDANCE_THRESHOLDS_M}
        invalid = [x for x in values if round(float(x), 6) not in allowed]
        if invalid:
            raise ValueError(
                "Unsupported exceedance threshold(s) "
                f"{invalid}. Allowed thresholds: {list(ALLOWED_EXCEEDANCE_THRESHOLDS_M)}."
            )
        return values

    @app.get("/api/model-bundle")
    def model_bundle():
        return orchestrator.bundle.public_metadata()

    @app.get("/api/me")
    def me(user: User = Depends(_current_user)):
        allowed = _require_allowed(user)
        return {
            "email": allowed.email,
            "user_id": allowed.user_id,
            "is_admin": allowed.is_admin,
            "disclaimer_acknowledged": allowed.disclaimer_acknowledged,
        }

    @app.post("/api/me/disclaimer")
    def acknowledge_disclaimer(user: User = Depends(_current_user)):
        acknowledged = orchestrator.access_policy.acknowledge_disclaimer(user)
        return {
            "email": acknowledged.email,
            "user_id": acknowledged.user_id,
            "is_admin": acknowledged.is_admin,
            "disclaimer_acknowledged": acknowledged.disclaimer_acknowledged,
        }

    @app.get("/api/forcing-template")
    def forcing_template():
        return Response(
            content=build_forcing_template_csv(orchestrator.bundle),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=coastal_fgn_forcing_template.csv"},
        )

    @app.post("/api/forcing/validate")
    async def validate_forcing(
        file: UploadFile = File(...),
        forecast_steps: int | None = Form(default=None),
        user: User = Depends(_current_user),
    ):
        _require_allowed(user)
        try:
            data = await file.read()
            forcing = parse_forcing_csv(data, bundle=orchestrator.bundle, requested_forecast_steps=forecast_steps)
            return {"valid": True, "summary": forcing.summary(), "messages": []}
        except Exception as exc:
            return {"valid": False, "summary": None, "messages": [str(exc)]}

    @app.post("/api/runs")
    async def create_run(
        file: UploadFile = File(...),
        label: str | None = Form(default=None),
        forecast_steps: int | None = Form(default=None),
        output_detail: str = Form(default="standard"),
        exceedance_thresholds_m: str | None = Form(default=None),
        request_full_hdf5: bool = Form(default=False),
        request_animation: bool = Form(default=False),
        user: User = Depends(_current_user),
    ):
        try:
            data = await file.read()
            record = orchestrator.submit(
                user=user,
                forcing_csv=data,
                label=label,
                forecast_steps=forecast_steps,
                output_detail=output_detail,
                exceedance_thresholds_m=_parse_thresholds(exceedance_thresholds_m),
                request_full_hdf5=request_full_hdf5,
                request_animation=request_animation,
            )
            return {"run_id": record.spec.run_id, "status": record.status.value}
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs")
    def list_runs(user: User = Depends(_current_user)):
        user = _require_allowed(user)
        return [_run_payload(record) for record in orchestrator.repository.list_for_user(user.user_id)]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, user: User = Depends(_current_user)):
        record = orchestrator.repository.get(run_id)
        _require_run_read(user, record)
        return _run_payload(record)

    @app.get("/api/runs/{run_id}/artifacts")
    def list_artifacts(run_id: str, user: User = Depends(_current_user)):
        record = orchestrator.repository.get(run_id)
        _require_run_read(user, record)
        return [
            {"artifact_id": ref.artifact_id, "content_type": ref.content_type, "size_bytes": ref.size_bytes}
            for ref in orchestrator.artifact_store.list(run_id)
        ]

    @app.get("/api/runs/{run_id}/artifacts/{artifact_id}")
    def get_artifact(run_id: str, artifact_id: str, user: User = Depends(_current_user)):
        record = orchestrator.repository.get(run_id)
        _require_run_read(user, record)
        data = orchestrator.artifact_store.read_bytes(run_id, artifact_id)
        refs = {ref.artifact_id: ref for ref in orchestrator.artifact_store.list(run_id)}
        content_type = refs.get(artifact_id).content_type if artifact_id in refs else "application/octet-stream"
        return Response(content=data, media_type=content_type)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str, user: User = Depends(_current_user)):
        record = orchestrator.repository.get(run_id)
        _require_run_read(user, record)
        return _run_payload(orchestrator.cancel(run_id))

    @app.get("/api/admin/runs")
    def admin_list_runs(user: User = Depends(_current_user)):
        _require_admin(user)
        list_all = getattr(orchestrator.repository, "list_all", None)
        if not callable(list_all):
            raise HTTPException(status_code=500, detail="Run repository does not support admin list_all.")
        return [_run_payload(record) for record in list_all()]

    @app.post("/api/admin/runs/{run_id}/pin")
    def admin_pin_run(run_id: str, user: User = Depends(_current_user)):
        _require_admin(user)
        return _run_payload(orchestrator.repository.set_pinned(run_id, True))

    @app.post("/api/admin/runs/{run_id}/unpin")
    def admin_unpin_run(run_id: str, user: User = Depends(_current_user)):
        _require_admin(user)
        return _run_payload(orchestrator.repository.set_pinned(run_id, False))

    @app.post("/api/admin/runs/{run_id}/cancel")
    def admin_cancel_run(run_id: str, user: User = Depends(_current_user)):
        _require_admin(user)
        record = orchestrator.repository.get(run_id)
        if record.status in TERMINAL_STATUSES:
            return _run_payload(record)
        return _run_payload(orchestrator.cancel(run_id))

    @app.get("/api/admin/users")
    def admin_list_users(user: User = Depends(_current_user)):
        _require_admin(user)
        return orchestrator.access_policy.list_users()

    @app.post("/api/admin/users")
    async def admin_add_user(request: Request, user: User = Depends(_current_user)):
        _require_admin(user)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
        if not isinstance(body, dict) or "email" not in body:
            raise HTTPException(status_code=400, detail="Body must be {email: str, is_admin?: bool}.")
        try:
            return orchestrator.access_policy.add_allowed(
                body["email"],
                admin=bool(body.get("is_admin", False)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/admin/users/{email}")
    def admin_remove_user(email: str, user: User = Depends(_current_user)):
        _require_admin(user)
        try:
            removed = orchestrator.access_policy.remove_allowed(email)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not removed:
            raise HTTPException(status_code=404, detail=f"User not in allowlist: {email}")
        return {"email": email.strip().lower(), "removed": True}

    return app
