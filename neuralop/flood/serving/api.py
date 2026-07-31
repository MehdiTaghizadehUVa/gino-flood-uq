"""FastAPI adapter for the flood serving application."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from neuralop.flood.serving.access import User
from neuralop.flood.serving.forcing import build_forcing_template_csv, parse_forcing_csv
from neuralop.flood.serving.orchestrator import RunOrchestrator
from neuralop.flood.serving.repository import (
    default_progress_for_status,
    default_progress_label_for_status,
)
from neuralop.flood.serving.run_spec import (
    ALLOWED_EXCEEDANCE_THRESHOLDS_M,
    RunStatus,
    TERMINAL_STATUSES,
)


MAX_PUBLIC_VALIDATION_BYTES = 2 * 1024 * 1024


def _count_by(items, key_fn):
    counts: dict[str, int] = {}
    for item in items:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _finite_number(value) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
    }


def _week_key(value: datetime) -> str:
    iso = value.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _compute_error_descriptors(
    fgn_descriptors: dict[str, object],
    hecras_descriptors: dict[str, object],
) -> dict[str, float | None]:
    errors: dict[str, float | None] = {}
    for key in sorted(set(fgn_descriptors) | set(hecras_descriptors)):
        fgn_val = _finite_number(fgn_descriptors.get(key))
        hec_val = _finite_number(hecras_descriptors.get(key))
        if fgn_val is None or hec_val is None:
            continue
        signed = fgn_val - hec_val
        errors[f"{key}_signed"] = signed
        errors[f"{key}_absolute"] = abs(signed)
        if abs(hec_val) > 1e-9:
            errors[f"{key}_relative"] = signed / hec_val
    return errors


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

    def _get_run_or_404(run_id: str):
        try:
            return orchestrator.repository.get(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc

    def _require_admin(user: User) -> None:
        try:
            orchestrator.access_policy.require_admin(user)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

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
            "monitoring_pre_run": "monitoring_report_pre_run.json" in ids,
            "monitoring_post_run": "monitoring_report_post_run.json" in ids,
        }

    def _read_json_artifact(run_id: str, artifact_id: str) -> dict:
        try:
            payload = json.loads(orchestrator.artifact_store.read_bytes(run_id, artifact_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Missing artifact {artifact_id} for run {run_id}.") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not read artifact {artifact_id}: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=500, detail=f"Artifact {artifact_id} is not a JSON object.")
        return payload

    def _summary_delta(a: dict, b: dict) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for key, label, unit in (
            ("max_mean_wd_m", "Peak mean depth", "m"),
            ("peak_expected_flooded_area_km2_gt_0.05m", "Peak expected flooded area > 0.05 m", "km2"),
            ("peak_expected_flooded_area_fraction_wettable_gt_0.05m", "Peak expected flooded-area fraction > 0.05 m", "fraction"),
            ("peak_expected_flooded_area_lead_hours_gt_0.05m", "Lead time of peak expected flooded area", "h"),
            ("peak_area_weighted_iqr_wd_m", "Peak area-weighted IQR", "m"),
            ("peak_area_weighted_central_90_wd_m", "Peak area-weighted p95-p05", "m"),
            ("uncertainty_to_signal_ratio", "Uncertainty-to-signal ratio", "ratio"),
        ):
            av = a.get(key)
            bv = b.get(key)
            if not isinstance(av, (int, float)) or not isinstance(bv, (int, float)):
                continue
            delta = float(bv) - float(av)
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "unit": unit,
                    "a": float(av),
                    "b": float(bv),
                    "delta": delta,
                    "percent_delta": (100.0 * delta / float(av)) if abs(float(av)) > 1.0e-12 else None,
                }
            )
        for threshold in (0.05, 0.30):
            key = f"{threshold:g}"
            a_by_threshold = a.get("exceedance_by_threshold_m")
            b_by_threshold = b.get("exceedance_by_threshold_m")
            if not isinstance(a_by_threshold, dict) or not isinstance(b_by_threshold, dict):
                continue
            a_thr = a_by_threshold.get(key, {})
            b_thr = b_by_threshold.get(key, {})
            if not isinstance(a_thr, dict) or not isinstance(b_thr, dict):
                continue
            av = a_thr.get("peak_expected_area_fraction_wettable")
            bv = b_thr.get("peak_expected_area_fraction_wettable")
            if not isinstance(av, (int, float)) or not isinstance(bv, (int, float)):
                continue
            delta = float(bv) - float(av)
            rows.append(
                {
                    "key": f"peak_expected_area_fraction_wettable_gt_{threshold:g}m",
                    "label": f"Peak expected exceedance footprint > {threshold:.2f} m",
                    "unit": "fraction",
                    "a": float(av),
                    "b": float(bv),
                    "delta": delta,
                    "percent_delta": (100.0 * delta / float(av)) if abs(float(av)) > 1.0e-12 else None,
                }
            )
        return rows

    def _runtime_stats(record) -> dict[str, object]:
        list_all = getattr(orchestrator.repository, "list_all", None)
        full_units = (
            int(orchestrator.bundle.max_forecast_steps)
            * int(orchestrator.bundle.n_checkpoints)
            * int(orchestrator.bundle.members_per_checkpoint)
        )
        run_units = (
            int(record.spec.forecast_steps)
            * int(record.spec.ensemble_count)
            * int(record.spec.members_per_ensemble)
        )
        current_fraction = (run_units / full_units) if full_units > 0 else None
        completed_full = []
        if callable(list_all):
            for candidate in list_all():
                if candidate.status != RunStatus.COMPLETED:
                    continue
                if candidate.runtime_seconds is None:
                    continue
                if candidate.spec.bundle_id != orchestrator.bundle.bundle_id:
                    continue
                if int(candidate.spec.forecast_steps) != int(orchestrator.bundle.max_forecast_steps):
                    continue
                if int(candidate.spec.ensemble_count) != int(orchestrator.bundle.n_checkpoints):
                    continue
                if int(candidate.spec.members_per_ensemble) != int(orchestrator.bundle.members_per_checkpoint):
                    continue
                completed_full.append(float(candidate.runtime_seconds))
        average = (sum(completed_full) / len(completed_full)) if completed_full else None
        return {
            "average_full_rollout_seconds": average,
            "average_full_rollout_sample_size": len(completed_full),
            "average_full_rollout_basis": (
                "completed runs with max forecast horizon and full "
                f"{orchestrator.bundle.n_checkpoints}x{orchestrator.bundle.members_per_checkpoint} ensemble"
            ),
            "current_run_work_fraction_of_full_rollout": current_fraction,
        }

    def _timing_payload(record, progress: float) -> dict[str, object]:
        now = datetime.now(timezone.utc)
        stats = _runtime_stats(record)
        elapsed = None
        if record.started_at is not None:
            started_at = _as_aware_datetime(record.started_at)
            end = _as_aware_datetime(record.completed_at) if record.completed_at is not None else now
            elapsed = max(0.0, (end - started_at).total_seconds())
        average_full = stats["average_full_rollout_seconds"]
        fraction = stats["current_run_work_fraction_of_full_rollout"]
        estimated_total = None
        if isinstance(average_full, (int, float)) and isinstance(fraction, (int, float)):
            # This is an operational estimate, not a scientific product: scale
            # the historical full-rollout average by requested members x steps.
            estimated_total = max(0.0, float(average_full) * float(fraction))
        remaining = None
        if estimated_total is not None and elapsed is not None and record.status not in TERMINAL_STATUSES:
            remaining = max(0.0, estimated_total - elapsed)
        return {
            "started_at": record.started_at.isoformat() if record.started_at is not None else None,
            "completed_at": record.completed_at.isoformat() if record.completed_at is not None else None,
            "elapsed_seconds": elapsed,
            "runtime_seconds": record.runtime_seconds,
            "estimated_total_seconds": estimated_total,
            "estimated_remaining_seconds": remaining,
            "estimated_basis": "scaled_from_average_full_rollout" if estimated_total is not None else None,
            "progress_fraction": progress,
            **stats,
        }

    def _as_aware_datetime(value: datetime) -> datetime:
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value

    def _run_payload(record, *, include_admin_cache: bool = False):
        progress = (
            float(record.progress)
            if isinstance(record.progress, (int, float))
            else default_progress_for_status(record.status)
        )
        progress = max(0.0, min(1.0, progress))
        cache_payload = orchestrator.cache_payload_for_run(
            record.spec.run_id,
            include_admin=include_admin_cache,
        )
        return {
            "run_id": record.spec.run_id,
            "label": record.spec.label,
            "status": record.status.value,
            "progress": progress,
            "progress_label": record.progress_label or default_progress_label_for_status(record.status),
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "started_at": record.started_at.isoformat() if record.started_at is not None else None,
            "completed_at": record.completed_at.isoformat() if record.completed_at is not None else None,
            "runtime_seconds": record.runtime_seconds,
            "timing": _timing_payload(record, progress),
            "pinned": record.pinned,
            "failure_reason": record.failure_reason,
            "validation_messages": [],
            "result_availability": _artifact_availability(record.spec.run_id),
            "spec": record.spec.manifest(),
            "cache": cache_payload,
        }

    def _submit_cache_status(record) -> dict[str, object]:
        cache = orchestrator.cache_payload_for_run(record.spec.run_id, include_admin=False)
        if not cache.get("enabled"):
            return {}
        if cache.get("waiting_for_cached_result"):
            status = "WAITING_FOR_SOURCE"
        elif cache.get("materialized_from_cache"):
            status = "HIT"
        else:
            status = "MISS"
        return {"cache_status": status, "cache_key_prefix": cache.get("cache_key_prefix")}

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

    def _inspector_thresholds(thresholds: Sequence[float] | None) -> tuple[float, ...]:
        configured = [float(x) for x in (thresholds or ())]
        selected: list[float] = []
        for reference in (0.05, 0.30):
            if any(abs(value - reference) < 1.0e-9 for value in configured):
                selected.append(reference)
        if not selected and configured:
            selected.append(configured[0])
            if (
                not any(abs(value - 0.30) < 1.0e-9 for value in selected)
                and any(abs(value - 0.30) < 1.0e-9 for value in ALLOWED_EXCEEDANCE_THRESHOLDS_M)
            ):
                selected.append(0.30)
        if not selected:
            selected = [0.05, 0.30]
        return tuple(selected)

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
    ):
        try:
            data = await file.read(MAX_PUBLIC_VALIDATION_BYTES + 1)
            if len(data) > MAX_PUBLIC_VALIDATION_BYTES:
                raise ValueError("Forcing CSV exceeds the 2 MiB validation limit.")
            forcing = parse_forcing_csv(data, bundle=orchestrator.bundle, requested_forecast_steps=forecast_steps)
            screening = {"available": False}
            if orchestrator.monitoring_orchestrator is not None:
                screening = orchestrator.monitoring_orchestrator.preview_forcing(
                    user_id="public-validation",
                    forcing=forcing,
                    bundle_id=orchestrator.bundle.bundle_id,
                ).screening_payload()
            return {"valid": True, "summary": forcing.summary(), "messages": [], "screening": screening}
        except Exception as exc:
            return {"valid": False, "summary": None, "messages": [str(exc)], "screening": {"available": False}}

    @app.post("/api/runs")
    async def create_run(
        file: UploadFile = File(...),
        label: str | None = Form(default=None),
        forecast_steps: int | None = Form(default=None),
        output_detail: str = Form(default="standard"),
        exceedance_thresholds_m: str | None = Form(default=None),
        # Defaulted on so the per-cell inspector works for every new run; the
        # SPA can still send "false" explicitly for callers that don't want
        # the ~5-20 MB ensemble HDF5 on disk.
        request_full_hdf5: bool = Form(default=True),
        request_animation: bool = Form(default=False),
        ensemble_count: int | None = Form(default=None),
        members_per_ensemble: int | None = Form(default=None),
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
                ensemble_count=ensemble_count,
                members_per_ensemble=members_per_ensemble,
            )
            return {"run_id": record.spec.run_id, "status": record.status.value, **_submit_cache_status(record)}
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs")
    def list_runs(user: User = Depends(_current_user)):
        user = _require_allowed(user)
        return [
            _run_payload(record)
            for record in orchestrator.repository.list_for_user(user.user_id)
            if record.status != RunStatus.DELETED
        ]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, user: User = Depends(_current_user)):
        record = _get_run_or_404(run_id)
        _require_run_read(user, record)
        return _run_payload(record, include_admin_cache=bool(user.is_admin))

    @app.get("/api/runs/{run_id}/monitoring")
    def get_run_monitoring(run_id: str, user: User = Depends(_current_user)):
        record = _get_run_or_404(run_id)
        _require_run_read(user, record)
        if orchestrator.monitoring_orchestrator is None:
            return {"available": False, "monitoring_bundle_id": None, "reports": [], "candidate": None}
        return orchestrator.monitoring_orchestrator.monitoring_payload_for_run(run_id)

    @app.get("/api/runs/{run_id}/artifacts")
    def list_artifacts(run_id: str, user: User = Depends(_current_user)):
        record = _get_run_or_404(run_id)
        _require_run_read(user, record)
        return [
            {"artifact_id": ref.artifact_id, "content_type": ref.content_type, "size_bytes": ref.size_bytes}
            for ref in orchestrator.artifact_store.list(run_id)
        ]

    @app.get("/api/runs/{run_id}/artifacts/{artifact_id}")
    def get_artifact(run_id: str, artifact_id: str, user: User = Depends(_current_user)):
        record = _get_run_or_404(run_id)
        _require_run_read(user, record)
        data = orchestrator.artifact_store.read_bytes(run_id, artifact_id)
        refs = {ref.artifact_id: ref for ref in orchestrator.artifact_store.list(run_id)}
        content_type = refs.get(artifact_id).content_type if artifact_id in refs else "application/octet-stream"
        return Response(content=data, media_type=content_type)

    @app.get("/api/runs/{run_id}/compare/{other_run_id}")
    def compare_runs(run_id: str, other_run_id: str, user: User = Depends(_current_user)):
        record_a = _get_run_or_404(run_id)
        record_b = _get_run_or_404(other_run_id)
        _require_run_read(user, record_a)
        _require_run_read(user, record_b)
        if record_a.status != RunStatus.COMPLETED or record_b.status != RunStatus.COMPLETED:
            raise HTTPException(status_code=409, detail="Both runs must be COMPLETED before comparison.")
        if record_a.spec.bundle_id != record_b.spec.bundle_id:
            raise HTTPException(status_code=409, detail="Runs use different model bundles and cannot be compared.")
        summary_a = _read_json_artifact(run_id, "calibrated_summary.json")
        summary_b = _read_json_artifact(other_run_id, "calibrated_summary.json")
        geometry_a = _read_json_artifact(run_id, "geometry_meta.json")
        geometry_b = _read_json_artifact(other_run_id, "geometry_meta.json")
        if int(geometry_a.get("n_cells", -1)) != int(geometry_b.get("n_cells", -2)):
            raise HTTPException(status_code=409, detail="Runs use incompatible geometry and cannot be compared.")

        label = other_run_id[:12]
        delta_frame_ids: dict[str, list[str]] = {"mean": [], "spread": [], "p_gt_0p30m": []}
        with tempfile.TemporaryDirectory(prefix="fgn-compare-") as tmp:
            try:
                paths = orchestrator.product_builder.write_compare_delta_maps_from_hdf5(
                    run_a_h5_bytes=orchestrator.artifact_store.read_bytes(run_id, "forecast_members.h5"),
                    run_b_h5_bytes=orchestrator.artifact_store.read_bytes(other_run_id, "forecast_members.h5"),
                    geometry_meta=geometry_a,
                    output_dir=Path(tmp),
                    label=label,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Both runs need forecast_members.h5 for map comparison.") from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Could not generate comparison maps: {exc}") from exc
            for path in paths:
                orchestrator.artifact_store.put_bytes(run_id, path.name, path.read_bytes(), content_type="image/png")
                for product in delta_frame_ids:
                    if f"_{product}_delta_" in path.name:
                        delta_frame_ids[product].append(path.name)
        for product in delta_frame_ids:
            delta_frame_ids[product].sort()
        lead_a = summary_a.get("lead_time_hours") if isinstance(summary_a.get("lead_time_hours"), list) else []
        lead_b = summary_b.get("lead_time_hours") if isinstance(summary_b.get("lead_time_hours"), list) else []
        common_n = min(len(lead_a), len(lead_b))
        return {
            "run_a": _run_payload(record_a),
            "run_b": _run_payload(record_b),
            "artifacts_a": [
                {"artifact_id": ref.artifact_id, "content_type": ref.content_type, "size_bytes": ref.size_bytes}
                for ref in orchestrator.artifact_store.list(run_id)
            ],
            "artifacts_b": [
                {"artifact_id": ref.artifact_id, "content_type": ref.content_type, "size_bytes": ref.size_bytes}
                for ref in orchestrator.artifact_store.list(other_run_id)
            ],
            "summary_a": summary_a,
            "summary_b": summary_b,
            "summary_delta": _summary_delta(summary_a, summary_b),
            "common_lead_time_hours": [float(x) for x in lead_a[:common_n]],
            "delta_frames": delta_frame_ids,
        }

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str, user: User = Depends(_current_user)):
        record = _get_run_or_404(run_id)
        _require_run_read(user, record)
        return _run_payload(orchestrator.cancel(run_id), include_admin_cache=bool(user.is_admin))

    @app.get("/api/runs/{run_id}/cell/{cell_index}/timeseries")
    def get_cell_timeseries(run_id: str, cell_index: int, user: User = Depends(_current_user)):
        """Per-cell ensemble inspector for completed forecast runs.

        Returns the full 60-member raw + calibrated depth trace at one cell
        plus the derived quantiles, exceedance probability traces, and
        per-member arrival distributions. Reads ``forecast_members.h5`` lazily
        — no per-run compute on submit, no large new artifact, and a single
        cell slice is ~50-100 KB on the wire.

        404 if the run did not produce the ensemble HDF5, 422 if
        ``cell_index`` is out of range, and 403 if the caller does not own the
        run.
        """
        record = _get_run_or_404(run_id)
        _require_run_read(user, record)
        try:
            h5_bytes = orchestrator.artifact_store.read_bytes(run_id, "forecast_members.h5")
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Run did not store the ensemble HDF5; per-cell inspection requires "
                    "request_full_hdf5=true at submission time."
                ),
            ) from exc

        from neuralop.flood.serving.products import ForecastProductBuilder

        thresholds = record.spec.exceedance_thresholds_m or (0.05, 0.30)
        # Restrict to the two reference thresholds when the run requested them;
        # otherwise keep the primary requested threshold plus 0.30 m when it is
        # available. This keeps the payload compact without silently ignoring
        # the run's threshold contract.
        inspector_thresholds = _inspector_thresholds(thresholds)
        try:
            payload = ForecastProductBuilder.summarize_cell_timeseries(
                h5_bytes=h5_bytes,
                cell_index=int(cell_index),
                thresholds_m=inspector_thresholds,
                calibration_adapter=orchestrator.calibration_adapter,
            )
            try:
                geometry_meta = json.loads(orchestrator.artifact_store.read_bytes(run_id, "geometry_meta.json"))
            except Exception:
                geometry_meta = {}
            if isinstance(geometry_meta, dict):
                for source_key, payload_key in (
                    ("elevation_m", "elevation_m"),
                    ("slope", "slope"),
                    ("flow_accumulation", "flow_accumulation"),
                ):
                    values = geometry_meta.get(source_key)
                    if isinstance(values, list) and 0 <= int(cell_index) < len(values):
                        payload[payload_key] = values[int(cell_index)]
        except IndexError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return payload

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str, user: User = Depends(_current_user)):
        user = _require_allowed(user)
        try:
            record = orchestrator.delete_run(run_id, user=user)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            # Active run — client should cancel first.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _run_payload(record)

    @app.post("/api/runs/delete")
    async def delete_runs_batch(request: Request, user: User = Depends(_current_user)):
        """Batch delete. Per-run outcomes returned so the UI can show what
        succeeded and why anything was skipped, rather than failing the whole
        batch on the first conflict."""
        user = _require_allowed(user)
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Body must be JSON.") from exc
        run_ids = payload.get("run_ids") if isinstance(payload, dict) else None
        if not isinstance(run_ids, list) or not all(isinstance(x, str) for x in run_ids):
            raise HTTPException(status_code=400, detail="Body must contain a 'run_ids' string list.")
        if len(run_ids) == 0:
            return {"deleted": [], "skipped": []}
        if len(run_ids) > 200:
            raise HTTPException(status_code=400, detail="At most 200 run_ids per batch delete.")
        return orchestrator.delete_runs(run_ids, user=user)

    @app.get("/api/admin/runs")
    def admin_list_runs(user: User = Depends(_current_user)):
        _require_admin(user)
        list_all = getattr(orchestrator.repository, "list_all", None)
        if not callable(list_all):
            raise HTTPException(status_code=500, detail="Run repository does not support admin list_all.")
        return [_run_payload(record) for record in list_all()]

    @app.get("/api/admin/retraining-candidates")
    def admin_list_retraining_candidates(user: User = Depends(_current_user)):
        _require_admin(user)
        if orchestrator.monitoring_orchestrator is None:
            return []
        return orchestrator.monitoring_orchestrator.list_candidates()

    @app.post("/api/admin/retraining-candidates/{candidate_id}/status")
    async def admin_update_retraining_candidate(candidate_id: str, request: Request, user: User = Depends(_current_user)):
        _require_admin(user)
        if orchestrator.monitoring_orchestrator is None:
            raise HTTPException(status_code=404, detail="Monitoring is not configured.")
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
        if not isinstance(body, dict) or "status" not in body:
            raise HTTPException(status_code=400, detail="Body must be {status: str, admin_notes?: str}.")
        try:
            return orchestrator.monitoring_orchestrator.update_candidate_status(
                candidate_id,
                str(body["status"]),
                reviewed_by=user.email,
                admin_notes=body.get("admin_notes"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/admin/retraining-candidates/{candidate_id}/simulated")
    async def admin_mark_simulated(candidate_id: str, request: Request, user: User = Depends(_current_user)):
        _require_admin(user)
        if orchestrator.monitoring_orchestrator is None:
            raise HTTPException(status_code=404, detail="Monitoring is not configured.")
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
        from neuralop.flood.serving.monitoring import HecrasErrorRecord, MonitoringPhase
        try:
            candidate = orchestrator.monitoring_orchestrator.repository.get_candidate(candidate_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        hecras_hdf_path = body.get("hecras_hdf_path")
        hecras_descriptors = body.get("hecras_descriptors", {})
        error_descriptors_input = body.get("error_descriptors")

        if hecras_hdf_path:
            from neuralop.flood.serving.monitoring import validate_hecras_path
            from neuralop.flood.serving.monitoring_build_bundle import hecras_descriptor_for_hdf
            try:
                hdf_path = validate_hecras_path(hecras_hdf_path)
            except ValueError as exc:
                raise HTTPException(status_code=501, detail=str(exc)) from exc
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            try:
                run_record = orchestrator.repository.get(candidate.run_id)
                hecras_descriptors = hecras_descriptor_for_hdf(
                    hdf_path,
                    start_timestep=int(orchestrator.bundle.skip_before_timestep) + int(orchestrator.bundle.n_history),
                    forecast_steps=int(run_record.spec.forecast_steps),
                    dt_seconds=int(orchestrator.bundle.dt_seconds),
                    thresholds_m=tuple(float(x) for x in run_record.spec.exceedance_thresholds_m),
                )
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Could not extract HEC-RAS descriptors: {exc}") from exc
        elif not isinstance(hecras_descriptors, dict):
            raise HTTPException(status_code=400, detail="Body must include hecras_hdf_path or hecras_descriptors object.")

        fgn_logs = orchestrator.monitoring_orchestrator.repository.list_descriptor_logs(
            phase=MonitoringPhase.POST_RUN,
        )
        fgn_post = [
            e for e in fgn_logs if e.run_id == candidate.run_id
        ]
        fgn_post.sort(key=lambda e: e.created_at, reverse=True)
        fgn_descriptors = fgn_post[0].descriptors if fgn_post else {}

        if error_descriptors_input is not None:
            error_descriptors = error_descriptors_input
        else:
            error_descriptors = _compute_error_descriptors(fgn_descriptors, hecras_descriptors)

        record = HecrasErrorRecord(
            error_record_id=f"error_{candidate_id}",
            candidate_id=candidate_id,
            run_id=candidate.run_id,
            monitoring_bundle_id=candidate.monitoring_bundle_id,
            fgn_descriptors=fgn_descriptors,
            hecras_descriptors=hecras_descriptors,
            error_descriptors=error_descriptors,
        )
        orchestrator.monitoring_orchestrator.repository.create_hecras_error_record(record)
        from neuralop.flood.serving.monitoring import CandidateStatus
        orchestrator.monitoring_orchestrator.update_candidate_status(
            candidate_id,
            CandidateStatus.SIMULATED.value,
            reviewed_by=user.email,
        )
        return {"status": "SIMULATED", "error_record_id": record.error_record_id}

    @app.get("/api/admin/monitoring-trends")
    def admin_monitoring_trends(user: User = Depends(_current_user)):
        _require_admin(user)
        if orchestrator.monitoring_orchestrator is None:
            return {"candidates": [], "descriptor_stats": {}}
        monitor = orchestrator.monitoring_orchestrator
        candidates = monitor.list_candidates()
        descriptor_logs = monitor.repository.list_descriptor_logs(limit=500)
        score_values: list[float] = []
        descriptor_values: dict[str, list[float]] = {}
        weekly_descriptor_means: dict[str, dict[str, list[float]]] = {}
        for entry in descriptor_logs:
            for value in entry.scores.values():
                numeric = _finite_number(value)
                if numeric is not None:
                    score_values.append(numeric)
            week = _week_key(entry.created_at)
            weekly_descriptor_means.setdefault(week, {})
            for key, value in entry.descriptors.items():
                numeric = _finite_number(value)
                if numeric is None:
                    continue
                descriptor_values.setdefault(key, []).append(numeric)
                weekly_descriptor_means[week].setdefault(key, []).append(numeric)
        top_flagged: dict[str, int] = {}
        for candidate in candidates[:100]:
            for report in monitor.repository.list_reports_for_run(str(candidate["run_id"])):
                for flag in report.flags:
                    key = flag.descriptor or flag.code
                    top_flagged[key] = top_flagged.get(key, 0) + 1
        return {
            "total_candidates": len(candidates),
            "candidates_by_status": _count_by(candidates, lambda c: c["status"]),
            "recent_candidates": candidates[:20],
            "candidate_counts_by_week": _count_by(candidates, lambda c: _week_key(datetime.fromisoformat(c["created_at"]))),
            "score_distribution": _summary(score_values),
            "descriptor_stats": {key: _summary(values) for key, values in sorted(descriptor_values.items())},
            "descriptor_means_by_week": {
                week: {key: _summary(values)["mean"] for key, values in sorted(raw.items())}
                for week, raw in sorted(weekly_descriptor_means.items())
            },
            "top_flagged_descriptors": [
                {"descriptor": key, "count": count}
                for key, count in sorted(top_flagged.items(), key=lambda item: item[1], reverse=True)[:20]
            ],
        }

    @app.get("/api/admin/drift-status")
    def admin_drift_status(user: User = Depends(_current_user)):
        _require_admin(user)
        if orchestrator.monitoring_orchestrator is None:
            return {"available": False, "results": [], "drift_detected": False}
        monitor = orchestrator.monitoring_orchestrator
        results = monitor.repository.latest_drift_tests()
        history = monitor.repository.list_drift_test_results(limit=500)
        from neuralop.flood.serving.drift import DriftDetector

        by_day: dict[str, list] = {}
        for result in sorted(history, key=lambda r: r.created_at):
            by_day.setdefault(result.created_at.date().isoformat(), []).append(result)
        persistent = DriftDetector().apply_persistence_filter(list(by_day.values()))
        persistent_keys = {(r.test_type, r.descriptor_name) for r in persistent}
        payload = [
            {
                "test_id": r.test_id,
                "test_type": r.test_type,
                "descriptor_name": r.descriptor_name,
                "monitoring_bundle_id": r.monitoring_bundle_id,
                "window_start": r.window_start.isoformat() if r.window_start else None,
                "window_end": r.window_end.isoformat() if r.window_end else None,
                "n_observations": r.n_observations,
                "drift_detected": r.drift_detected,
                "persistent_drift_detected": (r.test_type, r.descriptor_name) in persistent_keys,
                "test_statistic": r.test_statistic,
                "threshold": r.threshold,
                "details": r.details or {},
                "created_at": r.created_at.isoformat(),
            }
            for r in results
        ]
        detected = [item for item in payload if item["drift_detected"]]
        persistent_detected = [item for item in payload if item["persistent_drift_detected"]]
        return {
            "available": True,
            "drift_detected": bool(detected),
            "persistent_drift_detected": bool(persistent_detected),
            "n_results": len(payload),
            "n_detected": len(detected),
            "n_persistent_detected": len(persistent_detected),
            "results": payload,
            "message": (
                "No persistent drift signals detected."
                if not persistent_detected
                else f"{len(persistent_detected)} persistent drift signal(s) detected."
            ),
        }

    @app.get("/api/admin/hecras-errors")
    def admin_hecras_errors(user: User = Depends(_current_user)):
        _require_admin(user)
        if orchestrator.monitoring_orchestrator is None:
            return {"errors": []}
        records = orchestrator.monitoring_orchestrator.repository.list_hecras_error_records(limit=50)
        all_errors = orchestrator.monitoring_orchestrator.repository.list_hecras_error_records(limit=None)
        signed_by_descriptor: dict[str, list[float]] = {}
        absolute_by_descriptor: dict[str, list[float]] = {}
        for record in all_errors:
            for key, value in record.error_descriptors.items():
                numeric = _finite_number(value)
                if numeric is None:
                    continue
                if key.endswith("_signed"):
                    signed_by_descriptor.setdefault(key.removesuffix("_signed"), []).append(numeric)
                elif key.endswith("_absolute"):
                    absolute_by_descriptor.setdefault(key.removesuffix("_absolute"), []).append(numeric)
        threshold = float(os.environ.get("FGN_HECRAS_AREA_FRACTION_ERROR_THRESHOLD", "0.1"))
        degradation_flags = []
        for descriptor, values in signed_by_descriptor.items():
            if len(values) < 10 or "area_fraction" not in descriptor:
                continue
            mean_signed = sum(values) / len(values)
            if abs(mean_signed) >= threshold:
                degradation_flags.append(
                    {
                        "descriptor": descriptor,
                        "mean_signed_error": mean_signed,
                        "n": len(values),
                        "threshold": threshold,
                    }
                )
        return {
            "total": len(records),
            "aggregate": {
                descriptor: {
                    "mean_signed_error": sum(values) / len(values),
                    "rmse": math.sqrt(
                        sum(v * v for v in signed_by_descriptor.get(descriptor, []))
                        / max(1, len(signed_by_descriptor.get(descriptor, [])))
                    ),
                    "mean_absolute_error": (
                        sum(absolute_by_descriptor.get(descriptor, []))
                        / max(1, len(absolute_by_descriptor.get(descriptor, [])))
                    ),
                    "n": len(values),
                }
                for descriptor, values in sorted(signed_by_descriptor.items())
            },
            "degradation_flags": degradation_flags,
            "records": [
                {
                    "error_record_id": r.error_record_id,
                    "candidate_id": r.candidate_id,
                    "run_id": r.run_id,
                    "error_descriptors": r.error_descriptors,
                    "created_at": r.created_at.isoformat(),
                }
                for r in records
            ],
        }

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
        record = _get_run_or_404(run_id)
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
