"""Run orchestration module for flood serving."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from neuralop.flood.serving.access import AccessPolicy, User
from neuralop.flood.serving.calibration import CalibrationAdapter
from neuralop.flood.serving.forcing import parse_forcing_csv
from neuralop.flood.serving.inference import FGNInferenceService
from neuralop.flood.serving.model_bundle import FGNModelBundle
from neuralop.flood.serving.products import ForecastProductBuilder, ForecastResult
from neuralop.flood.serving.quota import QuotaPolicy
from neuralop.flood.serving.queue import JobQueue
from neuralop.flood.serving.repository import RunRepository
from neuralop.flood.serving.run_spec import RunSpec, RunStatus
from neuralop.flood.serving.storage import ArtifactStore


@dataclass
class RunOrchestrator:
    bundle: FGNModelBundle
    repository: RunRepository
    queue: JobQueue
    artifact_store: ArtifactStore
    access_policy: AccessPolicy
    inference_service: FGNInferenceService
    calibration_adapter: CalibrationAdapter
    product_builder: ForecastProductBuilder
    quota_policy: QuotaPolicy = QuotaPolicy()

    def submit(
        self,
        *,
        user: User,
        forcing_csv: str | bytes,
        label: str | None = None,
        forecast_steps: int | None = None,
        output_detail: str = "standard",
        exceedance_thresholds_m: Sequence[float] | None = None,
        request_full_hdf5: bool = False,
        request_animation: bool = False,
    ):
        user = self.access_policy.require_disclaimer(user)
        self.quota_policy.validate_submit(self.repository, user.user_id)
        forcing = parse_forcing_csv(forcing_csv, bundle=self.bundle, requested_forecast_steps=forecast_steps)
        spec = RunSpec.new(
            user_id=user.user_id,
            bundle_id=self.bundle.bundle_id,
            input_hash=forcing.input_hash,
            forecast_steps=forcing.forecast_steps,
            label=label,
            output_detail=output_detail,
            request_full_hdf5=request_full_hdf5,
            request_animation=request_animation,
            exceedance_thresholds_m=exceedance_thresholds_m,
        )
        self.repository.create(spec)
        payload = forcing_csv if isinstance(forcing_csv, bytes) else str(forcing_csv).encode("utf-8")
        self.artifact_store.put_bytes(spec.run_id, "forcing.csv", payload, content_type="text/csv")
        self.artifact_store.put_json(
            spec.run_id,
            "run_manifest",
            {"run": spec.manifest(), "forcing": forcing.summary(), "bundle": self.bundle.public_metadata()},
        )
        self.repository.transition(spec.run_id, RunStatus.VALIDATING)
        self.repository.transition(spec.run_id, RunStatus.QUEUED)
        self.queue.enqueue(spec.run_id)
        return self.repository.get(spec.run_id)

    def execute(self, run_id: str) -> None:
        if not self.quota_policy.can_start(self.repository, run_id):
            self.queue.enqueue(run_id)
            return
        record = self.repository.transition(run_id, RunStatus.RUNNING)
        try:
            forcing_bytes = self.artifact_store.read_bytes(run_id, "forcing.csv")
            forcing = parse_forcing_csv(forcing_bytes, bundle=self.bundle, requested_forecast_steps=record.spec.forecast_steps)
            raw = self.inference_service.run(record.spec, forcing)
            self.repository.transition(run_id, RunStatus.POSTPROCESSING)
            calibrated = self.calibration_adapter.apply(raw)
            product_builder = self._product_builder(record.spec.exceedance_thresholds_m)
            raw_summary = product_builder.build_summary(raw, label="raw")
            calibrated_summary = product_builder.build_summary(
                calibrated,
                label="calibrated",
                calibration_adapter=self.calibration_adapter,
            )
            self.artifact_store.put_json(run_id, "raw_summary", raw_summary)
            self.artifact_store.put_json(run_id, "calibrated_summary", calibrated_summary)
            self.artifact_store.put_json(
                run_id,
                "comparison_summary",
                {
                    "run_id": run_id,
                    "bundle_id": self.bundle.bundle_id,
                    "raw_mean_spread_wd_m": raw_summary["mean_spread_wd_m"],
                    "calibrated_mean_spread_wd_m": calibrated_summary["mean_spread_wd_m"],
                    "raw_max_mean_wd_m": raw_summary["max_mean_wd_m"],
                    "calibrated_max_mean_wd_m": calibrated_summary["max_mean_wd_m"],
                },
            )
            self._write_map_products(run_id, raw=raw, calibrated=calibrated)
            if record.spec.request_animation:
                self._write_animation(run_id, calibrated=calibrated)
            if record.spec.request_full_hdf5:
                self._write_forecast_hdf5(run_id, raw=raw, calibrated=calibrated)
            self.repository.transition(run_id, RunStatus.COMPLETED)
        except Exception as exc:
            try:
                self.repository.transition(run_id, RunStatus.FAILED, failure_reason=_safe_failure_reason(exc))
            finally:
                raise

    def cancel(self, run_id: str):
        record = self.repository.get(run_id)
        if record.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED, RunStatus.EXPIRED}:
            return record
        return self.repository.transition(run_id, RunStatus.CANCELED)

    def expire_due_runs(self, *, retention_days: int = 30, now=None):
        from neuralop.flood.serving.retention import ExpirationResult, RetentionManager

        manager = RetentionManager(
            repository=self.repository,
            artifact_store=self.artifact_store,
            retention_days=retention_days,
        )
        expired: list[str] = []
        skipped: list[str] = []
        for record in manager.due_records(now=now):
            self.artifact_store.delete_run_artifacts(record.spec.run_id)
            self.repository.transition(record.spec.run_id, RunStatus.EXPIRED)
            expired.append(record.spec.run_id)
        expired_set = set(expired)
        list_all = getattr(self.repository, "list_all", None)
        if callable(list_all):
            skipped = [r.spec.run_id for r in list_all() if r.spec.run_id not in expired_set]
        return ExpirationResult(tuple(expired), tuple(skipped))

    def _write_map_products(self, run_id: str, *, raw: ForecastResult, calibrated: ForecastResult) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            paths = []
            paths.extend(self.product_builder.write_map_pngs(raw, output_dir=tmp_dir, label="raw", max_times=3))
            paths.extend(
                self.product_builder.write_map_pngs(
                    calibrated,
                    output_dir=tmp_dir,
                    label="calibrated",
                    max_times=3,
                    calibration_adapter=self.calibration_adapter,
                )
            )
            for path in paths:
                self.artifact_store.put_bytes(run_id, path.name, path.read_bytes(), content_type="image/png")

    def _product_builder(self, thresholds_m: Sequence[float]) -> ForecastProductBuilder:
        return ForecastProductBuilder(thresholds_m=thresholds_m)

    def _write_animation(self, run_id: str, *, calibrated: ForecastResult) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "calibrated_mean_wd_animation.gif"
            written = self.product_builder.write_animation_gif(
                calibrated,
                output_path=out_path,
                label="calibrated",
            )
            if written is None or not written.exists():
                return
            self.artifact_store.put_bytes(
                run_id,
                "calibrated_mean_wd_animation.gif",
                written.read_bytes(),
                content_type="image/gif",
            )

    def _write_forecast_hdf5(self, run_id: str, *, raw: ForecastResult, calibrated: ForecastResult) -> None:
        try:
            import h5py
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("Full HDF5 ensemble export requires h5py in the serving worker environment.") from exc
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with h5py.File(tmp_path, "w") as h5:
                h5.create_dataset("raw_members_wd", data=raw.members_wd, compression="gzip")
                h5.create_dataset("calibrated_members_wd", data=calibrated.members_wd, compression="gzip")
                h5.create_dataset("lead_time_hours", data=calibrated.lead_time_hours)
                if calibrated.wettable_mask is not None:
                    h5.create_dataset("wettable_mask", data=calibrated.wettable_mask.astype("uint8"))
                h5.attrs["bundle_id"] = self.bundle.bundle_id
                h5.attrs["calibration"] = "crps_member_by_member"
            self.artifact_store.put_bytes(
                run_id,
                "forecast_members.h5",
                tmp_path.read_bytes(),
                content_type="application/x-hdf5",
            )
        finally:
            tmp_path.unlink(missing_ok=True)


def _safe_failure_reason(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    return message[:1000]
