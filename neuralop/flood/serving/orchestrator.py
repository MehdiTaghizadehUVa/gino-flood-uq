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
from neuralop.flood.serving.monitoring import MonitoringOrchestrator
from neuralop.flood.serving.products import ForecastProductBuilder, ForecastResult
from neuralop.flood.serving.quota import QuotaPolicy
from neuralop.flood.serving.queue import JobQueue
from neuralop.flood.serving.repository import RunRepository
from neuralop.flood.serving.run_spec import TERMINAL_STATUSES, RunSpec, RunStatus
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
    monitoring_orchestrator: MonitoringOrchestrator | None = None

    def submit(
        self,
        *,
        user: User,
        forcing_csv: str | bytes,
        label: str | None = None,
        forecast_steps: int | None = None,
        output_detail: str = "standard",
        exceedance_thresholds_m: Sequence[float] | None = None,
        # HDF5 ensemble export is required by the per-cell inspector
        # (GET /api/runs/{id}/cell/{idx}/timeseries reads it lazily), so the
        # default is True. Callers can still opt out for storage-sensitive
        # deployments; the inspector then 404s with a helpful message.
        request_full_hdf5: bool = True,
        request_animation: bool = False,
        ensemble_count: int | None = None,
        members_per_ensemble: int | None = None,
    ):
        user = self.access_policy.require_disclaimer(user)
        self.quota_policy.validate_submit(self.repository, user.user_id)
        forcing = parse_forcing_csv(forcing_csv, bundle=self.bundle, requested_forecast_steps=forecast_steps)
        ensemble_count, members_per_ensemble = self._validate_member_budget(
            ensemble_count=ensemble_count,
            members_per_ensemble=members_per_ensemble,
        )
        spec = RunSpec.new(
            user_id=user.user_id,
            bundle_id=self.bundle.bundle_id,
            input_hash=forcing.input_hash,
            forecast_steps=forcing.forecast_steps,
            label=label,
            output_detail=output_detail,
            request_full_hdf5=request_full_hdf5,
            request_animation=request_animation,
            ensemble_count=ensemble_count,
            members_per_ensemble=members_per_ensemble,
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
        if self.monitoring_orchestrator is not None:
            self.monitoring_orchestrator.screen_forcing(
                run_spec=spec,
                owner_email=user.email,
                forcing=forcing,
                forcing_csv=payload,
                model_bundle_metadata=self.bundle.public_metadata(),
            )
        self.repository.transition(spec.run_id, RunStatus.VALIDATING)
        self.repository.transition(spec.run_id, RunStatus.QUEUED)
        self.queue.enqueue(spec.run_id)
        return self.repository.get(spec.run_id)

    def _validate_member_budget(
        self,
        *,
        ensemble_count: int | None = None,
        members_per_ensemble: int | None = None,
    ) -> tuple[int, int]:
        """Normalize a per-run ensemble budget against the deployment bundle.

        The deployed coastal bundle remains fixed at 3 checkpoint ensembles x
        20 latent members. A run may request a smaller deterministic subset for
        faster exploratory work, but it cannot exceed the validated bundle.
        """
        max_ensembles = int(self.bundle.n_checkpoints)
        max_members = int(self.bundle.members_per_checkpoint)
        next_ensembles = max_ensembles if ensemble_count is None else int(ensemble_count)
        next_members = max_members if members_per_ensemble is None else int(members_per_ensemble)
        if next_ensembles < 1 or next_ensembles > max_ensembles:
            raise ValueError(f"ensemble_count must be between 1 and {max_ensembles}.")
        if next_members < 1 or next_members > max_members:
            raise ValueError(f"members_per_ensemble must be between 1 and {max_members}.")
        return next_ensembles, next_members

    def execute(self, run_id: str) -> None:
        if self._is_terminal(run_id):
            return
        if not self.quota_policy.can_start(self.repository, run_id):
            self.queue.enqueue(run_id)
            return
        if self._is_terminal(run_id):
            return
        record = self.repository.transition(run_id, RunStatus.RUNNING)
        self._set_progress(run_id, 0.38, "Preparing forcing and GPU tensors")
        try:
            forcing_bytes = self.artifact_store.read_bytes(run_id, "forcing.csv")
            forcing = parse_forcing_csv(forcing_bytes, bundle=self.bundle, requested_forecast_steps=record.spec.forecast_steps)
            last_inference_progress = [0.38]

            def _on_inference_progress(fraction: float, label: str) -> None:
                mapped = 0.38 + max(0.0, min(1.0, float(fraction))) * 0.38
                if mapped - last_inference_progress[0] >= 0.01 or mapped >= 0.76:
                    last_inference_progress[0] = mapped
                    self._set_progress(run_id, mapped, label)

            raw = self.inference_service.run(
                record.spec,
                forcing,
                progress_callback=_on_inference_progress,
            )
            if self._is_terminal(run_id):
                return
            self._set_progress(run_id, 0.78, "GPU rollout complete; applying calibration")
            self.repository.transition(run_id, RunStatus.POSTPROCESSING)
            self._set_progress(run_id, 0.82, "Applying calibration and building summaries")
            calibrated = self.calibration_adapter.apply(raw)
            product_builder = self._product_builder(record.spec.exceedance_thresholds_m)
            raw_summary = product_builder.build_summary(raw, label="raw")
            calibrated_summary = product_builder.build_summary(
                calibrated,
                label="calibrated",
                calibration_adapter=self.calibration_adapter,
            )
            comparison_summary = _comparison_summary(
                run_id=run_id,
                bundle_id=self.bundle.bundle_id,
                raw_summary=raw_summary,
                calibrated_summary=calibrated_summary,
                thresholds_m=record.spec.exceedance_thresholds_m,
            )
            self.artifact_store.put_json(run_id, "raw_summary", raw_summary)
            self.artifact_store.put_json(run_id, "calibrated_summary", calibrated_summary)
            self.artifact_store.put_json(run_id, "comparison_summary", comparison_summary)
            if self.monitoring_orchestrator is not None:
                self.monitoring_orchestrator.evaluate_completed_run(
                    run_spec=record.spec,
                    owner_email=record.spec.user_id,
                    raw_summary=raw_summary,
                    calibrated_summary=calibrated_summary,
                    comparison_summary=comparison_summary,
                    model_bundle_metadata=self.bundle.public_metadata(),
                )
            self._set_progress(run_id, 0.86, "Rendering publication-style UQ figures")
            self._write_figure_products(
                run_id,
                forcing=forcing,
                raw_summary=raw_summary,
                calibrated_summary=calibrated_summary,
                comparison_summary=comparison_summary,
                thresholds_m=record.spec.exceedance_thresholds_m,
            )
            self._set_progress(run_id, 0.89, "Rendering map products")
            self._write_map_products(run_id, raw=raw, calibrated=calibrated)
            # Per-cell envelope summaries power the Hazard tab's small-multiples
            # and feed the per-cell inspector's elevation/peak/arrival readouts.
            # Cheap (one .npy per array, no rendering); failure is non-fatal.
            self._set_progress(run_id, 0.92, "Writing hazard envelope maps")
            self._write_envelope_maps(run_id, calibrated=calibrated)
            # Empirical CRPS, between/within decomposition, and reliability
            # curves. Computed from the calibrated ensemble; isotonic adapter
            # passed through so the reliability card can show what calibration
            # actually did.
            self._set_progress(run_id, 0.94, "Writing uncertainty diagnostics")
            self._write_uncertainty_diagnostics(run_id, calibrated=calibrated)
            self._set_progress(run_id, 0.955, "Writing driver diagnostics")
            self._write_driver_products(run_id, calibrated=calibrated)
            if record.spec.request_animation:
                self._set_progress(run_id, 0.965, "Rendering animations and scrub frames")
                self._write_animation(run_id, calibrated=calibrated)
                # Frame-accurate scrub PNGs power the interactive Time Player.
                # Generated unconditionally with the GIF so the two stay in
                # sync; failure here is non-fatal because the run's primary
                # products and the GIF are already on disk.
                self._write_scrub_frames(run_id, calibrated=calibrated)
            if record.spec.request_full_hdf5:
                self._set_progress(run_id, 0.985, "Writing full ensemble HDF5 for cell inspection")
                self._write_forecast_hdf5(run_id, raw=raw, calibrated=calibrated)
            if self._is_terminal(run_id):
                return
            self.repository.transition(run_id, RunStatus.COMPLETED)
        except Exception as exc:
            if self._is_terminal(run_id):
                return
            try:
                self.repository.transition(run_id, RunStatus.FAILED, failure_reason=_safe_failure_reason(exc))
            finally:
                raise

    def cancel(self, run_id: str):
        record = self.repository.get(run_id)
        if record.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELED,
            RunStatus.EXPIRED,
            RunStatus.DELETED,
        }:
            return record
        return self.repository.transition(run_id, RunStatus.CANCELED)

    def _is_terminal(self, run_id: str) -> bool:
        return self.repository.get(run_id).status in TERMINAL_STATUSES

    def _set_progress(self, run_id: str, progress: float, label: str) -> None:
        if self._is_terminal(run_id):
            return
        update_progress = getattr(self.repository, "update_progress", None)
        if callable(update_progress):
            update_progress(run_id, progress, label=label)

    def delete_run(self, run_id: str, *, user):
        """Delete run artifacts while preserving auditable run metadata.

        Permission: the run's owner or an admin. State: only terminal runs
        (COMPLETED / FAILED / CANCELED / EXPIRED) are deletable; active runs
        must be cancelled first so the worker can stop writing files. The
        repository row is intentionally retained and tombstoned as DELETED so
        bundle ID, input hash, calibration settings, and timestamps remain
        available for audit after artifact bytes are removed.
        """
        record = self.repository.get(run_id)
        try:
            self.access_policy.require_run_read(user, record)
        except PermissionError as exc:
            raise PermissionError(f"User {user.email} is not allowed to delete run {run_id}.") from exc
        if record.status == RunStatus.DELETED:
            return record
        if record.status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED, RunStatus.EXPIRED}:
            raise ValueError(
                f"Cannot delete run {run_id} in status {record.status.value}. "
                "Cancel the run first, then delete."
            )
        self.artifact_store.delete_run_artifacts(run_id)
        return self.repository.transition(run_id, RunStatus.DELETED)

    def delete_runs(self, run_ids, *, user) -> dict:
        """Best-effort batch delete; per-run outcomes are returned not raised.

        UI callers want one click to dispatch many deletes and a per-row
        verdict to render. Raising would force the client to short-circuit on
        the first failure (e.g. one active run inside a batch of completed
        ones) which is not what users expect from a multi-select action.
        """
        deleted: list[str] = []
        skipped: list[dict] = []
        for run_id in run_ids:
            try:
                self.delete_run(run_id, user=user)
                deleted.append(run_id)
            except KeyError:
                skipped.append({"run_id": run_id, "reason": "not_found"})
            except PermissionError as exc:
                skipped.append({"run_id": run_id, "reason": "forbidden", "detail": str(exc)})
            except ValueError as exc:
                skipped.append({"run_id": run_id, "reason": "active", "detail": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive
                skipped.append({"run_id": run_id, "reason": "error", "detail": str(exc)})
        return {"deleted": deleted, "skipped": skipped}

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

    def _write_figure_products(
        self,
        run_id: str,
        *,
        forcing,
        raw_summary: dict,
        calibrated_summary: dict,
        comparison_summary: dict,
        thresholds_m: Sequence[float],
    ) -> None:
        from neuralop.flood.serving.figures import ForecastFigureBuilder

        with tempfile.TemporaryDirectory() as tmp:
            paths = ForecastFigureBuilder(
                dt_seconds=self.bundle.dt_seconds,
                skip_before_timestep=self.bundle.skip_before_timestep,
                n_history=self.bundle.n_history,
                thresholds_m=thresholds_m,
            ).write_figures(
                forcing=forcing,
                raw_summary=raw_summary,
                calibrated_summary=calibrated_summary,
                comparison_summary=comparison_summary,
                output_dir=Path(tmp),
            )
            for path in paths:
                self.artifact_store.put_bytes(
                    run_id,
                    path.name,
                    path.read_bytes(),
                    content_type="image/svg+xml",
                )

    def _write_animation(self, run_id: str, *, calibrated: ForecastResult) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_path = tmp_dir / "calibrated_mean_wd_animation.gif"
            written = self.product_builder.write_animation_gif(
                calibrated,
                output_path=out_path,
                label="calibrated",
            )
            if written is not None and written.exists():
                self.artifact_store.put_bytes(
                    run_id,
                    "calibrated_mean_wd_animation.gif",
                    written.read_bytes(),
                    content_type="image/gif",
                )
            threshold = _primary_animation_threshold(self.product_builder.thresholds_m)
            prob_path = tmp_dir / f"calibrated_p_gt_{ForecastProductBuilder._threshold_key(threshold)}m_animation.gif"
            prob_written = self.product_builder.write_exceedance_animation_gif(
                calibrated,
                output_path=prob_path,
                threshold_m=threshold,
                label="calibrated",
                calibration_adapter=self.calibration_adapter,
            )
            if prob_written is not None and prob_written.exists():
                self.artifact_store.put_bytes(
                    run_id,
                    prob_written.name,
                    prob_written.read_bytes(),
                    content_type="image/gif",
                )

    def _write_scrub_frames(self, run_id: str, *, calibrated: ForecastResult) -> None:
        """Render per-step scrub frames for every Time Player product.

        Produces three product streams in one tempdir: ``mean``, ``spread``,
        and the isotonic-calibrated ``p_gt_0p30m``. Each is independently
        try/excepted so a failure in one product (e.g. missing isotonic
        curves) doesn't block the others. The frontend's Time Player
        discovers what's available and renders the toggle accordingly.
        """
        builder = self.product_builder
        for product in builder.SCRUB_PRODUCTS:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    paths = builder.write_scrub_frames(
                        calibrated,
                        output_dir=Path(tmp),
                        label="calibrated",
                        product=product,
                        # Only matters when product == "p_gt_0p30m"; ignored
                        # otherwise. Hard-coded to 0.30 m to match the
                        # decision-summary headline threshold; per-run
                        # customisation can come in a later phase.
                        threshold_m=0.30,
                        calibration_adapter=self.calibration_adapter,
                    )
                    for path in paths:
                        self.artifact_store.put_bytes(
                            run_id, path.name, path.read_bytes(), content_type="image/png"
                        )
            except Exception:  # pragma: no cover - non-fatal supporting product
                continue

    def _write_envelope_maps(self, run_id: str, *, calibrated: ForecastResult) -> None:
        """Persist per-cell envelope maps for the Hazard tab small-multiples.

        Each map is written twice: once as a float32 ``.npy`` array (machine-
        readable; the per-cell inspector pulls from these), and
        once as a ``.png`` rendering using the same DEM-context renderer as
        the static map gallery (what the Hazard tab actually displays). Both
        live in the same artifact store under predictable names.

        Also emits ``geometry_meta.json`` — the compact geometry sidecar the
        click-to-inspect overlay uses to translate map clicks into a
        cell index. Storing it here keeps the spatial metadata co-located
        with the maps that depend on it. Failure is non-fatal.
        """
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                written = self.product_builder.write_envelope_maps(
                    calibrated,
                    output_dir=tmp_dir,
                    render_pngs=True,
                )
                # Persist .npy artifacts.
                for name, path in written.items():
                    self.artifact_store.put_bytes(
                        run_id, name, path.read_bytes(), content_type="application/octet-stream"
                    )
                # Persist the paired .png renderings (every <name>.npy has a <name>.png).
                for npy_name in written:
                    png_name = npy_name.replace(".npy", ".png")
                    png_path = tmp_dir / png_name
                    if png_path.exists():
                        self.artifact_store.put_bytes(
                            run_id, png_name, png_path.read_bytes(), content_type="image/png"
                        )
                # Geometry sidecar for click hit-testing.
                geometry_meta = self.product_builder.build_geometry_meta(calibrated)
                if geometry_meta is not None:
                    self.artifact_store.put_json(run_id, "geometry_meta", geometry_meta)
        except Exception:  # pragma: no cover - non-fatal supporting product
            pass

    def _write_uncertainty_diagnostics(self, run_id: str, *, calibrated: ForecastResult) -> None:
        """Persist uncertainty-tab diagnostics.

        Bundles together: empirical CRPS map (peak-time intra-ensemble
        spread), spread decomposition (between vs within checkpoint
        variance) with summary, and reliability curves JSON. PNG renderings
        of the three spatial maps are persisted alongside the .npy/.npz
        files so the frontend can ``<img>`` them directly. Failure is
        non-fatal — the Hazard / Drivers tabs still work without these.
        """
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                written = self.product_builder.write_uncertainty_diagnostics(
                    calibrated,
                    output_dir=tmp_dir,
                    calibration_adapter=self.calibration_adapter,
                    render_pngs=True,
                )
                # Persist data artifacts (.npy, .npz, .json) by their suffix.
                for name, path in written.items():
                    if name.endswith(".json"):
                        ctype = "application/json"
                    elif name.endswith(".svg"):
                        ctype = "image/svg+xml"
                    elif name.endswith(".png"):
                        ctype = "image/png"
                    else:
                        ctype = "application/octet-stream"
                    self.artifact_store.put_bytes(
                        run_id, name, path.read_bytes(), content_type=ctype
                    )
                # Persist the paired PNG renderings written next to the data files.
                for png_name in (
                    "empirical_crps_map.png",
                    "between_var_map.png",
                    "within_var_map.png",
                ):
                    png_path = tmp_dir / png_name
                    if png_path.exists():
                        self.artifact_store.put_bytes(
                            run_id, png_name, png_path.read_bytes(), content_type="image/png"
                        )
        except Exception:  # pragma: no cover - non-fatal supporting product
            pass

    def _write_driver_products(self, run_id: str, *, calibrated: ForecastResult) -> None:
        """Persist forcing/response support products for the Drivers tab."""
        try:
            leaderboard = self.product_builder.build_cell_contribution_leaderboard(
                calibrated,
                threshold_m=0.30,
                top_n=50,
                calibration_adapter=self.calibration_adapter,
            )
            self.artifact_store.put_json(run_id, "cell_contribution_leaderboard", leaderboard)
        except Exception:  # pragma: no cover - non-fatal supporting product
            pass

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


def _primary_animation_threshold(thresholds_m: Sequence[float]) -> float:
    values = [float(x) for x in thresholds_m]
    for value in values:
        if abs(value - 0.30) < 1.0e-9:
            return value
    return max(values) if values else 0.30


def _comparison_summary(
    *,
    run_id: str,
    bundle_id: str,
    raw_summary: dict,
    calibrated_summary: dict,
    thresholds_m: Sequence[float],
) -> dict:
    raw_peak_fraction = float(raw_summary.get("peak_expected_flooded_area_fraction_wettable_gt_0.05m") or 0.0)
    cal_peak_fraction = float(calibrated_summary.get("peak_expected_flooded_area_fraction_wettable_gt_0.05m") or 0.0)
    raw_iqr = float(raw_summary.get("peak_area_weighted_iqr_wd_m") or 0.0)
    cal_iqr = float(calibrated_summary.get("peak_area_weighted_iqr_wd_m") or 0.0)
    raw_exceedance = raw_summary.get("exceedance_by_threshold_m", {})
    cal_exceedance = calibrated_summary.get("exceedance_by_threshold_m", {})
    threshold_deltas: dict[str, float] = {}
    if isinstance(raw_exceedance, dict) and isinstance(cal_exceedance, dict):
        for threshold in thresholds_m:
            key = f"{float(threshold):g}"
            raw_item = raw_exceedance.get(key, {})
            cal_item = cal_exceedance.get(key, {})
            if not isinstance(raw_item, dict) or not isinstance(cal_item, dict):
                continue
            threshold_deltas[key] = 100.0 * (
                float(cal_item.get("peak_expected_area_fraction_wettable", 0.0) or 0.0)
                - float(raw_item.get("peak_expected_area_fraction_wettable", 0.0) or 0.0)
            )
    return {
        "run_id": run_id,
        "bundle_id": bundle_id,
        "raw_mean_spread_wd_m": raw_summary["mean_spread_wd_m"],
        "calibrated_mean_spread_wd_m": calibrated_summary["mean_spread_wd_m"],
        "raw_max_mean_wd_m": raw_summary["max_mean_wd_m"],
        "calibrated_max_mean_wd_m": calibrated_summary["max_mean_wd_m"],
        "raw_peak_expected_flooded_area_fraction_wettable_gt_0.05m": raw_peak_fraction,
        "calibrated_peak_expected_flooded_area_fraction_wettable_gt_0.05m": cal_peak_fraction,
        "delta_peak_expected_flooded_area_fraction_wettable_percentage_points_gt_0.05m": (
            100.0 * (cal_peak_fraction - raw_peak_fraction)
        ),
        "raw_peak_area_weighted_iqr_wd_m": raw_iqr,
        "calibrated_peak_area_weighted_iqr_wd_m": cal_iqr,
        "delta_peak_area_weighted_iqr_wd_m": cal_iqr - raw_iqr,
        "delta_peak_expected_area_fraction_wettable_percentage_points_by_threshold_m": threshold_deltas,
    }
