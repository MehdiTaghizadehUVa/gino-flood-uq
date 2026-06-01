import json
from pathlib import Path

import numpy as np
import pytest

from neuralop.flood.serving.access import AccessDenied, AccessPolicy, User
from neuralop.flood.serving.calibration import CalibrationAdapter
from neuralop.flood.serving.forcing import ForcingValidationError, parse_forcing_csv
from neuralop.flood.serving.forcing import build_forcing_template_csv
from neuralop.flood.serving.inference import DomainAssets, FakeFGNInferenceService, ProductionFGNInferenceService
from neuralop.flood.serving.model_bundle import FGNModelBundle, ModelBundleError
from neuralop.flood.serving.orchestrator import RunOrchestrator
from neuralop.flood.serving.products import ForecastProductBuilder, ForecastResult
from neuralop.flood.serving.queue import InMemoryJobQueue
from neuralop.flood.serving.repository import InMemoryRunRepository
from neuralop.flood.serving.result_cache import (
    InMemoryResultCacheRepository,
    LocalCacheArtifactStore,
    ResultCache,
    build_result_cache_key,
)
from neuralop.flood.serving.run_spec import RunSpec, RunStateError, RunStatus
from neuralop.flood.serving.storage import LocalArtifactStore


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _bundle(tmp_path: Path, *, paths: bool = True) -> FGNModelBundle:
    ckpts = []
    for idx in range(3):
        d = tmp_path / f"ckpt{idx}"
        d.mkdir(parents=True)
        _touch(d / "best_model_state_dict.pt")
        ckpts.append(d)
    normalizer = _touch(tmp_path / "normalizers_depth_only.pt")
    coeff = _touch(tmp_path / "crps.json", '{"lead_time_hours":[0,999],"wet_frequency_edges":[0,1],"wet_frequency_by_cell":[1,1],"coefficients":[[[0,1,1]]]}')
    iso = _touch(tmp_path / "iso.json", "{}")
    statics = [_touch(tmp_path / f"static{i}.txt") for i in range(5)]
    return FGNModelBundle(
        bundle_id="coastal-fgn-60-v1",
        domain_name="coastal",
        git_commit="abc123",
        checkpoint_dirs=ckpts,
        checkpoint_alias="best_model",
        normalizer_path=normalizer,
        static_files=statics,
        calibration_coefficients_path=coeff if paths else tmp_path / "missing_crps.json",
        isotonic_curves_path=iso,
        boundary_channels=["stage", "precipitation"],
        dt_seconds=900,
        n_history=3,
        skip_before_timestep=12,
        max_forecast_steps=94,
        fgn_noise_dim=32,
        members_per_checkpoint=20,
        mesh_hash="mesh-a",
        expected_mesh_hash="mesh-a",
    )


def _valid_csv(n=24):
    lines = ["time_seconds,stage,precipitation"]
    for i in range(n):
        lines.append(f"{i*900},{1.0 + i*0.01},{0.2}")
    return "\n".join(lines) + "\n"


def _fake_orchestrator(tmp_path: Path) -> RunOrchestrator:
    bundle = _bundle(tmp_path)
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    return RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
    )


def _cached_orchestrator(tmp_path: Path) -> tuple[RunOrchestrator, InMemoryJobQueue]:
    bundle = _bundle(tmp_path)
    queue = InMemoryJobQueue()
    run_store = LocalArtifactStore(tmp_path / "artifacts")
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    result_cache = ResultCache(
        repository=InMemoryResultCacheRepository(),
        run_artifact_store=run_store,
        cache_artifact_store=LocalCacheArtifactStore(tmp_path / "result-cache"),
    )
    return (
        RunOrchestrator(
            bundle=bundle,
            repository=InMemoryRunRepository(),
            queue=queue,
            artifact_store=run_store,
            access_policy=AccessPolicy(allowed_emails=["alice@example.com", "bob@example.com"]),
            inference_service=FakeFGNInferenceService(bundle, n_cells=8),
            calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
            product_builder=ForecastProductBuilder(),
            result_cache=result_cache,
        ),
        queue,
    )


def test_model_bundle_rejects_missing_calibration(tmp_path):
    bundle = _bundle(tmp_path, paths=False)
    with pytest.raises(ModelBundleError, match="missing required files"):
        bundle.validate(validate_paths=True)


def test_model_bundle_rejects_wrong_member_count(tmp_path):
    bundle = _bundle(tmp_path)
    bad = FGNModelBundle(**{**bundle.__dict__, "members_per_checkpoint": 10})
    with pytest.raises(ModelBundleError, match="pinned to 60"):
        bad.validate(validate_paths=False)


def test_forcing_csv_validation_and_hash(tmp_path):
    bundle = _bundle(tmp_path)
    forcing = parse_forcing_csv(_valid_csv(), bundle=bundle, requested_forecast_steps=5)
    assert forcing.forecast_steps == 5
    assert forcing.as_boundary_matrix().shape == (24, 2)
    assert len(forcing.input_hash) == 64


def test_forcing_template_is_valid_for_bundle(tmp_path):
    bundle = _bundle(tmp_path)
    forcing = parse_forcing_csv(build_forcing_template_csv(bundle), bundle=bundle)
    assert forcing.n_rows == bundle.min_required_forcing_rows
    assert forcing.forecast_steps == 1


def test_result_cache_key_uses_scientific_fingerprint_not_csv_bytes(tmp_path):
    bundle = _bundle(tmp_path)
    csv_a = _valid_csv(24)
    csv_b = "precipitation,stage,time_seconds\n" + "\n".join(
        f"0.2,{1.0 + i*0.01},{i*900}" for i in range(24)
    ) + "\n"
    forcing_a = parse_forcing_csv(csv_a, bundle=bundle, requested_forecast_steps=4)
    forcing_b = parse_forcing_csv(csv_b, bundle=bundle, requested_forecast_steps=4)
    spec_a = RunSpec.new(
        user_id="alice@example.com",
        bundle_id=bundle.bundle_id,
        input_hash=forcing_a.input_hash,
        forecast_steps=forcing_a.forecast_steps,
        label="A",
    )
    spec_b = RunSpec.new(
        user_id="bob@example.com",
        bundle_id=bundle.bundle_id,
        input_hash=forcing_b.input_hash,
        forecast_steps=forcing_b.forecast_steps,
        label="B",
    )

    assert forcing_a.input_hash != forcing_b.input_hash
    assert build_result_cache_key(run_spec=spec_a, forcing_input=forcing_a, bundle=bundle) == (
        build_result_cache_key(run_spec=spec_b, forcing_input=forcing_b, bundle=bundle)
    )

    changed = RunSpec.new(
        user_id="alice@example.com",
        bundle_id=bundle.bundle_id,
        input_hash=forcing_a.input_hash,
        forecast_steps=forcing_a.forecast_steps,
        exceedance_thresholds_m=(0.01, 0.05),
    )
    assert build_result_cache_key(run_spec=changed, forcing_input=forcing_a, bundle=bundle) != (
        build_result_cache_key(run_spec=spec_a, forcing_input=forcing_a, bundle=bundle)
    )


def test_completed_duplicate_materializes_private_run_from_result_cache(tmp_path):
    orchestrator, queue = _cached_orchestrator(tmp_path)
    alice = User(user_id="alice@example.com", email="alice@example.com", disclaimer_acknowledged=True)
    bob = User(user_id="bob@example.com", email="bob@example.com", disclaimer_acknowledged=True)

    first = orchestrator.submit(user=alice, forcing_csv=_valid_csv(24), forecast_steps=4, label="source")
    queue.drain(orchestrator.execute)
    assert orchestrator.repository.get(first.spec.run_id).status == RunStatus.COMPLETED

    second = orchestrator.submit(user=bob, forcing_csv=_valid_csv(24), forecast_steps=4, label="private-copy")

    assert second.status == RunStatus.COMPLETED
    assert second.spec.run_id != first.spec.run_id
    assert second.spec.user_id == "bob@example.com"
    assert len(queue.jobs) == 0
    artifact_ids = {ref.artifact_id for ref in orchestrator.artifact_store.list(second.spec.run_id)}
    assert {"cache_manifest.json", "calibrated_summary.json", "forcing.csv", "run_manifest.json"}.issubset(artifact_ids)
    cache_payload = orchestrator.cache_payload_for_run(second.spec.run_id)
    assert cache_payload["materialized_from_cache"] is True
    assert "producer_run_id" not in cache_payload


def test_in_flight_duplicate_waits_for_source_and_never_enqueues_gpu_twice(tmp_path):
    orchestrator, queue = _cached_orchestrator(tmp_path)
    alice = User(user_id="alice@example.com", email="alice@example.com", disclaimer_acknowledged=True)
    bob = User(user_id="bob@example.com", email="bob@example.com", disclaimer_acknowledged=True)

    source = orchestrator.submit(user=alice, forcing_csv=_valid_csv(24), forecast_steps=4)
    waiter = orchestrator.submit(user=bob, forcing_csv=_valid_csv(24), forecast_steps=4)

    assert source.status == RunStatus.QUEUED
    assert waiter.status == RunStatus.WAITING_FOR_CACHE
    assert [job.run_id for job in queue.jobs] == [source.spec.run_id]

    queue.drain(orchestrator.execute)

    assert orchestrator.repository.get(source.spec.run_id).status == RunStatus.COMPLETED
    assert orchestrator.repository.get(waiter.spec.run_id).status == RunStatus.COMPLETED
    assert len(queue.jobs) == 0
    assert orchestrator.cache_payload_for_run(waiter.spec.run_id)["materialized_from_cache"] is True


def test_waiting_cache_run_is_requeued_when_source_is_canceled(tmp_path):
    orchestrator, queue = _cached_orchestrator(tmp_path)
    alice = User(user_id="alice@example.com", email="alice@example.com", disclaimer_acknowledged=True)
    bob = User(user_id="bob@example.com", email="bob@example.com", disclaimer_acknowledged=True)

    source = orchestrator.submit(user=alice, forcing_csv=_valid_csv(24), forecast_steps=4)
    waiter = orchestrator.submit(user=bob, forcing_csv=_valid_csv(24), forecast_steps=4)

    orchestrator.cancel(source.spec.run_id)

    assert orchestrator.repository.get(source.spec.run_id).status == RunStatus.CANCELED
    assert orchestrator.repository.get(waiter.spec.run_id).status == RunStatus.QUEUED
    assert queue.jobs[-1].run_id == waiter.spec.run_id

    queue.drain(orchestrator.execute)

    assert orchestrator.repository.get(waiter.spec.run_id).status == RunStatus.COMPLETED
    assert orchestrator.cache_payload_for_run(waiter.spec.run_id)["mode"] == "producer"


class _FakeCudaUnavailable:
    @staticmethod
    def is_available():
        return False


class _FakeTorchNoCuda:
    """Stand-in for the ``torch`` module that ``_device()`` resolves through
    ``self._torch()``. Only the surface used by ``_device`` is implemented;
    keeping it minimal makes the contract between the test and production code
    explicit (any drift surfaces as an AttributeError, not as a silent pass).
    """

    cuda = _FakeCudaUnavailable()

    @staticmethod
    def device(name):
        # Mirrors torch.device's "addressable identifier" semantics for the
        # subset of behaviour _device returns. The actual torch.device object
        # is opaque enough that string identity is a safe proxy here.
        return name


def test_production_inference_refuses_cpu_fallback_when_cuda_missing(tmp_path, monkeypatch):
    service = ProductionFGNInferenceService(_bundle(tmp_path), device="cuda:0")
    monkeypatch.setenv("FGN_INFERENCE_MODE", "production")
    monkeypatch.setattr(service, "_torch", lambda: _FakeTorchNoCuda())

    with pytest.raises(RuntimeError, match="refuses to fall back to CPU"):
        service._device()


def test_inference_falls_back_to_cpu_when_not_production(tmp_path, monkeypatch, caplog):
    """Non-production callers (CLI smoke, local dev) must keep the documented
    silent CPU fallback so the worker still functions without a GPU. The
    fail-fast behaviour is gated strictly on ``FGN_INFERENCE_MODE=production``.
    """
    import logging

    service = ProductionFGNInferenceService(_bundle(tmp_path), device="cuda:0")
    monkeypatch.delenv("FGN_INFERENCE_MODE", raising=False)
    monkeypatch.setattr(service, "_torch", lambda: _FakeTorchNoCuda())

    with caplog.at_level(logging.WARNING, logger="neuralop.flood.serving.inference"):
        device = service._device()

    assert device == "cpu"
    assert any("Falling back to CPU" in record.getMessage() for record in caplog.records), (
        "Operators rely on this log line to know the worker silently downgraded; do not remove it."
    )


def test_production_inference_exposes_static_context_columns(tmp_path):
    """Static tensor columns should reach geometry_meta for cell inspection.

    The bundle builder writes static columns as elevation, cell area, slope,
    aspect, flow direction, curvature, flow accumulation. The inspector needs
    elevation/slope/flow accumulation for physical context at the clicked cell.
    """
    pytest.importorskip("torch")

    class _IdentityNormalizer:
        def transform(self, value):
            return value

        def to(self, _device):
            return self

    class _DummyModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    service = ProductionFGNInferenceService(
        _bundle(tmp_path),
        device="cpu",
        preloaded_models=[_DummyModel(), _DummyModel(), _DummyModel()],
        preloaded_normalizers={
            "geometry": _IdentityNormalizer(),
            "static": _IdentityNormalizer(),
            "boundary": _IdentityNormalizer(),
            "dynamic": _IdentityNormalizer(),
            "target": _IdentityNormalizer(),
        },
        preloaded_domain_assets=DomainAssets(
            geometry=np.array([[100.0, 200.0], [110.0, 210.0]], dtype=np.float32),
            static=np.array(
                [
                    [1.0, 100.0, 0.1, 10.0, 1.0, 0.01, 1000.0],
                    [2.0, 200.0, 0.2, 20.0, 2.0, 0.02, 2000.0],
                ],
                dtype=np.float32,
            ),
        ),
    )

    prepared = service._ensure_loaded()

    np.testing.assert_allclose(prepared["elevation_raw_np"], [1.0, 2.0])
    np.testing.assert_allclose(prepared["cell_area_m2_np"], [100.0, 200.0])
    np.testing.assert_allclose(prepared["slope_raw_np"], [0.1, 0.2])
    np.testing.assert_allclose(prepared["flow_accumulation_raw_np"], [1000.0, 2000.0])


def test_celery_worker_honors_preload_models_env(monkeypatch):
    pytest.importorskip("celery")
    from neuralop.flood.serving import celery_app

    calls = []
    sentinel = object()

    def fake_build_orchestrator(*, queue_override, preload_models):
        calls.append({"queue_override": queue_override, "preload_models": preload_models})
        return sentinel

    monkeypatch.setattr(celery_app, "_orchestrator", None)
    monkeypatch.setattr(celery_app, "build_orchestrator", fake_build_orchestrator)
    monkeypatch.setenv("FGN_PRELOAD_MODELS", "1")

    assert celery_app.get_orchestrator() is sentinel
    assert calls == [{"queue_override": None, "preload_models": True}]


def test_celery_worker_can_keep_lazy_loading_when_preload_disabled(monkeypatch):
    pytest.importorskip("celery")
    from neuralop.flood.serving import celery_app

    calls = []
    sentinel = object()

    def fake_build_orchestrator(*, queue_override, preload_models):
        calls.append({"queue_override": queue_override, "preload_models": preload_models})
        return sentinel

    monkeypatch.setattr(celery_app, "_orchestrator", None)
    monkeypatch.setattr(celery_app, "build_orchestrator", fake_build_orchestrator)
    monkeypatch.setenv("FGN_PRELOAD_MODELS", "0")

    assert celery_app.get_orchestrator() is sentinel
    assert calls == [{"queue_override": None, "preload_models": False}]


def test_orchestrator_execute_ignores_run_canceled_before_worker_start(tmp_path):
    orchestrator = _fake_orchestrator(tmp_path)
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)
    record = orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=2)

    class ExplodingInference(FakeFGNInferenceService):
        def run(self, spec, forcing):  # pragma: no cover - should not be called
            raise AssertionError("Canceled runs must not enter inference.")

    orchestrator.inference_service = ExplodingInference(orchestrator.bundle, n_cells=8)
    orchestrator.cancel(record.spec.run_id)

    orchestrator.execute(record.spec.run_id)

    assert orchestrator.repository.get(record.spec.run_id).status == RunStatus.CANCELED


def test_orchestrator_execute_records_progress_label_and_runtime(tmp_path):
    orchestrator = _fake_orchestrator(tmp_path)
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)
    record = orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=4)

    orchestrator.execute(record.spec.run_id)

    done = orchestrator.repository.get(record.spec.run_id)
    assert done.status == RunStatus.COMPLETED
    assert done.progress == 1.0
    assert done.progress_label == "Completed"
    assert done.started_at is not None
    assert done.completed_at is not None
    assert done.runtime_seconds is not None
    timing = json.loads(orchestrator.artifact_store.read_bytes(record.spec.run_id, "performance_timing.json"))
    assert timing["phases"]["rollout"]["seconds"] >= 0.0
    assert timing["phases"]["calibration"]["seconds"] >= 0.0
    assert timing["phases"]["total"]["seconds"] >= 0.0
    assert timing["inference"]["member_chunk_size"] == 1
    assert timing["run"]["forecast_steps"] == 4


def test_orchestrator_delete_run_tombstones_record_and_purges_artifacts(tmp_path):
    """Deleting a terminal run removes artifacts while preserving audit metadata.

    Run metadata is the audit surface for bundle ID, input hash, calibration
    settings, and timestamps, so user-facing delete must never physically
    remove the row.
    """
    bundle = _bundle(tmp_path)
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)
    record = orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=4)
    orchestrator.execute(record.spec.run_id)
    run_id = record.spec.run_id
    assert orchestrator.repository.get(run_id).status == RunStatus.COMPLETED
    assert list(orchestrator.artifact_store.list(run_id))  # has artifacts

    orchestrator.delete_run(run_id, user=user)

    tombstone = orchestrator.repository.get(run_id)
    assert tombstone.status == RunStatus.DELETED
    assert tombstone.spec.bundle_id == record.spec.bundle_id
    assert tombstone.spec.input_hash == record.spec.input_hash
    assert tombstone.spec.calibration_mode == record.spec.calibration_mode
    assert not list(orchestrator.artifact_store.list(run_id))


def test_orchestrator_delete_run_refuses_active_run(tmp_path):
    """Active runs must not be deletable — the worker is still writing files."""
    bundle = _bundle(tmp_path)
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(
            crps_mbm={
                "lead_time_hours": [0.0, 999.0],
                "wet_frequency_edges": [0.0, 1.0],
                "wet_frequency_by_cell": [1.0] * 8,
                "coefficients": [[[0.0, 1.0, 0.5]]],
            }
        ),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)
    record = orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=2)
    # Don't execute → status stays QUEUED → still active.
    with pytest.raises(ValueError, match="Cancel the run first"):
        orchestrator.delete_run(record.spec.run_id, user=user)
    # Run must still exist after the refused delete.
    assert orchestrator.repository.get(record.spec.run_id).status == RunStatus.QUEUED


def test_orchestrator_delete_run_refuses_other_users_run(tmp_path):
    """A non-admin user cannot delete a run owned by someone else."""
    bundle = _bundle(tmp_path)
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["owner@example.com", "intruder@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(
            crps_mbm={
                "lead_time_hours": [0.0, 999.0],
                "wet_frequency_edges": [0.0, 1.0],
                "wet_frequency_by_cell": [1.0] * 8,
                "coefficients": [[[0.0, 1.0, 0.5]]],
            }
        ),
        product_builder=ForecastProductBuilder(),
    )
    owner = User(user_id="owner", email="owner@example.com", disclaimer_acknowledged=True)
    intruder = User(user_id="intruder", email="intruder@example.com", disclaimer_acknowledged=True)
    record = orchestrator.submit(user=owner, forcing_csv=_valid_csv(24), forecast_steps=2)
    orchestrator.execute(record.spec.run_id)
    with pytest.raises(PermissionError):
        orchestrator.delete_run(record.spec.run_id, user=intruder)
    # Record and artifacts are still present.
    assert orchestrator.repository.get(record.spec.run_id).status == RunStatus.COMPLETED
    assert list(orchestrator.artifact_store.list(record.spec.run_id))


def test_orchestrator_delete_runs_batch_returns_per_id_outcomes(tmp_path):
    """Batch delete returns granular outcomes; one bad apple does not stop the rest.

    This is the contract the UI depends on: a multi-select Delete must
    succeed for completed runs even when an active or unknown id is in the
    same payload.
    """
    bundle = _bundle(tmp_path)
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(
            crps_mbm={
                "lead_time_hours": [0.0, 999.0],
                "wet_frequency_edges": [0.0, 1.0],
                "wet_frequency_by_cell": [1.0] * 8,
                "coefficients": [[[0.0, 1.0, 0.5]]],
            }
        ),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)

    done = orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=2)
    orchestrator.execute(done.spec.run_id)
    active = orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=2)
    # active is QUEUED, not yet executed.

    result = orchestrator.delete_runs(
        [done.spec.run_id, active.spec.run_id, "does-not-exist"],
        user=user,
    )

    assert result["deleted"] == [done.spec.run_id]
    skipped_ids = {row["run_id"]: row["reason"] for row in result["skipped"]}
    assert skipped_ids[active.spec.run_id] == "active"
    assert skipped_ids["does-not-exist"] == "not_found"
    # done is tombstoned; active still present.
    assert orchestrator.repository.get(done.spec.run_id).status == RunStatus.DELETED
    assert not list(orchestrator.artifact_store.list(done.spec.run_id))
    assert orchestrator.repository.get(active.spec.run_id).status == RunStatus.QUEUED


def test_serving_renderer_depends_on_eval_render_private_helpers():
    """Contract test: serving's map renderer reaches into private helpers in
    ``neuralop.flood.eval.render``. If any of these are renamed or removed,
    the worker will silently produce broken images at runtime — fail loudly
    here in CI where the breakage is one-line obvious to fix.
    """
    from neuralop.flood.eval import render as eval_render

    expected = (
        "_build_spatial_renderer",
        "_cartographic_context",
        "_plot_spatial_panel",
        "_wd_spatial_vmax",
        "_robust_nonnegative_vmax",
        "_mask_wd_dry_for_overlay",
        "_update_spatial_artist",
    )
    missing = [name for name in expected if not callable(getattr(eval_render, name, None))]
    assert not missing, (
        "Serving's map_rendering.py and products.py call these helpers; "
        f"missing or non-callable in eval/render.py: {missing}"
    )


def test_tri_renderer_recovers_from_rcparams_typeerror(monkeypatch, caplog):
    """Regression guard for Matplotlib RcParams failures in Celery workers.

    Root cause (reproduced standalone in 2026-05 diagnosis): inside long-running
    Celery prefork workers, a validated rcParam like ``image.cmap`` can be
    silently replaced by an ``RcParams`` instance. Matplotlib's
    ``cm._ensure_cmap`` then evaluates ``rcParams["image.cmap"] not in
    _colormaps`` which hashes the corrupted value and raises
    ``TypeError: unhashable type: 'RcParams'`` mid-``tripcolor``. This test
    forces that error on the first call and asserts the renderer recovers
    inside its ``rc_context`` retry without leaking rcParams changes.
    """
    import logging

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from neuralop.flood.eval import render as eval_render

    x = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    arr = np.asarray([0.0, 0.4, 0.7, 1.0], dtype=np.float64)
    renderer = {"mode": "tri", "triangulation": Triangulation(x, y)}
    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    original_tripcolor = ax.tripcolor
    calls = {"n": 0}

    def flaky_tripcolor(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("unhashable type: 'RcParams'")
        return original_tripcolor(*args, **kwargs)

    monkeypatch.setattr(ax, "tripcolor", flaky_tripcolor)
    # Capture a known custom rcParam outside the call to prove the workaround
    # no longer clobbers the worker's rcParams when it recovers.
    monkeypatch.setitem(mpl.rcParams, "figure.dpi", 137.0)
    try:
        with caplog.at_level(logging.WARNING, logger=eval_render.__name__):
            artist = eval_render._plot_spatial_field(
                ax=ax,
                x=x,
                y=y,
                arr=arr,
                renderer=renderer,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
        assert artist is not None
        assert calls["n"] == 2
        # Worker's custom rcParams must survive the retry path.
        assert mpl.rcParams["figure.dpi"] == 137.0
        # And the recovery must be observable in logs.
        assert any(
            "rcParams corruption" in record.getMessage() for record in caplog.records
        ), "expected warning when the RcParams workaround fires"
    finally:
        plt.close(fig)


def test_serving_renderers_isolate_rcparams_with_rc_context():
    """rcParams.update inside serving renderers must not leak globally.

    After 2026-05's diagnosis, all ``rcParams.update`` calls in
    ``serving/products.py``, ``serving/map_rendering.py`` and
    ``serving/figures.py`` are wrapped in ``mpl.rc_context``. This test
    asserts the source-level invariant so the protection cannot regress
    silently.
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root / "neuralop" / "flood" / "serving" / "products.py",
        repo_root / "neuralop" / "flood" / "serving" / "map_rendering.py",
        repo_root / "neuralop" / "flood" / "serving" / "figures.py",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        # rcParams.update must always appear inside a mpl.rc_context block.
        # We use a simple heuristic: every occurrence of "rcParams.update("
        # must be preceded (within the same file, lexically) by a matching
        # "rc_context" block opener that is not closed before it.
        if "rcParams.update(" not in text:
            continue
        for match in re.finditer(r"rcParams\.update\(", text):
            preceding = text[: match.start()]
            # Find the nearest enclosing 'with mpl.rc_context' or
            # 'with rc_context' before this point.
            last_ctx = max(
                preceding.rfind("with mpl.rc_context"),
                preceding.rfind("rc_context("),
            )
            assert last_ctx != -1, (
                f"{path.name}: rcParams.update() at offset {match.start()} "
                f"is not enclosed in mpl.rc_context — this would leak rcParams "
                f"globally and re-introduce the unhashable-RcParams class of bug"
            )


def test_forcing_rejects_bad_timestep(tmp_path):
    bundle = _bundle(tmp_path)
    csv = "time_seconds,stage,precipitation\n0,1,0\n900,1,0\n1700,1,0\n"
    with pytest.raises(ForcingValidationError, match="timestep"):
        parse_forcing_csv(csv, bundle=bundle)


def test_run_state_machine_rejects_invalid_transition():
    repo = InMemoryRunRepository()
    spec = RunSpec.new(user_id="u1", bundle_id="b", input_hash="h", forecast_steps=2)
    repo.create(spec)
    with pytest.raises(RunStateError):
        repo.transition(spec.run_id, RunStatus.COMPLETED)


def test_run_repository_records_live_progress_and_runtime():
    repo = InMemoryRunRepository()
    spec = RunSpec.new(user_id="u1", bundle_id="b", input_hash="h", forecast_steps=2)
    repo.create(spec)
    repo.transition(spec.run_id, RunStatus.VALIDATING)
    repo.transition(spec.run_id, RunStatus.QUEUED)
    running = repo.transition(spec.run_id, RunStatus.RUNNING)

    assert running.started_at is not None
    assert running.progress == pytest.approx(0.40)
    updated = repo.update_progress(spec.run_id, 0.57, label="GPU rollout model 1/3, lead 2/4")
    assert updated.progress == pytest.approx(0.57)
    assert updated.progress_label == "GPU rollout model 1/3, lead 2/4"

    repo.transition(spec.run_id, RunStatus.POSTPROCESSING)
    completed = repo.transition(spec.run_id, RunStatus.COMPLETED)
    assert completed.progress == 1.0
    assert completed.progress_label == "Completed"
    assert completed.completed_at is not None
    assert completed.runtime_seconds is not None


def test_sql_run_repository_persists_live_progress_and_runtime(tmp_path):
    pytest.importorskip("sqlalchemy")
    from neuralop.flood.serving.sql_repository import SqlRunRepository

    repo = SqlRunRepository(f"sqlite:///{tmp_path / 'runs.sqlite'}")
    spec = RunSpec.new(user_id="u1", bundle_id="b", input_hash="h", forecast_steps=2)
    repo.create(spec)
    repo.transition(spec.run_id, RunStatus.VALIDATING)
    repo.transition(spec.run_id, RunStatus.QUEUED)
    repo.transition(spec.run_id, RunStatus.RUNNING)
    repo.update_progress(spec.run_id, 0.63, label="GPU rollout model 2/3, lead 1/2")
    live = repo.get(spec.run_id)

    assert live.progress == pytest.approx(0.63)
    assert live.progress_label == "GPU rollout model 2/3, lead 1/2"
    assert live.started_at is not None

    repo.transition(spec.run_id, RunStatus.POSTPROCESSING)
    done = repo.transition(spec.run_id, RunStatus.COMPLETED)
    assert done.progress == 1.0
    assert done.runtime_seconds is not None


def test_sql_result_cache_repository_records_hits_and_waiters(tmp_path):
    pytest.importorskip("sqlalchemy")
    from neuralop.flood.serving.result_cache import ResultCacheLookupStatus
    from neuralop.flood.serving.sql_result_cache import SqlResultCacheRepository

    repo = SqlResultCacheRepository(f"sqlite:///{tmp_path / 'result_cache.sqlite'}")

    miss = repo.reserve_or_find("cache-a", "producer")
    assert miss.status == ResultCacheLookupStatus.MISS

    waiting = repo.reserve_or_find("cache-a", "waiter")
    assert waiting.status == ResultCacheLookupStatus.WAITING
    assert [link.run_id for link in repo.list_waiting_runs("cache-a")] == ["waiter"]

    ready = repo.publish_ready("producer", ["calibrated_summary.json", "map.png"])
    assert ready is not None
    assert ready.artifact_manifest == ("calibrated_summary.json", "map.png")

    hit = repo.reserve_or_find("cache-a", "third")
    assert hit.status == ResultCacheLookupStatus.HIT
    repo.mark_materialized("third", "cache-a", role=repo.link_for_run("third").role)
    assert repo.link_for_run("third").status.value == "MATERIALIZED"


def test_run_spec_rejects_unsupported_thresholds():
    with pytest.raises(ValueError, match="Unsupported|threshold"):
        RunSpec.new(
            user_id="u1",
            bundle_id="b",
            input_hash="h",
            forecast_steps=2,
            exceedance_thresholds_m=(0.02,),
        )


def test_run_spec_records_requested_member_budget():
    spec = RunSpec.new(
        user_id="u1",
        bundle_id="b",
        input_hash="h",
        forecast_steps=2,
        ensemble_count=2,
        members_per_ensemble=5,
    )

    assert spec.ensemble_count == 2
    assert spec.members_per_ensemble == 5
    assert spec.manifest()["ensemble_count"] == 2
    assert spec.manifest()["members_per_ensemble"] == 5


def test_orchestrator_member_budget_limits_fake_inference_members(tmp_path):
    bundle = _bundle(tmp_path)
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)

    record = orchestrator.submit(
        user=user,
        forcing_csv=_valid_csv(24),
        forecast_steps=3,
        ensemble_count=2,
        members_per_ensemble=5,
    )
    orchestrator.execute(record.spec.run_id)
    summary = json.loads(
        (tmp_path / "artifacts" / record.spec.run_id / "calibrated_summary.json").read_text()
    )

    assert record.spec.ensemble_count == 2
    assert record.spec.members_per_ensemble == 5
    assert summary["n_members"] == 10


def test_orchestrator_rejects_member_budget_above_bundle(tmp_path):
    bundle = _bundle(tmp_path)
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm={"method": "identity"}),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)

    with pytest.raises(ValueError, match="ensemble_count"):
        orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=2, ensemble_count=4)
    with pytest.raises(ValueError, match="members_per_ensemble"):
        orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=2, members_per_ensemble=21)


def test_access_policy_requires_allowed_email_and_disclaimer():
    policy = AccessPolicy(allowed_emails=["a@example.com"], admin_emails=["admin@example.com"])
    with pytest.raises(AccessDenied):
        policy.require_allowed(User(user_id="u", email="b@example.com"))
    with pytest.raises(AccessDenied, match="disclaimer"):
        policy.require_disclaimer(User(user_id="u", email="a@example.com"))
    user = policy.require_disclaimer(User(user_id="u", email="a@example.com", disclaimer_acknowledged=True))
    assert user.email == "a@example.com"


def test_access_policy_open_authenticated_access_auto_registers_non_admin_user():
    policy = AccessPolicy(
        allowed_emails=["existing@example.com"],
        admin_emails=["admin@example.com"],
        open_authenticated_access=True,
    )

    user = policy.require_allowed(User(user_id="new@example.com", email="new@example.com"))

    assert user.email == "new@example.com"
    assert user.is_admin is False
    assert policy.repository.get_user("new@example.com") is not None
    with pytest.raises(AccessDenied, match="disclaimer"):
        policy.require_disclaimer(User(user_id="new@example.com", email="new@example.com"))


def test_access_policy_open_authenticated_access_preserves_disclaimer_acknowledgement():
    policy = AccessPolicy(allowed_emails=[], open_authenticated_access=True)

    user = policy.require_disclaimer(
        User(user_id="new@example.com", email="new@example.com", disclaimer_acknowledged=True)
    )

    assert user.email == "new@example.com"
    assert user.disclaimer_acknowledged is True
    assert policy.repository.get_user("new@example.com").disclaimer_acknowledged is True  # type: ignore[union-attr]


def test_artifact_store_blocks_path_traversal(tmp_path):
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.put_bytes("run/evil", "x.txt", b"x", content_type="text/plain")
    with pytest.raises(ValueError):
        store.put_bytes("run", "../x.txt", b"x", content_type="text/plain")


def test_artifact_store_reports_common_content_types(tmp_path):
    store = LocalArtifactStore(tmp_path)
    store.put_bytes("run", "forcing.csv", b"a,b\n", content_type="text/csv")
    store.put_bytes("run", "map.png", b"x", content_type="image/png")
    store.put_bytes("run", "figure.svg", b"<svg />", content_type="image/svg+xml")
    refs = {ref.artifact_id: ref.content_type for ref in store.list("run")}
    assert refs["forcing.csv"] == "text/csv"
    assert refs["map.png"] == "image/png"
    assert refs["figure.svg"] == "image/svg+xml"


def test_forecast_product_builder_outputs_no_ground_truth_products():
    members = np.array([[[0.0, 0.2], [0.1, 0.29]], [[0.0, 0.4], [0.2, 0.41]]], dtype=np.float32)
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([1.0, 2.0]),
        wettable_mask=np.array([False, True]),
        metadata={"cell_area_m2": np.array([10.0, 20.0])},
    )
    summary = ForecastProductBuilder(thresholds_m=(0.05, 0.3)).build_summary(forecast)
    assert summary["n_members"] == 2
    assert summary["max_mean_wd_m"] == pytest.approx(0.35)
    assert "p_wd_gt_0.3m_mean" in summary
    assert summary["wettable_area_m2"] == pytest.approx(20.0)
    assert summary["peak_expected_flooded_area_km2_gt_0.05m"] == pytest.approx(20.0 / 1_000_000.0)
    assert summary["peak_expected_flooded_area_fraction_wettable_gt_0.05m"] == pytest.approx(1.0)
    assert summary["peak_expected_flooded_area_lead_hours_gt_0.05m"] == pytest.approx(1.0)
    assert summary["peak_area_weighted_iqr_wd_m"] >= 0.0
    assert summary["uncertainty_to_signal_ratio"] >= 0.0
    assert summary["exceedance_by_threshold_m"]["0.3"]["peak_expected_area_fraction_wettable"] == pytest.approx(0.5)


def test_forecast_product_builder_exceedance_area_is_area_weighted():
    members = np.array([[[0.4, 0.0]], [[0.5, 0.0]]], dtype=np.float32)
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([1.0]),
        wettable_mask=np.array([True, True]),
        metadata={"cell_area_m2": np.array([1.0, 9.0])},
    )

    summary = ForecastProductBuilder(thresholds_m=(0.30,)).build_summary(forecast)

    assert summary["p_wd_gt_0.3m_mean"] == pytest.approx(0.5)  # legacy unweighted probability.
    assert summary["exceedance_by_threshold_m"]["0.3"]["peak_expected_area_fraction_wettable"] == pytest.approx(0.1)
    assert summary["exceedance_by_threshold_m"]["0.3"]["peak_expected_area_km2"] == pytest.approx(1.0 / 1_000_000.0)


def test_forecast_product_builder_handles_no_flood_onset():
    forecast = ForecastResult(
        members_wd=np.zeros((2, 3, 4), dtype=np.float32),
        lead_time_hours=np.array([1.0, 2.0, 3.0]),
        wettable_mask=np.ones(4, dtype=bool),
        metadata={"cell_area_m2": np.ones(4)},
    )

    summary = ForecastProductBuilder(thresholds_m=(0.30,)).build_summary(forecast)

    assert summary["peak_expected_flooded_area_fraction_wettable_gt_0.05m"] == pytest.approx(0.0)
    assert summary["onset_lead_hours_expected_flooded_area_fraction_gt_1pct_gt_0.05m"] is None
    assert summary["peak_expected_flooded_area_lead_hours_gt_0.05m"] is None


def test_orchestrator_submit_execute_completes_with_fake_inference(tmp_path):
    bundle = _bundle(tmp_path)
    coeff = {"lead_time_hours": [0.0, 999.0], "wet_frequency_edges": [0.0, 1.0], "wet_frequency_by_cell": [1.0] * 8, "coefficients": [[[0.0, 1.0, 0.5]]]}
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)
    record = orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=4, label="smoke")
    assert record.status == RunStatus.QUEUED
    orchestrator.execute(record.spec.run_id)
    done = orchestrator.repository.get(record.spec.run_id)
    assert done.status == RunStatus.COMPLETED
    artifacts = {a.artifact_id for a in orchestrator.artifact_store.list(record.spec.run_id)}
    assert {
        "forcing.csv",
        "run_manifest.json",
        "raw_summary.json",
        "calibrated_summary.json",
        "comparison_summary.json",
        "forcing_hydrograph.svg",
        "uq_extent_by_time.svg",
        "uq_exceedance_bars.svg",
        "uq_uncertainty_width.svg",
        "calibration_effect.svg",
        "initial_condition_selection.json",
    }.issubset(artifacts)
    initial_selection = json.loads(orchestrator.artifact_store.read_bytes(record.spec.run_id, "initial_condition_selection.json"))
    assert initial_selection["mode"] == "dry"
    manifest = json.loads(orchestrator.artifact_store.read_bytes(record.spec.run_id, "run_manifest.json"))
    assert manifest["initial_condition"]["mode"] == "dry"


def test_quota_policy_rejects_excess_queued_runs(tmp_path):
    from neuralop.flood.serving.quota import QuotaError, QuotaPolicy

    repo = InMemoryRunRepository()
    policy = QuotaPolicy(max_user_queued=2, max_user_running=1)
    for _ in range(2):
        spec = RunSpec.new(user_id="u1", bundle_id="b", input_hash="h", forecast_steps=1)
        repo.create(spec)
        repo.transition(spec.run_id, RunStatus.VALIDATING)
        repo.transition(spec.run_id, RunStatus.QUEUED)
    with pytest.raises(QuotaError, match="queued jobs"):
        policy.validate_submit(repo, "u1")


def test_quota_policy_enforces_one_global_active_run():
    from neuralop.flood.serving.quota import QuotaPolicy

    repo = InMemoryRunRepository()
    first = RunSpec.new(user_id="u1", bundle_id="b", input_hash="h1", forecast_steps=1)
    second = RunSpec.new(user_id="u2", bundle_id="b", input_hash="h2", forecast_steps=1)
    for spec in (first, second):
        repo.create(spec)
        repo.transition(spec.run_id, RunStatus.VALIDATING)
        repo.transition(spec.run_id, RunStatus.QUEUED)
    repo.transition(first.run_id, RunStatus.RUNNING)
    assert QuotaPolicy(max_global_running=1).can_start(repo, second.run_id) is False


def test_trusted_header_auth_builds_allowlisted_user():
    from neuralop.flood.serving.auth import TrustedHeaderAuth

    auth = TrustedHeaderAuth.from_lists(allowed_emails=["user@example.com"], admin_emails=["admin@example.com"])
    user = auth.user_from_headers({"x-auth-request-email": "user@example.com", "x-fgn-disclaimer-accepted": "true"})
    assert user.email == "user@example.com"
    assert user.disclaimer_acknowledged is True


def test_repository_list_all_and_retention_expire_unpinned_terminal_run(tmp_path):
    from dataclasses import replace
    from datetime import datetime, timedelta, timezone

    from neuralop.flood.serving.retention import RetentionManager

    repo = InMemoryRunRepository()
    store = LocalArtifactStore(tmp_path / "artifacts")
    old = datetime.now(timezone.utc) - timedelta(days=45)
    pinned_spec = RunSpec.new(user_id="u1", bundle_id="b", input_hash="h1", forecast_steps=1)
    expire_spec = RunSpec.new(user_id="u1", bundle_id="b", input_hash="h2", forecast_steps=1)
    repo.create(pinned_spec)
    repo.create(expire_spec)
    for spec in (pinned_spec, expire_spec):
        repo.transition(spec.run_id, RunStatus.VALIDATING)
        repo.transition(spec.run_id, RunStatus.QUEUED)
        repo.transition(spec.run_id, RunStatus.RUNNING)
        repo.transition(spec.run_id, RunStatus.POSTPROCESSING)
        repo.transition(spec.run_id, RunStatus.COMPLETED)
        repo._runs[spec.run_id] = replace(repo._runs[spec.run_id], updated_at=old)
        store.put_bytes(spec.run_id, "x.txt", b"x", content_type="text/plain")
    repo.set_pinned(pinned_spec.run_id, True)

    result = RetentionManager(repository=repo, artifact_store=store, retention_days=30).expire_due_runs()

    assert result.expired_run_ids == (expire_spec.run_id,)
    assert repo.get(expire_spec.run_id).status == RunStatus.EXPIRED
    assert repo.get(pinned_spec.run_id).status == RunStatus.COMPLETED
    assert not list(store.list(expire_spec.run_id))
    assert {r.spec.run_id for r in repo.list_all()} == {pinned_spec.run_id, expire_spec.run_id}


def test_orchestrator_expires_runs_through_lifecycle_owner(tmp_path):
    from dataclasses import replace
    from datetime import datetime, timedelta, timezone

    bundle = _bundle(tmp_path)
    repo = InMemoryRunRepository()
    store = LocalArtifactStore(tmp_path / "artifacts")
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=repo,
        queue=InMemoryJobQueue(),
        artifact_store=store,
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm={"method": "identity"}),
        product_builder=ForecastProductBuilder(),
    )
    spec = RunSpec.new(user_id="u1", bundle_id="b", input_hash="h", forecast_steps=1)
    repo.create(spec)
    repo.transition(spec.run_id, RunStatus.VALIDATING)
    repo.transition(spec.run_id, RunStatus.QUEUED)
    repo.transition(spec.run_id, RunStatus.RUNNING)
    repo.transition(spec.run_id, RunStatus.POSTPROCESSING)
    repo.transition(spec.run_id, RunStatus.COMPLETED)
    repo._runs[spec.run_id] = replace(repo._runs[spec.run_id], updated_at=datetime.now(timezone.utc) - timedelta(days=45))
    store.put_bytes(spec.run_id, "x.txt", b"x", content_type="text/plain")
    result = orchestrator.expire_due_runs(retention_days=30)
    assert result.expired_run_ids == (spec.run_id,)
    assert repo.get(spec.run_id).status == RunStatus.EXPIRED


def test_serving_bundle_builder_helpers_hash_arrays_deterministically():
    from neuralop.flood.serving.bundle_builder import _stable_arrays_hash

    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert _stable_arrays_hash([a]) == _stable_arrays_hash([a.copy()])
    assert _stable_arrays_hash([a]) != _stable_arrays_hash([a + 1])


def test_calibration_adapter_applies_isotonic_to_exceedance_probabilities():
    """Isotonic curves shrink the per-cell exceedance probability map."""
    crps_mbm = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [0.5, 0.5, 0.5],
        "coefficients": [[[0.0, 1.0, 1.0]]],
    }
    # Step-function curve: predict(x) = ys[searchsorted(xs, x, side="left")] (clipped).
    isotonic = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "curves": {
            f"{0.30:.6g}": {
                "lead_0_wet_0": {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.05, 0.4]},
            }
        },
    }
    adapter = CalibrationAdapter(crps_mbm=crps_mbm, isotonic=isotonic)
    raw = np.array([0.2, 0.6, 1.0])
    cal = adapter.apply_isotonic_exceedance(
        raw,
        threshold_m=0.30,
        lead_time_hour=0.5,
    )
    assert cal.shape == (3,)
    assert (cal <= raw + 1e-9).all()
    assert cal[0] == pytest.approx(0.05)
    assert cal[1] == pytest.approx(0.4)
    assert cal[2] == pytest.approx(0.4)


def test_calibration_adapter_isotonic_falls_back_when_curves_missing():
    """No isotonic curves means raw probabilities pass through unchanged."""
    adapter = CalibrationAdapter(crps_mbm={"wet_frequency_by_cell": [0.5, 0.5]}, isotonic={})
    raw = np.array([0.1, 0.5])
    cal = adapter.apply_isotonic_exceedance(raw, threshold_m=0.30, lead_time_hour=1.0)
    np.testing.assert_array_equal(cal, raw)
    assert adapter.has_isotonic_curves() is False


def test_calibration_adapter_isotonic_handles_2d_per_time_input():
    """2D input (n_time, n_wettable) is calibrated with per-time lead values."""
    crps_mbm = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [0.5, 0.5],
        "coefficients": [[[0.0, 1.0, 1.0]]],
    }
    isotonic = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "curves": {
            f"{0.30:.6g}": {
                "lead_0_wet_0": {"x": [0.0, 1.0], "y": [0.0, 0.5]},
            }
        },
    }
    adapter = CalibrationAdapter(crps_mbm=crps_mbm, isotonic=isotonic)
    raw_2d = np.array([[0.2, 1.0], [0.6, 0.8]])
    cal = adapter.apply_isotonic_exceedance(
        raw_2d,
        threshold_m=0.30,
        lead_time_hour=np.array([0.5, 1.5]),
    )
    assert cal.shape == (2, 2)
    assert cal[0, 1] == pytest.approx(0.5)


def test_product_builder_build_summary_with_isotonic_adjusts_exceedance():
    """Calibrated build_summary applies isotonic to exceedance probability mean."""
    members = np.tile(
        np.array([0.0, 0.4, 0.8, 1.0]).reshape(1, 1, 4),
        (4, 2, 1),
    ).astype(np.float32)
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([1.0, 2.0]),
        wettable_mask=np.array([True, True, True, True]),
    )
    crps_mbm = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [0.5, 0.5, 0.5, 0.5],
        "coefficients": [[[0.0, 1.0, 1.0]]],
    }
    isotonic = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "curves": {
            f"{0.30:.6g}": {
                "lead_0_wet_0": {"x": [0.0, 1.0], "y": [0.0, 0.5]},
            }
        },
    }
    adapter = CalibrationAdapter(crps_mbm=crps_mbm, isotonic=isotonic)
    builder = ForecastProductBuilder(thresholds_m=(0.30,))
    summary_raw = builder.build_summary(forecast, label="raw")
    summary_cal = builder.build_summary(forecast, label="calibrated", calibration_adapter=adapter)
    assert summary_raw["p_wd_gt_0.3m_mean"] == pytest.approx(0.75)
    assert summary_raw["isotonic_calibration_applied"] is False
    assert summary_cal["p_wd_gt_0.3m_mean"] == pytest.approx(0.375)
    assert summary_cal["isotonic_calibration_applied"] is True


def test_forecast_product_builder_writes_animation_gif(tmp_path):
    members = np.array(
        [[[0.0, 0.2], [0.1, 0.3], [0.2, 0.4]]],
        dtype=np.float32,
    )
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        wettable_mask=np.array([True, True]),
        metadata={"geometry_xy": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)},
    )
    out = ForecastProductBuilder().write_animation_gif(
        forecast,
        output_path=tmp_path / "anim.gif",
        label="calibrated",
        fps=2,
    )
    assert out is not None
    assert out.exists() and out.stat().st_size > 0
    assert out.suffix == ".gif"


def test_orchestrator_submit_with_animation_writes_gif_artifact(tmp_path):
    bundle = _bundle(tmp_path)
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)
    record = orchestrator.submit(
        user=user,
        forcing_csv=_valid_csv(24),
        forecast_steps=4,
        request_animation=True,
    )
    orchestrator.execute(record.spec.run_id)
    artifacts = {a.artifact_id for a in orchestrator.artifact_store.list(record.spec.run_id)}
    assert "calibrated_mean_wd_animation.gif" in artifacts
    assert "calibrated_p_gt_0p30m_animation.gif" in artifacts


def test_empirical_crps_matches_brute_force_pairwise_mean():
    """Order-statistic implementation must agree with the naive O(M^2)
    pairwise mean to float tolerance. The Uncertainty tab depends on this
    metric being correctly normalised so cells compare apples-to-apples."""
    rng = np.random.default_rng(42)
    ensemble = rng.uniform(0, 1, size=(8, 5)).astype(np.float64)
    fast = ForecastProductBuilder.empirical_crps_per_cell(ensemble)
    brute = np.zeros(ensemble.shape[1])
    m = ensemble.shape[0]
    for i in range(m):
        for j in range(m):
            brute += np.abs(ensemble[i] - ensemble[j])
    brute /= (m * m)
    np.testing.assert_allclose(fast, brute.astype(np.float32), rtol=1e-5)


def test_empirical_crps_zero_for_identical_members():
    """All-identical ensemble = zero spread = zero CRPS. Sanity floor."""
    ensemble = np.tile(np.array([0.4, 0.7, 1.0]), (10, 1))
    crps = ForecastProductBuilder.empirical_crps_per_cell(ensemble.astype(np.float32))
    assert np.allclose(crps, 0.0)


def test_spread_decomposition_uses_member_model_id_groups():
    """Between-variance = variance of group means. Within-variance = mean
    of within-group variances. Validate against hand-computed values."""
    ensemble = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [4.0, 1.0],
            [5.0, 2.0],
        ],
        dtype=np.float64,
    )
    member_model_id = ["A", "A", "A", "B", "B", "B"]
    out = ForecastProductBuilder.spread_decomposition_per_cell(ensemble, member_model_id)
    assert out is not None
    between, within, summary = out
    np.testing.assert_allclose(between, [2.25, 0.25], rtol=1e-5)
    np.testing.assert_allclose(within, [1.0 / 3.0, 1.0 / 3.0], rtol=1e-5)
    assert summary["n_groups"] == 2
    assert summary["groups"] == ["A", "B"]
    assert summary["between_share"] + summary["within_share"] == pytest.approx(1.0)


def test_spread_decomposition_returns_none_without_member_model_id():
    """No metadata = no decomposition. Caller skips the artifact rather
    than failing the run."""
    ensemble = np.zeros((4, 3), dtype=np.float64)
    assert ForecastProductBuilder.spread_decomposition_per_cell(ensemble, None) is None
    assert ForecastProductBuilder.spread_decomposition_per_cell(ensemble, ["A"] * 4) is None


def test_build_summary_includes_checkpoint_disagreement_metrics():
    """Run summaries carry checkpoint-vs-latent disagreement metrics so
    monitoring can flag high structural disagreement without reading HDF5."""
    members = np.array(
        [
            [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
            [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
            [[0.0, 0.0, 0.0], [0.7, 0.1, 0.7]],
            [[0.0, 0.0, 0.0], [0.7, 0.1, 0.7]],
        ],
        dtype=np.float64,
    )
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([0.0, 1.0]),
        wettable_mask=np.array([True, True, True]),
        metadata={
            "cell_area_m2": np.array([1.0, 3.0, 2.0]),
            "member_model_id": ["A", "A", "B", "B"],
        },
    )

    summary = ForecastProductBuilder().build_summary(forecast)

    assert summary["checkpoint_disagreement_available"] is True
    assert summary["checkpoint_disagreement_groups"] == ["A", "B"]
    assert summary["peak_area_weighted_between_checkpoint_variance_share"] == pytest.approx(0.5)
    assert summary["peak_high_checkpoint_disagreement_area_fraction_wettable"] == pytest.approx(0.5)
    assert summary["peak_between_checkpoint_disagreement_lead_hours"] == pytest.approx(1.0)
    assert summary["peak_area_weighted_between_checkpoint_spread_wd_m"] == pytest.approx(np.sqrt(0.045))


def test_reliability_curves_payload_shape_without_isotonic():
    """Without an isotonic adapter the calibrated mean per non-empty bin
    equals the bin centre (identity mapping); ``applied`` is False."""
    rng = np.random.default_rng(0)
    members = rng.uniform(0, 1, size=(20, 2, 50)).astype(np.float32)
    payload = ForecastProductBuilder.reliability_curves_payload(
        members=members,
        wettable_mask=np.ones(50, dtype=bool),
        lead_time_hours=np.array([0.5, 1.0]),
        peak_time_idx=1,
        thresholds_m=(0.30,),
        calibration_adapter=None,
    )
    assert payload["applied"] is False
    assert payload["n_wettable_cells"] == 50
    assert "0.3" in payload["curves"]
    curve = payload["curves"]["0.3"]
    assert curve["threshold_m"] == 0.30
    assert len(curve["raw_bin_centers"]) == 20
    assert len(curve["calibrated_means_per_bin"]) == 20
    assert len(curve["raw_distribution_counts"]) == 20
    for raw_x, cal_y in zip(curve["raw_bin_centers"], curve["calibrated_means_per_bin"]):
        if cal_y is not None:
            assert abs(cal_y - raw_x) < 0.05


def test_orchestrator_writes_uncertainty_diagnostics(tmp_path):
    """End-to-end: every Phase-4 uncertainty artifact must land on a normal
    run. Without this contract the Uncertainty tab silently goes empty."""
    bundle = _bundle(tmp_path)
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)
    record = orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=4)
    orchestrator.execute(record.spec.run_id)
    ids = {a.artifact_id for a in orchestrator.artifact_store.list(record.spec.run_id)}
    expected = {
        "empirical_crps_map.npy",
        "spread_decomposition.npz",
        "spread_decomposition_summary.json",
        "reliability_curves.json",
    }
    assert expected.issubset(ids), f"missing P4 artifacts: {expected - ids}"


def test_geometry_meta_payload_matches_forecast_geometry(tmp_path):
    """Geometry sidecar must carry all cells with consistent bounds and a
    wettable flag list of the same length. The web inspector overlays this
    directly on the map for click hit-testing.
    """
    members = np.zeros((2, 2, 3), dtype=np.float32)
    geometry = np.array([[100.0, 200.0], [110.0, 200.0], [110.0, 210.0]], dtype=np.float32)
    wettable = np.array([True, True, False])
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([0.5, 1.0], dtype=np.float32),
        wettable_mask=wettable,
        metadata={"geometry_xy": geometry},
    )
    meta = ForecastProductBuilder().build_geometry_meta(forecast)
    assert meta is not None
    assert meta["n_cells"] == 3
    assert meta["bounds"] == {"x_min": 100.0, "x_max": 110.0, "y_min": 200.0, "y_max": 210.0}
    assert meta["x"] == [100.0, 110.0, 110.0]
    assert meta["y"] == [200.0, 200.0, 210.0]
    assert meta["wettable"] == [True, True, False]


def test_geometry_meta_includes_static_context_and_pick_viewport(tmp_path):
    """The click overlay needs both cell context and the rendered map viewport.

    The PNG includes a title and colorbar, so frontend hit-testing cannot use
    the full image rectangle as the data plane. ``image_data_viewport`` is the
    server-provided bridge from image pixels to UTM coordinates.
    """
    members = np.zeros((2, 2, 3), dtype=np.float32)
    geometry = np.array([[100.0, 200.0], [110.0, 200.0], [110.0, 210.0]], dtype=np.float32)
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([0.5, 1.0], dtype=np.float32),
        wettable_mask=np.array([True, True, True]),
        metadata={
            "geometry_xy": geometry,
            "elevation_raw": np.array([1.25, 2.5, 3.75], dtype=np.float32),
            "slope_raw": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "flow_accumulation_raw": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        },
    )

    meta = ForecastProductBuilder().build_geometry_meta(forecast)

    assert meta is not None
    assert meta["elevation_m"] == [1.25, 2.5, 3.75]
    assert meta["slope"] == [0.1, 0.2, 0.3]
    assert meta["flow_accumulation"] == [10.0, 20.0, 30.0]
    viewport = meta["image_data_viewport"]
    assert 0.0 <= viewport["left"] < viewport["right"] <= 1.0
    assert 0.0 <= viewport["top"] < viewport["bottom"] <= 1.0
    assert viewport["left"] > 0.0 or viewport["right"] < 1.0
    assert viewport["top"] > 0.0 or viewport["bottom"] < 1.0
    data_bounds = meta["image_data_bounds"]
    assert data_bounds["x_min"] <= min(meta["x"]) <= max(meta["x"]) <= data_bounds["x_max"]
    assert data_bounds["y_min"] <= min(meta["y"]) <= max(meta["y"]) <= data_bounds["y_max"]


def test_geometry_meta_returns_none_without_geometry(tmp_path):
    """No geometry_xy in metadata = no sidecar (and no Phase 2 inspector).
    Caller treats None as "skip artifact" rather than failing the run.
    """
    forecast = ForecastResult(
        members_wd=np.zeros((1, 1, 2), dtype=np.float32),
        lead_time_hours=np.array([0.5], dtype=np.float32),
        wettable_mask=np.array([True, True]),
    )
    assert ForecastProductBuilder().build_geometry_meta(forecast) is None


def test_summarize_cell_timeseries_round_trips_via_hdf5(tmp_path):
    """End-to-end: HDF5 written by the orchestrator must be slice-readable by
    the inspector helper, returning the same per-cell values that come out of
    the in-memory forecast. This is the contract the API route depends on.
    """
    import io

    pytest.importorskip("h5py")
    import h5py

    members = np.array(
        [
            # member 0
            [[0.00, 0.00, 0.00], [0.10, 0.05, 0.00], [0.40, 0.20, 0.02], [0.20, 0.10, 0.01]],
            # member 1
            [[0.00, 0.00, 0.00], [0.30, 0.15, 0.00], [0.60, 0.30, 0.05], [0.40, 0.20, 0.02]],
        ],
        dtype=np.float32,
    )
    # Shape: [n_members=2, n_time=4, n_cells=3]
    lead = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    wettable = np.array([True, True, False])

    h5_path = tmp_path / "forecast_members.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("raw_members_wd", data=members)
        h5.create_dataset("calibrated_members_wd", data=members)
        h5.create_dataset("lead_time_hours", data=lead)
        h5.create_dataset("wettable_mask", data=wettable.astype(np.uint8))

    h5_bytes = h5_path.read_bytes()
    payload = ForecastProductBuilder.summarize_cell_timeseries(
        h5_bytes=h5_bytes,
        cell_index=0,
        thresholds_m=(0.05, 0.30),
    )
    assert payload["cell_index"] == 0
    assert payload["n_members"] == 2
    assert payload["n_time"] == 4
    assert payload["wettable"] is True
    assert payload["lead_time_hours"] == [0.5, 1.0, 1.5, 2.0]
    # 2 members × 4 timesteps
    assert len(payload["calibrated_members_wd"]) == 2
    assert len(payload["calibrated_members_wd"][0]) == 4
    # Peak depth at cell 0 is the mean across members at t=2 (lead=1.5h):
    # mean([0.40, 0.60]) = 0.50; peak lead = 1.5 h.
    assert payload["peak_calibrated_mean_wd_m"] == pytest.approx(0.50, abs=1e-5)
    assert payload["peak_calibrated_lead_hours"] == pytest.approx(1.5, abs=1e-5)
    # P(WD > 0.05) at t=1 (lead=1.0h) for cell 0: members are [0.10, 0.30],
    # both above 0.05 → P = 1.0
    assert payload["calibrated_exceedance_prob"]["0.05"][1] == pytest.approx(1.0)
    # Threshold key format matches f"{thr:g}" — 0.30 collapses to "0.3".
    # Documenting this here keeps the wire-format contract visible.
    assert set(payload["calibrated_exceedance_prob"].keys()) == {"0.05", "0.3"}
    # P(WD > 0.3) at t=1 for cell 0: members are [0.10, 0.30]. Strict > 0.3
    # is 0 because both members are <= 0.30 (one is exactly 0.30, but with
    # the strict inequality used in the helper this is 0/2 = 0).
    assert payload["calibrated_exceedance_prob"]["0.3"][1] == pytest.approx(0.0)
    # Per-member arrival at 0.05: both members first cross at t=1 (lead=1.0h).
    assert payload["calibrated_member_arrival_hours"]["0.05"] == [1.0, 1.0]


def test_summarize_cell_timeseries_applies_isotonic_probability_for_cell(tmp_path):
    """Cell inspector probability traces must match calibrated exceedance products.

    CRPS-MBM calibrates the member depths, but exceedance probabilities have a
    second isotonic calibration layer. The inspector's P(WD > threshold) trace
    must apply that layer too; otherwise a clicked cell disagrees with the maps
    and summary JSON for the same run.
    """
    pytest.importorskip("h5py")
    import h5py

    members = np.array(
        [
            [[0.4], [0.4]],
            [[0.4], [0.4]],
            [[0.0], [0.0]],
            [[0.0], [0.0]],
        ],
        dtype=np.float32,
    )
    h5_path = tmp_path / "forecast_members.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("raw_members_wd", data=members)
        h5.create_dataset("calibrated_members_wd", data=members)
        h5.create_dataset("lead_time_hours", data=np.array([0.5, 1.0], dtype=np.float32))
        h5.create_dataset("wettable_mask", data=np.array([True], dtype=np.uint8))

    adapter = CalibrationAdapter(
        crps_mbm={
            "lead_time_hours": [0.0, 999.0],
            "wet_frequency_edges": [0.0, 1.0],
            "wet_frequency_by_cell": [0.5],
            "coefficients": [[[0.0, 1.0, 1.0]]],
        },
        isotonic={
            "lead_time_hours": [0.0, 999.0],
            "wet_frequency_edges": [0.0, 1.0],
            "curves": {
                f"{0.30:.6g}": {
                    "lead_0_wet_0": {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.25, 0.75]},
                }
            },
        },
    )

    payload = ForecastProductBuilder.summarize_cell_timeseries(
        h5_bytes=h5_path.read_bytes(),
        cell_index=0,
        thresholds_m=(0.30,),
        calibration_adapter=adapter,
    )

    # Raw frequency is 2/4 = 0.5 at both times. The isotonic curve maps 0.5
    # to 0.25, proving the inspector did not merely return raw member counts.
    assert payload["calibrated_exceedance_prob"]["0.3"] == pytest.approx([0.25, 0.25])


def test_summarize_cell_timeseries_marks_dry_cell(tmp_path):
    """Wettable=False must be preserved through the HDF5 round-trip."""
    pytest.importorskip("h5py")
    import h5py

    members = np.zeros((2, 3, 4), dtype=np.float32)
    h5_path = tmp_path / "forecast_members.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("raw_members_wd", data=members)
        h5.create_dataset("calibrated_members_wd", data=members)
        h5.create_dataset("lead_time_hours", data=np.array([0.5, 1.0, 1.5], dtype=np.float32))
        h5.create_dataset("wettable_mask", data=np.array([True, True, True, False], dtype=np.uint8))

    payload = ForecastProductBuilder.summarize_cell_timeseries(
        h5_bytes=h5_path.read_bytes(),
        cell_index=3,
    )
    assert payload["wettable"] is False


def test_summarize_cell_timeseries_rejects_out_of_range(tmp_path):
    """Defence-in-depth: the API route validates this too, but the helper
    must refuse on its own so direct callers (tests, scripts) can't smuggle
    a bad index past h5py's silent slice clamp."""
    pytest.importorskip("h5py")
    import h5py

    members = np.zeros((1, 1, 2), dtype=np.float32)
    h5_path = tmp_path / "forecast_members.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("raw_members_wd", data=members)
        h5.create_dataset("calibrated_members_wd", data=members)
        h5.create_dataset("lead_time_hours", data=np.array([0.5], dtype=np.float32))
        h5.create_dataset("wettable_mask", data=np.array([True, True], dtype=np.uint8))

    with pytest.raises(IndexError, match="out of range"):
        ForecastProductBuilder.summarize_cell_timeseries(
            h5_bytes=h5_path.read_bytes(), cell_index=2
        )
    with pytest.raises(IndexError, match="out of range"):
        ForecastProductBuilder.summarize_cell_timeseries(
            h5_bytes=h5_path.read_bytes(), cell_index=-1
        )


def test_spread_to_peak_histogram_handles_no_signal_cells(tmp_path):
    forecast = ForecastResult(
        members_wd=np.zeros((4, 3, 5), dtype=np.float32),
        lead_time_hours=np.array([0.5, 1.0, 1.5], dtype=np.float32),
        wettable_mask=np.array([True, True, True, True, False]),
    )
    written = ForecastProductBuilder().write_spread_to_peak_histogram(forecast, output_dir=tmp_path)
    assert {"spread_to_peak_histogram.json", "spread_to_peak_histogram.svg"} == set(written)
    payload = json.loads(written["spread_to_peak_histogram.json"].read_text())
    assert payload["n_wettable_cells"] == 4
    assert payload["n_signal_cells"] == 0
    assert payload["median_ratio"] is None
    assert sum(payload["counts"]) == 0


def test_cell_contribution_leaderboard_is_area_weighted(tmp_path):
    members = np.array(
        [
            [[0.0, 0.0], [0.4, 0.4]],
            [[0.0, 0.0], [0.0, 0.4]],
        ],
        dtype=np.float32,
    )
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([0.5, 1.0], dtype=np.float32),
        wettable_mask=np.array([True, True]),
        metadata={
            "cell_area_m2": np.array([100.0, 10.0], dtype=np.float32),
            "geometry_xy": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        },
    )
    leaderboard = ForecastProductBuilder().build_cell_contribution_leaderboard(
        forecast,
        threshold_m=0.30,
        top_n=2,
    )
    assert leaderboard["rows"][0]["cell_index"] == 0
    assert leaderboard["rows"][0]["peak_probability"] == pytest.approx(0.5)
    assert leaderboard["rows"][0]["peak_expected_area_m2"] == pytest.approx(50.0)
    assert leaderboard["rows"][1]["peak_expected_area_m2"] == pytest.approx(10.0)


def test_orchestrator_writes_geometry_meta_and_hdf5_by_default(tmp_path):
    """Per the Phase 2 plan, submitting a run without overriding flags must
    produce both ``geometry_meta.json`` (for the click overlay) and
    ``forecast_members.h5`` (for the per-cell endpoint). Either missing
    breaks the inspector silently."""
    pytest.importorskip("h5py")
    bundle = _bundle(tmp_path)
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)
    # No explicit request_full_hdf5 → relies on the new default.
    record = orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=4)
    assert record.spec.request_full_hdf5 is True
    orchestrator.execute(record.spec.run_id)
    ids = {a.artifact_id for a in orchestrator.artifact_store.list(record.spec.run_id)}
    assert "geometry_meta.json" in ids
    assert "forecast_members.h5" in ids


def test_envelope_maps_have_expected_files_and_shape(tmp_path):
    """Envelope maps must produce exactly four file kinds with [n_cells] shape.

    The Hazard tab discovers these by filename and reads them as float32 .npy
    arrays. Any drift in the contract here silently empties the small-multiples.
    """
    members = np.array(
        [
            [[0.00, 0.10, 0.40, 0.20], [0.00, 0.05, 0.20, 0.10], [0.00, 0.02, 0.05, 0.02]],
            [[0.00, 0.20, 0.60, 0.30], [0.00, 0.10, 0.30, 0.15], [0.00, 0.03, 0.08, 0.03]],
        ],
        dtype=np.float32,
    )
    # members has shape [n_time=3, n_cells=4]? No: above is [n_members=2, n_time=3, n_cells=4]
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([0.5, 1.0, 1.5], dtype=np.float32),
        wettable_mask=np.array([True, True, True, False]),
    )
    written = ForecastProductBuilder().write_envelope_maps(forecast, output_dir=tmp_path)
    expected = {
        "peak_depth_map.npy",
        "quantile_envelope_at_peak.npy",
        "arrival_time_map_gt_0p05m.npy",
        "duration_map_gt_0p05m.npy",
        "arrival_time_map_gt_0p3m.npy",
        "duration_map_gt_0p3m.npy",
    }
    assert set(written) == expected
    for name in expected:
        arr = np.load(written[name])
        assert arr.shape == (4,), f"{name} shape={arr.shape}"
        assert arr.dtype == np.float32, f"{name} dtype={arr.dtype}"
        # Structurally-dry cell (index 3) must be NaN in every map.
        assert np.isnan(arr[3]), f"{name} expected NaN at structurally-dry cell"


def test_envelope_maps_are_nonneg_and_consistent_with_summary(tmp_path):
    """Sanity: peak_depth_map.max() must equal summary['max_mean_wd_m'] and
    duration must be a non-negative multiple of dt; arrival NaNs only when the
    cell never crosses threshold.
    """
    members = np.array(
        [
            [[0.0, 0.1, 0.5, 0.2], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.3, 0.7, 0.4], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    # cells: 0 stays dry, 1 wets then dries, 2 deeply floods, 3 mildly floods.
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        wettable_mask=np.ones(2, dtype=bool),
    )
    # members[:, :, c=0,1,2,3] — wait, dims mismatch. Rebuild properly.
    # Shape needed: [n_members=2, n_time=4, n_cells=2]
    members = np.array(
        [
            [[0.00, 0.00], [0.10, 0.00], [0.50, 0.20], [0.20, 0.05]],
            [[0.00, 0.00], [0.30, 0.00], [0.70, 0.40], [0.40, 0.15]],
        ],
        dtype=np.float32,
    )
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        wettable_mask=np.ones(2, dtype=bool),
    )
    builder = ForecastProductBuilder()
    written = builder.write_envelope_maps(forecast, output_dir=tmp_path)
    peak = np.load(written["peak_depth_map.npy"])
    arrival_05 = np.load(written["arrival_time_map_gt_0p05m.npy"])
    duration_05 = np.load(written["duration_map_gt_0p05m.npy"])
    summary = builder.build_summary(forecast, label="raw")
    # max_mean_wd_m is the spatio-temporal max of the ensemble mean.
    assert float(np.nanmax(peak)) == pytest.approx(summary["max_mean_wd_m"], rel=1e-5)
    # Cell 0 wets first at t=1.0 h (mean = 0.20 > 0.05), so arrival is 1.0 h.
    assert arrival_05[0] == pytest.approx(1.0, abs=1e-5)
    # Cell 1 wets first at t=1.5 h (mean = 0.30 > 0.05), so arrival is 1.5 h.
    assert arrival_05[1] == pytest.approx(1.5, abs=1e-5)
    # Cell 0 above 0.05 m at t ∈ {1.0, 1.5, 2.0} → 3 timesteps × dt=0.5 h = 1.5 h.
    assert duration_05[0] == pytest.approx(1.5, abs=1e-5)
    # Both durations must be non-negative.
    assert np.all(duration_05[~np.isnan(duration_05)] >= 0)


def test_envelope_maps_arrival_is_nan_when_never_wets(tmp_path):
    """A cell whose ensemble-mean depth never exceeds the threshold must get
    NaN for arrival; duration must be zero (not NaN, because the wet-ness
    integral is well-defined as zero)."""
    members = np.zeros((2, 3, 2), dtype=np.float32)
    members[:, :, 0] = 0.01  # below 0.05 m threshold everywhere
    members[:, 2, 1] = 0.20  # cell 1 wets only at t=3
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([0.5, 1.0, 1.5], dtype=np.float32),
        wettable_mask=np.ones(2, dtype=bool),
    )
    written = ForecastProductBuilder().write_envelope_maps(forecast, output_dir=tmp_path)
    arrival_05 = np.load(written["arrival_time_map_gt_0p05m.npy"])
    duration_05 = np.load(written["duration_map_gt_0p05m.npy"])
    assert np.isnan(arrival_05[0])
    assert duration_05[0] == pytest.approx(0.0)
    assert arrival_05[1] == pytest.approx(1.5, abs=1e-5)


def test_orchestrator_writes_envelope_maps_to_artifact_store(tmp_path):
    """Orchestrator.execute() must persist the envelope maps as run artifacts.

    Without this contract the Hazard tab can't discover them.
    """
    bundle = _bundle(tmp_path)
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)
    record = orchestrator.submit(user=user, forcing_csv=_valid_csv(24), forecast_steps=4)
    orchestrator.execute(record.spec.run_id)
    ids = {a.artifact_id for a in orchestrator.artifact_store.list(record.spec.run_id)}
    expected_envelope = {
        "peak_depth_map.npy",
        "quantile_envelope_at_peak.npy",
        "arrival_time_map_gt_0p05m.npy",
        "duration_map_gt_0p05m.npy",
        "arrival_time_map_gt_0p3m.npy",
        "duration_map_gt_0p3m.npy",
    }
    assert expected_envelope.issubset(ids), f"missing: {expected_envelope - ids}"


def test_forecast_product_builder_writes_scrub_frames(tmp_path):
    """Scrub frames must be one-per-timestep with deterministic naming.

    The web Time Player relies on the ``{label}_{product}_scrub_t{NNN}.png``
    naming contract to discover available frames from the artifact list. The
    default product is ``mean`` (preserves Phase-1/2 backwards compatibility).
    """
    members = np.array(
        [[[0.0, 0.2, 0.4], [0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.5, 0.7]]],
        dtype=np.float32,
    )
    geometry = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        wettable_mask=np.ones(3, dtype=bool),
        metadata={"geometry_xy": geometry},
    )
    paths = ForecastProductBuilder().write_scrub_frames(forecast, output_dir=tmp_path, label="calibrated")
    assert len(paths) == 4
    names = [p.name for p in paths]
    assert names == [
        "calibrated_mean_scrub_t001.png",
        "calibrated_mean_scrub_t002.png",
        "calibrated_mean_scrub_t003.png",
        "calibrated_mean_scrub_t004.png",
    ]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_scrub_frames_support_spread_and_exceedance_products(tmp_path):
    """Phase 3 contract: the Time Player toggles between three product streams.

    Every supported product must produce per-timestep frames with the
    ``{label}_{product}_scrub_t{NNN}.png`` naming pattern. Adding a product
    here without updating the frontend matcher leaves orphaned files; the
    inverse leaves an empty toggle. Either is a P3 regression.
    """
    members = np.array(
        [
            [[0.00, 0.10, 0.40], [0.10, 0.30, 0.60], [0.20, 0.40, 0.80]],
            [[0.00, 0.20, 0.50], [0.30, 0.50, 0.70], [0.40, 0.60, 0.90]],
        ],
        dtype=np.float32,
    )
    geometry = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([0.5, 1.0, 1.5], dtype=np.float32),
        wettable_mask=np.ones(3, dtype=bool),
        metadata={"geometry_xy": geometry},
    )
    builder = ForecastProductBuilder()
    for product, expected_prefix in [
        ("mean", "calibrated_mean_scrub_t"),
        ("spread", "calibrated_spread_scrub_t"),
        ("p95", "calibrated_p95_scrub_t"),
        ("iqr", "calibrated_iqr_scrub_t"),
        ("p_gt_0p30m", "calibrated_p_gt_0p30m_scrub_t"),
    ]:
        paths = builder.write_scrub_frames(
            forecast, output_dir=tmp_path / product, product=product
        )
        assert len(paths) == 3, f"{product}: expected 3 frames, got {len(paths)}"
        for i, path in enumerate(paths):
            assert path.name == f"{expected_prefix}{i + 1:03d}.png", path.name
            assert path.stat().st_size > 0


def test_scrub_frames_reject_unknown_product(tmp_path):
    """Unknown product = explicit ValueError, not silent empty output."""
    forecast = ForecastResult(
        members_wd=np.zeros((1, 1, 2), dtype=np.float32),
        lead_time_hours=np.array([0.5], dtype=np.float32),
        wettable_mask=np.ones(2, dtype=bool),
        metadata={"geometry_xy": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="Unsupported scrub product"):
        ForecastProductBuilder().write_scrub_frames(
            forecast, output_dir=tmp_path, product="foo"
        )


def test_orchestrator_emits_all_scrub_product_streams(tmp_path):
    """End-to-end: requesting an animation must populate scrub frames for
    every product in SCRUB_PRODUCTS. The Forecast-maps slider and Time
    Player's toggle both depend on this multi-product artifact set being
    complete; missing one silently shrinks the toggle."""
    bundle = _bundle(tmp_path)
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)
    record = orchestrator.submit(
        user=user,
        forcing_csv=_valid_csv(24),
        forecast_steps=3,
        request_animation=True,
    )
    orchestrator.execute(record.spec.run_id)
    ids = {a.artifact_id for a in orchestrator.artifact_store.list(record.spec.run_id)}
    for product in ("mean", "spread", "p95", "iqr", "p_gt_0p30m"):
        prefix = f"calibrated_{product}_scrub_t"
        matches = sorted(name for name in ids if name.startswith(prefix) and name.endswith(".png"))
        assert len(matches) == 3, f"{product}: expected 3 scrub frames, got {len(matches)} ({matches})"


def test_orchestrator_submit_with_animation_writes_scrub_frames(tmp_path):
    """When animation is requested, scrub frames must land alongside the GIF."""
    bundle = _bundle(tmp_path)
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    orchestrator = RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        access_policy=AccessPolicy(allowed_emails=["user@example.com"]),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
    )
    user = User(user_id="u1", email="user@example.com", disclaimer_acknowledged=True)
    record = orchestrator.submit(
        user=user,
        forcing_csv=_valid_csv(24),
        forecast_steps=4,
        request_animation=True,
    )
    orchestrator.execute(record.spec.run_id)
    artifacts = {a.artifact_id for a in orchestrator.artifact_store.list(record.spec.run_id)}
    expected_scrub = {f"calibrated_mean_scrub_t{i:03d}.png" for i in range(1, 5)}
    assert expected_scrub.issubset(artifacts), f"missing scrub frames: {expected_scrub - artifacts}"


def test_forecast_product_builder_writes_map_pngs(tmp_path):
    members = np.array([[[0.0, 0.2, 0.4, 0.6]], [[0.1, 0.3, 0.5, 0.7]]], dtype=np.float32)
    geometry = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([1.0], dtype=np.float32),
        wettable_mask=np.ones(4, dtype=bool),
        metadata={
            "geometry_xy": geometry,
            "elevation_raw": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        },
    )
    paths = ForecastProductBuilder().write_map_pngs(forecast, output_dir=tmp_path, label="calibrated", max_times=1)
    assert len(paths) >= 4
    assert {path.name for path in paths}.issuperset(
        {
            "calibrated_p_gt_0p30m_t001.png",
            "calibrated_iqr_t001.png",
            "calibrated_p95_t001.png",
            "calibrated_mean_t001.png",
            "calibrated_spread_t001.png",
        }
    )
    assert all(path.exists() and path.stat().st_size > 0 and path.suffix == ".png" for path in paths)
    basemap_metadata = tmp_path / "cartographic_context" / "basemap_metadata.json"
    assert basemap_metadata.exists()
    assert json.loads(basemap_metadata.read_text())["mode"] == "dem_elevation"
