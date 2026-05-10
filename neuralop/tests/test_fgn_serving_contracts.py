from pathlib import Path

import numpy as np
import pytest

from neuralop.flood.serving.access import AccessDenied, AccessPolicy, User
from neuralop.flood.serving.calibration import CalibrationAdapter
from neuralop.flood.serving.forcing import ForcingValidationError, parse_forcing_csv
from neuralop.flood.serving.forcing import build_forcing_template_csv
from neuralop.flood.serving.inference import FakeFGNInferenceService
from neuralop.flood.serving.model_bundle import FGNModelBundle, ModelBundleError
from neuralop.flood.serving.orchestrator import RunOrchestrator
from neuralop.flood.serving.products import ForecastProductBuilder, ForecastResult
from neuralop.flood.serving.queue import InMemoryJobQueue
from neuralop.flood.serving.repository import InMemoryRunRepository
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
        dt_seconds=1200,
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
        lines.append(f"{i*1200},{1.0 + i*0.01},{0.2}")
    return "\n".join(lines) + "\n"


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


def test_forcing_rejects_bad_timestep(tmp_path):
    bundle = _bundle(tmp_path)
    csv = "time_seconds,stage,precipitation\n0,1,0\n900,1,0\n1200,1,0\n"
    with pytest.raises(ForcingValidationError, match="timestep"):
        parse_forcing_csv(csv, bundle=bundle)


def test_run_state_machine_rejects_invalid_transition():
    repo = InMemoryRunRepository()
    spec = RunSpec.new(user_id="u1", bundle_id="b", input_hash="h", forecast_steps=2)
    repo.create(spec)
    with pytest.raises(RunStateError):
        repo.transition(spec.run_id, RunStatus.COMPLETED)


def test_run_spec_rejects_unsupported_thresholds():
    with pytest.raises(ValueError, match="Unsupported|threshold"):
        RunSpec.new(
            user_id="u1",
            bundle_id="b",
            input_hash="h",
            forecast_steps=2,
            exceedance_thresholds_m=(0.02,),
        )


def test_access_policy_requires_allowed_email_and_disclaimer():
    policy = AccessPolicy(allowed_emails=["a@example.com"], admin_emails=["admin@example.com"])
    with pytest.raises(AccessDenied):
        policy.require_allowed(User(user_id="u", email="b@example.com"))
    with pytest.raises(AccessDenied, match="disclaimer"):
        policy.require_disclaimer(User(user_id="u", email="a@example.com"))
    user = policy.require_disclaimer(User(user_id="u", email="a@example.com", disclaimer_acknowledged=True))
    assert user.email == "a@example.com"


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
    refs = {ref.artifact_id: ref.content_type for ref in store.list("run")}
    assert refs["forcing.csv"] == "text/csv"
    assert refs["map.png"] == "image/png"


def test_forecast_product_builder_outputs_no_ground_truth_products():
    members = np.array([[[0.0, 0.2], [0.1, 0.3]], [[0.0, 0.4], [0.2, 0.4]]], dtype=np.float32)
    forecast = ForecastResult(members_wd=members, lead_time_hours=np.array([1.0, 2.0]), wettable_mask=np.array([False, True]))
    summary = ForecastProductBuilder(thresholds_m=(0.05, 0.3)).build_summary(forecast)
    assert summary["n_members"] == 2
    assert summary["max_mean_wd_m"] == pytest.approx(0.35)
    assert "p_wd_gt_0.3m_mean" in summary


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
    assert {"forcing.csv", "run_manifest.json", "raw_summary.json", "calibrated_summary.json", "comparison_summary.json"}.issubset(artifacts)


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


def test_forecast_product_builder_writes_map_pngs(tmp_path):
    members = np.array([[[0.0, 0.2, 0.4]], [[0.1, 0.3, 0.5]]], dtype=np.float32)
    geometry = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    forecast = ForecastResult(
        members_wd=members,
        lead_time_hours=np.array([1.0], dtype=np.float32),
        wettable_mask=np.ones(3, dtype=bool),
        metadata={"geometry_xy": geometry},
    )
    paths = ForecastProductBuilder().write_map_pngs(forecast, output_dir=tmp_path, label="calibrated", max_times=1)
    assert len(paths) == 4
    assert all(path.exists() and path.stat().st_size > 0 and path.suffix == ".png" for path in paths)
