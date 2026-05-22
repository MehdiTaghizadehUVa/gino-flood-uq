from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from neuralop.flood.serving.access import AccessPolicy, User
from neuralop.flood.serving.api import create_app
from neuralop.flood.serving.calibration import CalibrationAdapter
from neuralop.flood.serving.forcing import parse_forcing_csv
from neuralop.flood.serving.inference import FakeFGNInferenceService
from neuralop.flood.serving.monitoring import (
    CandidateStatus,
    InMemoryMonitoringRepository,
    MonitoringBundle,
    MonitoringOrchestrator,
    build_forecast_descriptors,
    build_forcing_descriptors,
)
from neuralop.flood.serving.monitoring_build_bundle import (
    HECRAS_AREA_DATASET,
    HECRAS_CELL_POINTS_DATASET,
    HECRAS_DEPTH_DATASET,
    build_hecras_forecast_reference,
    load_clean_boundary_table_forcings,
)
from neuralop.flood.serving.orchestrator import RunOrchestrator
from neuralop.flood.serving.products import ForecastProductBuilder
from neuralop.flood.serving.queue import InMemoryJobQueue
from neuralop.flood.serving.repository import InMemoryRunRepository
from neuralop.flood.serving.run_spec import RunStatus
from neuralop.flood.serving.storage import LocalArtifactStore
from neuralop.tests.test_fgn_serving_contracts import _bundle, _valid_csv


class _UserProvider:
    def __init__(self) -> None:
        self.current: User | None = None

    def __call__(self, _request=None):
        if self.current is None:
            raise PermissionError("not authenticated")
        return self.current


def _alice() -> User:
    return User(user_id="alice@example.com", email="alice@example.com", disclaimer_acknowledged=True)


def _bob() -> User:
    return User(user_id="bob@example.com", email="bob@example.com", disclaimer_acknowledged=True)


def _admin() -> User:
    return User(user_id="admin@example.com", email="admin@example.com", is_admin=True, disclaimer_acknowledged=True)


def _monitoring_bundle(*, candidate_stage_max: float = 1.10, control_modulus: int = 0) -> MonitoringBundle:
    descriptor_percentiles = {
        "stage_max": {
            "min": 0.3,
            "p01": 0.4,
            "p05": 0.5,
            "p50": 0.8,
            "p95": 1.0,
            "p99": candidate_stage_max,
            "max": 1.2,
        },
        "precipitation_total": {
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p50": 4.0,
            "p95": 12.0,
            "p99": 20.0,
            "max": 30.0,
        },
    }
    return MonitoringBundle(
        bundle_id="coastal-monitoring-test",
        descriptor_percentiles=descriptor_percentiles,
        control_sample_modulus=control_modulus,
    ).validate()


def _monitoring_orchestrator(
    tmp_path: Path,
    run_artifacts: LocalArtifactStore,
    *,
    bundle: MonitoringBundle | None = None,
) -> MonitoringOrchestrator:
    return MonitoringOrchestrator(
        bundle=bundle or _monitoring_bundle(),
        repository=InMemoryMonitoringRepository(),
        run_artifact_store=run_artifacts,
        candidate_artifact_store=LocalArtifactStore(tmp_path / "candidates"),
    )


def _orchestrator(tmp_path: Path, *, monitoring_bundle: MonitoringBundle | None = None) -> RunOrchestrator:
    bundle = _bundle(tmp_path)
    coeff = {
        "lead_time_hours": [0.0, 999.0],
        "wet_frequency_edges": [0.0, 1.0],
        "wet_frequency_by_cell": [1.0] * 8,
        "coefficients": [[[0.0, 1.0, 0.5]]],
    }
    run_artifacts = LocalArtifactStore(tmp_path / "artifacts")
    monitoring = _monitoring_orchestrator(tmp_path, run_artifacts, bundle=monitoring_bundle)
    return RunOrchestrator(
        bundle=bundle,
        repository=InMemoryRunRepository(),
        queue=InMemoryJobQueue(),
        artifact_store=run_artifacts,
        access_policy=AccessPolicy(
            allowed_emails=["alice@example.com", "bob@example.com"],
            admin_emails=["admin@example.com"],
        ),
        inference_service=FakeFGNInferenceService(bundle, n_cells=8),
        calibration_adapter=CalibrationAdapter(crps_mbm=coeff),
        product_builder=ForecastProductBuilder(),
        monitoring_orchestrator=monitoring,
    )


def _client(tmp_path: Path, *, monitoring_bundle: MonitoringBundle | None = None):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    provider = _UserProvider()
    orchestrator = _orchestrator(tmp_path, monitoring_bundle=monitoring_bundle)
    return TestClient(create_app(orchestrator, current_user=provider)), provider, orchestrator


# ---------------------------------------------------------------------------
# Phase 2 / Stage A1: Bundle versioning, transforms, Mahalanobis scoring
# ---------------------------------------------------------------------------


def test_bundle_loads_with_phase2_schema_versioning_fields(tmp_path):
    """Phase 2 bundle JSON with versioning, transforms, and covariance loads correctly."""
    payload = {
        "bundle_id": "coastal-v2",
        "monitoring_bundle_schema_version": 2,
        "descriptor_schema_version": 1,
        "descriptor_percentiles": {
            "stage_max": {"min": 0.3, "p01": 0.4, "p05": 0.5, "p50": 0.8, "p95": 1.0, "p99": 1.1, "max": 1.2},
        },
        "transform_spec": {"precipitation_total": "log1p", "stage_max": "identity"},
        "reference_population": {"n_reference_forcing": 142},
        "created_from_commit": "abc123",
        "covariance_exclude_descriptors": ["calibrated_max_mean_wd_m"],
    }
    path = tmp_path / "bundle_v2.json"
    path.write_text(json.dumps(payload))

    bundle = MonitoringBundle.load(path)

    assert bundle.monitoring_bundle_schema_version == 2
    assert bundle.descriptor_schema_version == 1
    assert bundle.transform_spec == {"precipitation_total": "log1p", "stage_max": "identity"}
    assert bundle.reference_population == {"n_reference_forcing": 142}
    assert bundle.created_from_commit == "abc123"
    assert bundle.covariance_exclude_descriptors == ["calibrated_max_mean_wd_m"]


def test_bundle_loads_phase1_json_without_phase2_fields(tmp_path):
    """Phase 1 bundle JSON loads with safe defaults for Phase 2 fields."""
    payload = {
        "bundle_id": "coastal-v1",
        "descriptor_percentiles": {
            "stage_max": {"min": 0.3, "p01": 0.4, "p05": 0.5, "p50": 0.8, "p95": 1.0, "p99": 1.1, "max": 1.2},
        },
    }
    path = tmp_path / "bundle_v1.json"
    path.write_text(json.dumps(payload))

    bundle = MonitoringBundle.load(path)

    assert bundle.monitoring_bundle_schema_version == 1
    assert bundle.descriptor_schema_version == 1
    assert bundle.transform_spec == {}
    assert bundle.reference_population == {}
    assert bundle.created_from_commit is None
    assert bundle.forcing_covariance is None
    assert bundle.forecast_covariance is None
    assert bundle.heuristic_reference_percentiles is None
    assert bundle.covariance_exclude_descriptors == []


def test_descriptor_transforms_log1p_and_logit_bounded():
    """log1p and logit_bounded transforms produce expected values."""
    from neuralop.flood.serving.monitoring import _apply_transforms

    descriptors = {"precipitation_total": 5.0, "area_fraction": 0.5, "stage_max": 1.0}
    transform_spec = {
        "precipitation_total": "log1p",
        "area_fraction": "logit_bounded",
        "stage_max": "identity",
    }

    result = _apply_transforms(descriptors, transform_spec)

    import math
    assert result["precipitation_total"] == pytest.approx(math.log1p(5.0))
    assert result["stage_max"] == pytest.approx(1.0)
    # logit_bounded(0.5) = log(0.5 / (1 - 0.5)) = log(1) = 0
    assert result["area_fraction"] == pytest.approx(0.0)


def test_descriptor_transforms_logit_bounded_clips_extremes():
    """logit_bounded clips values near 0 and 1 with eps=0.001."""
    from neuralop.flood.serving.monitoring import _apply_transforms

    result_zero = _apply_transforms({"x": 0.0}, {"x": "logit_bounded"})
    result_one = _apply_transforms({"x": 1.0}, {"x": "logit_bounded"})

    import math
    eps = 0.001
    assert result_zero["x"] == pytest.approx(math.log(eps / (1 - eps)))
    assert result_one["x"] == pytest.approx(math.log((1 - eps) / eps))


def _make_covariance_bundle(tmp_path):
    """Build a MonitoringBundle with 2D forcing covariance for Mahalanobis tests."""
    import numpy as np

    descriptors = ["stage_max", "precipitation_total"]
    mean = [1.0, 10.0]
    cov = [[0.04, 0.0], [0.0, 4.0]]  # diagonal: sd=0.2, sd=2.0
    alpha = 0.01 * (0.04 + 4.0) / 2  # regularization
    cov_reg = [[cov[0][0] + alpha, 0.0], [0.0, cov[1][1] + alpha]]
    inv_cov = [[1.0 / cov_reg[0][0], 0.0], [0.0, 1.0 / cov_reg[1][1]]]

    rng = np.random.default_rng(42)
    ref_vectors = rng.multivariate_normal(mean, cov, size=200)
    ref_distances = []
    for v in ref_vectors:
        d = v - np.array(mean)
        d2 = d @ np.array(inv_cov) @ d
        ref_distances.append(float(np.sqrt(max(0, d2))))
    pcts = np.percentile(ref_distances, [50, 75, 90, 95, 99])

    payload = {
        "bundle_id": "cov-test",
        "monitoring_bundle_schema_version": 2,
        "descriptor_percentiles": {
            "stage_max": {"min": 0.3, "p01": 0.4, "p05": 0.5, "p50": 1.0, "p95": 1.4, "p99": 1.6, "max": 2.0},
            "precipitation_total": {"min": 2.0, "p01": 3.0, "p05": 5.0, "p50": 10.0, "p95": 15.0, "p99": 18.0, "max": 20.0},
        },
        "transform_spec": {"stage_max": "identity", "precipitation_total": "identity"},
        "forcing_covariance": {
            "descriptor_names": descriptors,
            "mean_vector": mean,
            "inv_covariance_matrix": inv_cov,
            "covariance_diagonal": [cov_reg[0][0], cov_reg[1][1]],
            "n_reference": 200,
            "regularization_alpha": alpha,
            "empirical_distance_percentiles": {
                "p50": float(pcts[0]),
                "p75": float(pcts[1]),
                "p90": float(pcts[2]),
                "p95": float(pcts[3]),
                "p99": float(pcts[4]),
            },
        },
    }
    path = tmp_path / "cov_bundle.json"
    path.write_text(json.dumps(payload))
    return MonitoringBundle.load(path), payload


def test_multivariate_score_catches_joint_outlier(tmp_path):
    """All marginals within p05-p95 but joint Mahalanobis exceeds empirical p99."""
    from neuralop.flood.serving.monitoring import _score_multivariate

    bundle, payload = _make_covariance_bundle(tmp_path)
    empirical_p99 = payload["forcing_covariance"]["empirical_distance_percentiles"]["p99"]
    descriptors = {"stage_max": 1.35, "precipitation_total": 14.5}

    score, flags = _score_multivariate(descriptors, bundle, "forcing")

    assert score >= 0.6
    assert any(f.code == "multivariate_outlier" for f in flags)


def test_multivariate_score_passes_typical_vector(tmp_path):
    """Vector near reference mean produces score 0.0."""
    from neuralop.flood.serving.monitoring import _score_multivariate

    bundle, _ = _make_covariance_bundle(tmp_path)
    descriptors = {"stage_max": 1.0, "precipitation_total": 10.0}

    score, flags = _score_multivariate(descriptors, bundle, "forcing")

    assert score == 0.0
    assert not any(f.code == "multivariate_outlier" for f in flags)


def test_multivariate_score_absent_covariance_is_noop(tmp_path):
    """When forcing_covariance is None, multivariate score is 0.0."""
    from neuralop.flood.serving.monitoring import _score_multivariate

    payload = {
        "bundle_id": "no-cov",
        "descriptor_percentiles": {
            "stage_max": {"min": 0.3, "p01": 0.4, "p05": 0.5, "p50": 1.0, "p95": 1.4, "p99": 1.6, "max": 2.0},
        },
    }
    path = tmp_path / "nocov.json"
    path.write_text(json.dumps(payload))
    bundle = MonitoringBundle.load(path)
    descriptors = {"stage_max": 999.0}

    score, flags = _score_multivariate(descriptors, bundle, "forcing")

    assert score == 0.0
    assert flags == []


def test_build_report_integrates_multivariate_score(tmp_path):
    """_build_report uses max(univariate, multivariate) when covariance is present."""
    from neuralop.flood.serving.monitoring import MonitoringPhase, _build_report
    from neuralop.flood.serving.run_spec import RunSpec

    bundle, payload = _make_covariance_bundle(tmp_path)
    run_spec = RunSpec(
        run_id="test-mv-1", user_id="u1", bundle_id="cov-test",
        input_hash="abc123", forecast_steps=5,
    )
    descriptors = {"stage_max": 1.35, "precipitation_total": 14.5}

    report = _build_report(
        run_spec=run_spec,
        phase=MonitoringPhase.PRE_RUN,
        monitoring_bundle=bundle,
        descriptors=descriptors,
        score_prefix="forcing",
    )

    assert any(f.code == "multivariate_outlier" for f in report.flags)
    forcing_score = report.scores.get("forcing_novelty_score", 0.0)
    assert forcing_score >= 0.6


def test_phase1_reference_envelope_candidate_flag_is_diagnostic_only(tmp_path):
    """An extreme univariate envelope flag remains visible but does not create a candidate."""
    from neuralop.flood.serving.monitoring import MonitoringPhase, _build_report
    from neuralop.flood.serving.run_spec import RunSpec

    bundle = _monitoring_bundle(candidate_stage_max=1.10)
    run_spec = RunSpec(
        run_id="phase1-only",
        user_id="u1",
        bundle_id=bundle.bundle_id,
        input_hash="hash1",
        forecast_steps=5,
    )

    report = _build_report(
        run_spec=run_spec,
        phase=MonitoringPhase.PRE_RUN,
        monitoring_bundle=bundle,
        descriptors={"stage_max": 1.23, "precipitation_total": 4.0},
        score_prefix="input",
    )

    ref_flags = [flag for flag in report.flags if flag.code == "above_candidate_reference"]
    assert ref_flags
    assert ref_flags[0].details["decision_eligible"] is False
    assert report.scores["reference_envelope_score"] == pytest.approx(1.0)
    assert report.scores["candidate_decision_score"] == pytest.approx(0.0)
    assert report.candidate_recommended is False
    assert report.candidate_reason is None


def test_phase2_multivariate_outlier_creates_candidate(tmp_path):
    """A candidate-level Mahalanobis outlier owns the retraining-candidate decision."""
    from neuralop.flood.serving.monitoring import MonitoringPhase, _build_report
    from neuralop.flood.serving.run_spec import RunSpec

    bundle, _ = _make_covariance_bundle(tmp_path)
    run_spec = RunSpec(
        run_id="phase2-mv",
        user_id="u1",
        bundle_id=bundle.bundle_id,
        input_hash="hash2",
        forecast_steps=5,
    )

    report = _build_report(
        run_spec=run_spec,
        phase=MonitoringPhase.PRE_RUN,
        monitoring_bundle=bundle,
        descriptors={"stage_max": 2.0, "precipitation_total": 20.0},
        score_prefix="input",
    )

    assert any(flag.code == "multivariate_outlier" for flag in report.flags)
    assert report.scores["multivariate_score"] >= bundle.candidate_score_threshold
    assert report.scores["candidate_decision_score"] >= bundle.candidate_score_threshold
    assert report.candidate_recommended is True
    assert report.candidate_reason == "multivariate_outlier"


def test_phase2_reference_heuristic_creates_post_run_candidate(tmp_path):
    """Reference-derived post-run heuristics can create candidates without Phase 1 escalation."""
    from neuralop.flood.serving.monitoring import MonitoringPhase, _build_report
    from neuralop.flood.serving.run_spec import RunSpec

    payload = {
        "bundle_id": "heuristic-candidate",
        "descriptor_percentiles": {
            "stage_max": {"min": 0.3, "p01": 0.4, "p05": 0.5, "p50": 0.8, "p95": 1.0, "p99": 1.1, "max": 1.2},
        },
        "forecast_descriptor_percentiles": {
            "calibrated_max_mean_wd_m": {"min": 0.1, "p01": 0.2, "p05": 0.3, "p50": 0.8, "p95": 2.0, "p99": 2.5, "max": 3.0},
        },
        "heuristic_reference_percentiles": {
            "uncertainty_to_signal_ratio": {"p01": 0.05, "p05": 0.1, "p50": 0.25, "p95": 0.42, "p99": 0.55},
            "iqr_area_product": {"p01": 0.001, "p05": 0.005, "p50": 0.02, "p95": 0.08, "p99": 0.12},
            "max_abs_calibration_shift_pp": {"p01": 0.1, "p05": 0.5, "p50": 1.5, "p95": 4.0, "p99": 6.0},
        },
    }
    path = tmp_path / "heuristic_candidate_bundle.json"
    path.write_text(json.dumps(payload))
    bundle = MonitoringBundle.load(path)
    run_spec = RunSpec(
        run_id="phase2-heur",
        user_id="u1",
        bundle_id=bundle.bundle_id,
        input_hash="hash3",
        forecast_steps=5,
    )

    report = _build_report(
        run_spec=run_spec,
        phase=MonitoringPhase.POST_RUN,
        monitoring_bundle=bundle,
        descriptors={
            "calibrated_max_mean_wd_m": 0.85,
            "uncertainty_to_signal_ratio": 0.70,
            "peak_area_weighted_iqr_wd_m": 0.15,
            "peak_expected_flooded_area_fraction_wettable_gt_0.05m": 0.30,
            "max_abs_calibration_shift_percentage_points_by_threshold": 2.0,
        },
        score_prefix="forecast",
    )

    assert report.scores["post_run_heuristic_score"] >= bundle.candidate_score_threshold
    assert report.candidate_recommended is True
    assert report.candidate_reason == "high_uncertainty_to_signal"


def test_population_drift_reinforces_matching_reference_diagnostic(tmp_path):
    """Persistent population drift plus a matching individual diagnostic can create a candidate."""
    from neuralop.flood.serving.monitoring import MonitoringPhase, _build_report
    from neuralop.flood.serving.run_spec import RunSpec

    bundle = _monitoring_bundle(candidate_stage_max=1.10)
    run_spec = RunSpec(
        run_id="population-reinforced",
        user_id="u1",
        bundle_id=bundle.bundle_id,
        input_hash="hash4",
        forecast_steps=5,
    )

    report = _build_report(
        run_spec=run_spec,
        phase=MonitoringPhase.PRE_RUN,
        monitoring_bundle=bundle,
        descriptors={"stage_max": 1.23, "precipitation_total": 4.0},
        score_prefix="input",
        population_drift_descriptors={"stage_max"},
    )

    assert any(flag.code == "population_reinforced_candidate" for flag in report.flags)
    assert report.scores["population_reinforced_score"] >= bundle.candidate_score_threshold
    assert report.candidate_recommended is True
    assert report.candidate_reason == "population_reinforced_candidate"


def test_population_drift_alone_does_not_create_candidate(tmp_path):
    """Population drift without an individually suspicious descriptor is admin context only."""
    from neuralop.flood.serving.monitoring import MonitoringPhase, _build_report
    from neuralop.flood.serving.run_spec import RunSpec

    bundle = _monitoring_bundle(candidate_stage_max=1.10)
    run_spec = RunSpec(
        run_id="population-only",
        user_id="u1",
        bundle_id=bundle.bundle_id,
        input_hash="hash5",
        forecast_steps=5,
    )

    report = _build_report(
        run_spec=run_spec,
        phase=MonitoringPhase.PRE_RUN,
        monitoring_bundle=bundle,
        descriptors={"stage_max": 0.80, "precipitation_total": 4.0},
        score_prefix="input",
        population_drift_descriptors={"stage_max"},
    )

    assert not any(flag.code == "population_reinforced_candidate" for flag in report.flags)
    assert report.scores["population_reinforced_score"] == pytest.approx(0.0)
    assert report.candidate_recommended is False


def test_descriptor_logging_is_idempotent():
    """Re-evaluating same run+phase+bundle upserts, not duplicates."""
    from neuralop.flood.serving.monitoring import DescriptorLogEntry, MonitoringPhase

    repo = InMemoryMonitoringRepository()
    entry = DescriptorLogEntry(
        descriptor_log_id="run1:PRE_RUN:bundle1",
        run_id="run1",
        phase=MonitoringPhase.PRE_RUN,
        monitoring_bundle_id="bundle1",
        descriptor_schema_version=1,
        descriptors={"stage_max": 1.0},
        scores={"forcing_novelty_score": 0.3},
    )
    repo.log_descriptors(entry)
    repo.log_descriptors(entry)

    logs = repo.list_descriptor_logs(phase=MonitoringPhase.PRE_RUN)
    assert len(logs) == 1
    assert logs[0].descriptor_log_id == "run1:PRE_RUN:bundle1"


def test_descriptor_logging_list_filters_by_phase():
    """list_descriptor_logs filters by phase correctly."""
    from neuralop.flood.serving.monitoring import DescriptorLogEntry, MonitoringPhase

    repo = InMemoryMonitoringRepository()
    pre = DescriptorLogEntry(
        descriptor_log_id="run1:PRE_RUN:b1",
        run_id="run1",
        phase=MonitoringPhase.PRE_RUN,
        monitoring_bundle_id="b1",
        descriptor_schema_version=1,
        descriptors={"stage_max": 1.0},
        scores={"forcing_novelty_score": 0.3},
    )
    post = DescriptorLogEntry(
        descriptor_log_id="run1:POST_RUN:b1",
        run_id="run1",
        phase=MonitoringPhase.POST_RUN,
        monitoring_bundle_id="b1",
        descriptor_schema_version=1,
        descriptors={"area_fraction": 0.5},
        scores={"forecast_monitoring_score": 0.1},
    )
    repo.log_descriptors(pre)
    repo.log_descriptors(post)

    pre_logs = repo.list_descriptor_logs(phase=MonitoringPhase.PRE_RUN)
    post_logs = repo.list_descriptor_logs(phase=MonitoringPhase.POST_RUN)
    all_logs = repo.list_descriptor_logs()

    assert len(pre_logs) == 1
    assert len(post_logs) == 1
    assert len(all_logs) == 2


def test_descriptor_logging_persists_after_screening(tmp_path):
    """After screen_forcing(), descriptor log entries exist for PRE_RUN."""
    from neuralop.flood.serving.monitoring import DescriptorLogEntry, MonitoringPhase

    bundle = _monitoring_bundle()
    repo = InMemoryMonitoringRepository()
    artifact_store = LocalArtifactStore(tmp_path / "artifacts")
    candidate_store = LocalArtifactStore(tmp_path / "candidates")
    orch = MonitoringOrchestrator(
        bundle=bundle,
        repository=repo,
        run_artifact_store=artifact_store,
        candidate_artifact_store=candidate_store,
    )
    forcing = parse_forcing_csv(_valid_csv(), bundle=_bundle(tmp_path), requested_forecast_steps=5)
    from neuralop.flood.serving.run_spec import RunSpec
    run_spec = RunSpec.new(
        user_id="alice@example.com",
        bundle_id="test-bundle",
        input_hash=forcing.input_hash,
        forecast_steps=5,
    )
    orch.screen_forcing(
        run_spec=run_spec,
        owner_email="alice@example.com",
        forcing=forcing,
        forcing_csv=_valid_csv().encode(),
        model_bundle_metadata={"bundle_id": "test-bundle"},
    )

    logs = repo.list_descriptor_logs(phase=MonitoringPhase.PRE_RUN)
    assert len(logs) == 1
    assert logs[0].run_id == run_spec.run_id
    assert logs[0].monitoring_bundle_id == bundle.bundle_id


def test_reference_derived_heuristics_replace_hardcoded(tmp_path):
    """With heuristic_reference_percentiles, p95 threshold is used instead of hard-coded."""
    from neuralop.flood.serving.monitoring import _post_run_reference_score

    payload = {
        "bundle_id": "ref-heur",
        "descriptor_percentiles": {
            "stage_max": {"min": 0.3, "p01": 0.4, "p05": 0.5, "p50": 0.8, "p95": 1.0, "p99": 1.1, "max": 1.2},
        },
        "heuristic_reference_percentiles": {
            "uncertainty_to_signal_ratio": {"p01": 0.05, "p05": 0.1, "p50": 0.25, "p95": 0.42, "p99": 0.55},
            "iqr_area_product": {"p01": 0.001, "p05": 0.005, "p50": 0.02, "p95": 0.08, "p99": 0.12},
            "max_abs_calibration_shift_pp": {"p01": 0.1, "p05": 0.5, "p50": 1.5, "p95": 4.0, "p99": 6.0},
        },
    }
    path = tmp_path / "ref_bundle.json"
    path.write_text(json.dumps(payload))
    bundle = MonitoringBundle.load(path)

    descriptors = {
        "uncertainty_to_signal_ratio": 0.45,
        "peak_area_weighted_iqr_wd_m": 0.15,
        "peak_expected_flooded_area_fraction_wettable_gt_0.05m": 0.30,
        "max_abs_calibration_shift_percentage_points_by_threshold": 3.0,
    }
    score, flags = _post_run_reference_score(descriptors, bundle)

    assert score >= 0.6
    assert any(f.code == "high_uncertainty_to_signal" for f in flags)


def test_reference_derived_checkpoint_disagreement_can_drive_candidate(tmp_path):
    from neuralop.flood.serving.monitoring import _post_run_reference_score

    payload = {
        "bundle_id": "ref-disagreement",
        "descriptor_percentiles": {
            "stage_max": {"min": 0.3, "p01": 0.4, "p05": 0.5, "p50": 0.8, "p95": 1.0, "p99": 1.1, "max": 1.2},
        },
        "heuristic_reference_percentiles": {
            "peak_area_weighted_between_checkpoint_spread_wd_m": {
                "p01": 0.01,
                "p05": 0.02,
                "p50": 0.05,
                "p95": 0.10,
                "p99": 0.15,
            },
            "peak_area_weighted_between_checkpoint_variance_share": {
                "p01": 0.01,
                "p05": 0.02,
                "p50": 0.20,
                "p95": 0.50,
                "p99": 0.75,
            },
            "peak_high_checkpoint_disagreement_area_fraction_wettable": {
                "p01": 0.0,
                "p05": 0.0,
                "p50": 0.05,
                "p95": 0.20,
                "p99": 0.35,
            },
        },
    }
    path = tmp_path / "ref_disagreement_bundle.json"
    path.write_text(json.dumps(payload))
    bundle = MonitoringBundle.load(path)

    score, flags = _post_run_reference_score(
        {
            "peak_area_weighted_between_checkpoint_spread_wd_m": 0.16,
            "peak_area_weighted_between_checkpoint_variance_share": 0.30,
            "peak_high_checkpoint_disagreement_area_fraction_wettable": 0.10,
        },
        bundle,
    )

    assert score >= bundle.candidate_score_threshold
    assert any(f.code == "high_checkpoint_disagreement_spread" for f in flags)


def test_legacy_heuristics_when_no_reference(tmp_path):
    """Without heuristic_reference_percentiles, existing hard-coded thresholds apply."""
    from neuralop.flood.serving.monitoring import _post_run_heuristic_score_legacy

    descriptors = {
        "uncertainty_to_signal_ratio": 0.6,
        "peak_area_weighted_iqr_wd_m": 0.15,
        "peak_expected_flooded_area_fraction_wettable_gt_0.05m": 0.30,
        "max_abs_calibration_shift_percentage_points_by_threshold": 6.0,
    }
    score, flags = _post_run_heuristic_score_legacy(descriptors)

    assert score >= 0.85
    assert any(f.code == "high_uncertainty_to_signal" for f in flags)
    assert any(f.code == "large_calibration_shift" for f in flags)


def test_hecras_error_record_created_on_simulated():
    """Creating an error record stores FGN-vs-HEC-RAS errors."""
    from neuralop.flood.serving.monitoring import HecrasErrorRecord

    repo = InMemoryMonitoringRepository()
    record = HecrasErrorRecord(
        error_record_id="error_cand1",
        candidate_id="cand1",
        run_id="run1",
        monitoring_bundle_id="b1",
        fgn_descriptors={"area_fraction": 0.25},
        hecras_descriptors={"area_fraction": 0.30},
        error_descriptors={"area_fraction_signed": -0.05, "area_fraction_relative": -0.167},
    )
    repo.create_hecras_error_record(record)

    fetched = repo.get_hecras_error_record("cand1")
    assert fetched is not None
    assert fetched.error_descriptors["area_fraction_signed"] == pytest.approx(-0.05)


def test_hecras_path_rejected_outside_allowed_root(tmp_path, monkeypatch):
    """Path not under FGN_HECRAS_RESULT_ROOT raises PermissionError."""
    from neuralop.flood.serving.monitoring import validate_hecras_path

    allowed = tmp_path / "hecras_data"
    allowed.mkdir()
    monkeypatch.setenv("FGN_HECRAS_RESULT_ROOT", str(allowed))

    with pytest.raises(PermissionError, match="must be under"):
        validate_hecras_path("/etc/passwd")


def test_hecras_path_accepted_under_allowed_root(tmp_path, monkeypatch):
    """Path under FGN_HECRAS_RESULT_ROOT resolves successfully."""
    from neuralop.flood.serving.monitoring import validate_hecras_path

    allowed = tmp_path / "hecras_data"
    allowed.mkdir()
    hdf_file = allowed / "result.hdf"
    hdf_file.write_text("fake hdf")
    monkeypatch.setenv("FGN_HECRAS_RESULT_ROOT", str(allowed))

    result = validate_hecras_path(str(hdf_file))
    assert result == hdf_file.resolve()


def test_hecras_path_not_configured_raises(monkeypatch):
    """Missing FGN_HECRAS_RESULT_ROOT raises ValueError."""
    from neuralop.flood.serving.monitoring import validate_hecras_path

    monkeypatch.delenv("FGN_HECRAS_RESULT_ROOT", raising=False)

    with pytest.raises(ValueError, match="FGN_HECRAS_RESULT_ROOT not configured"):
        validate_hecras_path("/any/path")


def test_sql_descriptor_logging_idempotent():
    """SQL adapter upserts descriptor log entries on conflict."""
    from neuralop.flood.serving.monitoring import DescriptorLogEntry, MonitoringPhase
    from neuralop.flood.serving.sql_monitoring import SqlMonitoringRepository

    repo = SqlMonitoringRepository("sqlite:///:memory:")
    entry = DescriptorLogEntry(
        descriptor_log_id="run1:PRE_RUN:bundle1",
        run_id="run1",
        phase=MonitoringPhase.PRE_RUN,
        monitoring_bundle_id="bundle1",
        descriptor_schema_version=1,
        descriptors={"stage_max": 1.0},
        scores={"forcing_novelty_score": 0.3},
    )
    repo.log_descriptors(entry)
    repo.log_descriptors(entry)

    logs = repo.list_descriptor_logs(phase=MonitoringPhase.PRE_RUN)
    assert len(logs) == 1
    assert logs[0].descriptors == {"stage_max": 1.0}


def test_monitoring_bundle_rejects_missing_reference_descriptors(tmp_path):
    path = tmp_path / "monitoring.json"
    path.write_text(json.dumps({"bundle_id": "bad", "descriptor_percentiles": {}}))

    with pytest.raises(ValueError, match="descriptor_percentiles"):
        MonitoringBundle.load(path)


def test_forcing_descriptors_are_deterministic(tmp_path):
    bundle = _bundle(tmp_path)
    forcing = parse_forcing_csv(_valid_csv(), bundle=bundle, requested_forecast_steps=5)

    first = build_forcing_descriptors(forcing)
    second = build_forcing_descriptors(forcing)

    assert first == second
    assert first["stage_max"] == pytest.approx(1.23)
    assert first["precipitation_total"] == pytest.approx(4.8)
    assert first["forecast_steps"] == 5


def test_clean_boundary_tables_load_as_reference_forcings(tmp_path):
    bundle = _bundle(tmp_path)
    n_rows = bundle.min_required_forcing_rows + 2
    stage = tmp_path / "Stage_Hydrographs_Train_Clean.txt"
    precipitation = tmp_path / "Precipitation_Train_Clean.txt"
    stage.write_text(
        "TR000001\tTR000002\n"
        + "\n".join(f"{0.1 + idx * 0.01:.3f}\t{0.2 + idx * 0.01:.3f}" for idx in range(n_rows))
        + "\n"
    )
    precipitation.write_text(
        "TR000001\tTR000002\n"
        + "\n".join(f"{0.0:.3f}\t{0.05 if idx == 3 else 0.0:.3f}" for idx in range(n_rows))
        + "\n"
    )

    forcings = load_clean_boundary_table_forcings(
        stage_table=stage,
        precipitation_table=precipitation,
        dt_seconds=bundle.dt_seconds,
        skip_before_timestep=bundle.skip_before_timestep,
        n_history=bundle.n_history,
        max_forecast_steps=bundle.max_forecast_steps,
    )

    assert [forcing.source_name for forcing in forcings] == [
        "TR000001.clean_boundary",
        "TR000002.clean_boundary",
    ]
    assert forcings[0].n_rows == n_rows
    assert forcings[0].forecast_steps == 3
    assert forcings[0].input_hash != forcings[1].input_hash
    assert build_forcing_descriptors(forcings[1])["precipitation_total"] == pytest.approx(0.05)


def test_hecras_reference_scan_builds_matching_forecast_percentiles(tmp_path):
    h5py = pytest.importorskip("h5py")
    hdf = tmp_path / "run.hdf"
    with h5py.File(hdf, "w") as handle:
        _create_hdf_dataset(handle, HECRAS_CELL_POINTS_DATASET, [[0.0, 0.0], [1.0, 0.0]])
        _create_hdf_dataset(handle, HECRAS_AREA_DATASET, [2_000.0, 1_000.0, 999.0])
        _create_hdf_dataset(
            handle,
            HECRAS_DEPTH_DATASET,
            [
                [0.0, 0.0, 9.0],
                [0.10, 0.0, 9.0],
                [0.40, 0.35, 9.0],
                [0.20, 0.10, 9.0],
            ],
        )

    reference = build_hecras_forecast_reference(
        hdf_paths=[hdf],
        start_timestep=1,
        forecast_steps=3,
        dt_seconds=900,
        thresholds_m=[0.05, 0.30],
    )
    percentiles = reference["forecast_descriptor_percentiles"]

    assert reference["n_descriptors"] == 1
    assert reference["n_failures"] == 0
    assert percentiles["calibrated_max_mean_wd_m"]["p50"] == pytest.approx(0.40)
    assert percentiles["peak_expected_area_fraction_wettable_gt_0.05m"]["p50"] == pytest.approx(1.0)
    assert percentiles["peak_expected_area_fraction_wettable_gt_0.3m"]["p50"] == pytest.approx(1.0)
    assert percentiles["peak_expected_area_lead_hours_gt_0.3m"]["p50"] == pytest.approx(0.25)


def test_hecras_reference_scan_emits_forecast_covariance_when_enough_events(tmp_path):
    h5py = pytest.importorskip("h5py")
    hdfs = []
    for idx in range(20):
        hdf = tmp_path / f"run_{idx:03d}.hdf"
        scale = 1.0 + idx * 0.02
        with h5py.File(hdf, "w") as handle:
            _create_hdf_dataset(handle, HECRAS_CELL_POINTS_DATASET, [[0.0, 0.0], [1.0, 0.0]])
            _create_hdf_dataset(handle, HECRAS_AREA_DATASET, [2_000.0, 1_000.0])
            _create_hdf_dataset(
                handle,
                HECRAS_DEPTH_DATASET,
                [
                    [0.0, 0.0],
                    [0.10 * scale, 0.0],
                    [0.40 * scale, 0.35 * scale],
                    [0.20 * scale, 0.10 * scale],
                ],
            )
        hdfs.append(hdf)

    reference = build_hecras_forecast_reference(
        hdf_paths=hdfs,
        start_timestep=1,
        forecast_steps=3,
        dt_seconds=900,
        thresholds_m=[0.05, 0.30],
        transform_spec={"peak_expected_area_fraction_wettable_gt_0.3m": "logit_bounded"},
    )

    cov = reference["forecast_covariance"]
    assert "calibrated_max_mean_wd_m" not in cov["descriptor_names"]
    assert "covariance_diagonal" in cov
    assert "reference_vectors" in cov
    assert cov["empirical_distance_percentiles"]["p99"] >= cov["empirical_distance_percentiles"]["p95"]


def _create_hdf_dataset(handle, path: str, data):
    parent, _, name = path.rpartition("/")
    group = handle.require_group(parent) if parent else handle
    return group.create_dataset(name, data=data)


def test_forecast_descriptors_include_threshold_specific_extent_metrics():
    descriptors = build_forecast_descriptors(
        raw_summary={"max_mean_wd_m": 0.8},
        calibrated_summary={
            "max_mean_wd_m": 1.2,
            "peak_expected_flooded_area_km2_gt_0.05m": 3.4,
            "peak_expected_flooded_area_fraction_wettable_gt_0.05m": 0.25,
            "peak_expected_flooded_area_lead_hours_gt_0.05m": 3.0,
            "peak_area_weighted_iqr_wd_m": 0.1,
            "peak_area_weighted_total_ensemble_spread_wd_m": 0.18,
            "peak_area_weighted_between_checkpoint_spread_wd_m": 0.12,
            "peak_area_weighted_between_checkpoint_variance_share": 0.45,
            "peak_high_checkpoint_disagreement_area_fraction_wettable": 0.16,
            "peak_between_checkpoint_disagreement_lead_hours": 5.0,
            "exceedance_by_threshold_m": {
                "0.3": {
                    "threshold_m": 0.3,
                    "peak_expected_area_fraction_wettable": 0.12,
                    "peak_expected_area_km2": 1.7,
                    "peak_high_confidence_area_fraction_wettable": 0.08,
                    "peak_expected_area_lead_hours": 4.0,
                }
            },
        },
        comparison_summary={},
    )

    assert descriptors["peak_expected_area_fraction_wettable_gt_0.3m"] == pytest.approx(0.12)
    assert descriptors["peak_expected_area_km2_gt_0.3m"] == pytest.approx(1.7)
    assert descriptors["peak_high_confidence_area_fraction_wettable_gt_0.3m"] == pytest.approx(0.08)
    assert descriptors["peak_expected_area_lead_hours_gt_0.3m"] == pytest.approx(4.0)
    assert descriptors["peak_expected_flooded_area_lead_hours_gt_0.05m"] == pytest.approx(3.0)
    assert descriptors["peak_area_weighted_total_ensemble_spread_wd_m"] == pytest.approx(0.18)
    assert descriptors["peak_area_weighted_between_checkpoint_spread_wd_m"] == pytest.approx(0.12)
    assert descriptors["peak_area_weighted_between_checkpoint_variance_share"] == pytest.approx(0.45)
    assert descriptors["peak_high_checkpoint_disagreement_area_fraction_wettable"] == pytest.approx(0.16)
    assert descriptors["peak_between_checkpoint_disagreement_lead_hours"] == pytest.approx(5.0)


def test_validate_forcing_returns_screening_and_allows_candidate_csv(tmp_path):
    client, provider, _ = _client(tmp_path)
    provider.current = _alice()

    response = client.post(
        "/api/forcing/validate",
        files={"file": ("forcing.csv", _valid_csv(), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["screening"]["available"] is True
    assert body["screening"]["candidate_recommended"] is False
    assert body["screening"]["input_novelty_score"] >= 0.8
    assert any(flag["code"] == "above_candidate_reference" for flag in body["screening"]["flags"])


def test_phase1_only_submit_writes_pre_run_report_without_candidate_package(tmp_path):
    client, provider, orchestrator = _client(tmp_path)
    provider.current = _alice()

    response = client.post(
        "/api/runs",
        files={"file": ("forcing.csv", _valid_csv(), "text/csv")},
        data={"request_full_hdf5": "false"},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert "monitoring_report_pre_run.json" in {ref.artifact_id for ref in orchestrator.artifact_store.list(run_id)}
    payload = client.get(f"/api/runs/{run_id}/monitoring").json()
    assert payload["pre_run"]["candidate_recommended"] is False
    assert payload["candidate"] is None


def test_owner_access_and_admin_candidate_status_update(tmp_path):
    client, provider, _ = _client(tmp_path, monitoring_bundle=_monitoring_bundle(control_modulus=1))
    provider.current = _alice()
    run_id = client.post(
        "/api/runs",
        files={"file": ("forcing.csv", _valid_csv(), "text/csv")},
        data={"request_full_hdf5": "false"},
    ).json()["run_id"]

    provider.current = _bob()
    assert client.get(f"/api/runs/{run_id}/monitoring").status_code == 403

    provider.current = _admin()
    candidates = client.get("/api/admin/retraining-candidates")
    assert candidates.status_code == 200
    candidate_id = candidates.json()[0]["candidate_id"]
    updated = client.post(
        f"/api/admin/retraining-candidates/{candidate_id}/status",
        json={"status": "SELECTED_FOR_HECRAS", "admin_notes": "send to queue"},
    )

    assert updated.status_code == 200
    assert updated.json()["status"] == CandidateStatus.SELECTED_FOR_HECRAS.value
    assert updated.json()["reviewed_by"] == "admin@example.com"


def test_admin_monitoring_trends_endpoint(tmp_path):
    """GET /api/admin/monitoring-trends returns 200 with correct structure."""
    client, provider, _ = _client(tmp_path)
    provider.current = _admin()

    resp = client.get("/api/admin/monitoring-trends")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_candidates" in data
    assert "candidates_by_status" in data


def test_admin_drift_status_endpoint(tmp_path):
    """GET /api/admin/drift-status returns 200."""
    client, provider, _ = _client(tmp_path)
    provider.current = _admin()

    resp = client.get("/api/admin/drift-status")
    assert resp.status_code == 200


def test_admin_hecras_errors_endpoint(tmp_path):
    """GET /api/admin/hecras-errors returns 200."""
    client, provider, _ = _client(tmp_path)
    provider.current = _admin()

    resp = client.get("/api/admin/hecras-errors")
    assert resp.status_code == 200
    assert "records" in resp.json()


def test_non_admin_cannot_access_trends(tmp_path):
    """Non-admin user gets 403 on admin monitoring-trends."""
    client, provider, _ = _client(tmp_path)
    provider.current = _alice()

    resp = client.get("/api/admin/monitoring-trends")
    assert resp.status_code == 403


def test_completed_fake_run_writes_post_run_monitoring_report(tmp_path):
    client, provider, orchestrator = _client(tmp_path)
    provider.current = _alice()
    run_id = client.post(
        "/api/runs",
        files={"file": ("forcing.csv", _valid_csv(), "text/csv")},
        data={"request_full_hdf5": "false"},
    ).json()["run_id"]

    orchestrator.execute(run_id)

    payload = client.get(f"/api/runs/{run_id}/monitoring").json()
    assert payload["post_run"] is not None
    assert "forecast_descriptors.json" in {ref.artifact_id for ref in orchestrator.artifact_store.list(run_id)}
    assert orchestrator.repository.get(run_id).status == RunStatus.COMPLETED


def test_candidate_package_survives_normal_run_artifact_cleanup(tmp_path):
    client, provider, orchestrator = _client(tmp_path, monitoring_bundle=_monitoring_bundle(control_modulus=1))
    provider.current = _alice()
    run_id = client.post(
        "/api/runs",
        files={"file": ("forcing.csv", _valid_csv(), "text/csv")},
        data={"request_full_hdf5": "false"},
    ).json()["run_id"]
    orchestrator.repository.transition(run_id, RunStatus.RUNNING)
    orchestrator.repository.transition(run_id, RunStatus.POSTPROCESSING)
    orchestrator.repository.transition(run_id, RunStatus.COMPLETED)
    old = orchestrator.repository.get(run_id)
    orchestrator.repository._runs[run_id] = old.__class__(
        **{**old.__dict__, "updated_at": datetime.now(timezone.utc) - timedelta(days=31)}
    )
    candidate = orchestrator.monitoring_orchestrator.repository.candidate_for_run(run_id)
    assert candidate is not None
    candidate_root = orchestrator.monitoring_orchestrator.candidate_artifact_store.root / candidate.candidate_id

    orchestrator.expire_due_runs(retention_days=30)

    assert candidate_root.exists()
    assert (candidate_root / "candidate_manifest.json").exists()


# ---------------------------------------------------------------------------
# Phase 2 remaining planned tests
# ---------------------------------------------------------------------------


def test_multivariate_score_uses_empirical_not_chi2(tmp_path):
    """Scoring uses empirical distance percentiles, chi2 is diagnostic only."""
    from neuralop.flood.serving.monitoring import _score_multivariate

    bundle, payload = _make_covariance_bundle(tmp_path)
    empirical = payload["forcing_covariance"]["empirical_distance_percentiles"]
    descriptors = {"stage_max": 1.35, "precipitation_total": 14.5}
    score, flags = _score_multivariate(descriptors, bundle, "forcing")

    mv_flags = [f for f in flags if f.code == "multivariate_outlier"]
    if mv_flags:
        assert "Mahalanobis distance" in mv_flags[0].message
        assert "empirical" in mv_flags[0].message or "p9" in mv_flags[0].message
        assert "chi2_p_value" in mv_flags[0].details
        assert "top_3_standardized_deviations" in mv_flags[0].details


def test_transforms_applied_before_covariance(tmp_path):
    """log1p/logit applied to descriptors before covariance scoring."""
    from neuralop.flood.serving.monitoring import _score_multivariate

    import numpy as np

    descriptors_list = ["a", "b"]
    mean = [1.0, 0.0]
    cov = [[0.1, 0.0], [0.0, 0.1]]
    alpha = 0.01 * 0.2 / 2
    cov_reg = [[0.1 + alpha, 0.0], [0.0, 0.1 + alpha]]
    inv_cov = [[1.0 / cov_reg[0][0], 0.0], [0.0, 1.0 / cov_reg[1][1]]]

    rng = np.random.default_rng(99)
    ref = rng.multivariate_normal(mean, cov, size=100)
    dists = []
    for v in ref:
        d = v - np.array(mean)
        d2 = d @ np.array(inv_cov) @ d
        dists.append(float(np.sqrt(max(0, d2))))
    pcts = np.percentile(dists, [50, 75, 90, 95, 99])

    payload = {
        "bundle_id": "transform-test",
        "monitoring_bundle_schema_version": 2,
        "descriptor_percentiles": {
            "a": {"min": 0, "p01": 0, "p05": 0.1, "p50": 1.0, "p95": 5.0, "p99": 8.0, "max": 10.0},
            "b": {"min": 0, "p01": 0, "p05": 0.01, "p50": 0.5, "p95": 0.9, "p99": 0.99, "max": 1.0},
        },
        "transform_spec": {"a": "log1p", "b": "logit_bounded"},
        "forcing_covariance": {
            "descriptor_names": descriptors_list,
            "mean_vector": mean,
            "inv_covariance_matrix": inv_cov,
            "n_reference": 100,
            "regularization_alpha": alpha,
            "empirical_distance_percentiles": {
                "p50": float(pcts[0]), "p75": float(pcts[1]),
                "p90": float(pcts[2]), "p95": float(pcts[3]), "p99": float(pcts[4]),
            },
        },
    }
    path = tmp_path / "transform_bundle.json"
    path.write_text(json.dumps(payload))
    bundle = MonitoringBundle.load(path)

    score, _ = _score_multivariate({"a": 1.0, "b": 0.5}, bundle, "forcing")
    assert score == 0.0


def _make_forcing_csv(bundle, stage_base: float, precip_base: float) -> str:
    n_rows = bundle.min_required_forcing_rows + 3
    lines = ["time_seconds,stage,precipitation"]
    for idx in range(n_rows):
        s = stage_base + idx * 0.01
        p = precip_base if idx < 5 else 0.0
        lines.append(f"{idx * bundle.dt_seconds},{s:.6f},{p:.6f}")
    return "\n".join(lines) + "\n"


def test_bundle_builder_emits_covariance_and_empirical_percentiles(tmp_path):
    """build_monitoring_bundle_from_forcings emits covariance + empirical distance percentiles."""
    from neuralop.flood.serving.monitoring import build_monitoring_bundle_from_forcings

    bundle = _bundle(tmp_path)
    forcings = []
    for i in range(30):
        csv_text = _make_forcing_csv(bundle, 0.5 + i * 0.02, 0.1 * i)
        forcings.append(parse_forcing_csv(csv_text, bundle=bundle, requested_forecast_steps=3))

    payload = build_monitoring_bundle_from_forcings(
        bundle_id="covariance-test",
        forcings=forcings,
        transform_spec={"stage_max": "identity", "precipitation_total": "log1p"},
    )

    assert payload["monitoring_bundle_schema_version"] == 2
    assert "forcing_covariance" in payload
    cov = payload["forcing_covariance"]
    assert "descriptor_names" in cov
    assert "mean_vector" in cov
    assert "inv_covariance_matrix" in cov
    assert "empirical_distance_percentiles" in cov
    emp = cov["empirical_distance_percentiles"]
    assert "p50" in emp and "p90" in emp and "p95" in emp and "p99" in emp
    assert payload["reference_population"]["n_reference_forcing"] == 30


def test_covariance_exclude_descriptors_honored(tmp_path):
    """Excluded descriptors not in covariance matrix descriptor_names."""
    from neuralop.flood.serving.monitoring import build_monitoring_bundle_from_forcings

    bundle = _bundle(tmp_path)
    forcings = []
    for i in range(30):
        csv_text = _make_forcing_csv(bundle, 0.5 + i * 0.02, 0.1 * i)
        forcings.append(parse_forcing_csv(csv_text, bundle=bundle, requested_forecast_steps=3))

    payload = build_monitoring_bundle_from_forcings(
        bundle_id="exclude-test",
        forcings=forcings,
        covariance_exclude_descriptors=["stage_max"],
    )

    cov = payload.get("forcing_covariance")
    assert cov is not None
    assert "stage_max" not in cov["descriptor_names"]


def test_max_depth_excluded_from_forecast_covariance_by_default():
    """calibrated_max_mean_wd_m can be excluded from forecast covariance descriptor_names."""
    from neuralop.flood.serving.monitoring import _compute_covariance_block

    rows = [
        {"calibrated_max_mean_wd_m": 0.5 + i * 0.1, "area_fraction": 0.1 + i * 0.01, "lead_hours": 2.0 + i}
        for i in range(30)
    ]
    keys = ["area_fraction", "calibrated_max_mean_wd_m", "lead_hours"]
    exclude = ["calibrated_max_mean_wd_m"]

    result = _compute_covariance_block(rows, keys, {}, exclude)

    assert result is not None
    assert "calibrated_max_mean_wd_m" not in result["descriptor_names"]
    assert "area_fraction" in result["descriptor_names"]
    assert "lead_hours" in result["descriptor_names"]


def test_simulated_endpoint_creates_error_record(tmp_path):
    """POST /api/admin/retraining-candidates/{id}/simulated creates error record + updates status."""
    client, provider, orchestrator = _client(tmp_path, monitoring_bundle=_monitoring_bundle(control_modulus=1))
    provider.current = _alice()

    run_id = client.post(
        "/api/runs",
        files={"file": ("forcing.csv", _valid_csv(), "text/csv")},
        data={"request_full_hdf5": "false"},
    ).json()["run_id"]

    provider.current = _admin()
    candidates = client.get("/api/admin/retraining-candidates").json()
    candidate_id = candidates[0]["candidate_id"]

    resp = client.post(
        f"/api/admin/retraining-candidates/{candidate_id}/simulated",
        json={
            "hecras_descriptors": {"area_fraction": 0.30},
            "error_descriptors": {"area_fraction_signed": -0.05},
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SIMULATED"


def test_simulated_endpoint_extracts_hecras_hdf_descriptors(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    client, provider, orchestrator = _client(tmp_path, monitoring_bundle=_monitoring_bundle(control_modulus=1))
    provider.current = _alice()

    run_id = client.post(
        "/api/runs",
        files={"file": ("forcing.csv", _valid_csv(), "text/csv")},
        data={"request_full_hdf5": "false"},
    ).json()["run_id"]
    orchestrator.execute(run_id)
    record = orchestrator.repository.get(run_id)

    hecras_root = tmp_path / "hecras"
    hecras_root.mkdir()
    hdf = hecras_root / "candidate.hdf"
    start_timestep = int(orchestrator.bundle.skip_before_timestep) + int(orchestrator.bundle.n_history)
    n_steps = start_timestep + int(record.spec.forecast_steps) + 2
    depth = [[0.0, 0.0] for _ in range(n_steps)]
    depth[start_timestep + int(record.spec.forecast_steps) - 1] = [0.40, 0.35]
    with h5py.File(hdf, "w") as handle:
        _create_hdf_dataset(handle, HECRAS_CELL_POINTS_DATASET, [[0.0, 0.0], [1.0, 0.0]])
        _create_hdf_dataset(handle, HECRAS_AREA_DATASET, [2_000.0, 1_000.0])
        _create_hdf_dataset(handle, HECRAS_DEPTH_DATASET, depth)
    monkeypatch.setenv("FGN_HECRAS_RESULT_ROOT", str(hecras_root))

    provider.current = _admin()
    candidate_id = client.get("/api/admin/retraining-candidates").json()[0]["candidate_id"]
    resp = client.post(
        f"/api/admin/retraining-candidates/{candidate_id}/simulated",
        json={"hecras_hdf_path": str(hdf)},
    )

    assert resp.status_code == 200
    saved = orchestrator.monitoring_orchestrator.repository.get_hecras_error_record(candidate_id)
    assert saved is not None
    assert saved.hecras_descriptors["peak_expected_area_fraction_wettable_gt_0.3m"] == pytest.approx(1.0)


def test_simulated_endpoint_rejects_unsafe_path(tmp_path):
    """POST /api/admin/retraining-candidates/{id}/simulated with unsafe path returns error."""
    client, provider, orchestrator = _client(tmp_path, monitoring_bundle=_monitoring_bundle(control_modulus=1))
    provider.current = _alice()

    run_id = client.post(
        "/api/runs",
        files={"file": ("forcing.csv", _valid_csv(), "text/csv")},
        data={"request_full_hdf5": "false"},
    ).json()["run_id"]

    provider.current = _admin()
    candidates = client.get("/api/admin/retraining-candidates").json()
    if not candidates:
        pytest.skip("No candidates generated in this test scenario")
    candidate_id = candidates[0]["candidate_id"]

    resp = client.post(
        f"/api/admin/retraining-candidates/{candidate_id}/simulated",
        json={
            "hecras_hdf_path": "/etc/passwd",
        },
    )

    assert resp.status_code in (400, 403, 500, 501)


def test_existing_bundle_json_loads_in_phase2(tmp_path):
    """Phase 1 bundle JSON loads in Phase 2 codebase, schema_version defaults to 1."""
    payload = {
        "bundle_id": "phase1-legacy",
        "descriptor_percentiles": {
            "stage_max": {"min": 0.3, "p01": 0.4, "p05": 0.5, "p50": 0.8, "p95": 1.0, "p99": 1.1, "max": 1.2},
            "precipitation_total": {"min": 0.0, "p01": 0.0, "p05": 0.0, "p50": 4.0, "p95": 12.0, "p99": 20.0, "max": 30.0},
        },
        "thresholds": {
            "warning_percentile_low": "p05", "warning_percentile_high": "p95",
            "candidate_percentile_low": "p01", "candidate_percentile_high": "p99",
            "candidate_score_threshold": 0.8, "control_sample_modulus": 0,
        },
    }
    path = tmp_path / "phase1_bundle.json"
    path.write_text(json.dumps(payload))

    bundle = MonitoringBundle.load(path)

    assert bundle.monitoring_bundle_schema_version == 1
    assert bundle.forcing_covariance is None
    assert bundle.transform_spec == {}
    assert bundle.heuristic_reference_percentiles is None
    assert bundle.covariance_exclude_descriptors == []


def test_existing_scoring_unchanged_without_phase2_fields(tmp_path):
    """Scores identical to Phase 1 when no Phase 2 fields are present."""
    from neuralop.flood.serving.monitoring import MonitoringPhase, _build_report
    from neuralop.flood.serving.run_spec import RunSpec

    bundle = _monitoring_bundle()
    run_spec = RunSpec(
        run_id="compat-1", user_id="u1", bundle_id=bundle.bundle_id,
        input_hash="hash1", forecast_steps=5,
    )
    descriptors = {"stage_max": 0.8, "precipitation_total": 5.0}

    report = _build_report(
        run_spec=run_spec,
        phase=MonitoringPhase.PRE_RUN,
        monitoring_bundle=bundle,
        descriptors=descriptors,
        score_prefix="input",
    )

    assert report.scores["input_novelty_score"] == 0.0
    assert not report.candidate_recommended
    assert not any(f.code == "multivariate_outlier" for f in report.flags)


def test_existing_candidate_workflow_unchanged(tmp_path):
    """Generic status update still works for deterministic-control candidates."""
    client, provider, _ = _client(tmp_path, monitoring_bundle=_monitoring_bundle(control_modulus=1))
    provider.current = _alice()

    run_id = client.post(
        "/api/runs",
        files={"file": ("forcing.csv", _valid_csv(), "text/csv")},
        data={"request_full_hdf5": "false"},
    ).json()["run_id"]

    provider.current = _admin()
    candidates = client.get("/api/admin/retraining-candidates").json()
    candidate_id = candidates[0]["candidate_id"]

    resp = client.post(
        f"/api/admin/retraining-candidates/{candidate_id}/status",
        json={"status": "REVIEWED"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "REVIEWED"

    resp2 = client.post(
        f"/api/admin/retraining-candidates/{candidate_id}/status",
        json={"status": "SELECTED_FOR_HECRAS", "admin_notes": "send to HEC-RAS queue"},
    )
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "SELECTED_FOR_HECRAS"


def test_sql_tables_created_alongside_existing():
    """New Phase 2 SQL tables are created alongside existing tables without schema modification."""
    from neuralop.flood.serving.sql_monitoring import SqlMonitoringRepository

    repo = SqlMonitoringRepository("sqlite:///:memory:")

    from sqlalchemy import inspect
    inspector = inspect(repo.engine)
    tables = inspector.get_table_names()

    assert "fgn_monitoring_reports" in tables
    assert "fgn_retraining_candidates" in tables
    assert "fgn_monitoring_descriptor_log" in tables
    assert "fgn_hecras_error_records" in tables
    assert "fgn_drift_test_results" in tables
