from __future__ import annotations

import csv
import math

import pytest
import torch

from neuralop.flood.eval.neon_phase5 import (
    add_noninferiority_margin_sensitivity,
    aggregate_pilot_seed_decisions,
    assess_p1b_pilot_seed,
    assess_p2_pilot_seed,
    bootstrap_scale_interval,
    build_stage2_variant_predictions,
    common_anchor_gradient_geometry,
    evaluate_gd0,
    evaluate_gp1,
    hermite_basis_covariance,
    gradient_row_geometry,
    phase5_predeclared_protocol,
    paired_family_noninferiority,
    particle_risk_differentiation,
    raw_probit_weight_rank,
    rung_attribution_row,
    stratified_whitened_cancellation,
    verify_checksummed_artifact,
    verify_phase5_decision_artifact,
    verify_phase5_evidence_artifact,
    verify_phase5_ood_ranking_artifact,
    validate_pilot_rungs_for_gp1_decision,
    whitened_cancellation_diagnostics,
    write_rung_attribution_csv,
    write_checksummed_artifact,
)


def _write_phase5_provenance(tmp_path, *, head="abc123", protocol="protocol123"):
    write_checksummed_artifact(
        tmp_path / "PROVENANCE.json",
        {
            "schema_version": "neon_phase5_provenance_v1",
            "git_head": head,
            "protocol_sha256": protocol,
        },
    )


def test_phase5_evidence_requires_matching_checksummed_provenance(tmp_path):
    result = tmp_path / "RESULT.json"
    write_checksummed_artifact(result, {"metric": 1.0})
    _write_phase5_provenance(tmp_path)

    payload = verify_phase5_evidence_artifact(
        result, expected_head="abc123", protocol_sha256="protocol123"
    )

    assert payload == {"metric": 1.0}
    with pytest.raises(ValueError, match="Git HEAD"):
        verify_phase5_evidence_artifact(
            result, expected_head="other", protocol_sha256="protocol123"
        )
    with pytest.raises(ValueError, match="protocol"):
        verify_phase5_evidence_artifact(
            result, expected_head="abc123", protocol_sha256="other"
        )


def test_phase5_decision_requires_matching_head_and_protocol(tmp_path):
    decision = tmp_path / "DECISION.json"
    write_checksummed_artifact(
        decision,
        {
            "gate": "GP1",
            "analysis_git_head": "abc123",
            "protocol_sha256": "protocol123",
        },
    )

    payload = verify_phase5_decision_artifact(
        decision, expected_head="abc123", protocol_sha256="protocol123"
    )

    assert payload["gate"] == "GP1"
    with pytest.raises(ValueError, match="Git HEAD"):
        verify_phase5_decision_artifact(
            decision, expected_head="other", protocol_sha256="protocol123"
        )


def test_phase5_ood_ranking_is_pinned_to_checkpoint_head_and_protocol(tmp_path):
    ranking = tmp_path / "ranking.json"
    write_checksummed_artifact(
        ranking,
        {
            "schema_version": "neon_ood_ranking_v1",
            "analysis_git_head": "abc123",
            "protocol_sha256": "protocol123",
            "stage2_checkpoint_sha256": "checkpoint123",
        },
    )

    payload = verify_phase5_ood_ranking_artifact(
        ranking,
        expected_head="abc123",
        protocol_sha256="protocol123",
        checkpoint_sha256="checkpoint123",
    )

    assert payload["schema_version"] == "neon_ood_ranking_v1"
    with pytest.raises(ValueError, match="checkpoint"):
        verify_phase5_ood_ranking_artifact(
            ranking,
            expected_head="abc123",
            protocol_sha256="protocol123",
            checkpoint_sha256="other",
        )


def test_pilot_rungs_must_match_the_replicated_gp1_mechanism_decision():
    validate_pilot_rungs_for_gp1_decision(
        "continuous_amortization_failure", {"P1B_A", "P1B_B", "P1B_C"}
    )
    validate_pilot_rungs_for_gp1_decision("indeterminate", {"P1B_A"})
    validate_pilot_rungs_for_gp1_decision("shared_subspace_pathology", {"P2"})

    with pytest.raises(ValueError, match="incompatible"):
        validate_pilot_rungs_for_gp1_decision("shared_subspace_pathology", {"P1B_A"})
    with pytest.raises(ValueError, match="does not admit"):
        validate_pilot_rungs_for_gp1_decision("contraction_confirmation", {"P1B_A"})


def test_stage2_variants_keep_frozen_model0_separate_from_deterministic_head():
    base = torch.full((2, 3, 4, 1), 1.0)
    deterministic = torch.full_like(base, 0.5)
    trainable = torch.full((3, *base.shape), 0.2)
    prior = torch.full_like(trainable, 0.3)

    variants = build_stage2_variant_predictions(base, deterministic, trainable, prior)

    assert torch.equal(variants["base"], torch.ones_like(trainable))
    assert torch.equal(variants["deterministic"], torch.full_like(trainable, 1.5))
    assert torch.allclose(variants["both"], torch.full_like(trainable, 2.0))


def test_uniform_weights_give_zero_common_anchor_gradient():
    gradients = torch.randn(7, 5, dtype=torch.float64)
    weights = torch.ones(7, 4, dtype=torch.float64)

    result = common_anchor_gradient_geometry(gradients, weights)

    torch.testing.assert_close(result.delta_gradients, torch.zeros(4, 5, dtype=torch.float64))
    assert result.gradient_effective_rank == 0.0


def test_particle_risk_differentiation_compares_signal_to_minibatch_noise():
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    differentiated = particle_risk_differentiation(
        scores,
        torch.tensor(
            [[2.0, 0.1], [2.0, 0.1], [0.1, 2.0], [0.1, 2.0]],
            dtype=torch.float64,
        ),
        minibatch_size=2,
        replicates=512,
        seed=19,
    )
    collapsed = particle_risk_differentiation(
        scores,
        torch.ones(4, 2, dtype=torch.float64),
        minibatch_size=2,
        replicates=512,
        seed=19,
    )

    assert differentiated["between_particle_risk_rms"] > 0.0
    assert differentiated["minibatch_noise_rms"] > 0.0
    assert differentiated["signal_to_noise"] > 0.0
    assert collapsed["between_particle_risk_rms"] == pytest.approx(0.0)
    assert collapsed["signal_to_noise"] == pytest.approx(0.0)


def test_common_anchor_geometry_detects_synthetic_distinct_optima():
    # Two independent family-gradient directions and particles emphasizing
    # opposite family groups imply distinct local optima under H=I.
    gradients = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64
    )
    weights = torch.tensor(
        [[2.0, 0.5], [0.5, 2.0], [0.5, 0.5]],
        dtype=torch.float64,
    )

    result = common_anchor_gradient_geometry(
        gradients,
        weights,
        hessian=torch.eye(2, dtype=torch.float64),
        validation_jacobian=torch.eye(2, dtype=torch.float64),
        damping=(0.0, 0.1),
    )

    assert result.gradient_effective_rank > 1.2
    assert result.functional_displacement_rms[0.0] > 0.2
    assert result.hessian["negative_eigenvalue_fraction"] == 0.0
    assert 0.0 in result.displacements


def test_gradient_row_geometry_reports_rank_and_pairwise_cosines():
    rows = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

    result = gradient_row_geometry(rows)

    assert result["numerical_rank"] == 2
    assert result["effective_rank"] == pytest.approx(2.0)
    assert result["pairwise_absolute_cosine_mean"] == pytest.approx(0.0)


def test_raw_probit_transform_recovers_low_rank_logits():
    torch.manual_seed(2)
    z = torch.randn(64, 3, dtype=torch.float64)
    directions = torch.randn(25, 3, dtype=torch.float64)
    logits = 0.2 * directions @ z.T
    u = 0.5 * (1.0 + torch.erf(logits / math.sqrt(2.0)))
    raw_weights = -torch.log1p(-u.clamp(max=1.0 - 1.0e-12))

    rank = raw_probit_weight_rank(raw_weights, epistemic_indices=z)

    assert rank["numerical_rank"] == 3
    assert rank["participation_ratio"] <= 3.0 + 1.0e-8
    assert rank["jacobian_numerical_rank"] == 3
    assert rank["jacobian_relative_reconstruction_error"] < 1.0e-10
    assert 0.0 <= rank["off_diagonal_family_logit_abs_correlation_mean"] <= 1.0


def test_whitened_cancellation_is_invariant_to_basis_rotation():
    torch.manual_seed(4)
    m, d, n = 256, 4, 31
    psi = torch.randn(m, d, dtype=torch.float64)
    covariance = torch.eye(d, dtype=torch.float64)
    prior_coeff = torch.randn(d, n, dtype=torch.float64)
    train_coeff = -0.8 * prior_coeff + 0.1 * torch.randn(d, n, dtype=torch.float64)
    prior = psi @ prior_coeff
    train = psi @ train_coeff
    q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))

    original = whitened_cancellation_diagnostics(
        train,
        prior,
        psi,
        covariance,
    )
    rotated = whitened_cancellation_diagnostics(
        train,
        prior,
        psi @ q,
        q.T @ covariance @ q,
    )

    for key in (
        "train_prior_cosine",
        "regression_slope",
        "prior_variance",
        "trainable_variance",
        "retained_variance",
        "residual_variance_after_linear_cancellation",
    ):
        assert abs(original[key] - rotated[key]) < 1.0e-9


def test_analytic_hermite_covariance_has_correct_blocks():
    q = torch.tensor([[1.0, 0.0], [2.0**-0.5, 2.0**-0.5]], dtype=torch.float64)
    covariance = hermite_basis_covariance(2, q, linear_terms=True)
    torch.testing.assert_close(covariance[:2, :2], torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(covariance[:2, 2:], torch.zeros(2, 2, dtype=torch.float64))
    assert covariance[2, 2] == 2.0
    assert abs(float(covariance[2, 3]) - 1.0) < 1.0e-12


def test_stratified_cancellation_reports_wetness_and_lead_bins():
    torch.manual_seed(8)
    m, t, n = 64, 6, 5
    psi = torch.randn(m, 2, dtype=torch.float64)
    prior = torch.einsum("md,dtn->mtn", psi, torch.randn(2, t, n, dtype=torch.float64))
    train = -0.7 * prior
    reference_depth = torch.tensor(
        [
            [0.0, 0.005, 0.02, 0.06, 0.2],
            [0.0, 0.005, 0.02, 0.06, 0.2],
            [0.0, 0.005, 0.02, 0.06, 0.2],
            [0.0, 0.005, 0.02, 0.06, 0.2],
            [0.0, 0.005, 0.02, 0.06, 0.2],
            [0.0, 0.005, 0.02, 0.06, 0.2],
        ],
        dtype=torch.float64,
    )

    result = stratified_whitened_cancellation(
        train,
        prior,
        psi,
        torch.eye(2, dtype=torch.float64),
        reference_depth=reference_depth,
        wettable_mask=torch.ones(n, dtype=torch.bool),
        lead_bins={"early": (0, 2), "mid": (2, 4), "late": (4, 6)},
    )

    assert "all_wettable__early" in result["strata"]
    assert "wet_gt_0p01__mid" in result["strata"]
    assert "front_0p01_0p10__late" in result["strata"]
    assert abs(result["strata"]["all_wettable__early"]["regression_slope"] + 0.7) < 1e-8


def test_stratified_cancellation_respects_cell_area_weights():
    torch.manual_seed(18)
    m, t, n = 128, 2, 2
    psi = torch.randn(m, 1, dtype=torch.float64)
    prior = psi[:, 0, None, None].expand(m, t, n).clone()
    train = prior.clone()
    train[:, :, 0] = -prior[:, :, 0]
    result = stratified_whitened_cancellation(
        train,
        prior,
        psi,
        torch.eye(1, dtype=torch.float64),
        reference_depth=torch.full((t, n), 0.05, dtype=torch.float64),
        wettable_mask=torch.ones(n, dtype=torch.bool),
        lead_bins={"all": (0, t)},
        score_weights=torch.tensor(
            [[[100.0], [1.0]], [[100.0], [1.0]]], dtype=torch.float64
        ),
    )

    assert result["strata"]["all_wettable__all"]["train_prior_cosine"] < -0.95


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression test")
def test_stratified_cancellation_accepts_cpu_weights_with_cuda_fields():
    """Evaluation metadata stays on CPU while correction fields live on the GPU."""

    device = torch.device("cuda")
    m, t, n = 8, 2, 3
    psi = torch.randn(m, 1, dtype=torch.float64, device=device)
    prior = psi[:, 0, None, None].expand(m, t, n).clone()
    train = -0.5 * prior

    result = stratified_whitened_cancellation(
        train,
        prior,
        psi,
        torch.eye(1, dtype=torch.float64),
        reference_depth=torch.full((t, n), 0.05, dtype=torch.float64, device=device),
        wettable_mask=torch.ones(n, dtype=torch.bool, device=device),
        lead_bins={"all": (0, t)},
        score_weights=torch.ones(t, n, 1, dtype=torch.float64),
    )

    assert result["strata"]["all_wettable__all"]["regression_slope"] == pytest.approx(
        -0.5
    )


def test_bootstrap_scale_interval_resamples_families_and_draws_deterministically():
    # [families, draws] nonnegative epistemic variance contributions.
    variance = torch.tensor(
        [[1.0, 4.0, 9.0], [4.0, 9.0, 16.0]], dtype=torch.float64
    )
    first = bootstrap_scale_interval(variance, replicates=200, seed=12)
    second = bootstrap_scale_interval(variance, replicates=200, seed=12)
    assert first == second
    assert first["estimate"] > 0.0
    assert first["ci95_lower"] <= first["estimate"] <= first["ci95_upper"]


def test_gd0_has_three_way_outcomes_and_near_zero_guard():
    contraction = evaluate_gd0(
        observed_scale_m={"estimate": 1.0, "ci95_lower": 0.9, "ci95_upper": 1.1},
        direct_scale_m={"estimate": 1.1, "ci95_lower": 1.0, "ci95_upper": 1.2},
        differentiation_snr=0.4,
        functional_displacement_m=1.0e-5,
        noise_snr_threshold=1.0,
        meaningful_displacement_m=1.0e-3,
    )
    assert contraction["decision"] == "contraction_consistent"
    assert (
        contraction["ratio_interval_method"]
        == "conservative_ratio_of_marginal_95pct_endpoints"
    )

    under = evaluate_gd0(
        observed_scale_m={"estimate": 0.2, "ci95_lower": 0.15, "ci95_upper": 0.25},
        direct_scale_m={"estimate": 1.0, "ci95_lower": 0.9, "ci95_upper": 1.1},
        differentiation_snr=3.0,
        functional_displacement_m=0.01,
        noise_snr_threshold=1.0,
        meaningful_displacement_m=1.0e-3,
    )
    assert under["decision"] == "under_delivery"

    guarded = evaluate_gd0(
        observed_scale_m={"estimate": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0},
        direct_scale_m={"estimate": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0},
        differentiation_snr=0.0,
        functional_displacement_m=0.0,
    )
    assert guarded["decision"] == "indeterminate"
    assert guarded["near_zero_scale_guard"] is True
    assert (
        guarded["ratio_interval_method"]
        == "conservative_ratio_of_marginal_95pct_endpoints"
    )


def test_gp1_operationalizes_confirmation_repair_and_shared_pathology():
    confirmation = evaluate_gp1(
        p1a_to_direct_ratio=1.1,
        p1a_to_continuous_ratio=1.2,
        differentiation_snr=0.5,
        skill_noninferior=True,
        impact_noninferior=True,
        direct_to_p1a_ratio=0.9,
        functional_displacement_m=0.0,
        independent_prior_cancellation=0.2,
        skill_without_cancellation=False,
    )
    assert confirmation["decision"] == "contraction_confirmation"

    repair = evaluate_gp1(
        p1a_to_direct_ratio=0.9,
        p1a_to_continuous_ratio=3.0,
        differentiation_snr=2.0,
        skill_noninferior=True,
        impact_noninferior=True,
        direct_to_p1a_ratio=1.1,
        functional_displacement_m=0.01,
        independent_prior_cancellation=0.2,
        skill_without_cancellation=False,
    )
    assert repair["decision"] == "continuous_amortization_failure"

    pathology = evaluate_gp1(
        p1a_to_direct_ratio=0.3,
        p1a_to_continuous_ratio=1.5,
        differentiation_snr=2.0,
        skill_noninferior=True,
        impact_noninferior=True,
        direct_to_p1a_ratio=3.0,
        functional_displacement_m=0.01,
        independent_prior_cancellation=0.9,
        skill_without_cancellation=True,
    )
    assert pathology["decision"] == "shared_subspace_pathology"


def test_phase5_protocol_is_predeclared_and_checksummed(tmp_path):
    protocol = phase5_predeclared_protocol()
    assert protocol["primary_impact_threshold_m"] == pytest.approx(0.1)
    assert protocol["gd0"]["equivalence_factor"] == pytest.approx(2.0)
    assert protocol["noninferiority"]["crps_margin_m"] == pytest.approx(1.0e-4)
    assert protocol["noninferiority"]["impact_area_crps_margin_km2"] == 0.0
    assert protocol["direct"]["gd0_primary_mode"] == "data"
    assert protocol["direct"]["gp1_primary_law"] == "dirichlet"
    assert protocol["pilot"]["minimum_acceptance_seeds"] == 3
    path = tmp_path / "PROTOCOL.json"
    digest = write_checksummed_artifact(path, protocol)
    assert verify_checksummed_artifact(path) == digest
    path.write_text(path.read_text().replace("0.0001", "0.0002"))
    with pytest.raises(ValueError, match="checksum"):
        verify_checksummed_artifact(path)


def test_crps_margin_sensitivity_reuses_one_paired_interval():
    result = {
        "mean_difference": 2.0e-5,
        "ci95_lower": -4.0e-5,
        "ci95_upper": 8.0e-5,
        "margin": 1.0e-4,
        "pass": True,
    }

    augmented = add_noninferiority_margin_sensitivity(
        result, primary_margin=1.0e-4
    )

    assert augmented["mean_difference"] == result["mean_difference"]
    assert augmented["ci95_upper"] == result["ci95_upper"]
    assert augmented["sensitivity_table"] == [
        {"label": "strict", "margin": 0.0, "pass": False},
        {"label": "primary_predeclared", "margin": 1.0e-4, "pass": True},
    ]


def test_skill_without_cancellation_requires_paired_depth_rmse_and_impact_gates():
    passing = paired_family_noninferiority(
        variant_crps=[0.09, 0.10, 0.11, 0.10],
        base_crps=[0.10, 0.10, 0.10, 0.10],
        variant_rmse=[0.20, 0.20, 0.20, 0.20],
        base_rmse=[0.20, 0.20, 0.20, 0.20],
        variant_impact=[0.30, 0.29, 0.31, 0.30],
        base_impact=[0.30, 0.30, 0.30, 0.30],
        crps_margin=0.02,
        rmse_margin=0.001,
        impact_margin=0.02,
        replicates=500,
        seed=17,
    )
    assert passing["pass"] is True
    assert passing["depth_skill_pass"] is True
    assert passing["impact_skill_pass"] is True

    failing_rmse = paired_family_noninferiority(
        variant_crps=[0.09] * 4,
        base_crps=[0.10] * 4,
        variant_rmse=[0.22] * 4,
        base_rmse=[0.20] * 4,
        variant_impact=[0.29] * 4,
        base_impact=[0.30] * 4,
        crps_margin=0.02,
        rmse_margin=0.001,
        impact_margin=0.02,
        replicates=100,
        seed=19,
    )
    assert failing_rmse["crps"]["pass"] is True
    assert failing_rmse["rmse"]["pass"] is False
    assert failing_rmse["impact_area"]["pass"] is True
    assert failing_rmse["pass"] is False


def _pilot_report(*, seed: int, crps_pass: bool = True) -> dict:
    return {
        "ladder_rung": "P1B_A",
        "prior_seed": seed,
        "posterior_scale_m": {
            "estimate": 0.008,
            "ci95_lower": 0.007,
            "ci95_upper": 0.009,
        },
        "noninferiority": {
            "crps": {"pass": crps_pass, "ci95_upper": 8.0e-5},
            "rmse": {"pass": True, "ci95_upper": 7.0e-4},
            "impact_area": {"pass": True, "ci95_upper": -0.01},
            "depth_skill_pass": crps_pass,
            "impact_skill_pass": True,
        },
        "differentiation": {
            "signal_to_noise": 1.4,
            "functional_displacement_m": 0.002,
        },
        "stratified_cancellation": {
            "strata": {
                "all_wettable__early": {
                    "train_prior_cosine": -0.6,
                    "prior_variance": 4.0,
                    "retained_variance": 1.0,
                },
                "front_0p01_0p10__late": {
                    "train_prior_cosine": -0.5,
                    "prior_variance": 4.0,
                    "retained_variance": 1.2,
                },
            }
        },
    }


def test_rung_attribution_row_has_stable_physical_skill_and_geometry_schema():
    report = _pilot_report(seed=17)
    report.update(
        {
            "sampling_design": "crossed_common_random_numbers",
            "dirichlet_particle_seed": None,
            "posterior_scale_m": {
                "estimate": 0.008,
                "ci95_lower": 0.007,
                "ci95_upper": 0.009,
            },
            "skill": {
                "base": {"fair_crps_m": 0.020, "rmse_m": 0.051},
                "both": {"fair_crps_m": 0.019, "rmse_m": 0.0505},
            },
            "impact": {
                "base": {"area_crps_km2": 0.30},
                "both": {"area_crps_km2": 0.29},
            },
            "noninferiority": {
                "crps": {
                    "mean_difference": -0.001,
                    "ci95_upper": 8.0e-5,
                    "pass": True,
                },
                "rmse": {
                    "mean_difference": -0.0005,
                    "ci95_upper": 7.0e-4,
                    "pass": True,
                },
                "impact_area": {
                    "mean_difference": -0.01,
                    "ci95_upper": -0.002,
                    "pass": True,
                },
                "depth_skill_pass": True,
                "impact_skill_pass": True,
                "pass": True,
            },
            "cancellation_fraction": 0.35,
            "train_prior_cosine": -0.4,
            "skill_without_cancellation": True,
            "epistemic_error_association": {
                "epistemic_std_abs_error_partial_depth_all_wettable": 0.12,
                "epistemic_std_abs_error_partial_depth_ref_wet": 0.18,
                "epistemic_std_abs_error_partial_depth_wet_front": 0.21,
            },
        }
    )

    row = rung_attribution_row(report)

    assert row["schema_version"] == "neon_phase5_rung_attribution_v1"
    assert row["ladder_rung"] == "P1B_A"
    assert row["prior_seed"] == 17
    assert row["posterior_scale_m"] == pytest.approx(0.008)
    assert row["base_fair_crps_m"] == pytest.approx(0.020)
    assert row["full_fair_crps_m"] == pytest.approx(0.019)
    assert row["delta_fair_crps_m"] == pytest.approx(-0.001)
    assert row["delta_rmse_m"] == pytest.approx(-0.0005)
    assert row["delta_area_crps_km2"] == pytest.approx(-0.01)
    assert row["epistemic_std_error_partial_front"] == pytest.approx(0.21)
    assert row["all_noninferiority_pass"] is True


def test_write_rung_attribution_csv_is_checksumming_and_round_trippable(tmp_path):
    report = _pilot_report(seed=23)
    report.update(
        {
            "sampling_design": "crossed_common_random_numbers",
            "skill": {
                "base": {"fair_crps_m": 0.02, "rmse_m": 0.05},
                "both": {"fair_crps_m": 0.02, "rmse_m": 0.05},
            },
            "impact": {
                "base": {"area_crps_km2": 0.3},
                "both": {"area_crps_km2": 0.3},
            },
            "cancellation_fraction": 0.4,
            "train_prior_cosine": -0.2,
            "skill_without_cancellation": False,
            "epistemic_error_association": {},
        }
    )
    destination = tmp_path / "rung_attribution.csv"

    digest = write_rung_attribution_csv(destination, [report])

    assert len(digest) == 64
    assert destination.with_suffix(".csv.sha256").is_file()
    rows = list(csv.DictReader(destination.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["ladder_rung"] == "P1B_A"
    assert rows[0]["prior_seed"] == "23"


def _baseline_strata() -> dict:
    return {
        "all_wettable__early": {
            "train_prior_cosine": -0.9,
            "prior_variance": 4.0,
            "retained_variance": 0.2,
        },
        "front_0p01_0p10__late": {
            "train_prior_cosine": -0.8,
            "prior_variance": 4.0,
            "retained_variance": 0.3,
        },
    }


def test_p1b_pilot_seed_requires_skill_direct_scale_and_cancellation_direction():
    result = assess_p1b_pilot_seed(
        _pilot_report(seed=11),
        direct_scale_m={"estimate": 0.01},
        baseline_strata=_baseline_strata(),
    )

    assert result["pass"] is True
    assert result["checks"]["scale_approaches_direct"] is True
    assert result["checks"]["cancellation_moves_in_expected_direction"] is True
    assert result["cancellation_direction"]["joint_improvement_fraction"] == 1.0


def test_replicated_pilot_gate_requires_three_unique_consistent_seeds():
    rows = [
        assess_p1b_pilot_seed(
            _pilot_report(seed=seed),
            direct_scale_m={"estimate": 0.01},
            baseline_strata=_baseline_strata(),
        )
        for seed in (11, 12, 13)
    ]

    accepted = aggregate_pilot_seed_decisions(rows, minimum_seeds=3)
    assert accepted["decision"] == "pilot_accepted"
    assert accepted["verdict_status"] == "acceptance_replicated"

    rows[2]["pass"] = False
    rejected = aggregate_pilot_seed_decisions(rows, minimum_seeds=3)
    assert rejected["decision"] == "pilot_rejected"
    assert rejected["verdict_status"] == "replicated_inconsistent"

    with pytest.raises(ValueError, match="unique"):
        aggregate_pilot_seed_decisions([rows[0], rows[0], rows[1]], minimum_seeds=3)


def test_p2_seed_requires_controlled_association_and_ood_ranking_improvement():
    report = _pilot_report(seed=21)
    report["ladder_rung"] = "P2"
    report["amplitude_tuned_on_evaluation_events"] = False
    report["epistemic_error_association"] = {
        "epistemic_std_abs_error_partial_depth_all_wettable": 0.15,
        "epistemic_std_abs_error_partial_depth_ref_wet": 0.12,
        "epistemic_std_abs_error_partial_depth_wet_front": 0.08,
    }
    ood = {
        "spearman_epistemic_std_rmse": 0.4,
        "mean_sparsification_gap_m": 0.01,
        "leave_one_out": [{"spearman": 0.2}, {"spearman": 0.1}],
    }
    baseline = {"mean_sparsification_gap_m": 0.02}

    result = assess_p2_pilot_seed(
        report,
        ood_evidence=ood,
        baseline_ood_evidence=baseline,
        amplitude_tuned_on_evaluation_events=False,
    )

    assert result["pass"] is True
    assert result["checks"]["hydraulically_controlled_association_positive"] is True
    assert result["checks"]["ood_risk_coverage_improved"] is True
