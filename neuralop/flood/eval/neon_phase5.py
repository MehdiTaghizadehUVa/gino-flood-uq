"""Phase-5 diagnostics for NEON Stage-2 epistemic attribution.

The functions in this module are deliberately independent of the training
runner. They operate on frozen-feature gradients, saved correction tensors,
or per-family variance contributions so diagnosis cannot silently mutate the
Stage-2 estimator being studied.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


RUNG_ATTRIBUTION_FIELDS = (
    "schema_version",
    "ladder_rung",
    "prior_seed",
    "dirichlet_particle_seed",
    "sampling_design",
    "prior_scale_mode",
    "prior_scale_target_std_m",
    "prior_scale_alpha_normalized",
    "posterior_scale_m",
    "posterior_scale_ci95_lower_m",
    "posterior_scale_ci95_upper_m",
    "base_fair_crps_m",
    "full_fair_crps_m",
    "delta_fair_crps_m",
    "fair_crps_ucb_m",
    "fair_crps_noninferior",
    "base_rmse_m",
    "full_rmse_m",
    "delta_rmse_m",
    "rmse_ucb_m",
    "rmse_noninferior",
    "base_area_crps_km2",
    "full_area_crps_km2",
    "delta_area_crps_km2",
    "area_crps_ucb_km2",
    "area_crps_noninferior",
    "depth_noninferiority_pass",
    "impact_noninferiority_pass",
    "all_noninferiority_pass",
    "cancellation_fraction",
    "train_prior_cosine",
    "differentiation_snr",
    "functional_displacement_m",
    "epistemic_std_error_partial_all_wettable",
    "epistemic_std_error_partial_ref_wet",
    "epistemic_std_error_partial_front",
    "skill_without_cancellation",
    "amplitude_tuned_on_evaluation_events",
)


def _required_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Phase-5 attribution report is missing mapping {key!r}.")
    return value


def _metric_delta(
    comparison: Mapping[str, Any],
    *,
    base: float,
    full: float,
) -> float:
    value = comparison.get("mean_difference", full - base)
    return float(value)


def rung_attribution_row(report: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one signed pilot report into the stable attribution schema.

    The row intentionally contains only physical-space aggregate estimands and
    preregistered decision diagnostics. Per-family values remain in the signed
    JSON report, which is the source of truth for any later reanalysis.
    """

    skill = _required_mapping(report, "skill")
    base_skill = _required_mapping(skill, "base")
    full_skill = _required_mapping(skill, "both")
    impact = _required_mapping(report, "impact")
    base_impact = _required_mapping(impact, "base")
    full_impact = _required_mapping(impact, "both")
    noninferiority = _required_mapping(report, "noninferiority")
    crps = _required_mapping(noninferiority, "crps")
    rmse = _required_mapping(noninferiority, "rmse")
    area = _required_mapping(noninferiority, "impact_area")
    scale = _required_mapping(report, "posterior_scale_m")
    differentiation = dict(report.get("differentiation") or {})
    association = dict(report.get("epistemic_error_association") or {})

    base_crps = float(base_skill["fair_crps_m"])
    full_crps = float(full_skill["fair_crps_m"])
    base_rmse = float(base_skill["rmse_m"])
    full_rmse = float(full_skill["rmse_m"])
    base_area = float(base_impact["area_crps_km2"])
    full_area = float(full_impact["area_crps_km2"])
    row = {
        "schema_version": "neon_phase5_rung_attribution_v1",
        "ladder_rung": str(report.get("ladder_rung", "")).upper(),
        "prior_seed": report.get("prior_seed", ""),
        "dirichlet_particle_seed": report.get("dirichlet_particle_seed", ""),
        "sampling_design": str(report.get("sampling_design", "")),
        "prior_scale_mode": report.get("prior_scale_mode", ""),
        "prior_scale_target_std_m": report.get("prior_scale_target_std_m", ""),
        "prior_scale_alpha_normalized": report.get(
            "prior_scale_alpha_normalized", ""
        ),
        "posterior_scale_m": float(scale["estimate"]),
        "posterior_scale_ci95_lower_m": float(scale["ci95_lower"]),
        "posterior_scale_ci95_upper_m": float(scale["ci95_upper"]),
        "base_fair_crps_m": base_crps,
        "full_fair_crps_m": full_crps,
        "delta_fair_crps_m": _metric_delta(crps, base=base_crps, full=full_crps),
        "fair_crps_ucb_m": float(crps["ci95_upper"]),
        "fair_crps_noninferior": bool(crps["pass"]),
        "base_rmse_m": base_rmse,
        "full_rmse_m": full_rmse,
        "delta_rmse_m": _metric_delta(rmse, base=base_rmse, full=full_rmse),
        "rmse_ucb_m": float(rmse["ci95_upper"]),
        "rmse_noninferior": bool(rmse["pass"]),
        "base_area_crps_km2": base_area,
        "full_area_crps_km2": full_area,
        "delta_area_crps_km2": _metric_delta(
            area, base=base_area, full=full_area
        ),
        "area_crps_ucb_km2": float(area["ci95_upper"]),
        "area_crps_noninferior": bool(area["pass"]),
        "depth_noninferiority_pass": bool(noninferiority["depth_skill_pass"]),
        "impact_noninferiority_pass": bool(noninferiority["impact_skill_pass"]),
        "all_noninferiority_pass": bool(noninferiority.get("pass", False)),
        "cancellation_fraction": float(report.get("cancellation_fraction", math.nan)),
        "train_prior_cosine": float(report.get("train_prior_cosine", math.nan)),
        "differentiation_snr": float(
            differentiation.get("signal_to_noise", math.nan)
        ),
        "functional_displacement_m": float(
            differentiation.get("functional_displacement_m", math.nan)
        ),
        "epistemic_std_error_partial_all_wettable": float(
            association.get(
                "epistemic_std_abs_error_partial_depth_all_wettable", math.nan
            )
        ),
        "epistemic_std_error_partial_ref_wet": float(
            association.get(
                "epistemic_std_abs_error_partial_depth_ref_wet", math.nan
            )
        ),
        "epistemic_std_error_partial_front": float(
            association.get(
                "epistemic_std_abs_error_partial_depth_wet_front", math.nan
            )
        ),
        "skill_without_cancellation": bool(
            report.get("skill_without_cancellation", False)
        ),
        "amplitude_tuned_on_evaluation_events": bool(
            report.get("amplitude_tuned_on_evaluation_events", False)
        ),
    }
    if not row["ladder_rung"]:
        raise ValueError("Phase-5 attribution report is missing ladder_rung.")
    return {key: row[key] for key in RUNG_ATTRIBUTION_FIELDS}


def write_rung_attribution_csv(
    path: str | Path, reports: Sequence[Mapping[str, Any]]
) -> str:
    """Atomically write one or more pilot reports and a SHA-256 sidecar."""

    rows = [rung_attribution_row(report) for report in reports]
    if not rows:
        raise ValueError("At least one Phase-5 attribution report is required.")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(RUNG_ATTRIBUTION_FIELDS))
    writer.writeheader()
    writer.writerows(rows)
    encoded = buffer.getvalue().encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, destination)
    digest = hashlib.sha256(encoded).hexdigest()
    checksum = destination.with_suffix(destination.suffix + ".sha256")
    checksum.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return digest


def phase5_predeclared_protocol() -> dict[str, Any]:
    """Return the frozen Phase-5 statistics, strata, margins, and gates."""

    return {
        "schema_version": "neon_phase5_protocol_v1",
        "primary_scale": "sqrt(weighted_domain_mean_epistemic_variance)",
        "physical_space_required": True,
        "primary_impact_threshold_m": 0.1,
        "impact_area_unit": "km2",
        "arrival_time_censor": "right_censor_at_T_plus_1",
        "wetness_strata_m": {
            "all_wettable": None,
            "wet": [0.01, None],
            "front": [0.01, 0.10],
        },
        "lead_strata": {
            "early": [0, 31],
            "mid": [31, 63],
            "late": [63, 94],
        },
        "noninferiority": {
            "crps_margin_m": 1.0e-4,
            "rmse_margin_m": 1.0e-3,
            "impact_area_crps_margin_km2": 0.0,
            "confidence": 0.95,
            "paired_family_bootstrap": True,
        },
        "gd0": {
            "equivalence_factor": 2.0,
            "noise_snr_threshold": 1.0,
            "meaningful_displacement_m": 1.0e-3,
            "scale_floor_m": 1.0e-8,
        },
        "gp1": {
            "equivalence_interval": [0.5, 2.0],
            "substantial_retention_ratio": 2.0,
            "cancellation_threshold": 0.8,
        },
        "pilot": {
            "minimum_acceptance_seeds": 3,
            "direct_scale_equivalence_interval": [0.5, 2.0],
            "differentiation_snr_threshold": 1.0,
            # The plan prescribes a directional cancellation check but does
            # not justify a magnitude threshold. Requiring a strict joint
            # improvement in a majority of matched hydraulic strata avoids
            # introducing an unregistered effect-size margin.
            "cancellation_joint_improvement_fraction": 0.5,
        },
        "direct": {
            "last_layer_draws": 32,
            "full_head_draws": 8,
            "multistart_spot_checks": 2,
            "multistart_last_layer_draws": 4,
            "multistart_full_head_draws": 2,
            "multistart_relative_objective_tolerance": 0.02,
            "multistart_initialization_perturbation": 1.0e-3,
            "data_prior_alpha": 0.0,
            "maximum_gradient_reduction_ratio": 0.5,
            "head_sensitivity_factor": 2.0,
            # GD0 asks whether family reweighting implies distinct fitted
            # predictors. The alpha=0 direct-data fit isolates that estimand;
            # indexed-prior fits remain representability diagnostics only.
            "gd0_primary_mode": "data",
            "gd0_primary_law": "probit",
            "gp1_primary_mode": "rpf",
            "gp1_primary_law": "dirichlet",
        },
    }


def add_noninferiority_margin_sensitivity(
    result: Mapping[str, Any], *, primary_margin: float
) -> dict[str, Any]:
    """Attach strict and preregistered-margin readings to one paired CI.

    The raw paired difference and confidence interval remain unchanged; only
    their predeclared interpretation changes with the margin.
    """

    output = dict(result)
    upper = float(output["ci95_upper"])
    output["sensitivity_table"] = [
        {"label": "strict", "margin": 0.0, "pass": upper <= 0.0},
        {
            "label": "primary_predeclared",
            "margin": float(primary_margin),
            "pass": upper <= float(primary_margin),
        },
    ]
    return output


def _paired_family_upper_confidence_bound(
    variant: Sequence[float],
    baseline: Sequence[float],
    *,
    margin: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a paired family-level difference without treating cells as iid."""

    variant_values = torch.as_tensor(variant, dtype=torch.float64).reshape(-1)
    baseline_values = torch.as_tensor(baseline, dtype=torch.float64).reshape(-1)
    if variant_values.shape != baseline_values.shape or variant_values.numel() < 2:
        raise ValueError("variant and baseline require at least two paired families.")
    if not torch.isfinite(variant_values).all() or not torch.isfinite(baseline_values).all():
        raise ValueError("paired family metrics must be finite.")
    if int(replicates) < 1:
        raise ValueError("replicates must be >= 1.")
    differences = variant_values - baseline_values
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    index = torch.randint(
        differences.numel(),
        (int(replicates), differences.numel()),
        generator=generator,
    )
    samples = differences[index].mean(dim=1)
    lower = float(torch.quantile(samples, 0.025))
    upper = float(torch.quantile(samples, 0.975))
    return {
        "mean_difference": float(differences.mean()),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "margin": float(margin),
        "pass": upper <= float(margin),
    }


def paired_family_noninferiority(
    *,
    variant_crps: Sequence[float],
    base_crps: Sequence[float],
    variant_rmse: Sequence[float],
    base_rmse: Sequence[float],
    variant_impact: Sequence[float],
    base_impact: Sequence[float],
    crps_margin: float,
    rmse_margin: float,
    impact_margin: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Apply the complete paired skill contract to one prediction variant.

    A non-cancelling variant is scientifically acceptable only when its depth
    distribution, ensemble-mean RMSE, and inundated-area distribution all pass
    their predeclared family-level upper-confidence-bound tests.
    """

    crps = add_noninferiority_margin_sensitivity(
        _paired_family_upper_confidence_bound(
            variant_crps,
            base_crps,
            margin=crps_margin,
            replicates=replicates,
            seed=seed,
        ),
        primary_margin=crps_margin,
    )
    rmse = _paired_family_upper_confidence_bound(
        variant_rmse,
        base_rmse,
        margin=rmse_margin,
        replicates=replicates,
        seed=seed + 1,
    )
    impact = _paired_family_upper_confidence_bound(
        variant_impact,
        base_impact,
        margin=impact_margin,
        replicates=replicates,
        seed=seed + 2,
    )
    depth_pass = bool(crps["pass"] and rmse["pass"])
    impact_pass = bool(impact["pass"])
    return {
        "crps": crps,
        "rmse": rmse,
        "impact_area": impact,
        "depth_skill_pass": depth_pass,
        "impact_skill_pass": impact_pass,
        "pass": bool(depth_pass and impact_pass),
    }


def build_stage2_variant_predictions(
    base_prediction: torch.Tensor,
    deterministic_correction: torch.Tensor,
    trainable_correction: torch.Tensor,
    prior_correction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Expose Stage-1, mean-head, and epistemic variants without relabeling.

    ``base_prediction`` and ``deterministic_correction`` are ``[K,T,N,C]``;
    trainable/prior corrections are ``[M,K,T,N,C]``. The frozen Stage-1
    baseline is intentionally preserved as ``base`` so skill non-inferiority
    cannot silently move to the deterministic Stage-2 head.
    """

    base = torch.as_tensor(base_prediction)
    deterministic = torch.as_tensor(deterministic_correction)
    trainable = torch.as_tensor(trainable_correction)
    prior = torch.as_tensor(prior_correction)
    if base.shape != deterministic.shape:
        raise ValueError("base_prediction and deterministic_correction shapes differ.")
    if trainable.shape != prior.shape or trainable.ndim != base.ndim + 1:
        raise ValueError("trainable/prior corrections must share [M,K,T,N,C] shape.")
    if trainable.shape[1:] != base.shape:
        raise ValueError("correction member/time/mesh shape does not match the base prediction.")
    stage1 = base.unsqueeze(0).expand(trainable.shape[0], *base.shape)
    mean = stage1 + deterministic.unsqueeze(0)
    return {
        "base": stage1,
        "deterministic": mean,
        "prior": mean + prior,
        "trainable": mean + trainable,
        "both": mean + trainable + prior,
    }


def _stratum_retention_std(row: Mapping[str, Any], *, eps: float = 1.0e-15) -> float:
    prior = float(row.get("prior_variance", math.nan))
    retained = float(row.get("retained_variance", math.nan))
    if not math.isfinite(prior) or not math.isfinite(retained) or prior <= eps:
        return math.nan
    return math.sqrt(max(retained, 0.0) / prior)


def cancellation_direction_against_baseline(
    pilot_strata: Mapping[str, Mapping[str, Any]],
    baseline_strata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare whitened pilot cancellation with B3 by hydraulic stratum.

    A stratum improves only when its trainable/prior cosine becomes less
    negative *and* its retained posterior standard deviation increases. This
    is a direction-of-effect check, not an effect-size claim.
    """

    common = sorted(set(pilot_strata).intersection(baseline_strata))
    rows: list[dict[str, Any]] = []
    for name in common:
        pilot = pilot_strata[name]
        baseline = baseline_strata[name]
        pilot_cosine = float(pilot.get("train_prior_cosine", math.nan))
        baseline_cosine = float(baseline.get("train_prior_cosine", math.nan))
        pilot_retention = _stratum_retention_std(pilot)
        baseline_retention = _stratum_retention_std(baseline)
        finite = all(
            math.isfinite(value)
            for value in (
                pilot_cosine,
                baseline_cosine,
                pilot_retention,
                baseline_retention,
            )
        )
        cosine_improved = bool(finite and pilot_cosine > baseline_cosine)
        retention_improved = bool(finite and pilot_retention > baseline_retention)
        rows.append(
            {
                "stratum": name,
                "pilot_train_prior_cosine": pilot_cosine,
                "baseline_train_prior_cosine": baseline_cosine,
                "pilot_retention_std_ratio": pilot_retention,
                "baseline_retention_std_ratio": baseline_retention,
                "cosine_improved": cosine_improved,
                "retention_improved": retention_improved,
                "joint_improvement": bool(cosine_improved and retention_improved),
            }
        )
    valid = [row for row in rows if math.isfinite(row["pilot_retention_std_ratio"])]
    fraction = (
        sum(bool(row["joint_improvement"]) for row in valid) / len(valid)
        if valid
        else 0.0
    )
    return {
        "matched_strata": len(common),
        "valid_strata": len(valid),
        "joint_improvement_fraction": float(fraction),
        "strata": rows,
    }


def assess_p1b_pilot_seed(
    report: Mapping[str, Any],
    *,
    direct_scale_m: Mapping[str, Any],
    baseline_strata: Mapping[str, Mapping[str, Any]],
    equivalence_interval: tuple[float, float] = (0.5, 2.0),
    differentiation_snr_threshold: float = 1.0,
    cancellation_improvement_fraction: float = 0.5,
) -> dict[str, Any]:
    """Apply the predeclared P1b criteria to one independently seeded pilot."""

    rung = str(report.get("ladder_rung", "")).upper()
    if rung not in {"P1B_A", "P1B_B", "P1B_C"}:
        raise ValueError(f"P1b report has unsupported ladder_rung={rung!r}.")
    seed = report.get("prior_seed")
    if seed is None:
        raise ValueError("P1b report must record prior_seed.")
    pilot_scale = float(report["posterior_scale_m"]["estimate"])
    direct_scale = float(direct_scale_m["estimate"])
    if not math.isfinite(pilot_scale) or not math.isfinite(direct_scale) or direct_scale <= 0:
        raise ValueError("pilot and direct scales must be finite and direct scale positive.")
    lower, upper = (float(value) for value in equivalence_interval)
    ratio = pilot_scale / direct_scale
    noninferiority = dict(report.get("noninferiority") or {})
    depth_pass = bool(noninferiority.get("depth_skill_pass", False))
    impact_pass = bool(noninferiority.get("impact_skill_pass", False))
    snr = float((report.get("differentiation") or {}).get("signal_to_noise", math.nan))
    cancellation = cancellation_direction_against_baseline(
        ((report.get("stratified_cancellation") or {}).get("strata") or {}),
        baseline_strata,
    )
    cancellation_pass = bool(
        cancellation["valid_strata"] > 0
        and cancellation["joint_improvement_fraction"]
        > float(cancellation_improvement_fraction)
    )
    checks = {
        "depth_skill_noninferior": depth_pass,
        "impact_skill_noninferior": impact_pass,
        "scale_approaches_direct": lower <= ratio <= upper,
        "differentiation_exceeds_noise": (
            math.isfinite(snr) and snr >= float(differentiation_snr_threshold)
        ),
        "cancellation_moves_in_expected_direction": cancellation_pass,
    }
    return {
        "ladder_rung": rung,
        "prior_seed": int(seed),
        "pass": all(checks.values()),
        "checks": checks,
        "posterior_to_direct_scale_ratio": ratio,
        "differentiation_snr": snr,
        "cancellation_direction": cancellation,
        "thresholds": {
            "direct_scale_equivalence_interval": [lower, upper],
            "differentiation_snr_threshold": float(differentiation_snr_threshold),
            "cancellation_joint_improvement_fraction_strictly_above": float(
                cancellation_improvement_fraction
            ),
        },
    }


def assess_p2_pilot_seed(
    report: Mapping[str, Any],
    *,
    ood_evidence: Mapping[str, Any],
    baseline_ood_evidence: Mapping[str, Any],
    amplitude_tuned_on_evaluation_events: bool,
) -> dict[str, Any]:
    """Apply the stricter P2 association/OOD firewall to one prior seed."""

    if str(report.get("ladder_rung", "")).upper() != "P2":
        raise ValueError("P2 assessment requires ladder_rung='P2'.")
    seed = report.get("prior_seed")
    if seed is None:
        raise ValueError("P2 report must record prior_seed.")
    noninferiority = dict(report.get("noninferiority") or {})
    association = dict(report.get("epistemic_error_association") or {})
    association_keys = (
        "epistemic_std_abs_error_partial_depth_all_wettable",
        "epistemic_std_abs_error_partial_depth_ref_wet",
        "epistemic_std_abs_error_partial_depth_wet_front",
    )
    association_values = [float(association.get(key, math.nan)) for key in association_keys]
    association_pass = all(math.isfinite(value) and value > 0.0 for value in association_values)
    spearman = float(ood_evidence.get("spearman_epistemic_std_rmse", math.nan))
    leave_one_out = list(ood_evidence.get("leave_one_out") or [])
    loeo_pass = bool(
        leave_one_out
        and all(float(row.get("spearman", math.nan)) > 0.0 for row in leave_one_out)
    )
    pilot_gap = float(ood_evidence.get("mean_sparsification_gap_m", math.nan))
    baseline_gap = float(
        baseline_ood_evidence.get("mean_sparsification_gap_m", math.nan)
    )
    risk_pass = bool(
        math.isfinite(pilot_gap)
        and math.isfinite(baseline_gap)
        and pilot_gap < baseline_gap
    )
    checks = {
        "depth_skill_noninferior": bool(noninferiority.get("depth_skill_pass", False)),
        "impact_skill_noninferior": bool(noninferiority.get("impact_skill_pass", False)),
        "hydraulically_controlled_association_positive": association_pass,
        "ood_event_ranking_positive": math.isfinite(spearman) and spearman > 0.0,
        "ood_leave_one_event_out_stable": loeo_pass,
        "ood_risk_coverage_improved": risk_pass,
        "amplitude_not_tuned_on_evaluation_events": not bool(
            amplitude_tuned_on_evaluation_events
        ),
    }
    return {
        "ladder_rung": "P2",
        "prior_seed": int(seed),
        "pass": all(checks.values()),
        "checks": checks,
        "controlled_associations": dict(zip(association_keys, association_values)),
        "ood_spearman": spearman,
        "ood_min_loeo_spearman": (
            min(float(row["spearman"]) for row in leave_one_out)
            if leave_one_out
            else math.nan
        ),
        "pilot_sparsification_gap_m": pilot_gap,
        "baseline_sparsification_gap_m": baseline_gap,
    }


def aggregate_pilot_seed_decisions(
    seed_results: Sequence[Mapping[str, Any]], *, minimum_seeds: int = 3
) -> dict[str, Any]:
    """Require an arm to pass on every one of at least three unique seeds."""

    rows = [dict(row) for row in seed_results]
    if len(rows) < int(minimum_seeds):
        return {
            "decision": "pilot_provisional",
            "verdict_status": "under_replicated",
            "seed_count": len(rows),
            "minimum_seeds": int(minimum_seeds),
            "seed_results": rows,
        }
    seeds = [int(row["prior_seed"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("pilot acceptance requires unique prior seeds.")
    rungs = {str(row["ladder_rung"]).upper() for row in rows}
    if len(rungs) != 1:
        raise ValueError("pilot acceptance cannot mix ladder rungs.")
    passed = [bool(row.get("pass", False)) for row in rows]
    accepted = all(passed)
    return {
        "decision": "pilot_accepted" if accepted else "pilot_rejected",
        "verdict_status": (
            "acceptance_replicated" if accepted else "replicated_inconsistent"
        ),
        "ladder_rung": next(iter(rungs)),
        "seed_count": len(rows),
        "minimum_seeds": int(minimum_seeds),
        "passing_seed_count": sum(passed),
        "seed_results": rows,
    }


def write_checksummed_artifact(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Atomically write canonical JSON and its SHA-256 sidecar."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = destination.with_name(f"{destination.name}.tmp{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, destination)
    checksum = destination.with_suffix(destination.suffix + ".sha256")
    checksum.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return digest


def verify_checksummed_artifact(path: str | Path) -> str:
    """Verify one Phase-5 JSON artifact against its sidecar."""

    source = Path(path)
    checksum = source.with_suffix(source.suffix + ".sha256")
    if not source.is_file() or not checksum.is_file():
        raise ValueError(f"artifact or checksum is missing for {source}.")
    expected = checksum.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch for {source}: {actual} != {expected}.")
    return actual


def _load_verified_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    verify_checksummed_artifact(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Phase-5 artifact must contain a JSON object: {source}.")
    return payload


def verify_phase5_evidence_artifact(
    path: str | Path,
    *,
    expected_head: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Load evidence only when its signed provenance matches this analysis.

    Checksumming the result alone prevents mutation but not accidental mixing of
    outputs from different code revisions or predeclared protocols. Every GPU
    evidence directory therefore carries a separately checksummed provenance
    object, and mechanism gates fail closed on either mismatch.
    """

    source = Path(path)
    payload = _load_verified_json(source)
    provenance = _load_verified_json(source.parent / "PROVENANCE.json")
    actual_head = str(provenance.get("git_head", ""))
    if actual_head != str(expected_head):
        raise ValueError(
            f"Phase-5 evidence Git HEAD mismatch: {actual_head} != {expected_head}."
        )
    actual_protocol = str(provenance.get("protocol_sha256", ""))
    if actual_protocol != str(protocol_sha256):
        raise ValueError(
            "Phase-5 evidence protocol mismatch: "
            f"{actual_protocol} != {protocol_sha256}."
        )
    return payload


def verify_phase5_decision_artifact(
    path: str | Path,
    *,
    expected_head: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Load a decision artifact pinned to the current code and protocol."""

    payload = _load_verified_json(path)
    actual_head = str(payload.get("analysis_git_head", ""))
    if actual_head != str(expected_head):
        raise ValueError(
            f"Phase-5 decision Git HEAD mismatch: {actual_head} != {expected_head}."
        )
    actual_protocol = str(payload.get("protocol_sha256", ""))
    if actual_protocol != str(protocol_sha256):
        raise ValueError(
            "Phase-5 decision protocol mismatch: "
            f"{actual_protocol} != {protocol_sha256}."
        )
    return payload


def verify_phase5_ood_ranking_artifact(
    path: str | Path,
    *,
    expected_head: str,
    protocol_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Load OOD ranking evidence pinned to code, protocol, and Stage-2 fit."""

    payload = _load_verified_json(path)
    expected = {
        "analysis_git_head": str(expected_head),
        "protocol_sha256": str(protocol_sha256),
        "stage2_checkpoint_sha256": str(checkpoint_sha256),
    }
    for key, value in expected.items():
        actual = str(payload.get(key, ""))
        if actual != value:
            label = key.replace("_sha256", "").replace("_", " ")
            raise ValueError(
                f"Phase-5 OOD ranking {label} mismatch: {actual} != {value}."
            )
    return payload


def validate_pilot_rungs_for_gp1_decision(
    gp1_decision: str,
    pilot_rungs: Sequence[str],
) -> None:
    """Reject pilot evidence that does not follow the replicated GP1 branch."""

    decision = str(gp1_decision).strip().lower()
    rungs = {str(value).strip().upper() for value in pilot_rungs}
    if not rungs or "" in rungs:
        raise ValueError("pilot evidence must declare at least one ladder rung.")
    if decision == "contraction_confirmation":
        raise ValueError("GP1 contraction confirmation does not admit a pilot branch.")
    allowed = {
        "continuous_amortization_failure": {"P1B_A", "P1B_B", "P1B_C"},
        "indeterminate": {"P1B_A", "P1B_B", "P1B_C"},
        "shared_subspace_pathology": {"P2"},
    }.get(decision)
    if allowed is None:
        raise ValueError(f"unsupported replicated GP1 decision: {gp1_decision!r}.")
    if not rungs.issubset(allowed):
        raise ValueError(
            f"pilot rungs {sorted(rungs)} are incompatible with GP1 decision "
            f"{gp1_decision!r}; allowed={sorted(allowed)}."
        )


def _effective_rank(values: torch.Tensor, *, eps: float = 1.0e-15) -> float:
    values = values.detach().double().clamp_min(0.0)
    total = values.sum()
    if float(total) <= eps:
        return 0.0
    probabilities = values / total
    return float(torch.exp(-(probabilities * probabilities.clamp_min(eps).log()).sum()))


def _participation_ratio(values: torch.Tensor, *, eps: float = 1.0e-15) -> float:
    values = values.detach().double().clamp_min(0.0)
    denominator = values.square().sum()
    if float(denominator) <= eps:
        return 0.0
    return float(values.sum().square() / denominator)


def _off_diagonal_cosines(rows: torch.Tensor, *, eps: float = 1.0e-15) -> torch.Tensor:
    rows = rows.double()
    if rows.shape[0] < 2:
        return rows.new_empty((0,))
    normalized = rows / rows.norm(dim=1, keepdim=True).clamp_min(eps)
    matrix = normalized @ normalized.T
    mask = ~torch.eye(rows.shape[0], dtype=torch.bool, device=rows.device)
    return matrix[mask]


def gradient_row_geometry(
    rows: torch.Tensor,
    *,
    relative_rank_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Summarize a collection of gradient directions in common coordinates."""

    values = torch.as_tensor(rows).double()
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("gradient rows must be a finite [draws,parameters] tensor.")
    if float(relative_rank_tolerance) <= 0.0:
        raise ValueError("relative_rank_tolerance must be positive.")
    singular = torch.linalg.svdvals(values)
    tolerance = (
        float(singular.max()) * float(relative_rank_tolerance)
        if singular.numel()
        else 0.0
    )
    energy = singular.square()
    cosines = _off_diagonal_cosines(values)
    return {
        "numerical_rank": int((singular > tolerance).sum()),
        "effective_rank": _effective_rank(energy),
        "participation_ratio": _participation_ratio(energy),
        "singular_values": singular,
        "relative_rank_tolerance": float(relative_rank_tolerance),
        "rank_tolerance": tolerance,
        "row_norm_mean": float(values.norm(dim=1).mean()),
        "row_norm_min": float(values.norm(dim=1).min()),
        "row_norm_max": float(values.norm(dim=1).max()),
        "pairwise_cosine_mean": (
            float(cosines.mean()) if cosines.numel() else math.nan
        ),
        "pairwise_absolute_cosine_mean": (
            float(cosines.abs().mean()) if cosines.numel() else math.nan
        ),
    }


def particle_risk_differentiation(
    per_family_scores: torch.Tensor,
    family_particle_weights: torch.Tensor,
    *,
    minibatch_size: int,
    replicates: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare between-particle empirical-risk signal with minibatch noise.

    Scores are measured for every fit family at one common prediction state.
    This isolates bootstrap-risk differentiation from indexed priors and
    particle-specific predictions. The noise reference repeatedly estimates
    each particle's own weighted risk at the optimizer's family-batch size.
    """

    scores = torch.as_tensor(per_family_scores).detach().double().cpu().reshape(-1)
    weights = torch.as_tensor(family_particle_weights).detach().double().cpu()
    if weights.ndim != 2 or weights.shape[0] != scores.numel():
        raise ValueError("family_particle_weights must be [N_families,M].")
    if scores.numel() < 2 or weights.shape[1] < 2:
        raise ValueError("risk differentiation requires at least two families and particles.")
    if not torch.isfinite(scores).all() or not torch.isfinite(weights).all():
        raise ValueError("risk differentiation inputs must be finite.")
    if bool((weights <= 0).any()):
        raise ValueError("family-particle weights must be strictly positive.")
    batch_size = int(minibatch_size)
    replicates = int(replicates)
    if batch_size < 1 or replicates < 1:
        raise ValueError("minibatch_size and replicates must be positive.")

    full_risk = (weights * scores[:, None]).sum(dim=0) / weights.sum(dim=0)
    signal = (full_risk - full_risk.mean()).square().mean().sqrt()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        scores.numel(), (replicates, batch_size), generator=generator
    )
    sampled_scores = scores[indices, None]
    sampled_weights = weights[indices]
    minibatch_risk = (sampled_weights * sampled_scores).sum(dim=1) / sampled_weights.sum(
        dim=1
    )
    noise = (minibatch_risk - full_risk.unsqueeze(0)).square().mean().sqrt()
    snr = signal / noise.clamp_min(1.0e-15)
    return {
        "particle_weighted_risks": [float(value) for value in full_risk],
        "between_particle_risk_rms": float(signal),
        "minibatch_noise_rms": float(noise),
        "signal_to_noise": float(snr),
        "minibatch_size": batch_size,
        "replicates": replicates,
        "seed": int(seed),
        "prediction_state": "common_uniform_anchor",
    }


@dataclass(frozen=True)
class CommonAnchorGradientGeometry:
    """Geometry induced by bootstrap weights at one shared uniform anchor."""

    delta_gradients: torch.Tensor
    gradient_effective_rank: float
    gradient_participation_ratio: float
    pairwise_cosine_mean: float
    pairwise_absolute_cosine_mean: float
    gradient_signal_to_noise: float | None
    hessian: dict[str, float]
    displacements: dict[float, torch.Tensor]
    functional_displacement_rms: dict[float, float]


def common_anchor_gradient_geometry(
    per_family_gradients: torch.Tensor,
    particle_weights: torch.Tensor,
    *,
    minibatch_noise_gradients: torch.Tensor | None = None,
    hessian: torch.Tensor | None = None,
    validation_jacobian: torch.Tensor | None = None,
    damping: Sequence[float] = (1.0e-6, 1.0e-4, 1.0e-2),
) -> CommonAnchorGradientGeometry:
    r"""Diagnose weight-induced optima at a common uniform anchor.

    ``per_family_gradients[i]`` is :math:`g_i(\hat\beta_0)`. The returned
    rows are exactly ``mean_i[(w_i(u)-1) g_i]``; no indexed-model state or
    prior gradient is mixed into this attribution.
    """

    gradients = torch.as_tensor(per_family_gradients).double()
    weights = torch.as_tensor(particle_weights).double()
    if gradients.ndim != 2 or weights.ndim != 2:
        raise ValueError("gradients and weights must be [N,P] and [N,M].")
    if gradients.shape[0] != weights.shape[0]:
        raise ValueError("gradient and weight family dimensions differ.")
    if not torch.isfinite(gradients).all() or not torch.isfinite(weights).all():
        raise ValueError("gradient geometry inputs must be finite.")

    delta = (weights - 1.0).T @ gradients / float(gradients.shape[0])
    singular = torch.linalg.svdvals(delta)
    energy = singular.square()
    cosines = _off_diagonal_cosines(delta)
    cosine_mean = float(cosines.mean()) if cosines.numel() else math.nan
    abs_cosine_mean = float(cosines.abs().mean()) if cosines.numel() else math.nan

    signal_to_noise = None
    if minibatch_noise_gradients is not None:
        noise = torch.as_tensor(minibatch_noise_gradients).double()
        if noise.ndim != 2 or noise.shape[1] != gradients.shape[1]:
            raise ValueError("minibatch_noise_gradients must be [Q,P].")
        signal = delta.norm(dim=1).square().mean().sqrt()
        noise_floor = noise.norm(dim=1).square().mean().sqrt()
        signal_to_noise = float(signal / noise_floor.clamp_min(1.0e-15))

    hessian_report: dict[str, float] = {}
    displacements: dict[float, torch.Tensor] = {}
    functional_rms: dict[float, float] = {}
    if hessian is not None:
        H = torch.as_tensor(hessian).double()
        if H.shape != (gradients.shape[1], gradients.shape[1]):
            raise ValueError("hessian must be square with the gradient parameter dimension.")
        H = 0.5 * (H + H.T)
        eigenvalues = torch.linalg.eigvalsh(H)
        positive = eigenvalues[eigenvalues > 1.0e-12]
        condition = (
            float(positive.max() / positive.min()) if positive.numel() else math.inf
        )
        hessian_report = {
            "minimum_eigenvalue": float(eigenvalues.min()),
            "maximum_eigenvalue": float(eigenvalues.max()),
            "negative_eigenvalue_fraction": float((eigenvalues < -1.0e-12).double().mean()),
            "near_zero_eigenvalue_fraction": float((eigenvalues.abs() <= 1.0e-12).double().mean()),
            "positive_subspace_condition_number": condition,
            "spectral_effective_rank": _effective_rank(eigenvalues.clamp_min(0.0)),
        }
        identity = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
        jacobian = None
        if validation_jacobian is not None:
            jacobian = torch.as_tensor(validation_jacobian).double()
            if jacobian.ndim != 2 or jacobian.shape[1] != H.shape[0]:
                raise ValueError("validation_jacobian must be [outputs,P].")
        for raw_damping in damping:
            value = float(raw_damping)
            if value < 0.0:
                raise ValueError("damping values must be nonnegative.")
            try:
                displacement = -torch.linalg.solve(H + value * identity, delta.T).T
            except torch.linalg.LinAlgError:
                displacement = -(torch.linalg.pinv(H + value * identity) @ delta.T).T
            displacements[value] = displacement
            if jacobian is not None:
                functional = displacement @ jacobian.T
                functional_rms[value] = float(functional.square().mean().sqrt())
            else:
                functional_rms[value] = math.nan
        finite_rms = [value for value in functional_rms.values() if math.isfinite(value)]
        if finite_rms:
            hessian_report["functional_damping_sensitivity_ratio"] = max(finite_rms) / max(
                min(finite_rms), 1.0e-15
            )

    return CommonAnchorGradientGeometry(
        delta_gradients=delta,
        gradient_effective_rank=_effective_rank(energy),
        gradient_participation_ratio=_participation_ratio(energy),
        pairwise_cosine_mean=cosine_mean,
        pairwise_absolute_cosine_mean=abs_cosine_mean,
        gradient_signal_to_noise=signal_to_noise,
        hessian=hessian_report,
        displacements=displacements,
        functional_displacement_rms=functional_rms,
    )


def raw_probit_weight_rank(
    raw_weights: torch.Tensor,
    *,
    epistemic_indices: torch.Tensor | None = None,
    eps: float = 1.0e-12,
    relative_rank_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Rank diagnostics after exactly inverting raw probit-exponential weights.

    The input must be raw, positive, unclipped, untempered, and unnormalized.
    Applying this inverse to production-normalized weights has no rank meaning.
    """

    weights = torch.as_tensor(raw_weights).double()
    if weights.ndim != 2 or not torch.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("raw_weights must be a finite positive [families,draws] matrix.")
    u = (1.0 - torch.exp(-weights)).clamp(float(eps), 1.0 - float(eps))
    logits = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    singular = torch.linalg.svdvals(logits)
    if float(relative_rank_tolerance) <= 0.0:
        raise ValueError("relative_rank_tolerance must be positive.")
    tolerance = (
        float(singular.max()) * float(relative_rank_tolerance)
        if singular.numel()
        else 0.0
    )
    energy = singular.square()
    centered = logits - logits.mean(dim=1, keepdim=True)
    normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(float(eps))
    correlation = normalized @ normalized.T
    off_diagonal = correlation[
        ~torch.eye(correlation.shape[0], dtype=torch.bool, device=correlation.device)
    ]
    result: dict[str, Any] = {
        "numerical_rank": int((singular > tolerance).sum()),
        "participation_ratio": _participation_ratio(energy),
        "effective_rank": _effective_rank(energy),
        "singular_values": singular,
        "inverse_logits": logits,
        "tolerance": tolerance,
        "relative_rank_tolerance": float(relative_rank_tolerance),
        "off_diagonal_family_logit_correlation_mean": (
            float(off_diagonal.mean()) if off_diagonal.numel() else math.nan
        ),
        "off_diagonal_family_logit_abs_correlation_mean": (
            float(off_diagonal.abs().mean()) if off_diagonal.numel() else math.nan
        ),
    }
    if epistemic_indices is not None:
        indices = torch.as_tensor(epistemic_indices).double()
        if indices.ndim != 2 or indices.shape[0] != logits.shape[1]:
            raise ValueError(
                "epistemic_indices must be [draws,d] aligned with raw-weight columns."
            )
        # logits.T = Z B.T, so B is the exact Jacobian of family logits with
        # respect to the Gaussian epistemic index under the raw law.
        solution = torch.linalg.lstsq(indices, logits.T).solution
        jacobian = solution.T
        jacobian_singular = torch.linalg.svdvals(jacobian)
        jacobian_tolerance = (
            float(jacobian_singular.max()) * float(relative_rank_tolerance)
            if jacobian_singular.numel()
            else 0.0
        )
        reconstructed = indices @ solution
        relative_error = (reconstructed - logits.T).norm() / logits.T.norm().clamp_min(
            float(eps)
        )
        jacobian_energy = jacobian_singular.square()
        result.update(
            {
                "jacobian": jacobian,
                "jacobian_singular_values": jacobian_singular,
                "jacobian_numerical_rank": int(
                    (jacobian_singular > jacobian_tolerance).sum()
                ),
                "jacobian_participation_ratio": _participation_ratio(jacobian_energy),
                "jacobian_effective_rank": _effective_rank(jacobian_energy),
                "jacobian_relative_reconstruction_error": float(relative_error),
                "jacobian_tolerance": jacobian_tolerance,
                "expected_random_direction_abs_correlation": math.sqrt(
                    2.0 / (math.pi * float(indices.shape[1]))
                ),
            }
        )
    return result


def hermite_basis_covariance(
    epistemic_dim: int,
    quadratic_vectors: torch.Tensor,
    *,
    linear_terms: bool = True,
) -> torch.Tensor:
    """Analytic covariance of linear and projected quadratic Hermite terms."""

    q = torch.as_tensor(quadratic_vectors).double()
    d = int(epistemic_dim)
    if q.ndim != 2 or q.shape[1] != d:
        raise ValueError("quadratic_vectors must be [J,epistemic_dim].")
    if q.numel():
        norms = q.norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=1.0e-5, atol=1.0e-6):
            raise ValueError("quadratic Hermite vectors must be unit normalized.")
    quadratic = 2.0 * (q @ q.T).square()
    if not linear_terms:
        return quadratic
    covariance = q.new_zeros((d + q.shape[0], d + q.shape[0]))
    covariance[:d, :d] = torch.eye(d, dtype=q.dtype, device=q.device)
    covariance[d:, d:] = quadratic
    return covariance


def whitened_cancellation_diagnostics(
    trainable_correction: torch.Tensor,
    scaled_prior_correction: torch.Tensor,
    basis_values: torch.Tensor,
    basis_covariance: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    """Coordinate-free cancellation diagnostics in covariance-whitened modes."""

    train_model, prior_model, basis_summary = _whitened_modeled_fields(
        trainable_correction,
        scaled_prior_correction,
        basis_values,
        basis_covariance,
        eps=eps,
    )
    result = _cancellation_from_modeled_fields(train_model, prior_model, eps=eps)
    result.update(basis_summary)
    return result


def _whitened_modeled_fields(
    trainable_correction: torch.Tensor,
    scaled_prior_correction: torch.Tensor,
    basis_values: torch.Tensor,
    basis_covariance: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    train = torch.as_tensor(trainable_correction).double()
    prior = torch.as_tensor(scaled_prior_correction).double()
    psi = torch.as_tensor(basis_values).double()
    covariance = torch.as_tensor(basis_covariance).double()
    if train.shape != prior.shape or train.ndim < 2:
        raise ValueError("trainable and prior corrections must share [M,...] shape.")
    if psi.ndim != 2 or psi.shape[0] != train.shape[0]:
        raise ValueError("basis_values must be [M,D].")
    if covariance.shape != (psi.shape[1], psi.shape[1]):
        raise ValueError("basis_covariance must be [D,D].")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    # Whiten only numerically identifiable covariance modes. An absolute-only
    # cutoff can amplify near-null Hermite directions by many orders of
    # magnitude and turn Monte Carlo noise into apparent cancellation.
    largest = eigenvalues.clamp_min(0.0).max()
    threshold = max(float(eps), float(largest) * 1.0e-8)
    keep = eigenvalues > threshold
    if not bool(keep.any()):
        raise ValueError("basis covariance has no positive modes.")
    transform = eigenvectors[:, keep] / eigenvalues[keep].sqrt().unsqueeze(0)
    white = psi @ transform
    train_flat = train.reshape(train.shape[0], -1)
    prior_flat = prior.reshape(prior.shape[0], -1)
    # Regressing both fields on the same orthogonal basis removes Monte Carlo
    # components outside the declared epistemic function class.
    pinv = torch.linalg.pinv(white)
    train_model = white @ (pinv @ train_flat)
    prior_model = white @ (pinv @ prior_flat)
    return train_model.reshape_as(train), prior_model.reshape_as(prior), {
        "basis_positive_modes": float(keep.sum()),
        "basis_total_modes": float(eigenvalues.numel()),
        "basis_discarded_modes": float((~keep).sum()),
        "basis_eigenvalue_threshold": threshold,
        "basis_condition_number": float(eigenvalues[keep].max() / eigenvalues[keep].min()),
        "basis_retained_variance_fraction": float(
            eigenvalues[keep].sum() / eigenvalues.clamp_min(0.0).sum().clamp_min(float(eps))
        ),
    }


def _cancellation_from_modeled_fields(
    train_model: torch.Tensor,
    prior_model: torch.Tensor,
    *,
    location_weights: torch.Tensor | None = None,
    eps: float,
) -> dict[str, float]:
    train_centered = train_model - train_model.mean(dim=0, keepdim=True)
    prior_centered = prior_model - prior_model.mean(dim=0, keepdim=True)
    total_centered = train_centered + prior_centered
    if location_weights is None:
        def reduce(value: torch.Tensor) -> torch.Tensor:
            return value.mean()
    else:
        weights = torch.as_tensor(location_weights).double()
        if weights.shape != train_centered.shape[1:]:
            raise ValueError("location_weights must match the correction field without M.")
        if not torch.isfinite(weights).all() or bool((weights < 0).any()):
            raise ValueError("location_weights must be finite and nonnegative.")
        denominator = float(train_centered.shape[0]) * weights.sum().clamp_min(float(eps))

        def reduce(value: torch.Tensor) -> torch.Tensor:
            return (value * weights.unsqueeze(0)).sum() / denominator

    train_var = reduce(train_centered.square())
    prior_var = reduce(prior_centered.square())
    retained_var = reduce(total_centered.square())
    cross = reduce(train_centered * prior_centered)
    cosine = cross / (train_var.sqrt() * prior_var.sqrt()).clamp_min(float(eps))
    slope = cross / prior_var.clamp_min(float(eps))
    residual = train_centered - slope * prior_centered
    return {
        "train_prior_cosine": float(cosine),
        "regression_slope": float(slope),
        "prior_variance": float(prior_var),
        "trainable_variance": float(train_var),
        "retained_variance": float(retained_var),
        "residual_variance_after_linear_cancellation": float(reduce(residual.square())),
    }


def stratified_whitened_cancellation(
    trainable_correction: torch.Tensor,
    scaled_prior_correction: torch.Tensor,
    basis_values: torch.Tensor,
    basis_covariance: torch.Tensor,
    *,
    reference_depth: torch.Tensor,
    wettable_mask: torch.Tensor,
    lead_bins: Mapping[str, tuple[int, int]],
    score_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Report coordinate-free cancellation by hydraulic regime and lead bin."""

    train = torch.as_tensor(trainable_correction).double()
    prior = torch.as_tensor(scaled_prior_correction).double()
    if train.shape != prior.shape or train.ndim not in {3, 4}:
        raise ValueError("corrections must share [M,T,Nv] or [M,T,Nv,C] shape.")
    if train.ndim == 4:
        if train.shape[-1] != 1:
            raise ValueError("stratified cancellation currently requires depth-only output.")
        train = train[..., 0]
        prior = prior[..., 0]
    depth = torch.as_tensor(reference_depth).double()
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape != train.shape[1:]:
        raise ValueError("reference_depth must match [T,Nv] correction fields.")
    wettable = torch.as_tensor(wettable_mask, dtype=torch.bool).reshape(-1)
    if wettable.shape[0] != train.shape[2]:
        raise ValueError("wettable_mask must contain one value per mesh cell.")
    if score_weights is None:
        spatial_weights = torch.ones_like(depth)
    else:
        spatial_weights = torch.as_tensor(score_weights).double()
        if spatial_weights.ndim == 3 and spatial_weights.shape[-1] == 1:
            spatial_weights = spatial_weights[..., 0]
        if spatial_weights.shape != depth.shape:
            raise ValueError("score_weights must match [T,Nv] or [T,Nv,1].")
        if not torch.isfinite(spatial_weights).all() or bool((spatial_weights < 0).any()):
            raise ValueError("score_weights must be finite and nonnegative.")

    train_model, prior_model, basis_summary = _whitened_modeled_fields(
        train,
        prior,
        basis_values,
        basis_covariance,
        eps=1.0e-12,
    )
    regimes = {
        "all_wettable": wettable.unsqueeze(0).expand_as(depth),
        "wet_gt_0p01": wettable.unsqueeze(0).expand_as(depth) & (depth > 0.01),
        "front_0p01_0p10": (
            wettable.unsqueeze(0).expand_as(depth)
            & (depth >= 0.01)
            & (depth <= 0.10)
        ),
    }
    strata: dict[str, dict[str, float]] = {}
    for lead_name, (start, stop) in lead_bins.items():
        start, stop = int(start), int(stop)
        if start < 0 or stop > train.shape[1] or start >= stop:
            raise ValueError(f"invalid lead bin {lead_name!r}: {(start, stop)}")
        lead_mask = torch.zeros_like(depth, dtype=torch.bool)
        lead_mask[start:stop] = True
        for regime_name, regime_mask in regimes.items():
            mask = lead_mask & regime_mask
            if int(mask.sum()) < 1:
                continue
            diagnostic = _cancellation_from_modeled_fields(
                train_model[:, mask],
                prior_model[:, mask],
                location_weights=spatial_weights[mask],
                eps=1.0e-12,
            )
            diagnostic.update(basis_summary)
            diagnostic["n_locations"] = float(mask.sum())
            diagnostic["location_weight_sum"] = float(spatial_weights[mask].sum())
            strata[f"{regime_name}__{lead_name}"] = diagnostic

    per_lead_slope = []
    lead_index = []
    for t in range(train.shape[1]):
        mask = regimes["all_wettable"][t]
        if int(mask.sum()) < 1:
            continue
        diagnostic = _cancellation_from_modeled_fields(
            train_model[:, t, mask],
            prior_model[:, t, mask],
            location_weights=spatial_weights[t, mask],
            eps=1.0e-12,
        )
        per_lead_slope.append(diagnostic["regression_slope"])
        lead_index.append(float(t))
    if len(lead_index) >= 2:
        x = torch.tensor(lead_index, dtype=torch.float64)
        y = torch.tensor(per_lead_slope, dtype=torch.float64)
        x = x - x.mean()
        cancellation_vs_lead = float((x * (y - y.mean())).sum() / x.square().sum())
    else:
        cancellation_vs_lead = math.nan
    return {
        "strata": strata,
        "per_lead_regression_slope": per_lead_slope,
        "cancellation_regression_slope_per_lead_step": cancellation_vs_lead,
    }


def bootstrap_scale_interval(
    variance_contributions: torch.Tensor,
    *,
    family_weights: torch.Tensor | None = None,
    replicates: int = 2_000,
    seed: int = 0,
) -> dict[str, float]:
    r"""Estimate the pinned scale ``sqrt(sum w V / sum w)`` with a two-way CI.

    Rows are families and columns are epistemic draws. Bootstrap replicates
    resample both axes; mesh cells are never treated as independent units.
    """

    variance = torch.as_tensor(variance_contributions).double()
    if variance.ndim != 2 or not torch.isfinite(variance).all() or bool((variance < 0).any()):
        raise ValueError("variance_contributions must be finite nonnegative [families,draws].")
    if int(replicates) < 1:
        raise ValueError("replicates must be >= 1.")
    if family_weights is None:
        weights = torch.ones(variance.shape[0], dtype=torch.float64)
    else:
        weights = torch.as_tensor(family_weights).double().reshape(-1)
    if weights.shape[0] != variance.shape[0] or bool((weights < 0).any()):
        raise ValueError("family_weights must be nonnegative with one value per family.")

    def scale(values: torch.Tensor, selected_weights: torch.Tensor) -> float:
        weighted = values.mean(dim=1) * selected_weights
        return float((weighted.sum() / selected_weights.sum().clamp_min(1.0e-15)).sqrt())

    estimate = scale(variance, weights)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    samples = []
    for _ in range(int(replicates)):
        family_index = torch.randint(variance.shape[0], (variance.shape[0],), generator=generator)
        draw_index = torch.randint(variance.shape[1], (variance.shape[1],), generator=generator)
        sampled = variance.index_select(0, family_index).index_select(1, draw_index)
        samples.append(scale(sampled, weights.index_select(0, family_index)))
    ordered = torch.tensor(samples, dtype=torch.float64).sort().values
    lo = float(torch.quantile(ordered, 0.025))
    hi = float(torch.quantile(ordered, 0.975))
    return {"estimate": estimate, "ci95_lower": lo, "ci95_upper": hi}


def _scale_ratio_interval(
    numerator: Mapping[str, float],
    denominator: Mapping[str, float],
    *,
    floor: float,
) -> tuple[float, float, float] | None:
    required = ("estimate", "ci95_lower", "ci95_upper")
    if any(key not in numerator or key not in denominator for key in required):
        raise ValueError("scale summaries require estimate, ci95_lower, and ci95_upper.")
    if float(numerator["estimate"]) <= floor or float(denominator["estimate"]) <= floor:
        return None
    denominator_lower = float(denominator["ci95_lower"])
    if denominator_lower <= floor:
        return None
    ratio = float(numerator["estimate"]) / float(denominator["estimate"])
    lower = max(float(numerator["ci95_lower"]), floor) / float(denominator["ci95_upper"])
    upper = float(numerator["ci95_upper"]) / denominator_lower
    return ratio, lower, upper


def evaluate_gd0(
    *,
    observed_scale_m: Mapping[str, float],
    direct_scale_m: Mapping[str, float],
    differentiation_snr: float,
    functional_displacement_m: float,
    noise_snr_threshold: float = 1.0,
    meaningful_displacement_m: float = 1.0e-3,
    equivalence_factor: float = 2.0,
    scale_floor_m: float = 1.0e-8,
) -> dict[str, Any]:
    """Apply the predeclared three-way GD0 diagnostic gate."""

    ratio_interval_method = "conservative_ratio_of_marginal_95pct_endpoints"
    interval = _scale_ratio_interval(observed_scale_m, direct_scale_m, floor=scale_floor_m)
    if interval is None:
        return {
            "decision": "indeterminate",
            "near_zero_scale_guard": True,
            "ratio": None,
            "ratio_interval_method": ratio_interval_method,
            "thresholds": {
                "equivalence_factor": float(equivalence_factor),
                "noise_snr_threshold": float(noise_snr_threshold),
                "meaningful_displacement_m": float(meaningful_displacement_m),
                "scale_floor_m": float(scale_floor_m),
            },
        }
    ratio, lower, upper = interval
    log_ratio = math.log(ratio)
    log_lower = math.log(max(lower, scale_floor_m))
    log_upper = math.log(max(upper, scale_floor_m))
    bound = math.log(float(equivalence_factor))
    weak = float(differentiation_snr) < float(noise_snr_threshold)
    meaningful = float(functional_displacement_m) >= float(meaningful_displacement_m)
    equivalent = log_lower >= -bound and log_upper <= bound
    under_delivered = upper < 0.5 and not weak and meaningful
    if equivalent and weak:
        decision = "contraction_consistent"
    elif under_delivered:
        decision = "under_delivery"
    else:
        decision = "indeterminate"
    return {
        "decision": decision,
        "near_zero_scale_guard": False,
        "ratio": ratio,
        "ratio_interval_method": ratio_interval_method,
        "ratio_ci95_lower": lower,
        "ratio_ci95_upper": upper,
        "log_ratio": log_ratio,
        "log_ratio_ci95_lower": log_lower,
        "log_ratio_ci95_upper": log_upper,
        "checks": {
            "scale_equivalent_within_factor_two": equivalent,
            "differentiation_below_noise": weak,
            "functional_displacement_meaningful": meaningful,
            "ratio_ucb_below_half": upper < 0.5,
        },
        "thresholds": {
            "equivalence_factor": float(equivalence_factor),
            "noise_snr_threshold": float(noise_snr_threshold),
            "meaningful_displacement_m": float(meaningful_displacement_m),
            "scale_floor_m": float(scale_floor_m),
        },
    }


def evaluate_gp1(
    *,
    p1a_to_direct_ratio: float,
    p1a_to_continuous_ratio: float,
    differentiation_snr: float,
    skill_noninferior: bool,
    impact_noninferior: bool,
    direct_to_p1a_ratio: float,
    functional_displacement_m: float,
    independent_prior_cancellation: float,
    skill_without_cancellation: bool,
    equivalence_lower: float = 0.5,
    equivalence_upper: float = 2.0,
    substantial_retention_ratio: float = 2.0,
    noise_snr_threshold: float = 1.0,
    meaningful_displacement_m: float = 1.0e-3,
    cancellation_threshold: float = 0.8,
) -> dict[str, Any]:
    """Operationalize the hypothesis-dependent GP1 mechanism verdict."""

    approaches_direct = equivalence_lower <= p1a_to_direct_ratio <= equivalence_upper
    weak = differentiation_snr < noise_snr_threshold
    exceeds_noise = differentiation_snr >= noise_snr_threshold
    common_skill = bool(skill_noninferior and impact_noninferior)
    confirmation = approaches_direct and weak and common_skill
    repair = (
        approaches_direct
        and p1a_to_continuous_ratio >= substantial_retention_ratio
        and exceeds_noise
        and common_skill
    )
    shared = (
        direct_to_p1a_ratio >= substantial_retention_ratio
        and functional_displacement_m >= meaningful_displacement_m
        and independent_prior_cancellation >= cancellation_threshold
        and bool(skill_without_cancellation)
    )
    if shared:
        decision = "shared_subspace_pathology"
    elif repair:
        decision = "continuous_amortization_failure"
    elif confirmation:
        decision = "contraction_confirmation"
    else:
        decision = "indeterminate"
    return {
        "decision": decision,
        "checks": {
            "p1a_approaches_direct": approaches_direct,
            "p1a_substantially_exceeds_continuous": (
                p1a_to_continuous_ratio >= substantial_retention_ratio
            ),
            "differentiation_below_noise": weak,
            "differentiation_exceeds_noise": exceeds_noise,
            "skill_and_impact_noninferior": common_skill,
            "direct_materially_above_p1a": (
                direct_to_p1a_ratio >= substantial_retention_ratio
            ),
            "functional_displacement_meaningful": (
                functional_displacement_m >= meaningful_displacement_m
            ),
            "independent_particles_cancel_prior": (
                independent_prior_cancellation >= cancellation_threshold
            ),
            "skill_does_not_require_cancellation": bool(skill_without_cancellation),
        },
        "thresholds": {
            "equivalence_lower": float(equivalence_lower),
            "equivalence_upper": float(equivalence_upper),
            "substantial_retention_ratio": float(substantial_retention_ratio),
            "noise_snr_threshold": float(noise_snr_threshold),
            "meaningful_displacement_m": float(meaningful_displacement_m),
            "cancellation_threshold": float(cancellation_threshold),
        },
    }
