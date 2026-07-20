#!/usr/bin/env python3
"""Emit checksummed GD0 and GP1 decision artifacts from completed Phase-5 runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from neuralop.flood.eval.neon_phase5 import (
    evaluate_gd0,
    evaluate_gp1,
    phase5_predeclared_protocol,
    verify_checksummed_artifact,
    verify_phase5_evidence_artifact,
    write_checksummed_artifact,
)
from neon_phase5_runtime import require_clean_repository


def _load(
    path: str | Path, *, expected_head: str, protocol_sha256: str
) -> dict[str, Any]:
    return verify_phase5_evidence_artifact(
        path, expected_head=expected_head, protocol_sha256=protocol_sha256
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("gd0", "gp1"), required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--geometry")
    parser.add_argument("--cancellation")
    parser.add_argument("--direct-data")
    parser.add_argument("--direct-rpf-last")
    parser.add_argument("--direct-rpf-full")
    parser.add_argument("--p1a", nargs="+")
    parser.add_argument("--dirichlet-direct", nargs="+")
    return parser.parse_args()


def _require_direct_design(
    report: dict[str, Any], *, mode: str, law: str, head: str
) -> None:
    actual = (str(report.get("mode")), str(report.get("law")), str(report.get("head")))
    expected = (str(mode), str(law), str(head))
    if actual != expected:
        raise ValueError(f"direct comparator mismatch: got {actual}, expected {expected}.")
    if not bool(report.get("optimization_valid", False)):
        raise ValueError(f"direct comparator {actual} failed its optimization-validity gate.")


def _write_decision(output: Path, payload: dict[str, Any]) -> None:
    write_checksummed_artifact(output / "DECISION.json", payload)
    lines = [
        f"gate={payload['gate']}",
        f"decision={payload['decision']}",
        f"mandatory_next={payload['mandatory_next']}",
        f"protocol_sha256={payload['protocol_sha256']}",
    ]
    (output / "DECISION.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    head = require_clean_repository(expected_head=args.expected_head)
    protocol_sha = verify_checksummed_artifact(args.protocol)
    protocol = phase5_predeclared_protocol()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.gate == "gd0":
        if not all(
            (
                args.geometry,
                args.cancellation,
                args.direct_data,
                args.direct_rpf_last,
                args.direct_rpf_full,
            )
        ):
            raise ValueError(
                "GD0 requires geometry, cancellation, data-direct, RPF-last, and "
                "RPF-full results."
            )
        load = lambda path: _load(
            path, expected_head=head, protocol_sha256=protocol_sha
        )
        geometry = load(args.geometry)
        cancellation = load(args.cancellation)
        direct_data = load(args.direct_data)
        direct = load(args.direct_rpf_last)
        direct_full = load(args.direct_rpf_full)
        _require_direct_design(direct_data, mode="data", law="probit", head="last")
        _require_direct_design(direct, mode="rpf", law="probit", head="last")
        _require_direct_design(direct_full, mode="rpf", law="probit", head="full")
        result = evaluate_gd0(
            observed_scale_m=cancellation["observed_posterior_scale_m"],
            direct_scale_m=direct_data["direct_scale_m"],
            differentiation_snr=float(
                geometry["gradient_geometry"]["signal_to_minibatch_noise"]
            ),
            functional_displacement_m=float(
                direct_data["weight_induced_displacement_rms_m"]["estimate"]
            ),
            **protocol["gd0"],
        )
        last_scale = direct["direct_scale_m"]
        full_scale = direct_full["direct_scale_m"]
        head_ratio = float(full_scale["estimate"]) / max(
            float(last_scale["estimate"]), 1.0e-15
        )
        factor = float(protocol["direct"]["head_sensitivity_factor"])
        intervals_overlap = not (
            float(full_scale["ci95_lower"]) > float(last_scale["ci95_upper"])
            or float(last_scale["ci95_lower"]) > float(full_scale["ci95_upper"])
        )
        head_contradiction = bool(
            not intervals_overlap and not (1.0 / factor <= head_ratio <= factor)
        )
        # The full-head comparison probes RPF representability. It cannot
        # alter GD0, whose estimand is the alpha=0 data-bootstrap optimum.
        result["head_sensitivity_guard_triggered"] = False
        result["direct_head_sensitivity"] = {
            "full_to_last_scale_ratio": head_ratio,
            "ci95_intervals_overlap": intervals_overlap,
            "contradiction": head_contradiction,
            "factor": factor,
            "full_head_is_qualitative_only": True,
            "changes_gd0_decision": False,
        }
        result["functional_displacement_source"] = (
            "direct_data_weighted_reoptimization"
        )
        result["gradient_geometry_role"] = (
            "differentiation_screen_only; curvature-proxy displacement is non-gating"
        )
        payload = {
            "schema_version": "neon_phase5_gd0_v1",
            "gate": "GD0",
            "analysis_git_head": head,
            **result,
            "mandatory_next": "P1a_persistent_dirichlet_particles",
            "protocol_sha256": protocol_sha,
            "inputs": {
                "geometry": str(args.geometry),
                "cancellation": str(args.cancellation),
                "direct_data": str(args.direct_data),
                "direct_rpf_last": str(args.direct_rpf_last),
                "direct_rpf_full": str(args.direct_rpf_full),
            },
            "data_bootstrap_scale_m": direct_data["direct_scale_m"],
            "rpf_last_scale_m_descriptive": direct["direct_scale_m"],
            "rpf_full_scale_m_descriptive": direct_full["direct_scale_m"],
        }
    else:
        if not all((args.geometry, args.cancellation, args.p1a, args.dirichlet_direct)):
            raise ValueError(
                "GP1 requires geometry, continuous cancellation, P1a, and Dirichlet direct results."
            )
        load = lambda path: _load(
            path, expected_head=head, protocol_sha256=protocol_sha
        )
        geometry = load(args.geometry)
        continuous = load(args.cancellation)
        p1a_reports = [load(path) for path in args.p1a]
        direct_reports = [load(path) for path in args.dirichlet_direct]
        if len(p1a_reports) != len(direct_reports):
            raise ValueError("GP1 requires one direct comparator for every P1a support seed.")
        if len({int(report.get("dirichlet_particle_seed", -1)) for report in p1a_reports}) != len(p1a_reports):
            raise ValueError("GP1 P1a support seeds must be unique and explicitly recorded.")
        continuous_scale = float(
            continuous["observed_posterior_scale_m"]["estimate"]
        )
        seed_results = []
        for p1a, direct in zip(p1a_reports, direct_reports):
            _require_direct_design(direct, mode="rpf", law="dirichlet", head="last")
            if str(p1a.get("sampling_design")) not in {
                "fixed_persistent_dirichlet_support",
                "fixed_epistemic_support_common_random_numbers",
            }:
                raise ValueError("GP1 requires persistent-Dirichlet P1a evaluations.")
            if int(p1a["dirichlet_particle_seed"]) != int(
                direct.get("dirichlet_particle_seed", -1)
            ):
                raise ValueError("GP1 paired P1a/direct support seeds do not match.")
            p1a_scale = float(p1a["posterior_scale_m"]["estimate"])
            direct_scale = float(direct["direct_scale_m"]["estimate"])
            result = evaluate_gp1(
                p1a_to_direct_ratio=p1a_scale / max(direct_scale, 1e-15),
                p1a_to_continuous_ratio=p1a_scale / max(continuous_scale, 1e-15),
                differentiation_snr=float(p1a["differentiation"]["signal_to_noise"]),
                skill_noninferior=bool(p1a["noninferiority"]["depth_skill_pass"]),
                impact_noninferior=bool(p1a["noninferiority"]["impact_skill_pass"]),
                direct_to_p1a_ratio=direct_scale / max(p1a_scale, 1e-15),
                functional_displacement_m=float(
                    direct["weight_induced_displacement_rms_m"]["estimate"]
                ),
                independent_prior_cancellation=float(p1a["cancellation_fraction"]),
                skill_without_cancellation=bool(p1a["skill_without_cancellation"]),
                equivalence_lower=float(protocol["gp1"]["equivalence_interval"][0]),
                equivalence_upper=float(protocol["gp1"]["equivalence_interval"][1]),
                substantial_retention_ratio=float(
                    protocol["gp1"]["substantial_retention_ratio"]
                ),
                noise_snr_threshold=float(protocol["gd0"]["noise_snr_threshold"]),
                meaningful_displacement_m=float(
                    protocol["gd0"]["meaningful_displacement_m"]
                ),
                cancellation_threshold=float(protocol["gp1"]["cancellation_threshold"]),
            )
            seed_results.append(
                {
                    "dirichlet_particle_seed": int(p1a["dirichlet_particle_seed"]),
                    "functional_displacement_source": (
                        "support_matched_dirichlet_direct_reoptimization"
                    ),
                    "gradient_geometry_functional_displacement_m_descriptive": float(
                        p1a["differentiation"]["functional_displacement_m"]
                    ),
                    **result,
                }
            )
        decisions = {row["decision"] for row in seed_results}
        seed_count = len(seed_results)
        result = dict(seed_results[0])
        if seed_count >= 3 and len(decisions) == 1:
            result["decision"] = next(iter(decisions))
            verdict_status = "acceptance_replicated"
        elif seed_count >= 3:
            result["decision"] = "indeterminate"
            result["replicate_disagreement"] = sorted(decisions)
            verdict_status = "replicated_inconsistent"
        else:
            verdict_status = "provisional_single_seed" if seed_count == 1 else "provisional_under_replicated"
        next_by_decision = {
            "contraction_confirmation": "phase_S",
            "continuous_amortization_failure": "P1b_factorial",
            "shared_subspace_pathology": "P2_partial_nonrepresentable_prior",
            "indeterminate": "P1b_factorial_and_targeted_followup",
        }
        payload = {
            "schema_version": "neon_phase5_gp1_v1",
            "gate": "GP1",
            "analysis_git_head": head,
            **result,
            "mandatory_next": next_by_decision[result["decision"]],
            "protocol_sha256": protocol_sha,
            "verdict_status": verdict_status,
            "p1a_seed_count": seed_count,
            "seed_results": seed_results,
            "inputs": {
                "geometry": str(args.geometry),
                "continuous": str(args.cancellation),
                "p1a": [str(path) for path in args.p1a],
                "dirichlet_direct": [str(path) for path in args.dirichlet_direct],
            },
        }
        if seed_count < 3:
            payload["mandatory_next_after_screen"] = payload["mandatory_next"]
            payload["mandatory_next"] = "replicate_P1a_on_at_least_three_support_seeds"
    _write_decision(output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
