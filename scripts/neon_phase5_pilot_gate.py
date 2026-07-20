#!/usr/bin/env python3
"""Screen and replicate Phase-5 P1b pilots under one checksummed contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from neuralop.flood.eval.neon_phase5 import (
    aggregate_pilot_seed_decisions,
    assess_p1b_pilot_seed,
    assess_p2_pilot_seed,
    phase5_predeclared_protocol,
    validate_pilot_rungs_for_gp1_decision,
    verify_checksummed_artifact,
    verify_phase5_decision_artifact,
    verify_phase5_evidence_artifact,
    verify_phase5_ood_ranking_artifact,
    write_checksummed_artifact,
)
from neon_phase5_runtime import require_clean_repository


def _evidence_checkpoint_sha256(path: str | Path) -> str:
    provenance = Path(path).parent / "PROVENANCE.json"
    verify_checksummed_artifact(provenance)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    value = str(payload.get("checkpoint_sha256", ""))
    if len(value) != 64:
        raise ValueError(f"invalid checkpoint SHA-256 in {provenance}.")
    return value


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    noninferiority = row["ranking_metrics"]
    return (
        abs(
            math.log(
                max(float(row.get("posterior_to_direct_scale_ratio", 1.0)), 1.0e-15)
            )
        ),
        float(noninferiority["crps_ucb"]),
        float(noninferiority["rmse_ucb"]),
        float(noninferiority["impact_area_ucb"]),
        str(row["ladder_rung"]),
    )


def evaluate_pilot_gate(
    *,
    mode: str,
    reports: Sequence[dict[str, Any]],
    direct_report: dict[str, Any] | None,
    baseline_cancellation: dict[str, Any] | None,
    minimum_seeds: int,
    p2_ood_reports: Sequence[dict[str, Any]] = (),
    baseline_ood_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a screen or replicated-acceptance decision for P1b reports."""

    protocol = phase5_predeclared_protocol()
    pilot_protocol = protocol["pilot"]
    rungs = {str(report.get("ladder_rung", "")).upper() for report in reports}
    p2_mode = rungs == {"P2"}
    if "P2" in rungs and not p2_mode:
        raise ValueError("P2 reports cannot be mixed with P1b reports.")
    if p2_mode:
        if len(p2_ood_reports) != len(reports) or baseline_ood_report is None:
            raise ValueError("P2 gate requires one OOD report per seed and one B3 baseline.")
        direct_scale = None
        baseline_strata = None
    else:
        if direct_report is None or baseline_cancellation is None:
            raise ValueError("P1b gate requires direct and baseline-cancellation reports.")
        direct_scale = direct_report.get("direct_scale_m")
        baseline_strata = baseline_cancellation.get("strata")
        if not isinstance(direct_scale, dict) or not isinstance(baseline_strata, dict):
            raise ValueError("pilot gate requires direct_scale_m and baseline D3 strata.")
    assessments = []
    for report_index, report in enumerate(reports):
        if p2_mode:
            row = assess_p2_pilot_seed(
                report,
                ood_evidence=p2_ood_reports[report_index],
                baseline_ood_evidence=baseline_ood_report,
                # Fail closed: the signed pilot report, not this gate caller,
                # must attest that historical/OOD events were never used to
                # choose the prior amplitude.
                amplitude_tuned_on_evaluation_events=bool(
                    report.get("amplitude_tuned_on_evaluation_events", True)
                ),
            )
        else:
            assert direct_scale is not None and baseline_strata is not None
            row = assess_p1b_pilot_seed(
                report,
                direct_scale_m=direct_scale,
                baseline_strata=baseline_strata,
                equivalence_interval=tuple(
                    pilot_protocol["direct_scale_equivalence_interval"]
                ),
                differentiation_snr_threshold=float(
                    pilot_protocol["differentiation_snr_threshold"]
                ),
                cancellation_improvement_fraction=float(
                    pilot_protocol["cancellation_joint_improvement_fraction"]
                ),
            )
        noninferiority = report["noninferiority"]
        row["ranking_metrics"] = {
            "crps_ucb": float(noninferiority["crps"]["ci95_upper"]),
            "rmse_ucb": float(noninferiority["rmse"]["ci95_upper"]),
            "impact_area_ucb": float(
                noninferiority["impact_area"]["ci95_upper"]
            ),
        }
        assessments.append(row)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "screen":
        rungs = [str(row["ladder_rung"]) for row in assessments]
        if len(set(rungs)) != len(rungs):
            raise ValueError("screen mode requires exactly one report per candidate rung.")
        eligible = sorted(
            (row for row in assessments if bool(row["pass"])), key=_rank_key
        )
        if eligible:
            selected = eligible[0]
            decision = "pilot_screen_passed"
            mandatory_next = "replicate_selected_pilot_on_at_least_three_prior_seeds"
            selected_rung = selected["ladder_rung"]
        else:
            decision = "pilot_screen_failed"
            mandatory_next = "stop_no_predeclared_pilot_passed"
            selected_rung = None
        return {
            "schema_version": "neon_phase5_pilot_gate_v1",
            "gate": "PILOT_SCREEN",
            "decision": decision,
            "selected_rung": selected_rung,
            "mandatory_next": mandatory_next,
            "scaleout_eligible": False,
            "candidate_results": assessments,
            "ranking_policy": [
                "eligible_on_all_predeclared_checks",
                "absolute_log_distance_to_s_direct",
                "crps_ucb",
                "rmse_ucb",
                "impact_area_ucb",
                "rung_name_tiebreak",
            ],
        }
    if normalized_mode != "accept":
        raise ValueError("mode must be 'screen' or 'accept'.")
    aggregate = aggregate_pilot_seed_decisions(
        assessments, minimum_seeds=int(minimum_seeds)
    )
    accepted = aggregate["decision"] == "pilot_accepted"
    return {
        "schema_version": "neon_phase5_pilot_gate_v1",
        "gate": "PILOT_ACCEPTANCE",
        **aggregate,
        "mandatory_next": "phase_S" if accepted else "stop_or_revise_method",
        "scaleout_eligible": bool(accepted),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screen", "accept"), required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--gp1-decision", required=True)
    parser.add_argument("--direct")
    parser.add_argument("--baseline-cancellation")
    parser.add_argument("--pilot-result", action="append", required=True)
    parser.add_argument("--p2-ood-result", action="append", default=[])
    parser.add_argument("--baseline-ood")
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    head = require_clean_repository(expected_head=args.expected_head)
    protocol_sha = verify_checksummed_artifact(args.protocol)
    gp1 = verify_phase5_decision_artifact(
        args.gp1_decision,
        expected_head=head,
        protocol_sha256=protocol_sha,
    )
    if gp1.get("gate") != "GP1" or int(gp1.get("p1a_seed_count", 0)) < 3:
        raise ValueError("pilot gate requires a replicated GP1 decision.")
    load_evidence = lambda path: verify_phase5_evidence_artifact(
        path, expected_head=head, protocol_sha256=protocol_sha
    )
    direct = None if args.direct is None else load_evidence(args.direct)
    baseline = (
        None
        if args.baseline_cancellation is None
        else load_evidence(args.baseline_cancellation)
    )
    reports = [load_evidence(path) for path in args.pilot_result]
    validate_pilot_rungs_for_gp1_decision(
        str(gp1.get("decision", "")),
        {str(report.get("ladder_rung", "")) for report in reports},
    )
    if args.p2_ood_result and len(args.p2_ood_result) != len(args.pilot_result):
        raise ValueError("P2 requires one checkpoint-matched OOD result per pilot.")
    p2_ood = [
        verify_phase5_ood_ranking_artifact(
            ood_path,
            expected_head=head,
            protocol_sha256=protocol_sha,
            checkpoint_sha256=_evidence_checkpoint_sha256(pilot_path),
        )
        for ood_path, pilot_path in zip(args.p2_ood_result, args.pilot_result)
    ]
    if args.baseline_ood is not None and args.baseline_cancellation is None:
        raise ValueError("baseline OOD evidence requires baseline cancellation provenance.")
    baseline_ood = (
        None
        if args.baseline_ood is None
        else verify_phase5_ood_ranking_artifact(
            args.baseline_ood,
            expected_head=head,
            protocol_sha256=protocol_sha,
            checkpoint_sha256=_evidence_checkpoint_sha256(
                args.baseline_cancellation
            ),
        )
    )
    payload = evaluate_pilot_gate(
        mode=args.mode,
        reports=reports,
        direct_report=direct,
        baseline_cancellation=baseline,
        minimum_seeds=args.minimum_seeds,
        p2_ood_reports=p2_ood,
        baseline_ood_report=baseline_ood,
    )
    payload.update(
        {
            "protocol_sha256": protocol_sha,
            "analysis_git_head": head,
            "gp1_decision": str(args.gp1_decision),
            "gp1_mechanism_decision": gp1["decision"],
            "inputs": {
                "direct": args.direct,
                "baseline_cancellation": args.baseline_cancellation,
                "pilot_results": list(args.pilot_result),
                "p2_ood_results": list(args.p2_ood_result),
                "baseline_ood": args.baseline_ood,
            },
        }
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_checksummed_artifact(output / "DECISION.json", payload)
    (output / "DECISION.txt").write_text(
        "\n".join(
            [
                f"gate={payload['gate']}",
                f"decision={payload['decision']}",
                f"selected_rung={payload.get('selected_rung') or payload.get('ladder_rung')}",
                f"mandatory_next={payload['mandatory_next']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
