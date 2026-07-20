#!/usr/bin/env python3
"""Audit and summarize a Phase-S selected-rung prior-amplitude sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from neuralop.flood.eval.neon_phase5 import (
    verify_checksummed_artifact,
    verify_phase5_evidence_artifact,
    write_checksummed_artifact,
)


def _load(path: str | Path) -> dict[str, Any]:
    verify_checksummed_artifact(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_alpha_sweep(
    phase_s: dict[str, Any], reports: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Validate the conditional 0.5/1/2x DE sweep and select on ID skill only."""

    gate = dict(phase_s.get("amplitude_gate") or {})
    if not (
        gate.get("decision") == "alpha_sweep"
        and gate.get("alpha_sweep_eligible") is True
    ):
        raise ValueError("Phase-S evidence does not authorize an alpha sweep.")
    target = dict(phase_s.get("selected_evidence_target") or {})
    rung = str(target.get("ladder_rung", "")).upper()
    prior_seed = int(target["prior_seed"])
    support_seed = target.get("dirichlet_particle_seed")
    aggregate = dict((phase_s.get("deep_ensemble") or {}).get("aggregate") or {})
    deep_variance = float(aggregate["deep_epistemic_variance_mean_m2"])
    if not math.isfinite(deep_variance) or deep_variance <= 0.0:
        raise ValueError("Phase-S DE variance must be finite and positive.")
    deep_std = math.sqrt(deep_variance)
    expected = [0.5 * deep_std, deep_std, 2.0 * deep_std]
    rows = []
    for report in reports:
        if str(report.get("ladder_rung", "")).upper() != rung:
            raise ValueError("alpha-sweep report rung differs from selected Phase-S rung.")
        if int(report.get("prior_seed", -1)) != prior_seed:
            raise ValueError("alpha-sweep report prior seed differs from selected run.")
        actual_support = report.get("dirichlet_particle_seed")
        if actual_support != support_seed:
            raise ValueError("alpha-sweep Dirichlet support differs from selected run.")
        if bool(report.get("amplitude_tuned_on_evaluation_events", True)):
            raise ValueError("alpha sweep may not tune on OOD/evaluation events.")
        if str(report.get("prior_scale_mode")) != "de_spread_target":
            raise ValueError("alpha sweep requires de_spread_target calibration.")
        scale = float(report.get("prior_scale_target_std_m", math.nan))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("alpha sweep report lacks a valid physical target scale.")
        noninferiority = dict(report.get("noninferiority") or {})
        skill = dict(report.get("skill") or {})
        impact = dict(report.get("impact") or {})
        rows.append(
            {
                "target_std_m": scale,
                "target_to_de_std_ratio": scale / deep_std,
                "noninferiority_pass": bool(noninferiority.get("pass", False)),
                "fair_crps_m": float((skill.get("both") or {})["fair_crps_m"]),
                "rmse_m": float((skill.get("both") or {})["rmse_m"]),
                "area_crps_km2": float(
                    (impact.get("both") or {})["area_crps_km2"]
                ),
                "posterior_scale_m": float(report["posterior_scale_m"]["estimate"]),
            }
        )
    if len(rows) != 3:
        raise ValueError("alpha sweep requires exactly three reports.")
    actual = sorted(row["target_std_m"] for row in rows)
    if any(
        not math.isclose(got, want, rel_tol=1.0e-6, abs_tol=1.0e-10)
        for got, want in zip(actual, expected)
    ):
        raise ValueError("alpha sweep targets are not the predeclared 0.5/1/2x DE scale.")
    eligible = [row for row in rows if row["noninferiority_pass"]]
    selected = (
        min(
            eligible,
            key=lambda row: (
                row["fair_crps_m"],
                row["rmse_m"],
                row["area_crps_km2"],
                row["target_std_m"],
            ),
        )
        if eligible
        else None
    )
    return {
        "schema_version": "neon_phase_s_alpha_sweep_v1",
        "decision": "alpha_selected" if selected is not None else "alpha_sweep_failed",
        "ladder_rung": rung,
        "prior_seed": prior_seed,
        "dirichlet_particle_seed": support_seed,
        "deep_ensemble_target_std_m": deep_std,
        "selection_dataset": "fixed_50_family_training_package_validation_split",
        "evaluation_events_used_for_selection": False,
        "selection_policy": (
            "noninferiority_then_fair_crps_rmse_area_crps_target_tiebreak"
        ),
        "selected": selected,
        "runs": sorted(rows, key=lambda row: row["target_std_m"]),
        "mandatory_next": (
            "rerun_phase_s_evidence_for_selected_alpha"
            if selected is not None
            else "stop_no_noninferior_alpha"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    from neon_phase5_runtime import require_clean_repository

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-s-complete", required=True)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    require_clean_repository(expected_head=args.expected_head)
    phase_s = _load(args.phase_s_complete)
    if phase_s.get("git_head") != args.expected_head:
        raise ValueError("Phase-S completion Git HEAD differs from alpha finalization.")
    protocol_sha = str(
        (phase_s.get("source_artifacts") or {}).get("protocol_sha256", "")
    )
    if not protocol_sha:
        raise ValueError("Phase-S completion does not record its protocol checksum.")
    reports = [
        verify_phase5_evidence_artifact(
            path,
            expected_head=args.expected_head,
            protocol_sha256=protocol_sha,
        )
        for path in args.report
    ]
    payload = finalize_alpha_sweep(phase_s, reports)
    payload["phase_s_complete"] = str(args.phase_s_complete)
    payload["phase_s_complete_sha256"] = _sha256(args.phase_s_complete)
    payload["source_reports"] = list(args.report)
    payload["source_report_sha256"] = [_sha256(path) for path in args.report]
    payload["analysis_git_head"] = args.expected_head
    payload["protocol_sha256"] = protocol_sha
    write_checksummed_artifact(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
