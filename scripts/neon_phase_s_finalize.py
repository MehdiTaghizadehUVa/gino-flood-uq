#!/usr/bin/env python3
"""Fail-closed audit of the complete Phase-S evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile from no values.")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def assess_phase_s_amplitude_residual(
    deep_ensemble: dict[str, Any],
    *,
    replicates: int = 2_000,
    seed: int = 20260720,
    std_equivalence: tuple[float, float] = (0.5, 2.0),
) -> dict[str, Any]:
    """Distinguish an amplitude-only residual from a structural mismatch.

    Family-level bootstrap intervals prevent a noisy three-model DE reference
    from silently triggering additional tuning. Structure must exceed its
    chance/null boundaries before a variance-amplitude sweep is admissible.
    """

    rows = list(deep_ensemble.get("per_family") or [])
    if len(rows) < 2:
        raise ValueError("amplitude assessment requires paired per-family DE rows.")
    top_q = float((deep_ensemble.get("plan") or {}).get("top_q", 0.10))
    required = (
        "spatial_corr",
        "topq_overlap",
        "neon_epistemic_variance_mean_m2",
        "deep_epistemic_variance_mean_m2",
    )
    if any(any(key not in row for key in required) for row in rows):
        raise ValueError("deep-ensemble rows lack amplitude/structure diagnostics.")

    def summarize(sample: list[dict[str, Any]]) -> tuple[float, float, float]:
        correlation = sum(float(row["spatial_corr"]) for row in sample) / len(sample)
        overlap = sum(float(row["topq_overlap"]) for row in sample) / len(sample)
        neon_variance = sum(
            float(row["neon_epistemic_variance_mean_m2"]) for row in sample
        ) / len(sample)
        deep_variance = sum(
            float(row["deep_epistemic_variance_mean_m2"]) for row in sample
        ) / len(sample)
        if neon_variance < 0.0 or deep_variance <= 0.0:
            raise ValueError("epistemic variances must be nonnegative and DE positive.")
        return correlation, overlap, math.sqrt(neon_variance / deep_variance)

    estimate = summarize(rows)
    rng = random.Random(int(seed))
    samples = [
        summarize([rows[rng.randrange(len(rows))] for _ in rows])
        for _ in range(int(replicates))
    ]

    def interval(index: int) -> dict[str, float]:
        values = [row[index] for row in samples]
        return {
            "estimate": float(estimate[index]),
            "ci95_lower": _quantile(values, 0.025),
            "ci95_upper": _quantile(values, 0.975),
        }

    correlation = interval(0)
    overlap = interval(1)
    std_ratio = interval(2)
    structure_supported = bool(
        correlation["ci95_lower"] > 0.0 and overlap["ci95_lower"] > top_q
    )
    lower, upper = (float(value) for value in std_equivalence)
    amplitude_mismatch = bool(
        std_ratio["ci95_upper"] < lower or std_ratio["ci95_lower"] > upper
    )
    amplitude_equivalent = bool(
        std_ratio["ci95_lower"] >= lower and std_ratio["ci95_upper"] <= upper
    )
    if structure_supported and amplitude_mismatch:
        decision = "alpha_sweep"
        mandatory_next = "run_predeclared_selected_rung_alpha_sweep"
    elif not structure_supported:
        decision = "structure_residual"
        mandatory_next = "stop_alpha_tuning_and_revisit_epistemic_structure"
    elif amplitude_equivalent:
        decision = "no_alpha_sweep"
        mandatory_next = "phase_s_complete"
    else:
        decision = "amplitude_indeterminate"
        mandatory_next = "stop_alpha_tuning_due_to_interval_overlap"
    return {
        "schema_version": "neon_phase_s_amplitude_gate_v1",
        "decision": decision,
        "mandatory_next": mandatory_next,
        "alpha_sweep_eligible": decision == "alpha_sweep",
        "spatial_correlation": correlation,
        "topq_overlap": overlap,
        "epistemic_std_ratio_neon_to_de": std_ratio,
        "thresholds": {
            "spatial_correlation_null": 0.0,
            "topq_overlap_chance": top_q,
            "std_equivalence_interval": [lower, upper],
            "bootstrap_replicates": int(replicates),
            "bootstrap_seed": int(seed),
        },
        "checks": {
            "structure_supported_above_null": structure_supported,
            "amplitude_decisively_outside_equivalence": amplitude_mismatch,
            "amplitude_decisively_inside_equivalence": amplitude_equivalent,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"missing checksummed Phase-S artifact: {path}.")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = _sha256(path)
    if expected != actual:
        raise ValueError(f"checksum mismatch for Phase-S artifact {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(encoded)
    os.replace(tmp, path)
    digest = hashlib.sha256(encoded).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_tmp.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    os.replace(sidecar_tmp, sidecar)


def finalize_phase_s(
    submission_path: Path, *, expected_head: str
) -> dict[str, Any]:
    """Verify every mandatory Phase-S output and return its completion record."""

    submission_path = Path(submission_path)
    submission = _load(submission_path)
    if submission.get("schema_version") != "neon_phase5_scaleout_submission_v1":
        raise ValueError("invalid Phase-S submission schema.")
    if submission.get("git_head") != expected_head:
        raise ValueError("Phase-S submission Git HEAD differs from the finalizer.")
    terminal_jobs = list(submission.get("required_terminal_jobs") or [])
    if len(terminal_jobs) != 3 or any(not str(job) for job in terminal_jobs):
        raise ValueError("Phase-S submission must record three terminal evidence jobs.")
    protocol_root = Path(submission["protocol_root"])
    protocol_path = protocol_root / "PROTOCOL.json"
    _load(protocol_path)
    if _sha256(protocol_path) != submission.get("protocol_sha256"):
        raise ValueError("Phase-S protocol differs from the submitted protocol hash.")
    protocol_sha = _sha256(protocol_path)
    gate_path = Path(submission["governing_gate"])
    _load(gate_path)
    gate_sha = _sha256(gate_path)
    if gate_sha != submission.get("governing_gate_sha256"):
        raise ValueError("Phase-S governing gate differs from its submitted hash.")

    target_path = submission_path.parent / "PHASE_S_TARGET.json"
    target = _load(target_path)
    if target != submission.get("selected_evidence_target"):
        raise ValueError("embedded Phase-S evidence target differs from its signed artifact.")
    if target.get("analysis_git_head") != expected_head:
        raise ValueError("Phase-S target Git HEAD differs from the finalizer.")
    if target.get("protocol_sha256") != protocol_sha:
        raise ValueError("Phase-S target protocol differs from the submitted protocol.")
    if target.get("governing_gate_sha256") != gate_sha:
        raise ValueError("Phase-S target governing gate differs from the submission.")
    rung = str(target.get("ladder_rung", "")).upper()
    checkpoint_sha = str(target.get("stage2_checkpoint_sha256", ""))

    contraction_path = (
        Path(submission["scaleout_root"]) / "contraction_analysis.json"
    )
    contraction = _load(contraction_path)
    if contraction.get("schema_version") != "neon_contraction_analysis_v1":
        raise ValueError("invalid Phase-S contraction schema.")
    if str(contraction.get("ladder_rung", "")).upper() != rung:
        raise ValueError("contraction analysis rung differs from the selected Phase-S run.")
    if contraction.get("analysis_git_head") != expected_head:
        raise ValueError("contraction analysis Git HEAD differs from Phase-S.")
    if contraction.get("protocol_sha256") != protocol_sha:
        raise ValueError("contraction analysis protocol differs from Phase-S.")
    if contraction.get("governing_gate_sha256") != gate_sha:
        raise ValueError("contraction analysis governing gate differs from Phase-S.")
    if int(contraction.get("n_replicates", 0)) != 5 or contraction.get(
        "n_values"
    ) != [25, 50, 100, 250, 400]:
        raise ValueError("Phase-S contraction evidence is not the predeclared 5x5 design.")

    ood_path = Path(submission["ood_root"]) / "ranking.json"
    ood = _load(ood_path)
    if ood.get("schema_version") != "neon_ood_ranking_v1":
        raise ValueError("invalid Phase-S OOD ranking schema.")
    if int(ood.get("n_ood_events", 0)) != 13:
        raise ValueError("Phase-S requires exactly 13 OOD events.")
    if ood.get("analysis_git_head") != expected_head:
        raise ValueError("Phase-S OOD Git HEAD differs from Phase-S.")
    if ood.get("protocol_sha256") != protocol_sha:
        raise ValueError("Phase-S OOD protocol differs from Phase-S.")
    if ood.get("stage2_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Phase-S OOD checkpoint differs from the selected Stage-2 checkpoint.")

    de_path = (
        Path(submission["deep_ensemble_root"]) / "deep_ensemble_comparison.json"
    )
    de = _load(de_path)
    if de.get("schema_version") != "neon_deep_ensemble_comparison_v2":
        raise ValueError("invalid Phase-S deep-ensemble comparison schema.")
    if int(de.get("j_models", 0)) < 2:
        raise ValueError("deep-ensemble cross-check requires at least two Stage-1 models.")
    de_submission_path = Path(submission["deep_ensemble_root"]) / "SUBMITTED.json"
    de_submission = _load(de_submission_path)
    if de_submission.get("schema_version") != "neon_phase5_de_submission_v1":
        raise ValueError("invalid Phase-S deep-ensemble submission schema.")
    if de_submission.get("git_head") != expected_head:
        raise ValueError("deep-ensemble submission Git HEAD differs from Phase-S.")
    if de_submission.get("protocol_sha256") != protocol_sha:
        raise ValueError("deep-ensemble submission protocol differs from Phase-S.")
    if de_submission.get("stage2_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("deep-ensemble submission used a different Stage-2 checkpoint.")
    de_plan = dict(de.get("plan") or {})
    if not (
        de_plan.get("physical_space") is True
        and de_plan.get("common_aleatory_latent_bank") is True
    ):
        raise ValueError("deep-ensemble cross-check violates its physical/CRN contract.")
    if de_plan.get("stage2_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("deep-ensemble evidence used a different selected Stage-2 checkpoint.")
    amplitude_gate = assess_phase_s_amplitude_residual(de)

    return {
        "schema_version": "neon_phase_s_complete_v1",
        "git_head": expected_head,
        "evidence_complete": True,
        "ladder_rung": rung,
        "selection_policy": target.get("selection_policy"),
        "selected_evidence_target": target,
        "source_artifacts": {
            "protocol": str(protocol_path),
            "protocol_sha256": _sha256(protocol_path),
            "submission": str(submission_path),
            "submission_sha256": _sha256(submission_path),
            "target": str(target_path),
            "target_sha256": _sha256(target_path),
            "contraction": str(contraction_path),
            "contraction_sha256": _sha256(contraction_path),
            "ood": str(ood_path),
            "ood_sha256": _sha256(ood_path),
            "deep_ensemble": str(de_path),
            "deep_ensemble_sha256": _sha256(de_path),
            "deep_ensemble_submission": str(de_submission_path),
            "deep_ensemble_submission_sha256": _sha256(de_submission_path),
        },
        "contraction": {
            "gamma_mean": contraction["gamma_mean"],
            "gamma_bootstrap_95_ci": contraction["gamma_bootstrap_95_ci"],
            "n_replicates": contraction["n_replicates"],
        },
        "ood": {
            "n_events": ood["n_ood_events"],
            "spearman_epistemic_std_rmse": ood[
                "spearman_epistemic_std_rmse"
            ],
            "top3_error_event_recall": ood["top3_error_event_recall"],
        },
        "deep_ensemble": {
            "j_models": de["j_models"],
            "aggregate": de.get("aggregate"),
        },
        "amplitude_gate": amplitude_gate,
        "mandatory_next": amplitude_gate["mandatory_next"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = finalize_phase_s(args.submission, expected_head=args.expected_head)
    _atomic_write(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
