#!/usr/bin/env python3
"""Preflight and validate the conditional replicated NEON N-sweep."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

N_VALUES = (25, 50, 100, 250, 400)
N_REPLICATES = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute_no_resolve(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp.write_bytes(encoded)
    os.replace(tmp, path)
    digest = hashlib.sha256(encoded).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_tmp.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    os.replace(sidecar_tmp, sidecar)


def _validate_json_checksum(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing scale-out plan checksum: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"scale-out plan checksum mismatch: expected {expected}, got {actual}."
        )


def _load_checksummed_json(path: Path) -> dict[str, Any]:
    path = _absolute_no_resolve(path)
    _validate_json_checksum(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_stage2_run(stage2_dir: Path) -> dict[str, Any]:
    """Verify the immutable training products selected for Phase S."""

    products = {
        "checkpoint": stage2_dir / "neon_stage2_best.pt",
        "history": stage2_dir / "history.json",
        "preflight": stage2_dir / "preflight.json",
    }
    for candidate in products.values():
        if not candidate.is_file():
            raise ValueError(f"selected Phase-S run is incomplete: missing {candidate}.")
    completion_path = stage2_dir / "TRAINING_COMPLETE.json"
    if not completion_path.is_file():
        raise ValueError(
            "selected Phase-S run lacks its checksummed training-completion manifest: "
            f"{completion_path}."
        )
    completion = _load_checksummed_json(completion_path)
    if completion.get("schema_version") != "neon_stage2_training_complete_v1":
        raise ValueError("selected Phase-S run has an unsupported completion schema.")
    paths = dict(completion.get("input_paths") or {})
    hashes = dict(completion.get("input_sha256") or {})
    for key, candidate in products.items():
        expected_path = paths.get(key)
        if expected_path is None or _absolute_no_resolve(Path(expected_path)) != (
            _absolute_no_resolve(candidate)
        ):
            raise ValueError(
                f"training-completion {key} path does not match the selected Phase-S run."
            )
        expected_sha = hashes.get(key)
        actual_sha = _sha256(candidate)
        if expected_sha != actual_sha:
            raise ValueError(
                f"training-completion {key} SHA-256 mismatch: "
                f"expected {expected_sha}, got {actual_sha}."
            )
    return {
        "path": str(_absolute_no_resolve(completion_path)),
        "sha256": _sha256(completion_path),
        "git_head": completion.get("git_head"),
    }


def _load_phase5_report(
    path: Path,
    *,
    expected_head: str | None,
    protocol_sha256: str | None,
) -> dict[str, Any]:
    """Load signed evidence and, for production, verify its provenance contract."""

    payload = _load_checksummed_json(path)
    if expected_head is None and protocol_sha256 is None:
        return payload
    if not expected_head or not protocol_sha256:
        raise ValueError("expected_head and protocol_sha256 must be supplied together.")
    provenance = _load_checksummed_json(path.parent / "PROVENANCE.json")
    if provenance.get("git_head") != expected_head:
        raise ValueError("selected Phase-S report Git HEAD differs from the submission.")
    if provenance.get("protocol_sha256") != protocol_sha256:
        raise ValueError("selected Phase-S report protocol differs from the submission.")
    return payload


def resolve_phase_s_evidence_target(
    gate_report: Path,
    *,
    prior_seed: int | None = None,
    dirichlet_particle_seed: int | None = None,
    expected_head: str | None = None,
    protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Resolve a predeclared accepted run without outcome-based checkpoint selection."""

    gate_report = _absolute_no_resolve(gate_report)
    gate = (
        _load_checksummed_json(gate_report)
        if expected_head is not None or protocol_sha256 is not None
        else json.loads(gate_report.read_text(encoding="utf-8"))
    )
    if expected_head is not None or protocol_sha256 is not None:
        if not expected_head or not protocol_sha256:
            raise ValueError("expected_head and protocol_sha256 must be supplied together.")
        if gate.get("analysis_git_head") != expected_head:
            raise ValueError("governing Phase-S gate Git HEAD differs from the submission.")
        if gate.get("protocol_sha256") != protocol_sha256:
            raise ValueError("governing Phase-S gate protocol differs from the submission.")
    schema = gate.get("schema_version")
    if schema == "neon_phase5_gp1_v1":
        if not (
            gate.get("gate") == "GP1"
            and gate.get("decision") == "contraction_confirmation"
            and gate.get("verdict_status") == "acceptance_replicated"
            and int(gate.get("p1a_seed_count", 0)) >= 3
        ):
            raise ValueError("Phase-S target requires a replicated passing GP1 gate.")
        result_paths = [Path(item) for item in (gate.get("inputs") or {}).get("p1a", [])]
        if len(result_paths) < 3:
            raise ValueError("GP1 gate does not record its replicated P1a result paths.")
        reports = [
            (
                path,
                _load_phase5_report(
                    path,
                    expected_head=expected_head,
                    protocol_sha256=protocol_sha256,
                ),
            )
            for path in result_paths
        ]
        selected_seed = (
            int(reports[0][1]["dirichlet_particle_seed"])
            if dirichlet_particle_seed is None
            else int(dirichlet_particle_seed)
        )
        matches = [
            (path, report)
            for path, report in reports
            if int(report.get("dirichlet_particle_seed", -1)) == selected_seed
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one accepted P1a report for Dirichlet support seed "
                f"{selected_seed}; found {len(matches)}."
            )
        result_path, report = matches[0]
        if str(report.get("ladder_rung", "")).upper() != "B5":
            raise ValueError("GP1 Phase-S evidence target must be a B5 run.")
        selected_prior_seed = int(report["prior_seed"])
        rung = "B5"
    elif schema == "neon_phase5_pilot_gate_v1":
        if not (
            gate.get("gate") == "PILOT_ACCEPTANCE"
            and gate.get("decision") == "pilot_accepted"
            and gate.get("verdict_status") == "acceptance_replicated"
            and gate.get("scaleout_eligible") is True
            and int(gate.get("seed_count", 0)) >= 3
        ):
            raise ValueError("Phase-S target requires a replicated accepted pilot gate.")
        result_paths = [
            Path(item) for item in (gate.get("inputs") or {}).get("pilot_results", [])
        ]
        if len(result_paths) < 3:
            raise ValueError("pilot gate does not record its replicated result paths.")
        reports = [
            (
                path,
                _load_phase5_report(
                    path,
                    expected_head=expected_head,
                    protocol_sha256=protocol_sha256,
                ),
            )
            for path in result_paths
        ]
        selected_prior_seed = (
            int(reports[0][1]["prior_seed"])
            if prior_seed is None
            else int(prior_seed)
        )
        matches = [
            (path, report)
            for path, report in reports
            if int(report.get("prior_seed", -1)) == selected_prior_seed
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one accepted pilot report for prior seed "
                f"{selected_prior_seed}; found {len(matches)}."
            )
        result_path, report = matches[0]
        rung = str(gate.get("ladder_rung", "")).upper()
        if str(report.get("ladder_rung", "")).upper() != rung:
            raise ValueError("selected pilot report rung differs from the accepted gate rung.")
        selected_seed = None
    elif schema == "neon_phase_s_alpha_sweep_v1":
        if not (
            gate.get("decision") == "alpha_selected"
            and gate.get("evaluation_events_used_for_selection") is False
            and isinstance(gate.get("selected"), dict)
        ):
            raise ValueError("Phase-S alpha target requires a valid selected-alpha gate.")
        selected_target = float(gate["selected"]["target_std_m"])
        result_paths = [Path(item) for item in gate.get("source_reports", [])]
        reports = [
            (
                path,
                _load_phase5_report(
                    path,
                    expected_head=expected_head,
                    protocol_sha256=protocol_sha256,
                ),
            )
            for path in result_paths
        ]
        matches = [
            (path, report)
            for path, report in reports
            if math.isclose(
                float(report.get("prior_scale_target_std_m", -1.0)),
                selected_target,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ]
        if len(matches) != 1:
            raise ValueError("selected alpha target does not resolve to exactly one report.")
        result_path, report = matches[0]
        rung = str(gate.get("ladder_rung", "")).upper()
        if str(report.get("ladder_rung", "")).upper() != rung:
            raise ValueError("selected alpha report rung differs from its gate.")
        selected_prior_seed = int(gate["prior_seed"])
        selected_seed = gate.get("dirichlet_particle_seed")
        if prior_seed is not None and int(prior_seed) != selected_prior_seed:
            raise ValueError("requested prior seed differs from the selected alpha run.")
        if (
            dirichlet_particle_seed is not None
            and int(dirichlet_particle_seed) != selected_seed
        ):
            raise ValueError(
                "requested Dirichlet support seed differs from the selected alpha run."
            )
        if int(report.get("prior_seed", -1)) != selected_prior_seed or report.get(
            "dirichlet_particle_seed"
        ) != selected_seed:
            raise ValueError("selected alpha report seeds differ from its gate.")
    else:
        raise ValueError(f"unsupported Phase-S gate schema: {schema!r}.")

    stage2_dir = result_path.parent.parent
    training_completion = _require_stage2_run(stage2_dir)
    if expected_head is not None and training_completion.get("git_head") != expected_head:
        raise ValueError(
            "selected Phase-S run was trained at a different Git HEAD."
        )
    return {
        "schema_version": "neon_phase_s_evidence_target_v1",
        "governing_gate": str(gate_report),
        "ladder_rung": rung,
        "prior_seed": selected_prior_seed,
        "dirichlet_particle_seed": selected_seed,
        "stage2_dir": str(stage2_dir),
        "id_metrics": str(_absolute_no_resolve(result_path)),
        "id_metrics_sha256": _sha256(result_path),
        "stage2_checkpoint_sha256": _sha256(stage2_dir / "neon_stage2_best.pt"),
        "training_completion": training_completion,
        "analysis_git_head": expected_head,
        "protocol_sha256": protocol_sha256,
        "governing_gate_sha256": _sha256(gate_report),
        "selection_policy": (
            "fixed_id_validation_alpha_selection_no_evaluation_events"
            if schema == "neon_phase_s_alpha_sweep_v1"
            else "predeclared_seed_not_outcome_ranked"
        ),
        "prior_scale_target_std_m": report.get("prior_scale_target_std_m"),
    }


def _training_script_module():
    path = Path(__file__).resolve().with_name("neon_stage2_tr_train.py")
    spec = importlib.util.spec_from_file_location("neon_stage2_tr_train_scaleout", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import training script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_scaleout_plan(
    *,
    g1_report: Path,
    run_root: Path,
    cache_dir: Path,
    expected_head: str,
    prior_scale: str = "auto_0p10_base_rmse",
    d_e: int = 16,
    n_epochs: int = 30,
    ladder_rung: str = "B3",
    prior_seed: int = 20260703,
    dirichlet_particle_seed: int | None = None,
    protocol_sha256: str | None = None,
    governing_gate_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the governing gate and materialize 25 resolved task manifests."""
    g1_report = _absolute_no_resolve(g1_report)
    actual_gate_sha256 = _sha256(g1_report)
    if governing_gate_sha256 is not None and (
        str(governing_gate_sha256) != actual_gate_sha256
    ):
        raise ValueError("governing gate checksum differs from the scale-out request.")
    gate = json.loads(g1_report.read_text())
    phase5_pass = (
        gate.get("schema_version") == "neon_phase5_gp1_v1"
        and gate.get("gate") == "GP1"
        and gate.get("verdict_status") == "acceptance_replicated"
        and gate.get("decision") == "contraction_confirmation"
        and int(gate.get("p1a_seed_count", 0)) >= 3
    )
    pilot_pass = (
        gate.get("schema_version") == "neon_phase5_pilot_gate_v1"
        and gate.get("gate") == "PILOT_ACCEPTANCE"
        and gate.get("decision") == "pilot_accepted"
        and gate.get("verdict_status") == "acceptance_replicated"
        and gate.get("scaleout_eligible") is True
        and int(gate.get("seed_count", 0)) >= 3
    )
    alpha_pass = (
        gate.get("schema_version") == "neon_phase_s_alpha_sweep_v1"
        and gate.get("decision") == "alpha_selected"
        and gate.get("evaluation_events_used_for_selection") is False
        and isinstance(gate.get("selected"), dict)
    )
    if not (phase5_pass or pilot_pass or alpha_pass):
        raise ValueError(
            "scale-out requires a replicated Phase-5 GP1 contraction decision "
            "or a replicated accepted Phase-5 pilot gate, or a valid selected-alpha gate."
        )
    if phase5_pass and str(ladder_rung).upper() != "B5":
        raise ValueError(
            "a contraction-confirming GP1 decision authorizes only the "
            "persistent-particle B5 rung."
        )
    selected_dirichlet_seed = None
    if phase5_pass:
        seed_rows = list(gate.get("seed_results") or [])
        accepted_support_seeds = [
            int(row["dirichlet_particle_seed"])
            for row in seed_rows
            if row.get("dirichlet_particle_seed") is not None
            and row.get("decision") == "contraction_confirmation"
        ]
        if len(accepted_support_seeds) < 3 or len(set(accepted_support_seeds)) != len(
            accepted_support_seeds
        ):
            raise ValueError("GP1 scale-out requires three unique accepted Dirichlet support seeds.")
        selected_dirichlet_seed = (
            accepted_support_seeds[0]
            if dirichlet_particle_seed is None
            else int(dirichlet_particle_seed)
        )
        if selected_dirichlet_seed not in accepted_support_seeds:
            raise ValueError(
                f"dirichlet_particle_seed={selected_dirichlet_seed} is not among "
                f"accepted Dirichlet support seeds {accepted_support_seeds}."
            )
    if pilot_pass and str(gate.get("ladder_rung", "")).upper() != str(
        ladder_rung
    ).upper():
        raise ValueError("selected pilot rung differs from requested scale-out rung.")
    if pilot_pass:
        accepted_seeds = {
            int(row["prior_seed"])
            for row in gate.get("seed_results", [])
            if row.get("prior_seed") is not None
        }
        if int(prior_seed) not in accepted_seeds:
            raise ValueError(
                f"prior_seed={prior_seed} is not among accepted pilot prior seeds "
                f"{sorted(accepted_seeds)}."
            )
    selected_prior_target = None
    if alpha_pass:
        if str(gate.get("ladder_rung", "")).upper() != str(ladder_rung).upper():
            raise ValueError("selected alpha rung differs from requested scale-out rung.")
        if int(gate.get("prior_seed", -1)) != int(prior_seed):
            raise ValueError("selected alpha prior seed differs from scale-out request.")
        gate_support = gate.get("dirichlet_particle_seed")
        requested_support = (
            None if dirichlet_particle_seed is None else int(dirichlet_particle_seed)
        )
        if gate_support != requested_support:
            raise ValueError("selected alpha support seed differs from scale-out request.")
        selected_dirichlet_seed = requested_support
        selected_prior_target = float(gate["selected"]["target_std_m"])
    if not expected_head:
        raise ValueError("expected_head must be non-empty.")
    run_root = _absolute_no_resolve(run_root)
    cache_dir = _absolute_no_resolve(cache_dir)
    train_script = _training_script_module()
    config = train_script._resolved_ladder_config(
        ladder_rung,
        prior_scale=prior_scale,
        d_e=d_e,
        n_epochs=n_epochs,
        dirichlet_particle_seed=selected_dirichlet_seed,
        prior_target_std_m=selected_prior_target,
    )
    tasks = []
    for replicate in range(N_REPLICATES):
        for n_index, n_train in enumerate(N_VALUES):
            task_id = replicate * len(N_VALUES) + n_index
            output_dir = run_root / f"rep{replicate}" / f"n{n_train}"
            if (output_dir / "job_id.txt").exists():
                raise ValueError(f"scale-out task already submitted: {output_dir}")
            preflight = output_dir / "preflight.json"
            train_script._write_preflight_manifest(
                preflight,
                config=config,
                rung=ladder_rung,
                n_train=n_train,
                output_dir=output_dir,
                cache_dir=cache_dir,
                subset_replicate=replicate,
                prior_seed=int(prior_seed),
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "subset_replicate": replicate,
                    "n_train": n_train,
                    "output_dir": str(output_dir),
                    "preflight_path": str(preflight),
                    "config": asdict(config),
                }
            )
    plan = {
        "schema_version": "neon_scaleout_plan_v1",
        "expected_git_head": str(expected_head),
        "g1_report": str(g1_report),
        "g1_report_sha256": _sha256(g1_report),
        "protocol_sha256": protocol_sha256,
        "governing_gate_sha256": actual_gate_sha256,
        "ladder_rung": str(ladder_rung).upper(),
        "prior_seed": int(prior_seed),
        "dirichlet_particle_seed": selected_dirichlet_seed,
        "prior_scale_target_std_m": selected_prior_target,
        "cache_dir": str(cache_dir),
        "nested_subset_contract": "prefixes_of_one_seeded_permutation_per_replicate",
        "n_values": list(N_VALUES),
        "n_replicates": N_REPLICATES,
        "tasks": tasks,
    }
    _atomic_json(run_root / "scaleout_plan.json", plan)
    return plan


def validate_scaleout_task(
    plan_path: Path,
    *,
    task_id: int,
    expected_head: str,
    prior_seed: int | None = None,
    dirichlet_particle_seed: int | None = None,
    protocol_sha256: str | None = None,
    governing_gate_sha256: str | None = None,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    _validate_json_checksum(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan.get("schema_version") != "neon_scaleout_plan_v1":
        raise ValueError("invalid scale-out plan schema.")
    if plan.get("expected_git_head") != expected_head:
        raise ValueError("scale-out Git HEAD differs from preflight.")
    if protocol_sha256 is not None and plan.get("protocol_sha256") != protocol_sha256:
        raise ValueError("runtime protocol checksum differs from the scale-out plan.")
    if (
        governing_gate_sha256 is not None
        and plan.get("governing_gate_sha256") != governing_gate_sha256
    ):
        raise ValueError("runtime governing gate checksum differs from the scale-out plan.")
    if prior_seed is not None and int(plan.get("prior_seed", -1)) != int(prior_seed):
        raise ValueError("runtime prior seed differs from the scale-out plan.")
    if dirichlet_particle_seed is not None and int(
        plan.get("dirichlet_particle_seed", -1)
    ) != int(dirichlet_particle_seed):
        raise ValueError("runtime Dirichlet support seed differs from the scale-out plan.")
    if _sha256(Path(plan["g1_report"])) != plan.get("g1_report_sha256"):
        raise ValueError("G1 report changed after scale-out preflight.")
    gate = json.loads(Path(plan["g1_report"]).read_text())
    if not (
        (
            gate.get("schema_version") == "neon_phase5_gp1_v1"
            and gate.get("gate") == "GP1"
            and gate.get("verdict_status") == "acceptance_replicated"
            and gate.get("decision") == "contraction_confirmation"
            and int(gate.get("p1a_seed_count", 0)) >= 3
        )
        or (
            gate.get("schema_version") == "neon_phase5_pilot_gate_v1"
            and gate.get("gate") == "PILOT_ACCEPTANCE"
            and gate.get("decision") == "pilot_accepted"
            and gate.get("verdict_status") == "acceptance_replicated"
            and gate.get("scaleout_eligible") is True
            and int(gate.get("seed_count", 0)) >= 3
            and str(gate.get("ladder_rung", "")).upper()
            == str(plan.get("ladder_rung", "")).upper()
        )
        or (
            gate.get("schema_version") == "neon_phase_s_alpha_sweep_v1"
            and gate.get("decision") == "alpha_selected"
            and gate.get("evaluation_events_used_for_selection") is False
            and str(gate.get("ladder_rung", "")).upper()
            == str(plan.get("ladder_rung", "")).upper()
            and int(gate.get("prior_seed", -1)) == int(plan.get("prior_seed", -2))
            and gate.get("dirichlet_particle_seed")
            == plan.get("dirichlet_particle_seed")
            and float((gate.get("selected") or {}).get("target_std_m", -1.0))
            == float(plan.get("prior_scale_target_std_m", -2.0))
        )
    ):
        raise ValueError("governing scale-out report no longer passes.")
    matches = [row for row in plan["tasks"] if int(row["task_id"]) == int(task_id)]
    if len(matches) != 1:
        raise ValueError(f"expected one task_id={task_id}; got {len(matches)}.")
    task = matches[0]
    preflight = json.loads(Path(task["preflight_path"]).read_text())
    if preflight.get("config") != task.get("config"):
        raise ValueError("task config differs from its preflight manifest.")
    if int(preflight.get("prior_seed", -1)) != int(plan.get("prior_seed", -2)):
        raise ValueError("task prior seed differs from the scale-out plan.")
    return task


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--g1-report", type=Path, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument("--expected-head", required=True)
    prepare.add_argument("--ladder-rung", default="B3")
    prepare.add_argument("--prior-seed", type=int, default=20260703)
    prepare.add_argument("--dirichlet-particle-seed", type=int)
    prepare.add_argument("--protocol-sha256")
    prepare.add_argument("--governing-gate-sha256")
    validate = sub.add_parser("validate-task")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--task-id", type=int, required=True)
    validate.add_argument("--expected-head", required=True)
    validate.add_argument("--prior-seed", type=int)
    validate.add_argument("--dirichlet-particle-seed", type=int)
    validate.add_argument("--protocol-sha256")
    validate.add_argument("--governing-gate-sha256")
    resolve = sub.add_parser("resolve-evidence")
    resolve.add_argument("--gate", type=Path, required=True)
    resolve.add_argument("--prior-seed", type=int)
    resolve.add_argument("--dirichlet-particle-seed", type=int)
    resolve.add_argument("--expected-head")
    resolve.add_argument("--protocol-sha256")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_scaleout_plan(
            g1_report=args.g1_report,
            run_root=args.run_root,
            cache_dir=args.cache_dir,
            expected_head=args.expected_head,
            ladder_rung=args.ladder_rung,
            prior_seed=args.prior_seed,
            dirichlet_particle_seed=args.dirichlet_particle_seed,
            protocol_sha256=args.protocol_sha256,
            governing_gate_sha256=args.governing_gate_sha256,
        )
    elif args.command == "validate-task":
        payload = validate_scaleout_task(
            args.plan,
            task_id=args.task_id,
            expected_head=args.expected_head,
            prior_seed=args.prior_seed,
            dirichlet_particle_seed=args.dirichlet_particle_seed,
            protocol_sha256=args.protocol_sha256,
            governing_gate_sha256=args.governing_gate_sha256,
        )
    else:
        _validate_json_checksum(args.gate)
        payload = resolve_phase_s_evidence_target(
            args.gate,
            prior_seed=args.prior_seed,
            dirichlet_particle_seed=args.dirichlet_particle_seed,
            expected_head=args.expected_head,
            protocol_sha256=args.protocol_sha256,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
