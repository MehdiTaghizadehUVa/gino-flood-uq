#!/usr/bin/env python3
"""Preflight and validate the conditional replicated NEON N-sweep."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


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
) -> dict[str, Any]:
    """Validate G1 and materialize all 25 resolved B3 task manifests."""
    g1_report = Path(g1_report).resolve()
    gate = json.loads(g1_report.read_text())
    if gate.get("schema_version") != "neon_g1_gate_v1" or gate.get("gate_passed") is not True:
        raise ValueError("scale-out requires a passing G1 (neon_g1_gate_v1) report.")
    if not expected_head:
        raise ValueError("expected_head must be non-empty.")
    run_root = Path(run_root).resolve()
    cache_dir = Path(cache_dir).resolve()
    train_script = _training_script_module()
    config = train_script._resolved_ladder_config(
        "B3", prior_scale=prior_scale, d_e=d_e, n_epochs=n_epochs
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
                rung="B3",
                n_train=n_train,
                output_dir=output_dir,
                cache_dir=cache_dir,
                subset_replicate=replicate,
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
        "cache_dir": str(cache_dir),
        "nested_subset_contract": "prefixes_of_one_seeded_permutation_per_replicate",
        "n_values": list(N_VALUES),
        "n_replicates": N_REPLICATES,
        "tasks": tasks,
    }
    _atomic_json(run_root / "scaleout_plan.json", plan)
    return plan


def validate_scaleout_task(
    plan_path: Path, *, task_id: int, expected_head: str
) -> dict[str, Any]:
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema_version") != "neon_scaleout_plan_v1":
        raise ValueError("invalid scale-out plan schema.")
    if plan.get("expected_git_head") != expected_head:
        raise ValueError("scale-out Git HEAD differs from preflight.")
    if _sha256(Path(plan["g1_report"])) != plan.get("g1_report_sha256"):
        raise ValueError("G1 report changed after scale-out preflight.")
    gate = json.loads(Path(plan["g1_report"]).read_text())
    if gate.get("gate_passed") is not True:
        raise ValueError("G1 report no longer passes.")
    matches = [row for row in plan["tasks"] if int(row["task_id"]) == int(task_id)]
    if len(matches) != 1:
        raise ValueError(f"expected one task_id={task_id}; got {len(matches)}.")
    task = matches[0]
    preflight = json.loads(Path(task["preflight_path"]).read_text())
    if preflight.get("config") != task.get("config"):
        raise ValueError("task config differs from its preflight manifest.")
    return task


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--g1-report", type=Path, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument("--expected-head", required=True)
    validate = sub.add_parser("validate-task")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--task-id", type=int, required=True)
    validate.add_argument("--expected-head", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_scaleout_plan(
            g1_report=args.g1_report,
            run_root=args.run_root,
            cache_dir=args.cache_dir,
            expected_head=args.expected_head,
        )
    else:
        payload = validate_scaleout_task(
            args.plan, task_id=args.task_id, expected_head=args.expected_head
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
