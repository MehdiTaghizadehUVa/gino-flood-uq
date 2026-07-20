#!/usr/bin/env python3
"""Build and validate the immutable Phase-5 legacy N-sweep export plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


N_VALUES = (25, 50, 100, 250, 400)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(encoded)
    os.replace(tmp, path)
    digest = hashlib.sha256(encoded).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def _validate_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"plan checksum mismatch: expected {expected}, got {actual}")


def build_plan(
    *,
    config: Path,
    bundle: Path,
    source_root: Path,
    output_root: Path,
    expected_head: str,
    m_eval: int = 16,
    k_eval: int = 8,
    family_count: int = 50,
) -> dict[str, Any]:
    """Resolve every checkpoint/family task and pin its provenance."""

    config = _require_file(config)
    bundle = _require_file(bundle)
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if int(m_eval) < 2 or int(k_eval) < 2:
        raise ValueError("legacy crossed remap requires m_eval>=2 and k_eval>=2")
    if int(family_count) != 50:
        raise ValueError("the preliminary legacy sweep is defined on exactly 50 families")
    checkpoints = []
    tasks = []
    for n_train in N_VALUES:
        run = source_root / f"tr_n{n_train}"
        checkpoint = _require_file(run / "neon_stage2_best.pt")
        history = _require_file(run / "history.json")
        history_payload = json.loads(history.read_text(encoding="utf-8"))
        if int(history_payload.get("n_train", -1)) != int(n_train):
            raise ValueError(f"{history}: n_train does not match directory name")
        record = {
            "n_train": int(n_train),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "history": str(history),
            "history_sha256": _sha256(history),
            "output_dir": str(output_root / f"n{n_train}" / "output"),
        }
        checkpoints.append(record)
        # N-major task ordering lets the first checkpoint populate the shared
        # frozen-feature cache before later checkpoints consume it.
        for family_index in range(int(family_count)):
            tasks.append(
                {
                    "task_id": len(tasks),
                    "n_train": int(n_train),
                    "family_index": int(family_index),
                    "checkpoint": str(checkpoint),
                    "output_dir": record["output_dir"],
                }
            )
    return {
        "schema_version": "neon_phase5_legacy_export_plan_v1",
        "scientific_status": "preliminary_descriptive_nonreplicated",
        "git_head": str(expected_head),
        "config": str(config),
        "config_sha256": _sha256(config),
        "bundle": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "m_eval": int(m_eval),
        "k_eval": int(k_eval),
        "family_count": int(family_count),
        "sampling_design": "crossed_common_random_numbers",
        "checkpoints": checkpoints,
        "tasks": tasks,
    }


def load_and_validate_plan(
    path: Path, *, expected_head: str, task_id: int | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    path = _require_file(path)
    _validate_sidecar(path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "neon_phase5_legacy_export_plan_v1":
        raise ValueError("unsupported legacy export plan schema")
    if str(plan.get("git_head")) != str(expected_head):
        raise ValueError("legacy export plan Git head mismatch")
    for key in ("config", "bundle"):
        source = _require_file(Path(plan[key]))
        if _sha256(source) != str(plan[f"{key}_sha256"]):
            raise ValueError(f"legacy export {key} changed after preflight")
    selected = None
    if task_id is not None:
        tasks = plan.get("tasks") or []
        if int(task_id) < 0 or int(task_id) >= len(tasks):
            raise ValueError(f"task id {task_id} outside 0..{len(tasks) - 1}")
        selected = dict(tasks[int(task_id)])
        checkpoint = _require_file(Path(selected["checkpoint"]))
        checkpoint_record = next(
            row for row in plan["checkpoints"]
            if int(row["n_train"]) == int(selected["n_train"])
        )
        if _sha256(checkpoint) != str(checkpoint_record["checkpoint_sha256"]):
            raise ValueError("legacy checkpoint changed after preflight")
    return plan, selected


def validate_checkpoint_loads(plan: dict[str, Any], *, loader: Any) -> list[dict[str, Any]]:
    """Deserialize every unique legacy checkpoint before scheduler mutation."""

    records = []
    for row in plan.get("checkpoints") or []:
        checkpoint = _require_file(Path(row["checkpoint"]))
        if _sha256(checkpoint) != str(row["checkpoint_sha256"]):
            raise ValueError("legacy checkpoint changed after preflight")
        _module, metadata = loader(checkpoint, map_location="cpu")
        records.append(
            {
                "n_train": int(row["n_train"]),
                "checkpoint": str(checkpoint),
                "metadata_keys": sorted(str(key) for key in metadata),
            }
        )
    if len(records) != len(N_VALUES):
        raise ValueError(
            f"expected {len(N_VALUES)} unique legacy checkpoints, got {len(records)}"
        )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--bundle", type=Path, required=True)
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--expected-head", required=True)
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--m-eval", type=int, default=16)
    prepare.add_argument("--k-eval", type=int, default=8)
    prepare.add_argument("--family-count", type=int, default=50)
    validate = subparsers.add_parser("validate-task")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--expected-head", required=True)
    validate.add_argument("--task-id", type=int, required=True)
    validate.add_argument("--format", choices=("json", "tsv"), default="json")
    check = subparsers.add_parser("validate-final")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--expected-head", required=True)
    checkpoint_check = subparsers.add_parser("validate-checkpoints")
    checkpoint_check.add_argument("--plan", type=Path, required=True)
    checkpoint_check.add_argument("--expected-head", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = build_plan(
            config=args.config,
            bundle=args.bundle,
            source_root=args.source_root,
            output_root=args.output_root,
            expected_head=args.expected_head,
            m_eval=args.m_eval,
            k_eval=args.k_eval,
            family_count=args.family_count,
        )
        _atomic_json(args.plan, payload)
        print(json.dumps({"tasks": len(payload["tasks"]), "plan": str(args.plan)}))
        return 0
    plan, task = load_and_validate_plan(
        args.plan,
        expected_head=args.expected_head,
        task_id=(args.task_id if args.command == "validate-task" else None),
    )
    if args.command == "validate-task":
        assert task is not None
        if args.format == "tsv":
            print(
                "\t".join(
                    str(task[key])
                    for key in ("n_train", "family_index", "checkpoint", "output_dir")
                )
            )
        else:
            print(json.dumps(task, sort_keys=True))
    elif args.command == "validate-checkpoints":
        from neuralop.flood.neon import load_neon_stage2_checkpoint

        records = validate_checkpoint_loads(plan, loader=load_neon_stage2_checkpoint)
        print(json.dumps({"checkpoints": records, "valid": True}, sort_keys=True))
    else:
        print(json.dumps({"tasks": len(plan["tasks"]), "valid": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
