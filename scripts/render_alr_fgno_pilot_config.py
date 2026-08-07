#!/usr/bin/env python3
"""Render one grounded ALR-FGNO pilot training configuration."""

import argparse
import os
from pathlib import Path

import yaml


_RUNS = {
    "smoke": {
        "train_families": 2,
        "validation_families": 2,
        "warmup_epochs": 1,
        "joint_epochs": 1,
        "batch_size": 1,
        "k_eval": 3,
    },
    "pilot50": {"train_families": 50, "validation_families": 50},
    "n150": {"train_families": 150, "validation_families": 50},
    "full450": {"train_families": 450, "validation_families": 50},
}


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(str(path.expanduser())))


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("flood"), dict):
        raise ValueError(f"Expected a top-level flood mapping in {path}.")
    return payload


def _dump_compatible(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        try:
            yaml.safe_dump(payload, handle, sort_keys=False)
        except TypeError:
            yaml.safe_dump(payload, handle)


def render_config(*, base: Path, output: Path, run_dir: Path, run_kind: str) -> Path:
    if run_kind not in _RUNS:
        raise ValueError(f"Unknown run kind {run_kind!r}; choose from {sorted(_RUNS)}.")
    payload = _load(base)
    flood = payload["flood"]
    spec = _RUNS[run_kind]
    training = flood["training"]["anchored_low_rank"]
    training["train_family_limit"] = int(spec["train_families"])
    training["validation_family_count"] = int(spec["validation_families"])
    training["adapter_warmup_epochs"] = int(spec.get("warmup_epochs", 5))
    training["joint_finetune_epochs"] = int(spec.get("joint_epochs", 25))
    training["k_eval"] = int(spec.get("k_eval", 15))
    flood["data"]["batch_size"] = int(spec.get("batch_size", 16))
    flood["checkpoint"]["save_dir"] = str(run_dir / "checkpoints")
    flood["checkpoint"]["resume_from_dir"] = None
    flood["rollout"]["out_dir"] = str(run_dir / "rollout")
    flood["rollout"]["forecast_artifact_dir"] = str(
        run_dir / "rollout" / "forecast_artifacts" / "test_raw"
    )
    flood["log_file"] = str(run_dir / "training.log")
    flood["wandb"]["name"] = f"alr_fgno_{run_kind}_{run_dir.name}"
    flood["wandb"]["group"] = "coastal-alr-fgno-pilot"
    flood["wandb"]["log"] = False  # Cluster jobs must not depend on an unstaged API key.
    flood["use_progress_bar"] = False
    flood["alr_pilot"] = {
        "run_kind": run_kind,
        "train_families": int(spec["train_families"]),
        "validation_families": int(spec["validation_families"]),
    }
    _dump_compatible(payload, output)

    rendered = _load(output)["flood"]
    alr = rendered["model"]["anchored_low_rank"]
    # The gate protects the adapter parameter budget, not the literal 4x4 shape.
    # Adapter parameters scale as num_particles * rank, so trading rank for
    # particles keeps the budget fixed -- which is what Phase D needs, since the
    # diagnosis says the adapter subspace is already expressive enough and what
    # is short is the number of samples in the epistemic estimate.
    # operator_app enforces adapter_trainable_fraction < 0.25 at build time;
    # mirror that here in the units the renderer can see (M*rank, with the
    # 4x4 pilot at fraction 0.1157 as the reference point).
    if not bool(alr.get("enabled")):
        raise RuntimeError("Rendered pilot must enable ALR particles.")
    num_particles = int(alr.get("num_particles", 0))
    rank = int(alr.get("rank", 0))
    if num_particles < 2:
        raise RuntimeError("Rendered pilot requires at least two ALR particles.")
    if rank < 1:
        raise RuntimeError("Rendered pilot requires rank >= 1 adapters.")
    capacity = num_particles * rank
    reference_capacity = 4 * 4
    reference_fraction = 0.11573977388258551
    projected_fraction = reference_fraction * capacity / reference_capacity
    if projected_fraction >= 0.25:
        raise RuntimeError(
            f"Rendered pilot exceeds the adapter parameter gate: "
            f"num_particles={num_particles} rank={rank} projects to an adapter "
            f"fraction of {projected_fraction:.4f}, which is not below 0.25."
        )
    if not bool(rendered.get("inverse_test")):
        raise RuntimeError("Rendered pilot requires physical-space validation.")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-kind", choices=sorted(_RUNS), required=True)
    args = parser.parse_args()
    render_config(
        base=_absolute_without_symlink_resolution(args.base),
        output=_absolute_without_symlink_resolution(args.output),
        run_dir=_absolute_without_symlink_resolution(args.run_dir),
        run_kind=args.run_kind,
    )
    print(_absolute_without_symlink_resolution(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
