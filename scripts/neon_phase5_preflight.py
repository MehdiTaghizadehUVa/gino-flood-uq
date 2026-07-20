#!/usr/bin/env python3
"""Validate and freeze the immutable inputs for NEON Phase-5 diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from neuralop.flood.eval.neon_phase5 import (
    phase5_predeclared_protocol,
    write_checksummed_artifact,
)
from neon_phase5_runtime import file_sha256, require_clean_repository


V4_BASE_HEAD = "37633fe205334da1668cb30d491af5e0b0d6c761"
B3_TRAINING_HEAD = "663194621817b18491354eb64a8b5850796036c1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "config",
        "bundle",
        "checkpoint",
        "history",
        "preflight",
        "cache_dir",
        "output_dir",
        "expected_head",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    return parser.parse_args()


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _family_hash(values: list[str]) -> str:
    encoded = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = _args()
    head = require_clean_repository(expected_head=args.expected_head)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", V4_BASE_HEAD, head], check=False
    )
    if ancestor.returncode != 0:
        raise RuntimeError(f"Phase-5 HEAD {head} does not descend from {V4_BASE_HEAD}.")

    files = {
        "config": Path(args.config),
        "bundle": Path(args.bundle),
        "checkpoint": Path(args.checkpoint),
        "history": Path(args.history),
        "preflight": Path(args.preflight),
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Phase-5 inputs: {missing}")
    cache = Path(args.cache_dir)
    if not cache.is_dir():
        raise FileNotFoundError(f"missing Phase-5 cache: {cache}")

    source = files["checkpoint"].parent
    source_head_file = source / "git_head.txt"
    if not source_head_file.is_file():
        raise FileNotFoundError(f"missing B3 source git head: {source_head_file}")
    source_head = source_head_file.read_text(encoding="utf-8").strip()
    if source_head != B3_TRAINING_HEAD:
        raise RuntimeError(
            f"unexpected B3 training head {source_head}; expected {B3_TRAINING_HEAD}."
        )
    source_status = source / "git_status.txt"
    if source_status.is_file() and source_status.read_text(encoding="utf-8").strip():
        raise RuntimeError("B3 source was trained from a dirty repository.")

    history = _json(files["history"])
    training_ids = [str(value) for value in history.get("train_family_ids", [])]
    if len(training_ids) != 450 or len(set(training_ids)) != 450:
        raise ValueError("Phase-5 B3 history must contain 450 unique fit family IDs.")
    validated = _json(files["preflight"])
    if str(validated.get("ladder_rung", "")).upper() != "B3":
        raise ValueError("Phase-5 source preflight must be the selected B3 rung.")
    if str(validated.get("cache_dir")) != str(cache):
        raise ValueError("B3 preflight and requested cache directory differ.")

    canonical = sorted(cache.glob("canonical_*"))
    if len(canonical) != 1 or not canonical[0].is_dir():
        raise ValueError(
            "Phase-5 requires exactly one canonical deterministic-feature namespace."
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_decision = source.parent / f"g1_stop_loss_{source_head[:7]}" / "DECISION.txt"
    if not source_decision.is_file() or "STOP_AFTER_G1" not in source_decision.read_text(
        encoding="utf-8"
    ):
        raise ValueError("missing or invalid original STOP_AFTER_G1 decision artifact.")
    quarantine = output / "QUARANTINE.json"
    if not quarantine.is_file() or not quarantine.with_suffix(".json.sha256").is_file():
        raise ValueError("Phase-5 preflight requires a checksummed quarantine report.")
    protocol_path = output / "PROTOCOL.json"
    protocol_sha = write_checksummed_artifact(
        protocol_path, phase5_predeclared_protocol()
    )
    payload = {
        "schema_version": "neon_phase5_preflight_v1",
        "analysis_git_head": head,
        "v4_base_head": V4_BASE_HEAD,
        "b3_training_git_head": source_head,
        "input_paths": {key: str(path) for key, path in files.items()},
        "input_sha256": {key: file_sha256(path) for key, path in files.items()},
        "cache_dir": str(cache),
        "canonical_cache_dir": str(canonical[0]),
        "fit_family_count": len(training_ids),
        "fit_family_ids_sha256": _family_hash(training_ids),
        "validation_family_contract": "last_50_sorted_training_package_families",
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "source_decision": str(source_decision),
        "quarantine_report": str(quarantine),
        "quarantine_report_sha256": file_sha256(quarantine),
        "quarantine_policy": "pre-G1 B4/B5 products are descriptive-only and excluded",
    }
    write_checksummed_artifact(output / "PREFLIGHT.json", payload)
    print(output / "PREFLIGHT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
