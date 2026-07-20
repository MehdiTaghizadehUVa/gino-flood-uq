#!/usr/bin/env python3
"""Mark unsanctioned pre-G1 B4/B5 products as excluded, without deleting data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuralop.flood.eval.neon_phase5 import write_checksummed_artifact
from neon_phase5_runtime import file_sha256, require_clean_repository


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    head = require_clean_repository(expected_head=args.expected_head)
    decision = Path(args.decision)
    if not decision.is_file():
        raise FileNotFoundError(decision)
    text = decision.read_text(encoding="utf-8")
    if "STOP_AFTER_G1" not in text:
        raise ValueError("quarantine requires the original STOP_AFTER_G1 decision.")
    root = Path(args.run_root)
    candidates = sorted(
        path
        for pattern in ("b4_*", "b5_*")
        for path in root.glob(pattern)
        if path.is_dir()
    )
    rows = []
    for path in candidates:
        marker = path / "NEON_PHASE5_QUARANTINED.json"
        payload = {
            "schema_version": "neon_phase5_quarantine_marker_v1",
            "status": "descriptive_only_excluded_from_phase5_gates",
            "directory": str(path),
            "decision": str(decision),
            "decision_sha256": file_sha256(decision),
            "analysis_git_head": head,
            "data_deleted": False,
        }
        write_checksummed_artifact(marker, payload)
        rows.append({"directory": str(path), "marker": str(marker)})
    report = {
        "schema_version": "neon_phase5_quarantine_v1",
        "analysis_git_head": head,
        "decision": str(decision),
        "decision_sha256": file_sha256(decision),
        "count": len(rows),
        "products": rows,
    }
    write_checksummed_artifact(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
