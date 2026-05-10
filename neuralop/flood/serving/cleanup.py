"""Retention cleanup CLI for FGN serving deployments."""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

from neuralop.flood.serving.factory import build_orchestrator


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Expire old unpinned FGN serving run artifacts.")
    parser.add_argument("--retention-days", type=int, default=int(os.environ.get("FGN_RETENTION_DAYS", "30")))
    args = parser.parse_args(argv)
    orchestrator = build_orchestrator(preload_models=False)
    result = orchestrator.expire_due_runs(retention_days=int(args.retention_days))
    print(json.dumps({"expired_run_ids": list(result.expired_run_ids), "skipped_run_ids": list(result.skipped_run_ids)}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
