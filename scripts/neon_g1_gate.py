#!/usr/bin/env python3
"""Evaluate the NEON Stage-2 G1 skill/RMSE gate from saved B3 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(
    values: list[float], *, replicates: int, seed: int
) -> dict[str, float]:
    if not values:
        raise ValueError("paired RMSE differences are empty.")
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be >= 1.")
    rng = random.Random(int(seed))
    n = len(values)
    means = [fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(replicates)]
    return {
        "mean": fmean(values),
        "ci95_lower": _percentile(means, 0.025),
        "ci95_upper": _percentile(means, 0.975),
    }


def evaluate_g1(
    run_dir: Path,
    *,
    bootstrap_replicates: int = 20_000,
    seed: int = 20260713,
    expected_families: int = 50,
    max_cancellation_fraction: float = 0.80,
    crps_noninferiority_margin_m: float = 1.0e-4,
) -> dict[str, Any]:
    """Return the auditable G1 decision for a completed B3 run."""
    run_dir = Path(run_dir)
    history_path = run_dir / "history.json"
    payload = json.loads(history_path.read_text())
    best_epoch = int(payload["best_epoch"])
    rows = [row for row in payload["history"] if int(row["epoch"]) == best_epoch]
    if len(rows) != 1:
        raise ValueError(f"expected one history row for best_epoch={best_epoch}; got {len(rows)}.")
    row = rows[0]
    required = (
        "val_base_fair_crps_physical",
        "val_deterministic_head_fair_crps_physical",
        "val_mixture_fair_crps_physical",
        "val_base_rmse_physical",
        "val_deterministic_head_rmse_physical",
        "val_stage2_rmse_physical",
        "val_stage2_minus_base_rmse_physical",
        "val_cancellation_fraction",
        "val_prior_retention_ratio",
        "selection_rmse_margin_m",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"best-epoch history row is missing G1 metrics: {missing}.")
    metrics = {key: float(row[key]) for key in required}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("G1 metrics contain non-finite values.")

    pairs_path = run_dir / f"validation_rmse_pairs_epoch_{best_epoch + 1:04d}.json"
    pairs_payload = json.loads(pairs_path.read_text())
    pairs = list(pairs_payload.get("pairs", []))
    family_ids = [str(item["family_id"]) for item in pairs]
    if len(pairs) != int(expected_families):
        raise ValueError(
            f"expected {expected_families} paired validation families; got {len(pairs)}."
        )
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("paired validation family IDs are not unique.")
    required_pair_keys = (
        "stage2_minus_base_rmse_physical",
        "base_fair_crps_physical",
        "mixture_fair_crps_physical",
    )
    missing_pair_keys = sorted(
        {
            key
            for item in pairs
            for key in required_pair_keys
            if key not in item
        }
    )
    if missing_pair_keys:
        raise ValueError(
            "paired validation rows are missing Phase-5 metrics: "
            f"{missing_pair_keys}."
        )
    rmse_differences = [float(item["stage2_minus_base_rmse_physical"]) for item in pairs]
    crps_differences = [
        float(item["mixture_fair_crps_physical"])
        - float(item["base_fair_crps_physical"])
        for item in pairs
    ]
    paired_rmse = _paired_bootstrap(
        rmse_differences, replicates=int(bootstrap_replicates), seed=int(seed)
    )
    paired_crps = _paired_bootstrap(
        crps_differences,
        replicates=int(bootstrap_replicates),
        seed=int(seed) + 1,
    )
    rmse_margin_m = metrics["selection_rmse_margin_m"]
    crps_margin_m = float(crps_noninferiority_margin_m)
    if crps_margin_m < 0.0:
        raise ValueError("crps_noninferiority_margin_m must be nonnegative.")

    checks = {
        "crps_noninferior": paired_crps["ci95_upper"] <= crps_margin_m,
        "rmse_noninferior": paired_rmse["ci95_upper"] <= rmse_margin_m,
        "cancellation_below_warning_threshold": metrics["val_cancellation_fraction"]
        <= float(max_cancellation_fraction),
        "retention_finite_nonnegative": metrics["val_prior_retention_ratio"] >= 0.0,
    }
    table = [
        {
            "estimator": "Frozen model-0",
            "fair_crps_m": metrics["val_base_fair_crps_physical"],
            "rmse_m": metrics["val_base_rmse_physical"],
        },
        {
            "estimator": "Frozen model-0 + deterministic head",
            "fair_crps_m": metrics["val_deterministic_head_fair_crps_physical"],
            "rmse_m": metrics["val_deterministic_head_rmse_physical"],
        },
        {
            "estimator": "Frozen model-0 + full Stage-2",
            "fair_crps_m": metrics["val_mixture_fair_crps_physical"],
            "rmse_m": metrics["val_stage2_rmse_physical"],
        },
    ]
    return {
        "schema_version": "neon_g1_gate_v1",
        "run_dir": str(run_dir),
        "best_epoch_zero_based": best_epoch,
        "n_paired_families": len(pairs),
        "rmse_margin_m": rmse_margin_m,
        "crps_noninferiority_margin_m": crps_margin_m,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "paired_rmse_difference_m": {
            **paired_rmse,
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(seed),
            "margin_m": rmse_margin_m,
            "ucb95_within_margin": paired_rmse["ci95_upper"] <= rmse_margin_m,
            "zero_margin_sensitivity": paired_rmse["ci95_upper"] <= 0.0,
        },
        "paired_crps_difference_m": {
            **paired_crps,
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(seed) + 1,
            "margin_m": crps_margin_m,
            "ucb95_within_margin": paired_crps["ci95_upper"] <= crps_margin_m,
            "zero_margin_sensitivity": paired_crps["ci95_upper"] <= 0.0,
        },
        "diagnostics": {
            "cancellation_fraction": metrics["val_cancellation_fraction"],
            "prior_retention_ratio": metrics["val_prior_retention_ratio"],
        },
        "comparison_table": table,
        "deep_ensemble_reference": None,
    }


def _write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "g1_gate.json", report)
    with (output_dir / "g1_gate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("estimator", "fair_crps_m", "rmse_m"))
        writer.writeheader()
        writer.writerows(report["comparison_table"])
    lines = [
        "| Estimator | Fair CRPS (m) | RMSE (m) |",
        "|---|---:|---:|",
    ]
    for row in report["comparison_table"]:
        lines.append(
            f"| {row['estimator']} | {row['fair_crps_m']:.6f} | {row['rmse_m']:.6f} |"
        )
    lines.extend(("", f"**G1 passed:** `{str(report['gate_passed']).lower()}`"))
    (output_dir / "g1_gate.md").write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--expected-families", type=int, default=50)
    args = parser.parse_args(argv)
    report = evaluate_g1(
        args.run_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        expected_families=args.expected_families,
    )
    output_dir = args.output_dir or args.run_dir / "g1_gate"
    _write_reports(output_dir, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
