#!/usr/bin/env python3
"""Estimate replicated NEON epistemic-spread contraction with training-set size."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Sequence

N_VALUES = (25, 50, 100, 250, 400)
N_REPLICATES = 5
STD_KEY = "val_total_epistemic_std_physical"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _best_row(payload: dict[str, Any], *, path: Path) -> dict[str, Any]:
    best_epoch = int(payload["best_epoch"])
    rows = [row for row in payload.get("history", []) if int(row.get("epoch", -1)) == best_epoch]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one history row for best_epoch={best_epoch}.")
    return rows[0]


def _fit_gamma(points: Sequence[tuple[int, float]]) -> tuple[float, float]:
    xs = [math.log(float(n)) for n, _ in points]
    ys = [math.log(float(std)) for _, std in points]
    xbar = statistics.fmean(xs)
    ybar = statistics.fmean(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    if denominator <= 0:
        raise ValueError("contraction fit requires at least two distinct N values.")
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator
    intercept = ybar - slope * xbar
    return -slope, intercept


def _bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if samples < 20:
        raise ValueError("bootstrap_samples must be at least 20.")
    rng = random.Random(int(seed))
    n = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(int(samples))
    )
    lo = draws[max(0, int(0.025 * samples))]
    hi = draws[min(samples - 1, int(math.ceil(0.975 * samples)) - 1)]
    return float(lo), float(hi)


def analyze_scaleout(
    run_root: Path,
    *,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20260713,
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    replicate_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for replicate in range(N_REPLICATES):
        points: list[tuple[int, float]] = []
        previous_ids: list[str] | None = None
        for n_train in N_VALUES:
            path = run_root / f"rep{replicate}" / f"n{n_train}" / "history.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing scale-out history: {path}")
            payload = json.loads(path.read_text())
            if int(payload.get("n_train", -1)) != n_train:
                raise ValueError(f"{path}: n_train metadata mismatch.")
            if int(payload.get("subset_replicate", -1)) != replicate:
                raise ValueError(f"{path}: subset_replicate metadata mismatch.")
            if str(payload.get("ladder_rung", "")).upper() != "B3":
                raise ValueError(f"{path}: contraction analysis requires B3 histories.")
            family_ids = [str(value) for value in payload.get("train_family_ids", [])]
            if len(family_ids) != n_train or len(set(family_ids)) != n_train:
                raise ValueError(f"{path}: invalid train_family_ids.")
            if previous_ids is not None and family_ids[: len(previous_ids)] != previous_ids:
                raise ValueError(
                    f"{path}: training families violate the nested prefix contract."
                )
            previous_ids = family_ids
            row = _best_row(payload, path=path)
            if STD_KEY not in row:
                raise ValueError(f"{path}: best epoch is missing {STD_KEY}.")
            std = float(row[STD_KEY])
            if not math.isfinite(std) or std <= 0:
                raise ValueError(f"{path}: {STD_KEY} must be finite and positive.")
            points.append((n_train, std))
            point_rows.append(
                {
                    "replicate": replicate,
                    "n_train": n_train,
                    "best_epoch": int(payload["best_epoch"]),
                    "best_val_fit": float(payload["best_val_fit"]),
                    "epistemic_std_m": std,
                    "history_path": str(path),
                }
            )
        gamma, intercept = _fit_gamma(points)
        replicate_rows.append(
            {
                "replicate": replicate,
                "gamma": gamma,
                "log_intercept": intercept,
                "nested_prefixes_valid": True,
                "points": [
                    {"n_train": n_train, "epistemic_std_m": std}
                    for n_train, std in points
                ],
            }
        )
    gammas = [float(row["gamma"]) for row in replicate_rows]
    gamma_mean = statistics.fmean(gammas)
    gamma_std = statistics.stdev(gammas) if len(gammas) > 1 else 0.0
    ci_low, ci_high = _bootstrap_mean_ci(
        gammas, samples=int(bootstrap_samples), seed=int(bootstrap_seed)
    )
    return {
        "schema_version": "neon_contraction_analysis_v1",
        "run_root": str(run_root),
        "metric": STD_KEY,
        "model": "log(std_epi_m) = intercept_replicate - gamma * log(N)",
        "n_values": list(N_VALUES),
        "n_replicates": N_REPLICATES,
        "gamma_mean": gamma_mean,
        "gamma_std": gamma_std,
        "gamma_standard_error": gamma_std / math.sqrt(len(gammas)),
        "gamma_bootstrap_95_ci": [ci_low, ci_high],
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "replicates": replicate_rows,
        "points": point_rows,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    args = parser.parse_args(argv)
    result = analyze_scaleout(
        args.run_root,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    prefix = args.output_prefix or (args.run_root / "contraction_analysis")
    _atomic_json(prefix.with_suffix(".json"), result)
    _write_csv(prefix.with_suffix(".csv"), result["points"])
    print(json.dumps({
        "gamma_mean": result["gamma_mean"],
        "gamma_bootstrap_95_ci": result["gamma_bootstrap_95_ci"],
        "json": str(prefix.with_suffix(".json")),
        "csv": str(prefix.with_suffix(".csv")),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

