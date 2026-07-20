#!/usr/bin/env python3
"""Summarize the legacy non-replicated NEON N-sweep after CRN remapping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence


N_VALUES = (25, 50, 100, 250, 400)
REMAP_LABEL = "legacy frozen rollout, estimator repaired"


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _family_hash(values: Sequence[str]) -> str | None:
    if not values:
        return None
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _fit_gamma(rows: Sequence[dict[str, Any]]) -> tuple[float, float]:
    x = [math.log(float(row["n_train"])) for row in rows]
    y = [math.log(float(row["posterior_epistemic_std_m"])) for row in rows]
    xm = sum(x) / len(x)
    ym = sum(y) / len(y)
    denominator = sum((value - xm) ** 2 for value in x)
    if denominator <= 0.0:
        raise ValueError("legacy contraction fit requires distinct training sizes")
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / denominator
    return -slope, ym - slope * xm


def analyze_legacy_sweep(run_root: Path, source_root: Path) -> dict[str, Any]:
    """Return a provenance-explicit descriptive covariate table and slope."""

    run_root = Path(run_root)
    source_root = Path(source_root)
    rows = []
    for n_train in N_VALUES:
        run = run_root / f"n{n_train}"
        source = source_root / f"tr_n{n_train}"
        remap_path = run / "remap" / "legacy_estimator_remap.json"
        metrics_path = run / "output" / "neon_eval_metrics.json"
        history_path = source / "history.json"
        remap = _json(remap_path)
        metrics = _json(metrics_path)
        history = _json(history_path)
        if remap.get("label") != REMAP_LABEL:
            raise ValueError(f"{remap_path}: missing estimator-repair label")
        if int(remap.get("n_families", -1)) != 50:
            raise ValueError(f"{remap_path}: legacy remap must contain 50 families")
        variance = float(
            remap["aggregate"]["variance_epistemic_crossed_mean"]
        )
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError(f"{remap_path}: crossed epistemic variance must be positive")
        family_ids = [str(value) for value in history.get("train_family_ids", [])]
        subset_hash = _family_hash(family_ids)
        aggregate = dict(metrics.get("aggregate") or {})
        checkpoint = dict(metrics.get("checkpoint_metadata") or {})
        missing = []
        if subset_hash is None:
            missing.append("subset_sha256")
        # These were not exported by the legacy checkpoints/artifacts. Keeping
        # explicit nulls prevents a later analysis from treating guesses as
        # observed covariates.
        missing.extend(("prior_scale_m", "retention_fraction", "cancellation_fraction"))
        row = {
            "n_train": int(n_train),
            "posterior_epistemic_std_m": math.sqrt(variance),
            "base_rmse_m": aggregate.get("base_model0_rmse"),
            "stage2_rmse_m": aggregate.get("ensemble_mean_rmse"),
            "stage2_fair_crps_m": aggregate.get("marginal_fair_crps"),
            "prior_alpha_normalized": checkpoint.get("alpha"),
            "prior_scale_m": None,
            "retention_fraction": None,
            "cancellation_fraction": None,
            "best_epoch": int(history["best_epoch"]),
            "epoch_count": len(history.get("history", [])),
            "best_val_fit_normalized": float(history["best_val_fit"]),
            "converged_before_last_epoch": int(history["best_epoch"])
            < max(0, len(history.get("history", [])) - 1),
            "subset_sha256": subset_hash,
            "subset_family_ids_recorded": bool(family_ids),
            "checkpoint_history": str(history_path),
            "evaluation_metrics": str(metrics_path),
            "estimator_remap": str(remap_path),
            "missing_covariates": missing,
        }
        rows.append(row)
    gamma, intercept = _fit_gamma(rows)
    return {
        "schema_version": "neon_phase5_legacy_sweep_v1",
        "status": "preliminary_descriptive_nonreplicated",
        "approved_label": (
            "preliminary legacy N-sweep; frozen rollout, crossed estimator repaired"
        ),
        "replicate_count": 1,
        "gamma_hat": gamma,
        "log_intercept": intercept,
        "n_values": list(N_VALUES),
        "scientific_limitations": [
            "one training subset per N; no across-subset uncertainty",
            "legacy training histories omit modern prior/retention/cancellation covariates",
            "not eligible for GD0, GP1, pilot acceptance, or Phase-S claims",
        ],
        "covariates": rows,
    }


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


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = [
        {
            **row,
            "missing_covariates": ";".join(row["missing_covariates"]),
        }
        for row in rows
    ]
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
        writer.writeheader()
        writer.writerows(serial)
    os.replace(tmp, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_legacy_sweep(args.run_root, args.source_root)
    _atomic_json(args.output_prefix.with_suffix(".json"), result)
    _write_csv(args.output_prefix.with_suffix(".csv"), result["covariates"])
    print(json.dumps({"gamma_hat": result["gamma_hat"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
