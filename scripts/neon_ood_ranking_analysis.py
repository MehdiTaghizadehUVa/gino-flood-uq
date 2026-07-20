#!/usr/bin/env python3
"""Event-level OOD evidence for NEON epistemic uncertainty."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import statistics
import hashlib
from pathlib import Path
from typing import Any, Sequence

RMSE_KEY = "ensemble_mean_rmse"
VAR_KEY = "variance_epistemic_anova_corrected_mean"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    rows = payload.get("per_family")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty per_family list.")
    parsed = []
    for row in rows:
        event_id = str(row.get("family_id", ""))
        rmse = float(row[RMSE_KEY])
        variance = float(row[VAR_KEY])
        if not event_id or not math.isfinite(rmse) or rmse < 0:
            raise ValueError(f"{path}: invalid event RMSE row {row!r}.")
        if not math.isfinite(variance) or variance < 0:
            raise ValueError(f"{path}: invalid epistemic variance row {row!r}.")
        parsed.append(
            {
                "family_id": event_id,
                "rmse_m": rmse,
                "epistemic_std_m": math.sqrt(variance),
            }
        )
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xm, ym = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - xm) ** 2 for a in x) * sum((b - ym) ** 2 for b in y)
    )
    return float("nan") if denominator <= 0 else numerator / denominator


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = 0.5 * (start + 1 + stop)
        for pos in range(start, stop):
            ranks[order[pos]] = rank
        start = stop
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return _pearson(_ranks(x), _ranks(y))


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def _bootstrap_correlations(
    x: Sequence[float],
    y: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    if samples < 20:
        raise ValueError("bootstrap_samples must be at least 20.")
    rng = random.Random(int(seed))
    n = len(x)
    pearson, spearman = [], []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        xb = [x[i] for i in indices]
        yb = [y[i] for i in indices]
        p, s = _pearson(xb, yb), _spearman(xb, yb)
        if math.isfinite(p):
            pearson.append(p)
        if math.isfinite(s):
            spearman.append(s)
    return {
        "pearson_95_ci": [_percentile(pearson, 0.025), _percentile(pearson, 0.975)],
        "spearman_95_ci": [_percentile(spearman, 0.025), _percentile(spearman, 0.975)],
    }


def _permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    observed = abs(_spearman(x, y))
    n = len(y)
    if n <= 8:
        permutations = itertools.permutations(y)
        mode = "exact"
    else:
        if samples < 20:
            raise ValueError("permutation_samples must be at least 20.")
        rng = random.Random(int(seed))
        permutations = (
            [y[index] for index in rng.sample(range(n), n)] for _ in range(samples)
        )
        mode = "monte_carlo"
    exceed, count = 0, 0
    for permuted in permutations:
        score = abs(_spearman(x, permuted))
        if math.isfinite(score) and score >= observed - 1.0e-15:
            exceed += 1
        count += 1
    pvalue = (exceed + (1 if mode == "monte_carlo" else 0)) / (
        count + (1 if mode == "monte_carlo" else 0)
    )
    return {"mode": mode, "samples": count, "two_sided_pvalue": pvalue}


def _risk_coverage(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, float]], float]:
    by_uncertainty = sorted(rows, key=lambda row: row["epistemic_std_m"])
    by_oracle = sorted(rows, key=lambda row: row["rmse_m"])
    curve = []
    gaps = []
    n = len(rows)
    for retained in range(1, n + 1):
        risk = statistics.fmean(row["rmse_m"] for row in by_uncertainty[:retained])
        oracle = statistics.fmean(row["rmse_m"] for row in by_oracle[:retained])
        curve.append(
            {
                "coverage": retained / n,
                "retained_events": retained,
                "rmse_m": risk,
                "oracle_rmse_m": oracle,
            }
        )
        gaps.append(risk - oracle)
    return curve, statistics.fmean(gaps)


def analyze_ood_ranking(
    ood_metrics: Path,
    id_metrics: Path,
    *,
    expected_ood_events: int | None = 13,
    bootstrap_samples: int = 10000,
    permutation_samples: int = 100000,
    seed: int = 20260713,
    analysis_git_head: str | None = None,
    protocol_sha256: str | None = None,
    stage2_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    ood = _read_rows(ood_metrics)
    iid = _read_rows(id_metrics)
    if expected_ood_events is not None and len(ood) != int(expected_ood_events):
        raise ValueError(
            f"expected {expected_ood_events} OOD events, found {len(ood)}."
        )
    if len(ood) < 3:
        raise ValueError("OOD ranking requires at least three events.")
    x = [row["epistemic_std_m"] for row in ood]
    y = [row["rmse_m"] for row in ood]
    pearson, spearman = _pearson(x, y), _spearman(x, y)
    uncertainty_top3 = {
        row["family_id"] for row in sorted(ood, key=lambda row: row["epistemic_std_m"])[-3:]
    }
    error_top3 = {
        row["family_id"] for row in sorted(ood, key=lambda row: row["rmse_m"])[-3:]
    }
    leave_one_out = []
    for excluded in range(len(ood)):
        keep = [idx for idx in range(len(ood)) if idx != excluded]
        leave_one_out.append(
            {
                "excluded_family_id": ood[excluded]["family_id"],
                "pearson": _pearson([x[i] for i in keep], [y[i] for i in keep]),
                "spearman": _spearman([x[i] for i in keep], [y[i] for i in keep]),
            }
        )
    curve, sparsification_gap = _risk_coverage(ood)
    uncertainty_ranks = _ranks(x)
    error_ranks = _ranks(y)
    rank_table = [
        {
            **row,
            "epistemic_std_rank_ascending": uncertainty_ranks[idx],
            "rmse_rank_ascending": error_ranks[idx],
            "top3_epistemic": row["family_id"] in uncertainty_top3,
            "top3_error": row["family_id"] in error_top3,
        }
        for idx, row in enumerate(ood)
    ]
    ood_std = statistics.fmean(x)
    id_std = statistics.fmean(row["epistemic_std_m"] for row in iid)
    ood_rmse = statistics.fmean(y)
    id_rmse = statistics.fmean(row["rmse_m"] for row in iid)
    if id_std <= 0 or id_rmse <= 0:
        raise ValueError("ID means must be positive for OOD/ID ratios.")
    if stage2_checkpoint_sha256 is not None:
        ood_payload = json.loads(Path(ood_metrics).read_text(encoding="utf-8"))
        actual_checkpoint = str(
            (ood_payload.get("plan") or {}).get("stage2_checkpoint_sha256", "")
        )
        if actual_checkpoint != str(stage2_checkpoint_sha256):
            raise ValueError(
                "OOD metrics/checkpoint mismatch: "
                f"{actual_checkpoint} != {stage2_checkpoint_sha256}."
            )
    result = {
        "schema_version": "neon_ood_ranking_v1",
        "ood_metrics": str(Path(ood_metrics)),
        "id_metrics": str(Path(id_metrics)),
        "n_ood_events": len(ood),
        "n_id_events": len(iid),
        "pearson_epistemic_std_rmse": pearson,
        "spearman_epistemic_std_rmse": spearman,
        **_bootstrap_correlations(x, y, samples=bootstrap_samples, seed=seed),
        "spearman_permutation_test": _permutation_test(
            x, y, samples=permutation_samples, seed=seed + 1
        ),
        "top3_error_event_recall": len(uncertainty_top3 & error_top3) / 3.0,
        "top3_epistemic_event_ids": sorted(uncertainty_top3),
        "top3_error_event_ids": sorted(error_top3),
        "ood_mean_epistemic_std_m": ood_std,
        "id_mean_epistemic_std_m": id_std,
        "ood_to_id_epistemic_std_ratio": ood_std / id_std,
        "ood_mean_rmse_m": ood_rmse,
        "id_mean_rmse_m": id_rmse,
        "ood_to_id_rmse_ratio": ood_rmse / id_rmse,
        "risk_coverage_curve": curve,
        "mean_sparsification_gap_m": sparsification_gap,
        "leave_one_out": leave_one_out,
        "rank_table": rank_table,
        "bootstrap_samples": int(bootstrap_samples),
        "permutation_samples_requested": int(permutation_samples),
        "seed": int(seed),
        "analysis_git_head": analysis_git_head,
        "protocol_sha256": protocol_sha256,
        "stage2_checkpoint_sha256": stage2_checkpoint_sha256,
        "ood_metrics_sha256": _sha256_file(Path(ood_metrics)),
        "id_metrics_sha256": _sha256_file(Path(id_metrics)),
    }
    return result


def _write_rank_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ood-metrics", type=Path, required=True)
    parser.add_argument("--id-metrics", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-ood-events", type=int, default=13)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--analysis-git-head")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--stage2-checkpoint-sha256")
    args = parser.parse_args(argv)
    result = analyze_ood_ranking(
        args.ood_metrics,
        args.id_metrics,
        expected_ood_events=args.expected_ood_events,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
        analysis_git_head=args.analysis_git_head,
        protocol_sha256=args.protocol_sha256,
        stage2_checkpoint_sha256=args.stage2_checkpoint_sha256,
    )
    prefix = args.output_prefix
    _atomic_json(prefix.with_suffix(".json"), result)
    _write_rank_csv(prefix.with_suffix(".csv"), result["rank_table"])
    print(json.dumps({
        "pearson": result["pearson_epistemic_std_rmse"],
        "spearman": result["spearman_epistemic_std_rmse"],
        "top3_recall": result["top3_error_event_recall"],
        "json": str(prefix.with_suffix(".json")),
        "csv": str(prefix.with_suffix(".csv")),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
