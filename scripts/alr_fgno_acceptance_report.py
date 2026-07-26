#!/usr/bin/env python3
"""Build the ALR-FGNO pilot acceptance report from member-level artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from neuralop.flood.eval.alr_fgn import alr_crossed_variance_components
from neuralop.flood.eval.scientific_calibration import (
    empirical_crps_mean,
    list_forecast_artifacts,
    load_forecast_artifact,
)


def _nested_members(artifact):
    pred = np.asarray(artifact["pred_members_wd"], dtype=np.float64)
    epi = artifact.get("member_epistemic_id")
    ale = artifact.get("member_aleatory_id")
    if epi is None or ale is None:
        raise ValueError(f"ALR member IDs are missing from {artifact['path']}.")
    epi = np.asarray(epi, dtype=np.int64)
    ale = np.asarray(ale, dtype=np.int64)
    particle_ids = np.unique(epi)
    aleatory_ids = np.unique(ale)
    nested = np.empty((particle_ids.size, aleatory_ids.size, *pred.shape[1:]), dtype=np.float64)
    for m_index, m in enumerate(particle_ids):
        for k_index, k in enumerate(aleatory_ids):
            matches = np.flatnonzero((epi == m) & (ale == k))
            if matches.size != 1:
                raise ValueError(f"Expected one member for particle={m}, aleatory={k}.")
            nested[m_index, k_index] = pred[matches[0]]
    return nested


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) <= 0 or np.std(y[valid]) <= 0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _event_metrics(path):
    artifact = load_forecast_artifact(path, load_members=True)
    has_nested_ids = (
        artifact.get("member_epistemic_id") is not None
        and artifact.get("member_aleatory_id") is not None
    )
    nested = _nested_members(artifact) if has_nested_ids else None
    pred = (
        nested.reshape(-1, *nested.shape[2:])
        if nested is not None
        else np.asarray(artifact["pred_members_wd"], dtype=np.float64)
    )
    ref = np.asarray(artifact["ref_members_wd"], dtype=np.float64)
    wettable = np.asarray(artifact["wettable_mask"], dtype=bool)
    if pred.shape[1:] != ref.shape[1:]:
        raise ValueError(f"Forecast/reference shape mismatch in {path}.")
    pred_mean = pred.mean(axis=0)
    ref_mean = ref.mean(axis=0)
    error = np.abs(pred_mean - ref_mean)
    rmse = float(np.sqrt(np.mean((pred_mean[:, wettable] - ref_mean[:, wettable]) ** 2)))
    crps = float(np.mean([
        empirical_crps_mean(pred[:, t, wettable], ref[:, t, wettable])
        for t in range(pred.shape[1])
    ]))

    epi_spread = []
    ale_spread = []
    front_epi = []
    front_error = []
    for t in (range(pred.shape[1]) if nested is not None else ()):
        components = alr_crossed_variance_components(
            torch.from_numpy(nested[:, :, t].astype(np.float32, copy=False))
        )
        epi_t = np.sqrt(components["variance_epistemic"].numpy().clip(min=0))
        ale_t = np.sqrt(components["variance_aleatory"].numpy().clip(min=0))
        epi_spread.append(float(np.mean(epi_t[wettable])))
        ale_spread.append(float(np.mean(ale_t[wettable])))
        front = wettable & (ref_mean[t] > 0.01) & (ref_mean[t] <= 0.10)
        if np.count_nonzero(front) >= 3:
            front_epi.append(epi_t[front])
            front_error.append(error[t, front])

    particle_means = (
        nested.mean(axis=1)[:, :, wettable].reshape(nested.shape[0], -1)
        if nested is not None
        else np.empty((0, 0))
    )
    correlations = []
    for i in range(particle_means.shape[0]):
        for j in range(i + 1, particle_means.shape[0]):
            correlations.append(_pearson(particle_means[i], particle_means[j]))
    wetfront_correlation = (
        _pearson(np.concatenate(front_epi), np.concatenate(front_error))
        if front_epi else float("nan")
    )
    return {
        "event_id": str(artifact["hydrograph_id"]),
        "rmse_m": rmse,
        "fair_crps_m": crps,
        "epistemic_spread_m": float(np.mean(epi_spread)) if epi_spread else float("nan"),
        "aleatory_spread_m": float(np.mean(ale_spread)) if ale_spread else float("nan"),
        "wetfront_epistemic_error_correlation": wetfront_correlation,
        "particle_correlation_mean": float(np.nanmean(correlations)) if correlations else float("nan"),
    }


def _artifact_metrics(root):
    return [_event_metrics(path) for path in list_forecast_artifacts(root)]


def _bootstrap_mean_ci(values, repetitions, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(repetitions), values.size), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def _paired_comparison(alr_rows, base_rows, repetitions, seed):
    base_by_id = {row["event_id"]: row for row in base_rows}
    pairs = [(row, base_by_id[row["event_id"]]) for row in alr_rows if row["event_id"] in base_by_id]
    if len(pairs) != len(alr_rows):
        raise ValueError("ALR and baseline artifact roots do not contain identical event IDs.")
    out = {}
    for metric in ("rmse_m", "fair_crps_m"):
        delta = [alr[metric] - base[metric] for alr, base in pairs]
        out[metric] = _bootstrap_mean_ci(delta, repetitions, seed)
    base_crps = float(np.mean([base["fair_crps_m"] for _, base in pairs]))
    out["fair_crps_percent_change"] = 100.0 * out["fair_crps_m"]["mean"] / max(base_crps, 1e-12)
    return out


def _historical_ranking(rows):
    if len(rows) < 3:
        return None
    rmse = np.asarray([row["rmse_m"] for row in rows])
    spread = np.asarray([row["epistemic_spread_m"] for row in rows])
    spearman = _pearson(_rankdata(rmse), _rankdata(spread))
    high_error = set(np.argsort(rmse)[-3:].tolist())
    high_uncertainty = set(np.argsort(spread)[-3:].tolist())
    loo = []
    for held_out in range(len(rows)):
        keep = np.arange(len(rows)) != held_out
        loo.append(_pearson(_rankdata(rmse[keep]), _rankdata(spread[keep])))
    return {
        "n_events": len(rows),
        "spearman_rmse_vs_epistemic_spread": spearman,
        "top3_overlap": len(high_error & high_uncertainty),
        "leave_one_out_positive": int(np.sum(np.asarray(loo) > 0)),
        "event_ranking": sorted(
            [
                {
                    "event_id": row["event_id"],
                    "rmse_m": row["rmse_m"],
                    "epistemic_spread_m": row["epistemic_spread_m"],
                }
                for row in rows
            ],
            key=lambda value: value["rmse_m"],
            reverse=True,
        ),
    }


def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path):
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _alr_timing_summary(root):
    paths = sorted(Path(root).expanduser().rglob("forward_only_timing.json"))
    rows = []
    for path in paths:
        payload = _load_json(path)
        rows.extend(payload.get("events", []))
    if not rows:
        raise FileNotFoundError("No forward_only_timing.json events found under {}.".format(root))
    values = np.asarray([row["forward_rollout_seconds"] for row in rows], dtype=np.float64)
    return {
        "n_events": int(values.size),
        "mean_seconds": float(values.mean()),
        "std_seconds": float(values.std()),
        "ensemble_members": int(rows[0]["ensemble_members"]),
    }


def _baseline_timing_summary(path, target_members=60):
    payload = _load_json(path)
    source_members = int(payload["ensemble_members"])
    source_mean = float(payload["forward_rollout_seconds"]["mean"])
    if source_members == int(target_members):
        return {
            "mean_seconds": source_mean,
            "ensemble_members": source_members,
            "policy": "direct measurement",
        }
    scaled = source_mean * float(target_members) / float(source_members)
    return {
        "mean_seconds": scaled,
        "ensemble_members": int(target_members),
        "source_ensemble_members": source_members,
        "source_mean_seconds": source_mean,
        "policy": "linear extrapolation from measured forward-only timing",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alr-artifacts", required=True)
    parser.add_argument("--baseline-artifacts")
    parser.add_argument("--historical-artifacts")
    parser.add_argument("--n50-artifacts")
    parser.add_argument("--n150-artifacts")
    parser.add_argument("--n450-artifacts")
    parser.add_argument("--parameter-report")
    parser.add_argument("--alr-timing-root")
    parser.add_argument("--baseline-timing-json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    rows = _artifact_metrics(args.alr_artifacts)
    report = {
        "n_events": len(rows),
        "alr_metrics_mean": {
            key: float(np.nanmean([row[key] for row in rows]))
            for key in rows[0]
            if key != "event_id"
        },
    }
    acceptance = {}
    if args.parameter_report:
        parameters = _load_json(args.parameter_report)
        report["parameter_accounting"] = parameters
        acceptance["adapter_parameters_below_25_percent"] = bool(
            parameters["adapter_trainable_fraction"] < 0.25
        )
    if args.alr_timing_root and args.baseline_timing_json:
        alr_timing = _alr_timing_summary(args.alr_timing_root)
        baseline_timing = _baseline_timing_summary(
            args.baseline_timing_json, target_members=alr_timing["ensemble_members"]
        )
        runtime_ratio = alr_timing["mean_seconds"] / baseline_timing["mean_seconds"]
        report["forward_only_timing"] = {
            "alr": alr_timing,
            "single_checkpoint_fgno": baseline_timing,
            "runtime_ratio": runtime_ratio,
        }
        acceptance["runtime_below_2x_single_checkpoint_fgno"] = bool(runtime_ratio < 2.0)
    if args.baseline_artifacts:
        base_rows = _artifact_metrics(args.baseline_artifacts)
        report["paired_vs_single_checkpoint_fgno"] = _paired_comparison(
            rows, base_rows, args.bootstrap_repetitions, args.seed
        )
        paired = report["paired_vs_single_checkpoint_fgno"]
        acceptance["rmse_degradation_at_most_0p001m"] = bool(
            paired["rmse_m"]["mean"] <= 0.001
        )
        acceptance["fair_crps_degradation_at_most_1_percent"] = bool(
            paired["fair_crps_percent_change"] <= 1.0
        )
    wetfront = [row["wetfront_epistemic_error_correlation"] for row in rows]
    wetfront = [value for value in wetfront if np.isfinite(value)]
    if wetfront:
        report["wetfront_event_block_correlation"] = _bootstrap_mean_ci(
            wetfront, args.bootstrap_repetitions, args.seed + 1
        )
        acceptance["wetfront_correlation_ci_lower_above_zero"] = bool(
            report["wetfront_event_block_correlation"]["ci95_low"] > 0.0
        )
    if args.historical_artifacts:
        historical_rows = _artifact_metrics(args.historical_artifacts)
        report["historical_ranking"] = _historical_ranking(historical_rows)
        ranking = report["historical_ranking"]
        acceptance["historical_spearman_positive"] = bool(
            ranking["spearman_rmse_vs_epistemic_spread"] > 0.0
        )
        acceptance["historical_top3_overlap_at_least_two"] = bool(
            ranking["top3_overlap"] >= 2
        )
        acceptance["historical_loo_positive_at_least_10_of_13"] = bool(
            ranking["leave_one_out_positive"] >= 10
        )
        _write_csv(historical_rows, args.output_dir / "historical_event_metrics.csv")
    contraction_roots = [args.n50_artifacts, args.n150_artifacts, args.n450_artifacts]
    if all(contraction_roots):
        contraction = {}
        for count, root in zip((50, 150, 450), contraction_roots):
            values = _artifact_metrics(root)
            contraction[str(count)] = float(np.mean([row["epistemic_spread_m"] for row in values]))
        report["training_size_contraction"] = {
            "mean_epistemic_spread_m": contraction,
            "strictly_decreasing": contraction["50"] > contraction["150"] > contraction["450"],
        }
        acceptance["epistemic_spread_contracts_with_training_size"] = bool(
            report["training_size_contraction"]["strictly_decreasing"]
        )

    particle_correlation = report["alr_metrics_mean"].get("particle_correlation_mean")
    if np.isfinite(particle_correlation):
        acceptance["particles_not_numerical_duplicates"] = bool(
            particle_correlation < 0.99999
        )
    report["acceptance_gates"] = acceptance
    report["all_available_gates_pass"] = bool(acceptance) and all(acceptance.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.output_dir / "heldout_event_metrics.csv")
    with (args.output_dir / "alr_fgno_acceptance_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(args.output_dir / "alr_fgno_acceptance_report.json")


if __name__ == "__main__":
    main()
