"""Compute UQ metrics on production n_ref=100 forecast artifacts.

Metric definitions intentionally match neuralop/flood/eval/rollout.py so the
numbers are directly comparable to the smoke-eval uq_overall_metrics.json:

- spread is the std across the member axis (axis=0), averaged over wettable cells
- spread_ratio = spread_pred / max(spread_gt, eps)
- RMSE: per-timestep RMSE of ensemble-mean vs reference-mean, averaged
- CRPS: fair empirical CRPS comparing forecast and reference ensembles
  (delegated to the maintained scientific-calibration implementation)
- coverage@alpha: fraction of reference values inside [q_lo, q_hi] of forecast
- Brier@thr: mean squared error between forecast exceedance probability and
  reference exceedance probability at threshold thr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np

from neuralop.flood.eval.scientific_calibration import empirical_crps_per_location

MIN_EPS = 1e-12
UQ_EXCEEDANCE_THRESHOLD = 0.30  # metres; matches eval default


def _masked_mean(arr: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is None:
        return float(np.mean(arr))
    if not np.any(mask):
        return float("nan")
    return float(np.mean(arr[mask]))


def _empirical_crps(forecast: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Fair per-location CRPS for forecast/reference ensembles.

    This delegates to the project-wide implementation so the standalone
    production-metrics script uses the same finite-ensemble correction as the
    maintained evaluation/calibration path.
    """
    return empirical_crps_per_location(forecast, reference)


def process_artifact(path: Path) -> Dict[str, float]:
    with h5py.File(path, "r") as f:
        pred = np.asarray(f["pred_members_wd"][:], dtype=np.float64)  # [M, T, N]
        ref = np.asarray(f["ref_members_wd"][:], dtype=np.float64)    # [R, T, N]
        wettable = np.asarray(f["wettable_mask"][:], dtype=bool) if "wettable_mask" in f else None
        structural_dry = (
            np.asarray(f["structural_dry_mask"][:], dtype=bool)
            if "structural_dry_mask" in f
            else None
        )

    if pred.ndim != 3 or ref.ndim != 3:
        raise ValueError(f"Unexpected shapes pred={pred.shape} ref={ref.shape} in {path.name}")
    if pred.shape[1:] != ref.shape[1:]:
        raise ValueError(
            f"Pred/ref shape mismatch (T,N): pred={pred.shape}, ref={ref.shape} in {path.name}"
        )
    M, T, N = pred.shape
    R = ref.shape[0]

    # Apply structural dry clamp on both sides (consistency with rollout post-clamp).
    if structural_dry is not None:
        pred[..., structural_dry] = 0.0
        ref[..., structural_dry] = 0.0

    # Per-timestep aggregates over wettable cells.
    spread_pred_per_t: List[float] = []
    spread_gt_per_t: List[float] = []
    rmse_per_t: List[float] = []
    crps_per_t: List[float] = []
    coverage_per_t: List[float] = []
    brier_per_t: List[float] = []
    prob_mae_per_t: List[float] = []
    nominal_alpha = 0.90
    q_lo = 0.5 * (1.0 - nominal_alpha)
    q_hi = 1.0 - q_lo

    for t in range(T):
        pf = pred[:, t, :]  # [M, N]
        rf = ref[:, t, :]   # [R, N]
        pred_spread_loc = np.std(pf, axis=0)   # [N]
        gt_spread_loc = np.std(rf, axis=0)     # [N]
        spread_pred_per_t.append(_masked_mean(pred_spread_loc, wettable))
        spread_gt_per_t.append(_masked_mean(gt_spread_loc, wettable))

        pred_mean = np.mean(pf, axis=0)        # [N]
        ref_mean = np.mean(rf, axis=0)         # [N]
        sq = (pred_mean - ref_mean) ** 2
        rmse_per_t.append(float(np.sqrt(_masked_mean(sq, wettable))))

        crps_loc = _empirical_crps(pf, rf)
        crps_per_t.append(_masked_mean(crps_loc, wettable))

        # Coverage at nominal 90%: fraction of ref draws inside forecast central interval.
        q_lo_arr = np.quantile(pf, q_lo, axis=0)   # [N]
        q_hi_arr = np.quantile(pf, q_hi, axis=0)
        inside = (rf >= q_lo_arr[None, :]) & (rf <= q_hi_arr[None, :])
        cov_per_cell = inside.mean(axis=0)          # [N], averaged across R
        coverage_per_t.append(_masked_mean(cov_per_cell, wettable))

        # Brier at threshold (per-cell, then mean).
        p_pred = (pf > UQ_EXCEEDANCE_THRESHOLD).mean(axis=0)   # [N]
        p_ref = (rf > UQ_EXCEEDANCE_THRESHOLD).mean(axis=0)
        brier_per_t.append(_masked_mean((p_pred - p_ref) ** 2, wettable))
        prob_mae_per_t.append(_masked_mean(np.abs(p_pred - p_ref), wettable))

    spread_pred_arr = np.array(spread_pred_per_t, dtype=np.float64)
    spread_gt_arr = np.array(spread_gt_per_t, dtype=np.float64)
    spread_ratio_per_t = spread_pred_arr / np.clip(spread_gt_arr, MIN_EPS, None)

    return {
        "n_members": int(M),
        "n_ref": int(R),
        "n_time": int(T),
        "n_cells": int(N),
        "n_wettable": int(wettable.sum()) if wettable is not None else int(N),
        "rmse_overall_mean": float(np.nanmean(rmse_per_t)),
        "rmse_leadtime_last": float(rmse_per_t[-1]),
        "crps_overall_mean": float(np.nanmean(crps_per_t)),
        "crps_leadtime_last": float(crps_per_t[-1]),
        "spread_pred_overall_mean": float(np.nanmean(spread_pred_arr)),
        "spread_gt_overall_mean": float(np.nanmean(spread_gt_arr)),
        "spread_ratio_overall_mean": float(np.nanmean(spread_ratio_per_t)),
        "spread_ratio_leadtime_last": float(spread_ratio_per_t[-1]),
        "coverage_90_overall_mean": float(np.nanmean(coverage_per_t)),
        "coverage_90_leadtime_last": float(coverage_per_t[-1]),
        "brier_overall_mean": float(np.nanmean(brier_per_t)),
        "brier_leadtime_last": float(brier_per_t[-1]),
        "prob_mae_overall_mean": float(np.nanmean(prob_mae_per_t)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, help="Directory with *.calibration_artifact.h5 files")
    parser.add_argument("--ensemble-name", required=True, help="Label written into per-ensemble summary")
    parser.add_argument("--output-json", required=True, help="Where to write per-hydrograph + aggregate JSON")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on artifacts processed (smoke)")
    args = parser.parse_args(argv)

    root = Path(args.artifact_root)
    files = sorted(root.glob("*.calibration_artifact.h5"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        print(f"[error] no calibration_artifact.h5 files in {root}", file=sys.stderr)
        return 2
    print(f"[info] {args.ensemble_name}: processing {len(files)} hydrographs from {root}")

    per_hydro: Dict[str, Dict[str, float]] = {}
    for path in files:
        name = path.stem.replace(".calibration_artifact", "")
        try:
            per_hydro[name] = process_artifact(path)
            print(
                f"[done] {name} rmse={per_hydro[name]['rmse_overall_mean']:.4f} "
                f"crps={per_hydro[name]['crps_overall_mean']:.4f} "
                f"spread_ratio={per_hydro[name]['spread_ratio_overall_mean']:.2f} "
                f"cov90={per_hydro[name]['coverage_90_overall_mean']:.3f}",
                flush=True,
            )
        except Exception as exc:
            print(f"[fail] {name}: {exc}", file=sys.stderr)

    if not per_hydro:
        print("[error] no hydrographs processed", file=sys.stderr)
        return 3

    keys_to_aggregate = [
        "rmse_overall_mean", "crps_overall_mean",
        "spread_pred_overall_mean", "spread_gt_overall_mean", "spread_ratio_overall_mean",
        "coverage_90_overall_mean", "brier_overall_mean", "prob_mae_overall_mean",
        "rmse_leadtime_last", "spread_ratio_leadtime_last", "coverage_90_leadtime_last",
    ]
    aggregate = {}
    for key in keys_to_aggregate:
        vals = np.array([m[key] for m in per_hydro.values() if np.isfinite(m.get(key, np.nan))], dtype=np.float64)
        if vals.size > 0:
            aggregate[f"{key}_mean"] = float(vals.mean())
            aggregate[f"{key}_std"] = float(vals.std())
            aggregate[f"{key}_median"] = float(np.median(vals))
            aggregate[f"{key}_p10"] = float(np.quantile(vals, 0.10))
            aggregate[f"{key}_p90"] = float(np.quantile(vals, 0.90))
        else:
            aggregate[f"{key}_mean"] = float("nan")

    payload = {
        "ensemble_name": args.ensemble_name,
        "artifact_root": str(root),
        "n_hydrographs": len(per_hydro),
        "aggregate": aggregate,
        "per_hydrograph": per_hydro,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[ok] wrote {out}")
    print("\n--- AGGREGATE ---")
    for k in sorted(aggregate):
        if k.endswith("_mean"):
            print(f"  {k:50s} {aggregate[k]:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
