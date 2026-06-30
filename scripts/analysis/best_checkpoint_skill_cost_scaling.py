#!/usr/bin/env python3
"""Generate best-checkpoint skill-cost scaling from member-level HDF5 artifacts.

This is intentionally separate from the 3x20 full-ensemble analysis. It selects
one checkpoint/model group per method, then evaluates N={1,2,5,10,20} members
within that checkpoint only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    _PLOT_ERR = exc
else:
    _PLOT_ERR = None

DEFAULT_ROOTS = {
    "FGNO": "/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_fgn/eval_outputs/fgn_3x20_m100_all50_raw_20260530_032503/outputs/forecast_artifacts/test_raw_all50",
    "MC-dropout GINO": "/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_mcdropout/eval_outputs/mcdropout_3x20_m100_member_artifacts_skip1step_20260530_0316/forecast_artifacts/test_raw",
    "Diffusion": "/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_diffusion/eval_outputs/diffusion_3x20_m100_member_artifacts_20260528_181358/combined/forecast_artifacts/diffusion_raw",
}
COLORS = {"FGNO": "#059669", "MC-dropout GINO": "#D97706", "Diffusion": "#2563EB"}
MARKERS = {"FGNO": "o", "MC-dropout GINO": "s", "Diffusion": "^"}
LINESTYLES = {"FGNO": "-", "MC-dropout GINO": "-.", "Diffusion": ":"}
SIZES = (1, 2, 5, 10, 20)


@dataclass(frozen=True)
class EventTask:
    method: str
    event_id: str
    path: str
    model_id: str
    member_indices: Tuple[int, ...]
    wettable_mask: Optional[np.ndarray]
    structural_dry_mask: Optional[np.ndarray]
    subsamples: Mapping[str, List[List[int]]]
    threshold: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def event_id(path: Path) -> str:
    name = path.name
    suf = ".calibration_artifact.h5"
    return name[:-len(suf)] if name.endswith(suf) else path.stem


def as_str_list(ds: h5py.Dataset) -> List[str]:
    out = []
    for x in ds[:]:
        out.append(x.decode("utf-8") if isinstance(x, bytes) else str(x))
    return out


def group_member_indices(path: Path) -> Dict[str, List[int]]:
    with h5py.File(path, "r") as f:
        n = int(f["pred_members_wd"].shape[0])
        if "member_model_id" in f:
            labels = as_str_list(f["member_model_id"])
        else:
            # Conservative fallback for historical artifacts: assume contiguous 20-member groups.
            labels = [f"model_{i//20}" for i in range(n)]
    groups: Dict[str, List[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(str(label), []).append(i)
    return groups


def optional_mask(f: h5py.File, key: str) -> Optional[np.ndarray]:
    return np.asarray(f[key][:], dtype=bool) if key in f else None


def read_masks(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Tuple[int, int, int], Tuple[int, int, int]]:
    with h5py.File(path, "r") as f:
        pred_shape = tuple(int(x) for x in f["pred_members_wd"].shape)
        ref_shape = tuple(int(x) for x in f["ref_members_wd"].shape)
        return optional_mask(f, "wettable_mask"), optional_mask(f, "structural_dry_mask"), pred_shape, ref_shape


def list_artifacts(root: Path) -> List[Path]:
    paths = sorted(root.glob("*.calibration_artifact.h5")) or sorted(root.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No artifacts found under {root}")
    return paths


def validate_roots(roots: Mapping[str, Path], expected_events: int) -> Dict[str, Dict[str, Path]]:
    per_method: Dict[str, Dict[str, Path]] = {}
    for method, root in roots.items():
        paths = list_artifacts(root)
        if len(paths) != expected_events:
            raise ValueError(f"{method}: expected {expected_events} artifacts, found {len(paths)} in {root}")
        per_method[method] = {event_id(p): p for p in paths}
    base_events = set(next(iter(per_method.values())).keys())
    for method, events in per_method.items():
        if set(events) != base_events:
            raise ValueError(f"{method}: event set differs from base method")
    return per_method


def balanced_indices(n_members: int, n_select: int, rng: np.random.Generator) -> np.ndarray:
    if n_select == n_members:
        return np.arange(n_members, dtype=int)
    return np.sort(rng.choice(n_members, size=n_select, replace=False))


def make_subsamples(n_members: int, sizes: Sequence[int], reps: int, seed: int) -> Dict[str, List[List[int]]]:
    out: Dict[str, List[List[int]]] = {}
    for n in sizes:
        if n > n_members:
            raise ValueError(f"Cannot select {n} from {n_members} members")
        rows = []
        for r in range(reps):
            rng = np.random.default_rng(seed + 7919 * int(n) + r)
            rows.append([int(x) for x in balanced_indices(n_members, int(n), rng)])
        out[str(int(n))] = rows
    return out


def mask_from(wettable: Optional[np.ndarray], structural_dry: Optional[np.ndarray], n_cells: int) -> np.ndarray:
    mask = np.ones(n_cells, dtype=bool) if wettable is None else np.asarray(wettable, dtype=bool).copy()
    if structural_dry is not None:
        mask &= ~np.asarray(structural_dry, dtype=bool)
    return mask


def cross_member_mean(pred_flat: np.ndarray, ref_flat: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_flat, dtype=np.float64)
    ref = np.asarray(ref_flat, dtype=np.float64)
    k, n = pred.shape
    r = ref.shape[0]
    ref_sorted = np.sort(ref, axis=0)
    ref_prefix = np.cumsum(ref_sorted, axis=0)
    ref_sum = ref_prefix[-1]
    loc = np.arange(n)
    out = np.empty(k, dtype=np.float64)
    for i in range(k):
        x = pred[i]
        counts = np.sum(ref_sorted <= x[None, :], axis=0).astype(np.int64, copy=False)
        left = np.zeros(n, dtype=np.float64)
        has_left = counts > 0
        left[has_left] = ref_prefix[counts[has_left] - 1, loc[has_left]]
        right = ref_sum - left
        out[i] = float(np.mean((x * counts - left + right - x * (r - counts)) / float(r)))
    return out


def pairwise_abs_mean(pred_flat: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_flat, dtype=np.float32)
    k = pred.shape[0]
    pair = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        xi = pred[i]
        for j in range(i + 1, k):
            val = float(np.mean(np.abs(xi - pred[j]), dtype=np.float64))
            pair[i, j] = pair[j, i] = val
    return pair


def brier_stats(pred_flat: np.ndarray, ref_flat: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, float]:
    pred_bin = (np.asarray(pred_flat) > threshold).astype(np.float32, copy=False)
    ref_prob = np.mean(np.asarray(ref_flat) > threshold, axis=0).astype(np.float32, copy=False)
    n = float(pred_bin.shape[1])
    pred_ref = (pred_bin @ ref_prob) / n
    pred_pair = (pred_bin @ pred_bin.T) / n
    ref_sq = float(np.mean(ref_prob * ref_prob, dtype=np.float64))
    return pred_ref.astype(np.float64), pred_pair.astype(np.float64), ref_sq


def skill(idx: Sequence[int], cross: np.ndarray, pair: np.ndarray, pr: np.ndarray, pp: np.ndarray, ref_sq: float) -> Tuple[float, float]:
    idx = np.asarray(idx, dtype=np.int64)
    k = int(idx.size)
    crps = float(np.mean(cross[idx]))
    if k >= 2:
        crps -= 0.5 * float(np.sum(pair[np.ix_(idx, idx)], dtype=np.float64)) / float(k * (k - 1))
    brier = float(np.sum(pp[np.ix_(idx, idx)], dtype=np.float64)) / float(k * k) - 2.0 * float(np.mean(pr[idx])) + float(ref_sq)
    return crps, max(0.0, brier)


def process_event(task: EventTask) -> Dict[str, Any]:
    t0 = time.perf_counter()
    idx_abs = np.asarray(task.member_indices, dtype=np.int64)
    with h5py.File(task.path, "r") as f:
        pred = np.asarray(f["pred_members_wd"][idx_abs, :, :], dtype=np.float32)
        ref = np.asarray(f["ref_members_wd"][:, :, :], dtype=np.float32)
    if task.structural_dry_mask is not None:
        dry = np.asarray(task.structural_dry_mask, dtype=bool)
        pred[..., dry] = 0.0
        ref[..., dry] = 0.0
    _, n_time, n_cells = pred.shape
    mask = mask_from(task.wettable_mask, task.structural_dry_mask, n_cells)
    pred_flat = np.ascontiguousarray(pred[:, :, mask].reshape(pred.shape[0], -1), dtype=np.float32)
    ref_flat = np.ascontiguousarray(ref[:, :, mask].reshape(ref.shape[0], -1), dtype=np.float32)
    del pred, ref
    cross = cross_member_mean(pred_flat, ref_flat)
    pair = pairwise_abs_mean(pred_flat)
    pr, pp, ref_sq = brier_stats(pred_flat, ref_flat, task.threshold)
    sizes: Dict[str, List[Dict[str, float]]] = {}
    for size, reps in task.subsamples.items():
        rows = []
        for rep, local_idx in enumerate(reps):
            c, b = skill(local_idx, cross, pair, pr, pp, ref_sq)
            rows.append({"rep": int(rep), "crps_fair_m": float(c), "brier_wd_exceed_0p30m": float(b)})
        sizes[size] = rows
    return {
        "method": task.method,
        "event_id": task.event_id,
        "model_id": task.model_id,
        "n_eval_points": int(n_time * int(mask.sum())),
        "sizes": sizes,
        "seconds": float(time.perf_counter() - t0),
    }


def aggregate(method: str, model_id: str, results: Sequence[Mapping[str, Any]], subsamples: Mapping[str, List[List[int]]]) -> Dict[str, Any]:
    total = float(sum(int(x["n_eval_points"]) for x in results))
    out: Dict[str, Any] = {"method": method, "model_id": model_id, "n_events": len(results), "sizes": {}}
    for size, reps in subsamples.items():
        crps_vals = []
        brier_vals = []
        for rep_idx in range(len(reps)):
            cnum = bnum = 0.0
            for row in results:
                w = float(row["n_eval_points"])
                cnum += float(row["sizes"][size][rep_idx]["crps_fair_m"]) * w
                bnum += float(row["sizes"][size][rep_idx]["brier_wd_exceed_0p30m"]) * w
            crps_vals.append(cnum / total)
            brier_vals.append(bnum / total)
        ca = np.asarray(crps_vals, dtype=np.float64)
        ba = np.asarray(brier_vals, dtype=np.float64)
        out["sizes"][size] = {
            "n_members": int(size),
            "crps_fair_m_mean": float(np.mean(ca)),
            "crps_fair_m_std": float(np.std(ca, ddof=1)) if len(ca) > 1 else 0.0,
            "brier_wd_exceed_0p30m_mean": float(np.mean(ba)),
            "brier_wd_exceed_0p30m_std": float(np.std(ba, ddof=1)) if len(ba) > 1 else 0.0,
        }
    return out


def load_timing(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def attach_timing(metrics: Dict[str, Any], timing: Mapping[str, Any]) -> None:
    if not timing:
        return
    for method, payload in metrics["methods"].items():
        tp = timing.get("methods", {}).get(method, {})
        for size, row in payload["sizes"].items():
            entry = tp.get("sizes", {}).get(str(size))
            if entry:
                row["seconds_per_event_rollout_mean"] = float(entry["seconds_per_event_rollout_mean"])
                row["seconds_per_event_rollout_std"] = float(entry.get("seconds_per_event_rollout_std", 0.0))
                row["runtime_estimated"] = bool(entry.get("estimated", False))


def plot(metrics: Mapping[str, Any], out_base: Path) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {_PLOT_ERR}")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.3, "axes.labelsize": 8.4, "legend.fontsize": 7.2,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "axes.linewidth": 0.62,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05), constrained_layout=True)
    axA, axB, axC, axD = axes.flat
    for method, payload in metrics["methods"].items():
        rows = [payload["sizes"][k] for k in sorted(payload["sizes"], key=lambda x: int(x))]
        n = np.asarray([r["n_members"] for r in rows], dtype=float)
        crps = np.asarray([r["crps_fair_m_mean"] for r in rows], dtype=float)
        crps_std = np.asarray([r["crps_fair_m_std"] for r in rows], dtype=float)
        brier = np.asarray([r["brier_wd_exceed_0p30m_mean"] for r in rows], dtype=float)
        brier_std = np.asarray([r["brier_wd_exceed_0p30m_std"] for r in rows], dtype=float)
        runtime = np.asarray([r.get("seconds_per_event_rollout_mean", np.nan) for r in rows], dtype=float)
        runtime_std = np.asarray([r.get("seconds_per_event_rollout_std", np.nan) for r in rows], dtype=float)
        c = COLORS.get(method, "0.2")
        mk = MARKERS.get(method, "o")
        ls = LINESTYLES.get(method, "-")
        if np.isfinite(runtime).any():
            ok = np.isfinite(runtime)
            axA.errorbar(n[ok], runtime[ok], yerr=runtime_std[ok], color=c, marker=mk, linestyle=ls, lw=.95, ms=2.8, capsize=1.4, elinewidth=.55, label=method)
            axD.plot(runtime[ok], crps[ok], color=c, marker=mk, linestyle=ls, lw=.95, ms=2.8, label=method)
        axB.plot(n, crps, color=c, marker=mk, linestyle=ls, lw=.95, ms=2.8, label=method)
        axB.fill_between(n, crps-crps_std, crps+crps_std, color=c, alpha=.12, lw=0)
        axC.plot(n, brier, color=c, marker=mk, linestyle=ls, lw=.95, ms=2.8, label=method)
        axC.fill_between(n, brier-brier_std, brier+brier_std, color=c, alpha=.12, lw=0)
    axA.set_yscale("log")
    axD.set_xscale("log")
    axA.set_ylabel("Inference Time per Event Rollout (s)")
    axB.set_ylabel("Fair CRPS (m)")
    axC.set_ylabel("Brier Score, WD > 0.3 m")
    axD.set_ylabel("Fair CRPS (m)")
    for ax in [axA, axB, axC]:
        ax.set_xlabel("Number of Forecast Members from Best Checkpoint")
    axD.set_xlabel("Inference Time per Event Rollout (s)")
    for ax in [axA, axB, axC, axD]:
        ax.grid(True, alpha=.14, lw=.45)
    axA.legend(frameon=False, loc="best", handlelength=1.6)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], [axA, axB, axC, axD]):
        ax.text(.018, .975, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=9.2)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_base.with_suffix("." + ext), dpi=600 if ext == "png" else None)
    plt.close(fig)


def write_csv(metrics: Mapping[str, Any], path: Path) -> None:
    fields = ["method", "model_id", "n_members", "crps_fair_m_mean", "crps_fair_m_std", "brier_wd_exceed_0p30m_mean", "brier_wd_exceed_0p30m_std", "seconds_per_event_rollout_mean", "seconds_per_event_rollout_std", "runtime_estimated"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method, payload in metrics["methods"].items():
            for size, row in payload["sizes"].items():
                w.writerow({k: row.get(k, "") for k in fields} | {"method": method, "model_id": payload["model_id"], "n_members": int(size)})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fgno-root", default=DEFAULT_ROOTS["FGNO"])
    p.add_argument("--mcdropout-root", default=DEFAULT_ROOTS["MC-dropout GINO"])
    p.add_argument("--diffusion-root", default=DEFAULT_ROOTS["Diffusion"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--timing-json", default=None)
    p.add_argument("--ensemble-sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260613)
    p.add_argument("--expected-events", type=int, default=50)
    p.add_argument("--max-events", type=int, default=None, help="Optional cap after validation for smoke tests.")
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--selection-metric", choices=["crps", "brier"], default="crps")
    p.add_argument("--model-id", action="append", default=[], help="Override as METHOD=MODEL_ID; otherwise selected by lowest N=max-size metric.")
    p.add_argument("--skip-plot", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = {"FGNO": Path(args.fgno_root), "MC-dropout GINO": Path(args.mcdropout_root), "Diffusion": Path(args.diffusion_root)}
    events = validate_roots(roots, args.expected_events)
    if args.max_events is not None:
        keep = sorted(next(iter(events.values())).keys())[: int(args.max_events)]
        events = {method: {eid: per_event[eid] for eid in keep} for method, per_event in events.items()}
    overrides = dict(item.split("=", 1) for item in args.model_id)
    sizes = [int(x) for x in args.ensemble_sizes]
    max_size = max(sizes)
    timing = load_timing(Path(args.timing_json).expanduser().resolve() if args.timing_json else None)

    metrics: Dict[str, Any] = {"created_utc": utc_now(), "artifact_roots": {m: str(r) for m, r in roots.items()}, "ensemble_sizes": sizes, "repetitions": args.repetitions, "seed": args.seed, "selection_metric": args.selection_metric, "methods": {}}

    for method, event_paths in events.items():
        first_path = event_paths[sorted(event_paths)[0]]
        groups = group_member_indices(first_path)
        candidate_ids = sorted(k for k, v in groups.items() if len(v) >= max_size)
        if not candidate_ids:
            raise ValueError(f"{method}: no member_model_id group has at least {max_size} members; groups={ {k: len(v) for k,v in groups.items()} }")
        candidate_results: Dict[str, Any] = {}
        for model_id in candidate_ids:
            n_members = len(groups[model_id])
            subsamples = make_subsamples(n_members, sizes, args.repetitions, args.seed)
            tasks: List[EventTask] = []
            for eid, path in sorted(event_paths.items()):
                event_groups = group_member_indices(path)
                if model_id not in event_groups:
                    raise ValueError(f"{method} {eid}: model_id {model_id} missing")
                wet, dry, pred_shape, ref_shape = read_masks(path)
                tasks.append(EventTask(method, eid, str(path), model_id, tuple(event_groups[model_id]), wet, dry, subsamples, args.threshold))
            print(f"[method={method}] model_id={model_id} events={len(tasks)} workers={args.workers}", flush=True)
            results: List[Mapping[str, Any]] = []
            if args.workers <= 1:
                for task in tasks:
                    results.append(process_event(task))
                    print(f"  {model_id}: {len(results)}/{len(tasks)}", flush=True)
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futs = [pool.submit(process_event, t) for t in tasks]
                    for fut in as_completed(futs):
                        results.append(fut.result())
                        print(f"  {model_id}: {len(results)}/{len(tasks)}", flush=True)
            agg = aggregate(method, model_id, results, subsamples)
            candidate_results[model_id] = agg
            atomic_json({"candidate_results": candidate_results}, out_dir / f"best_checkpoint_candidates_{re.sub(r'[^a-z0-9]+','_',method.lower()).strip('_')}.json")
        if method in overrides:
            selected = overrides[method]
            if selected not in candidate_results:
                raise ValueError(f"Override {method}={selected} not in candidates {list(candidate_results)}")
        else:
            metric_key = "crps_fair_m_mean" if args.selection_metric == "crps" else "brier_wd_exceed_0p30m_mean"
            selected = min(candidate_results, key=lambda mid: candidate_results[mid]["sizes"][str(max_size)][metric_key])
        metrics["methods"][method] = candidate_results[selected]
        metrics["methods"][method]["candidate_model_ids"] = candidate_ids
    attach_timing(metrics, timing)
    atomic_json(metrics, out_dir / "best_checkpoint_skill_cost_metrics.json")
    write_csv(metrics, out_dir / "best_checkpoint_skill_cost_metrics.csv")
    if not args.skip_plot:
        plot(metrics, out_dir / "best_checkpoint_skill_cost_scaling")
    print(f"[ok] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
