#!/usr/bin/env python3
"""Build skill-cost Pareto curves from forecast-member flood artifacts.

The script is artifact-only for skill metrics: it never loads model checkpoints,
fits normalizers, or runs evaluation. It is optimized for the 3x20 coastal UQ
comparison by loading each event artifact once per method, reducing it to
sufficient statistics, and then evaluating all ensemble-size subsamples from
those cached statistics.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - exercised by CLI preflight
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuralop.flood.eval.scientific_calibration import empirical_crps_per_location

DEFAULT_METHOD_ROOTS = {
    "FGNO": "/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_fgn/eval_outputs/fgn_3x20_m100_all50_raw_20260530_032503/outputs/forecast_artifacts/test_raw_all50",
    "MC-dropout GINO": "/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_mcdropout/eval_outputs/mcdropout_3x20_m100_member_artifacts_skip1step_20260530_0316/forecast_artifacts/test_raw",
    "Diffusion": "/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_diffusion/eval_outputs/diffusion_3x20_m100_member_artifacts_20260528_181358/combined/forecast_artifacts/diffusion_raw",
}

METHOD_STYLE = {
    "FGNO": {"color": "#059669", "marker": "o", "linestyle": "-"},
    "MC-dropout GINO": {"color": "#D97706", "marker": "s", "linestyle": "-."},
    "Diffusion": {"color": "#2563EB", "marker": "^", "linestyle": ":"},
}

ENSEMBLE_SIZES_DEFAULT = (1, 2, 5, 10, 20, 40, 60)
MIN_EPS = 1e-12


@dataclass(frozen=True)
class ArtifactMeta:
    event_id: str
    path: Path
    pred_shape: Tuple[int, int, int]
    ref_shape: Tuple[int, int, int]
    wettable_mask: Optional[np.ndarray]
    structural_dry_mask: Optional[np.ndarray]
    cell_hash: str


@dataclass(frozen=True)
class EventTask:
    method: str
    event_id: str
    path: str
    wettable_mask: Optional[np.ndarray]
    structural_dry_mask: Optional[np.ndarray]
    subsamples: Mapping[str, List[List[int]]]
    threshold: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _event_id_from_path(path: Path) -> str:
    name = path.name
    suffix = ".calibration_artifact.h5"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _method_slug(method: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def list_artifacts(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Artifact root does not exist: {root}")
    paths = sorted(root.glob("*.calibration_artifact.h5"))
    if not paths:
        paths = sorted(root.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No HDF5 artifacts found under {root}")
    return paths


def _optional_bool_mask(handle: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in handle:
        return None
    return np.asarray(handle[key][...], dtype=bool)


def read_meta(path: Path) -> ArtifactMeta:
    with h5py.File(path, "r") as f:
        if "pred_members_wd" not in f or "ref_members_wd" not in f:
            raise ValueError(f"Missing pred_members_wd/ref_members_wd in {path}")
        pred_shape = tuple(int(x) for x in f["pred_members_wd"].shape)
        ref_shape = tuple(int(x) for x in f["ref_members_wd"].shape)
        wettable = _optional_bool_mask(f, "wettable_mask")
        structural_dry = _optional_bool_mask(f, "structural_dry_mask")
        cell_hash = str(f.attrs.get("cell_hash", ""))
    if len(pred_shape) != 3 or len(ref_shape) != 3:
        raise ValueError(f"Expected [members,time,cells] datasets in {path}; got {pred_shape} and {ref_shape}")
    return ArtifactMeta(_event_id_from_path(path), path, pred_shape, ref_shape, wettable, structural_dry, cell_hash)


def validate_artifacts(
    method_roots: Mapping[str, Path],
    *,
    expected_events: int,
    expected_members: int,
    expected_ref_members: int,
) -> Dict[str, Dict[str, ArtifactMeta]]:
    metas: Dict[str, Dict[str, ArtifactMeta]] = {}
    for method, root in method_roots.items():
        paths = list_artifacts(root)
        if len(paths) != expected_events:
            raise ValueError(f"{method}: expected {expected_events} artifacts under {root}, found {len(paths)}")
        per_event: Dict[str, ArtifactMeta] = {}
        for path in paths:
            meta = read_meta(path)
            if meta.event_id in per_event:
                raise ValueError(f"{method}: duplicate event ID {meta.event_id} under {root}")
            if meta.pred_shape[0] != expected_members:
                raise ValueError(f"{method} {meta.event_id}: expected {expected_members} forecast members, got {meta.pred_shape[0]}")
            if meta.ref_shape[0] != expected_ref_members:
                raise ValueError(f"{method} {meta.event_id}: expected {expected_ref_members} reference members, got {meta.ref_shape[0]}")
            if meta.pred_shape[1:] != meta.ref_shape[1:]:
                raise ValueError(f"{method} {meta.event_id}: pred/ref time-cell mismatch {meta.pred_shape} vs {meta.ref_shape}")
            per_event[meta.event_id] = meta
        metas[method] = per_event

    methods = list(metas)
    base_method = methods[0]
    base_events = set(metas[base_method])
    base_event_shapes = {event_id: metas[base_method][event_id].pred_shape[1:] for event_id in base_events}
    first_shape = next(iter(base_event_shapes.values()))
    for event_id, shape in base_event_shapes.items():
        if shape != first_shape:
            raise ValueError(f"{base_method}: rollout shape differs across events; {event_id} has {shape}, expected {first_shape}")

    for method in methods[1:]:
        missing = sorted(base_events - set(metas[method]))
        extra = sorted(set(metas[method]) - base_events)
        if missing or extra:
            raise ValueError(f"{method}: event IDs differ from {base_method}; missing={missing[:5]} extra={extra[:5]}")

    for event_id in sorted(base_events):
        base = metas[base_method][event_id]
        for method in methods[1:]:
            meta = metas[method][event_id]
            if meta.pred_shape[1:] != base.pred_shape[1:]:
                raise ValueError(f"{event_id}: {method} shape {meta.pred_shape[1:]} differs from {base_method} {base.pred_shape[1:]}")
            if meta.ref_shape != base.ref_shape:
                raise ValueError(f"{event_id}: {method} reference shape {meta.ref_shape} differs from {base_method} {base.ref_shape}")
            if base.cell_hash and meta.cell_hash and base.cell_hash != meta.cell_hash:
                raise ValueError(f"{event_id}: cell hash mismatch between {base_method} and {method}")
            if base.wettable_mask is not None and meta.wettable_mask is not None and not np.array_equal(base.wettable_mask, meta.wettable_mask):
                raise ValueError(f"{event_id}: wettable mask mismatch between {base_method} and {method}")
            if base.structural_dry_mask is not None and meta.structural_dry_mask is not None and not np.array_equal(base.structural_dry_mask, meta.structural_dry_mask):
                raise ValueError(f"{event_id}: structural dry mask mismatch between {base_method} and {method}")
    return metas


def balanced_member_indices(n_members: int, n_select: int, *, n_models: int, rng: np.random.Generator) -> np.ndarray:
    if n_select < 1 or n_select > n_members:
        raise ValueError(f"n_select must be in [1,{n_members}], got {n_select}")
    if n_members % n_models != 0:
        return np.sort(rng.choice(n_members, size=n_select, replace=False))
    per_model = n_members // n_models
    base = n_select // n_models
    remainder = n_select % n_models
    counts = np.full(n_models, base, dtype=int)
    if remainder:
        for model_idx in rng.choice(n_models, size=remainder, replace=False):
            counts[int(model_idx)] += 1
    selected: List[int] = []
    for model_idx, count in enumerate(counts):
        start = model_idx * per_model
        stop = start + per_model
        if count > 0:
            selected.extend(int(x) for x in rng.choice(np.arange(start, stop), size=int(count), replace=False))
    return np.sort(np.asarray(selected, dtype=int))


def make_subsamples(n_members: int, sizes: Sequence[int], *, repetitions: int, seed: int, n_models: int) -> Dict[str, List[List[int]]]:
    out: Dict[str, List[List[int]]] = {}
    for n_select in sizes:
        reps: List[List[int]] = []
        for rep in range(repetitions):
            rng = np.random.default_rng(seed + 1009 * int(n_select) + rep)
            idx = balanced_member_indices(n_members, int(n_select), n_models=n_models, rng=rng)
            reps.append([int(x) for x in idx])
        out[str(n_select)] = reps
    return out


def _mask_for_artifact(wettable: Optional[np.ndarray], structural_dry: Optional[np.ndarray], n_cells: int) -> np.ndarray:
    if wettable is None:
        mask = np.ones(n_cells, dtype=bool)
    else:
        mask = np.asarray(wettable, dtype=bool).copy()
    if structural_dry is not None:
        mask &= ~np.asarray(structural_dry, dtype=bool)
    return mask


def _fair_crps_masked(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    vals = empirical_crps_per_location(pred[:, mask], ref[:, mask])
    return float(np.mean(vals)) if vals.size else math.nan


def _brier_masked(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    p_pred = np.mean(pred[:, mask] > threshold, axis=0)
    p_ref = np.mean(ref[:, mask] > threshold, axis=0)
    return float(np.mean((p_pred - p_ref) ** 2)) if p_pred.size else math.nan


def _cross_member_mean_against_reference(pred_flat: np.ndarray, ref_flat: np.ndarray) -> np.ndarray:
    """Mean reference-ensemble absolute error for each forecast member.

    pred_flat: [K, L], ref_flat: [R, L]. Returns [K]. This is the first CRPS
    term averaged over all evaluated time-cell locations. Reference sorting is
    performed once, then reused for all forecast members.
    """
    pred = np.asarray(pred_flat, dtype=np.float64)
    ref = np.asarray(ref_flat, dtype=np.float64)
    n_members, n_locations = pred.shape
    n_ref = ref.shape[0]
    ref_sorted = np.sort(ref, axis=0)
    ref_prefix = np.cumsum(ref_sorted, axis=0)
    ref_sum = ref_prefix[-1]
    loc_idx = np.arange(n_locations)
    cross = np.empty(n_members, dtype=np.float64)
    for member_idx in range(n_members):
        xm = pred[member_idx]
        counts = np.sum(ref_sorted <= xm[None, :], axis=0).astype(np.int64, copy=False)
        left = np.zeros(n_locations, dtype=np.float64)
        has_left = counts > 0
        left[has_left] = ref_prefix[counts[has_left] - 1, loc_idx[has_left]]
        right = ref_sum - left
        per_location = (xm * counts - left + right - xm * (n_ref - counts)) / float(n_ref)
        cross[member_idx] = float(np.mean(per_location))
    return cross


def _forecast_pairwise_abs_mean(pred_flat: np.ndarray) -> np.ndarray:
    """Mean absolute difference for every ordered forecast-member pair."""
    pred = np.asarray(pred_flat, dtype=np.float32)
    n_members = pred.shape[0]
    pair = np.zeros((n_members, n_members), dtype=np.float64)
    for i in range(n_members):
        xi = pred[i]
        for j in range(i + 1, n_members):
            val = float(np.mean(np.abs(xi - pred[j]), dtype=np.float64))
            pair[i, j] = val
            pair[j, i] = val
    return pair


def _brier_sufficient_stats(pred_flat: np.ndarray, ref_flat: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, float]:
    pred_bin = (np.asarray(pred_flat) > threshold).astype(np.float32, copy=False)
    ref_prob = np.mean(np.asarray(ref_flat) > threshold, axis=0).astype(np.float32, copy=False)
    n_locations = float(pred_bin.shape[1])
    pred_ref = (pred_bin @ ref_prob.astype(np.float32)) / n_locations
    pred_pair = (pred_bin @ pred_bin.T) / n_locations
    ref_prob_sq_mean = float(np.mean(ref_prob * ref_prob, dtype=np.float64))
    return pred_ref.astype(np.float64), pred_pair.astype(np.float64), ref_prob_sq_mean


def _skill_from_sufficient_stats(
    member_indices: Sequence[int],
    *,
    crps_cross: np.ndarray,
    crps_pair: np.ndarray,
    brier_pred_ref: np.ndarray,
    brier_pred_pair: np.ndarray,
    brier_ref_sq_mean: float,
) -> Tuple[float, float]:
    idx = np.asarray(member_indices, dtype=np.int64)
    k = int(idx.size)
    if k < 1:
        raise ValueError("At least one member is required")
    crps = float(np.mean(crps_cross[idx]))
    if k >= 2:
        pair_sum = float(np.sum(crps_pair[np.ix_(idx, idx)], dtype=np.float64))
        crps -= 0.5 * pair_sum / float(k * (k - 1))
    pred_pair_mean = float(np.sum(brier_pred_pair[np.ix_(idx, idx)], dtype=np.float64)) / float(k * k)
    pred_ref_mean = float(np.mean(brier_pred_ref[idx], dtype=np.float64))
    brier = pred_pair_mean - 2.0 * pred_ref_mean + float(brier_ref_sq_mean)
    return crps, max(0.0, brier)


def _process_event_task(task: EventTask) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        with h5py.File(task.path, "r") as f:
            pred = np.asarray(f["pred_members_wd"][...], dtype=np.float32)
            ref = np.asarray(f["ref_members_wd"][...], dtype=np.float32)
        load_seconds = time.perf_counter() - start
        if pred.ndim != 3 or ref.ndim != 3:
            raise ValueError(f"Expected [members,time,cells] arrays, got pred={pred.shape}, ref={ref.shape}")
        if pred.shape[1:] != ref.shape[1:]:
            raise ValueError(f"Pred/ref time-cell mismatch: pred={pred.shape}, ref={ref.shape}")
        n_members, n_time, n_cells = pred.shape
        mask = _mask_for_artifact(task.wettable_mask, task.structural_dry_mask, n_cells)
        if not np.any(mask):
            raise ValueError("No wettable/evaluated cells remain after applying masks")
        if task.structural_dry_mask is not None:
            dry = np.asarray(task.structural_dry_mask, dtype=bool)
            pred[..., dry] = 0.0
            ref[..., dry] = 0.0
        pred_flat = np.ascontiguousarray(pred[:, :, mask].reshape(n_members, -1), dtype=np.float32)
        ref_flat = np.ascontiguousarray(ref[:, :, mask].reshape(ref.shape[0], -1), dtype=np.float32)
        del pred, ref

        precompute_start = time.perf_counter()
        crps_cross = _cross_member_mean_against_reference(pred_flat, ref_flat)
        crps_pair = _forecast_pairwise_abs_mean(pred_flat)
        brier_pred_ref, brier_pred_pair, brier_ref_sq_mean = _brier_sufficient_stats(pred_flat, ref_flat, task.threshold)
        precompute_seconds = time.perf_counter() - precompute_start

        eval_start = time.perf_counter()
        sizes_payload: Dict[str, Any] = {}
        for size_text, rep_indices in task.subsamples.items():
            rep_values: List[Dict[str, float]] = []
            for rep_idx, member_indices in enumerate(rep_indices):
                crps, brier = _skill_from_sufficient_stats(
                    member_indices,
                    crps_cross=crps_cross,
                    crps_pair=crps_pair,
                    brier_pred_ref=brier_pred_ref,
                    brier_pred_pair=brier_pred_pair,
                    brier_ref_sq_mean=brier_ref_sq_mean,
                )
                rep_values.append({
                    "rep": int(rep_idx),
                    "crps_fair_m": float(crps),
                    "brier_wd_exceed_0p30m": float(brier),
                })
            sizes_payload[str(size_text)] = rep_values
        eval_seconds = time.perf_counter() - eval_start
        return {
            "method": task.method,
            "event_id": task.event_id,
            "path": task.path,
            "n_time": int(n_time),
            "n_cells": int(n_cells),
            "n_eval_cells": int(np.sum(mask)),
            "n_eval_points": int(n_time * int(np.sum(mask))),
            "sizes": sizes_payload,
            "timing_seconds": {
                "load": float(load_seconds),
                "precompute": float(precompute_seconds),
                "subsample_eval": float(eval_seconds),
                "total": float(time.perf_counter() - start),
            },
        }
    except Exception as exc:  # keep remote multiprocessing errors actionable
        raise RuntimeError(f"{task.method} {task.event_id} failed while processing {task.path}: {exc}") from exc


def _aggregate_method_event_results(method: str, event_results: Sequence[Mapping[str, Any]], subsamples: Mapping[str, List[List[int]]]) -> Dict[str, Any]:
    if not event_results:
        raise ValueError(f"{method}: no event results to aggregate")
    sizes_payload: Dict[str, Any] = {}
    total_points = float(sum(int(ev["n_eval_points"]) for ev in event_results))
    for size_text, rep_indices in subsamples.items():
        rep_crps: List[float] = []
        rep_brier: List[float] = []
        for rep_idx in range(len(rep_indices)):
            crps_num = 0.0
            brier_num = 0.0
            for ev in event_results:
                weight = float(ev["n_eval_points"])
                row = ev["sizes"][str(size_text)][rep_idx]
                crps_num += float(row["crps_fair_m"]) * weight
                brier_num += float(row["brier_wd_exceed_0p30m"]) * weight
            rep_crps.append(crps_num / max(total_points, 1.0))
            rep_brier.append(brier_num / max(total_points, 1.0))
        crps_arr = np.asarray(rep_crps, dtype=np.float64)
        brier_arr = np.asarray(rep_brier, dtype=np.float64)
        sizes_payload[str(size_text)] = {
            "n_members": int(size_text),
            "crps_fair_m_mean": float(np.nanmean(crps_arr)),
            "crps_fair_m_std": float(np.nanstd(crps_arr, ddof=1)) if crps_arr.size > 1 else 0.0,
            "brier_wd_exceed_0p30m_mean": float(np.nanmean(brier_arr)),
            "brier_wd_exceed_0p30m_std": float(np.nanstd(brier_arr, ddof=1)) if brier_arr.size > 1 else 0.0,
            "repetition_values": [
                {"rep": int(i), "crps_fair_m": float(c), "brier_wd_exceed_0p30m": float(b)}
                for i, (c, b) in enumerate(zip(rep_crps, rep_brier))
            ],
        }
    return {
        "method": method,
        "n_events": int(len(event_results)),
        "n_eval_points": int(total_points),
        "event_timing_seconds": {
            "total_mean": float(np.mean([ev["timing_seconds"]["total"] for ev in event_results])),
            "total_max": float(np.max([ev["timing_seconds"]["total"] for ev in event_results])),
            "precompute_mean": float(np.mean([ev["timing_seconds"]["precompute"] for ev in event_results])),
        },
        "sizes": sizes_payload,
    }


def _write_method_progress(
    out_dir: Optional[Path],
    method: str,
    completed_results: Sequence[Mapping[str, Any]],
    total_events: int,
    subsamples: Mapping[str, List[List[int]]],
    *,
    status: str,
) -> None:
    if out_dir is None:
        return
    slug = _method_slug(method)
    completed_events = [str(ev["event_id"]) for ev in completed_results]
    payload: Dict[str, Any] = {
        "method": method,
        "status": status,
        "updated_utc": utc_now(),
        "completed_events": completed_events,
        "completed_count": len(completed_events),
        "total_events": int(total_events),
        "event_results": list(completed_results),
    }
    if completed_results:
        payload["aggregate_so_far"] = _aggregate_method_event_results(method, completed_results, subsamples)
    atomic_write_json(payload, out_dir / f"skill_cost_partial_{slug}.json")
    progress = {key: payload[key] for key in ["method", "status", "updated_utc", "completed_events", "completed_count", "total_events"]}
    atomic_write_json(progress, out_dir / f"skill_cost_progress_{slug}.json")


def compute_method_metrics(
    method: str,
    metas: Mapping[str, ArtifactMeta],
    subsamples: Mapping[str, List[List[int]]],
    *,
    threshold: float,
    workers: int,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ordered_events = sorted(metas)
    tasks = [
        EventTask(
            method=method,
            event_id=event_id,
            path=str(metas[event_id].path),
            wettable_mask=metas[event_id].wettable_mask,
            structural_dry_mask=metas[event_id].structural_dry_mask,
            subsamples=subsamples,
            threshold=float(threshold),
        )
        for event_id in ordered_events
    ]
    completed: List[Mapping[str, Any]] = []
    _write_method_progress(out_dir, method, completed, len(tasks), subsamples, status="running")
    if workers <= 1:
        for task in tasks:
            result = _process_event_task(task)
            completed.append(result)
            print(
                f"[progress] {method}: {len(completed)}/{len(tasks)} events; "
                f"last={result['event_id']} total={result['timing_seconds']['total']:.1f}s",
                flush=True,
            )
            _write_method_progress(out_dir, method, completed, len(tasks), subsamples, status="running")
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            future_to_event = {pool.submit(_process_event_task, task): task.event_id for task in tasks}
            for future in as_completed(future_to_event):
                result = future.result()
                completed.append(result)
                completed.sort(key=lambda row: str(row["event_id"]))
                print(
                    f"[progress] {method}: {len(completed)}/{len(tasks)} events; "
                    f"last={result['event_id']} total={result['timing_seconds']['total']:.1f}s",
                    flush=True,
                )
                _write_method_progress(out_dir, method, completed, len(tasks), subsamples, status="running")
    final = _aggregate_method_event_results(method, completed, subsamples)
    _write_method_progress(out_dir, method, completed, len(tasks), subsamples, status="complete")
    return final


def load_timing(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def timing_lookup(timing: Mapping[str, Any], method: str, n_members: int) -> Optional[Dict[str, float]]:
    if not timing:
        return None
    methods = timing.get("methods", timing)
    method_payload = methods.get(method) if isinstance(methods, Mapping) else None
    if method_payload is None:
        return None
    sizes = method_payload.get("sizes", method_payload)
    entry = sizes.get(str(n_members)) if isinstance(sizes, Mapping) else None
    if entry is None:
        return None
    mean = entry.get("seconds_per_event_rollout_mean", entry.get("seconds_mean", entry.get("mean")))
    std = entry.get("seconds_per_event_rollout_std", entry.get("seconds_std", entry.get("std", 0.0)))
    estimated = bool(entry.get("estimated", False))
    if mean is None:
        return None
    return {"seconds_mean": float(mean), "seconds_std": float(std or 0.0), "estimated": estimated}


def attach_timing(metrics: Dict[str, Any], timing: Mapping[str, Any]) -> None:
    for method, payload in metrics["methods"].items():
        for size_text, size_payload in payload["sizes"].items():
            t = timing_lookup(timing, method, int(size_text))
            if t is not None:
                size_payload["seconds_per_event_rollout_mean"] = t["seconds_mean"]
                size_payload["seconds_per_event_rollout_std"] = t["seconds_std"]
                size_payload["runtime_estimated"] = t["estimated"]


def write_csv(metrics: Mapping[str, Any], path: Path) -> None:
    fieldnames = [
        "method",
        "n_members",
        "crps_fair_m_mean",
        "crps_fair_m_std",
        "brier_wd_exceed_0p30m_mean",
        "brier_wd_exceed_0p30m_std",
        "seconds_per_event_rollout_mean",
        "seconds_per_event_rollout_std",
        "runtime_estimated",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, payload in metrics["methods"].items():
            for size_text, row in payload["sizes"].items():
                writer.writerow({key: row.get(key, "") for key in fieldnames} | {"method": method, "n_members": int(size_text)})


def _set_publication_style() -> None:
    if plt is None:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
    })


def plot_figure(metrics: Mapping[str, Any], out_base: Path) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib import failed: {_MATPLOTLIB_IMPORT_ERROR}")
    _set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    for method, payload in metrics["methods"].items():
        style = METHOD_STYLE.get(method, {"color": "0.2", "marker": "o", "linestyle": "-"})
        rows = [payload["sizes"][k] for k in sorted(payload["sizes"], key=lambda x: int(x))]
        x = np.asarray([r["n_members"] for r in rows], dtype=float)
        crps = np.asarray([r["crps_fair_m_mean"] for r in rows], dtype=float)
        crps_std = np.asarray([r["crps_fair_m_std"] for r in rows], dtype=float)
        brier = np.asarray([r["brier_wd_exceed_0p30m_mean"] for r in rows], dtype=float)
        brier_std = np.asarray([r["brier_wd_exceed_0p30m_std"] for r in rows], dtype=float)
        runtime = np.asarray([r.get("seconds_per_event_rollout_mean", np.nan) for r in rows], dtype=float)
        runtime_std = np.asarray([r.get("seconds_per_event_rollout_std", np.nan) for r in rows], dtype=float)

        valid_runtime = np.isfinite(runtime)
        if np.any(valid_runtime):
            ax_a.errorbar(
                x[valid_runtime],
                runtime[valid_runtime],
                yerr=runtime_std[valid_runtime],
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                capsize=2.5,
            )
            ax_d.plot(runtime[valid_runtime], crps[valid_runtime], label=method, color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=1.5)
            for n, tx, ty in zip(x[valid_runtime], runtime[valid_runtime], crps[valid_runtime]):
                if int(n) in {1, 20, 60}:
                    ax_d.annotate(str(int(n)), (tx, ty), textcoords="offset points", xytext=(3, 3), fontsize=7, color=style["color"])

        ax_b.plot(x, crps, label=method, color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=1.5)
        ax_b.fill_between(x, crps - crps_std, crps + crps_std, color=style["color"], alpha=0.16, linewidth=0)
        ax_c.plot(x, brier, label=method, color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=1.5)
        ax_c.fill_between(x, brier - brier_std, brier + brier_std, color=style["color"], alpha=0.16, linewidth=0)

    ax_a.set_xlabel("Number of Forecast Members")
    ax_a.set_ylabel("Inference Time per Event Rollout (s)")
    ax_a.set_yscale("log")
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(frameon=False, loc="best")

    ax_b.set_xlabel("Number of Forecast Members")
    ax_b.set_ylabel("Fair CRPS (m)")
    ax_b.grid(True, alpha=0.25)

    ax_c.set_xlabel("Number of Forecast Members")
    ax_c.set_ylabel("Brier Score, WD > 0.3 m")
    ax_c.grid(True, alpha=0.25)

    ax_d.set_xlabel("Inference Time per Event Rollout (s)")
    ax_d.set_ylabel("Fair CRPS (m)")
    ax_d.set_xscale("log")
    ax_d.grid(True, alpha=0.25)

    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], [ax_a, ax_b, ax_c, ax_d]):
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left", fontweight="bold")

    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".svg"))
    fig.savefig(out_base.with_suffix(".png"), dpi=600)
    plt.close(fig)


def write_readme(path: Path, metrics: Mapping[str, Any], timing_path: Optional[Path], args: argparse.Namespace) -> None:
    lines = [
        "Skill-cost Pareto analysis",
        "===========================",
        "",
        f"Created UTC: {metrics['created_utc']}",
        "Policy: probabilistic methods only; deterministic GINO omitted from main figure.",
        "Skill metrics: computed from existing physical-space member-level HDF5 artifacts; no model reruns.",
        "Computation: each event artifact is loaded once per method, reference terms are precomputed, and event workers run in parallel.",
        "CRPS: fair finite-ensemble forecast-vs-reference CRPS in meters.",
        "Brier: exceedance probability error for water depth > 0.3 m.",
        "Primary domain: wettable cells, with structural dry cells clamped/excluded when masks exist.",
        f"Subsampling repetitions: {args.repetitions}",
        f"Subsampling seed: {args.seed}",
        f"Workers: {args.workers}",
        f"Ensemble sizes: {','.join(str(x) for x in args.ensemble_sizes)}",
        "",
        "Artifact roots:",
    ]
    for method, root in metrics["artifact_roots"].items():
        lines.append(f"- {method}: {root}")
    lines.extend(["", "Timing:"])
    if timing_path is None:
        lines.append("- No timing JSON supplied; runtime panels are omitted or incomplete.")
    else:
        lines.append(f"- Timing JSON: {timing_path}")
        lines.append("- Runtime values are forward-only wall-clock seconds per full event rollout ensemble.")
        lines.append("- Entries marked runtime_estimated=true are extrapolated and should be described as such.")
    lines.extend([
        "",
        "Intermediate outputs:",
        "- skill_cost_progress_<method>.json records completed event IDs.",
        "- skill_cost_partial_<method>.json records event-level metrics and aggregate-so-far values.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fgno-root", default=DEFAULT_METHOD_ROOTS["FGNO"])
    parser.add_argument("--mcdropout-root", default=DEFAULT_METHOD_ROOTS["MC-dropout GINO"])
    parser.add_argument("--diffusion-root", default=DEFAULT_METHOD_ROOTS["Diffusion"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timing-json", default=None)
    parser.add_argument("--ensemble-sizes", type=int, nargs="+", default=list(ENSEMBLE_SIZES_DEFAULT))
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--n-models", type=int, default=3)
    parser.add_argument("--expected-events", type=int, default=50)
    parser.add_argument("--expected-members", type=int, default=60)
    parser.add_argument("--expected-ref-members", type=int, default=100)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap for smoke tests; still validates that roots contain expected-events before truncating.")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    method_roots = {
        "FGNO": Path(args.fgno_root).expanduser(),
        "MC-dropout GINO": Path(args.mcdropout_root).expanduser(),
        "Diffusion": Path(args.diffusion_root).expanduser(),
    }
    metas = validate_artifacts(
        method_roots,
        expected_events=args.expected_events,
        expected_members=args.expected_members,
        expected_ref_members=args.expected_ref_members,
    )
    if args.max_events is not None:
        keep = sorted(next(iter(metas.values())).keys())[: int(args.max_events)]
        metas = {method: {event_id: per_event[event_id] for event_id in keep} for method, per_event in metas.items()}

    subsamples = make_subsamples(
        args.expected_members,
        args.ensemble_sizes,
        repetitions=args.repetitions,
        seed=args.seed,
        n_models=args.n_models,
    )
    metrics: Dict[str, Any] = {
        "created_utc": utc_now(),
        "artifact_roots": {method: str(root) for method, root in method_roots.items()},
        "event_ids": sorted(next(iter(metas.values())).keys()),
        "ensemble_sizes": [int(x) for x in args.ensemble_sizes],
        "repetitions": int(args.repetitions),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "threshold_m": float(args.threshold),
        "subsample_indices": subsamples,
        "methods": {},
    }
    atomic_write_json(metrics, out_dir / "skill_cost_metrics_by_size.in_progress.json")
    for method in method_roots:
        print(f"[metrics] {method} workers={args.workers}", flush=True)
        metrics["methods"][method] = compute_method_metrics(
            method,
            metas[method],
            subsamples,
            threshold=float(args.threshold),
            workers=int(args.workers),
            out_dir=out_dir,
        )
        atomic_write_json(metrics, out_dir / "skill_cost_metrics_by_size.in_progress.json")
    timing_path = Path(args.timing_json).expanduser().resolve() if args.timing_json else None
    timing = load_timing(timing_path)
    if timing:
        attach_timing(metrics, timing)
        atomic_write_json(timing, out_dir / "skill_cost_timing.json")
    metrics_path = out_dir / "skill_cost_metrics_by_size.json"
    atomic_write_json(metrics, metrics_path)
    write_csv(metrics, out_dir / "skill_cost_metrics_by_size.csv")
    write_readme(out_dir / "README.txt", metrics, timing_path, args)
    if not args.skip_plot:
        plot_figure(metrics, out_dir / "skill_cost_pareto")
    in_progress = out_dir / "skill_cost_metrics_by_size.in_progress.json"
    if in_progress.exists():
        in_progress.unlink()
    print(f"[ok] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
