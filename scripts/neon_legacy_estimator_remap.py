#!/usr/bin/env python3
"""Recompute NEON epistemic diagnostics from immutable nested HDF5 artifacts.

This utility is intentionally inference-free. It compares the legacy
independent-nesting correction with the crossed common-random-number (CRN)
correction on exactly the same saved prediction tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


LABEL = "legacy frozen rollout, estimator repaired"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--time-chunk", type=int, default=3)
    parser.add_argument("--map-events", type=int, default=3)
    parser.add_argument("--map-hours", type=float, nargs="+", default=(6.0, 12.0, 23.5))
    return parser.parse_args()


def _member_index_matrix(epistemic_ids: np.ndarray, aleatory_ids: np.ndarray) -> np.ndarray:
    """Return artifact member indices as a complete ordered ``[M,K]`` grid."""

    epi_values = np.unique(epistemic_ids)
    ale_values = np.unique(aleatory_ids)
    matrix = np.empty((epi_values.size, ale_values.size), dtype=np.int64)
    for m, epi in enumerate(epi_values):
        for k, ale in enumerate(ale_values):
            found = np.flatnonzero((epistemic_ids == epi) & (aleatory_ids == ale))
            if found.size != 1:
                raise ValueError(
                    "nested artifact must contain exactly one member for each "
                    f"(epistemic, aleatory) pair; pair ({epi}, {ale}) has {found.size}"
                )
            matrix[m, k] = found[0]
    if matrix.size != epistemic_ids.size:
        raise ValueError("nested member IDs do not form a complete M x K crossed design")
    return matrix


def _corrected_fields(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw, legacy-independent, and crossed-CRN epistemic variances.

    ``y`` has shape ``[M,K,T,N]``.
    """

    m, k = y.shape[:2]
    if m < 2 or k < 2:
        zeros = np.zeros(y.shape[2:], dtype=np.float64)
        return zeros, zeros, zeros
    mean_k = y.mean(axis=1, dtype=np.float64)
    between = mean_k.var(axis=0, ddof=1)
    within = y.var(axis=1, ddof=1, dtype=np.float64).mean(axis=0)
    independent = np.maximum(between - within / float(k), 0.0)
    particle_mean = y.mean(axis=1, keepdims=True, dtype=np.float64)
    aleatory_mean = y.mean(axis=0, keepdims=True, dtype=np.float64)
    grand_mean = y.mean(axis=(0, 1), keepdims=True, dtype=np.float64)
    residual = y - particle_mean - aleatory_mean + grand_mean
    interaction_ms = np.square(residual, dtype=np.float64).sum(axis=(0, 1)) / float(
        (m - 1) * (k - 1)
    )
    crossed = np.maximum(between - interaction_ms / float(k), 0.0)
    return between, independent, crossed


def _safe_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    if int(selected.sum()) < 2:
        return 0.0
    xs = np.asarray(x[selected], dtype=np.float64)
    ys = np.asarray(y[selected], dtype=np.float64)
    xs -= xs.mean()
    ys -= ys.mean()
    denom = np.sqrt(np.square(xs).sum() * np.square(ys).sum())
    return 0.0 if denom <= 1.0e-12 else float(np.dot(xs, ys) / denom)


def _depth_residual(value: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    residual = np.zeros_like(value, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    d = np.asarray(depth[selected], dtype=np.float64)
    v = np.asarray(value[selected], dtype=np.float64)
    if d.size < 2:
        return residual
    unique = np.unique(d)
    if unique.size <= 10:
        labels = np.searchsorted(unique, d)
    else:
        edges = np.unique(np.quantile(d, np.linspace(0.0, 1.0, 11)))
        labels = np.digitize(d, edges[1:-1], right=False)
    r = np.empty_like(v)
    for idx in np.unique(labels):
        in_bin = labels == idx
        r[in_bin] = v[in_bin] - v[in_bin].mean()
    residual[selected] = r
    return residual


def _summarize_event(path: Path, *, time_chunk: int, map_hours, save_maps: bool, map_dir: Path):
    with h5py.File(path, "r") as handle:
        pred_ds = handle["pred_members_wd"]
        ref_ds = handle["ref_members_wd"]
        epi_ids = np.asarray(handle["member_epistemic_id"], dtype=np.int64)
        ale_ids = np.asarray(handle["member_aleatory_id"], dtype=np.int64)
        member_grid = _member_index_matrix(epi_ids, ale_ids)
        wettable = np.asarray(handle["wettable_mask"], dtype=bool)
        if "structural_dry_mask" in handle:
            wettable &= ~np.asarray(handle["structural_dry_mask"], dtype=bool)
        time_hours = np.asarray(handle["time_hours"], dtype=np.float64)
        event_id = str(handle.attrs.get("hydrograph_id", path.stem))
        n_time = int(pred_ds.shape[1])
        selected_times = {
            int(np.argmin(np.abs(time_hours - float(hour)))): float(hour) for hour in map_hours
        }
        sums = {"raw": 0.0, "independent": 0.0, "crossed": 0.0}
        counts = 0
        corr_rows = []
        partial_rows = []
        maps = {}
        for start in range(0, n_time, max(1, int(time_chunk))):
            stop = min(n_time, start + max(1, int(time_chunk)))
            flat = np.asarray(pred_ds[:, start:stop, :], dtype=np.float32)
            y = flat[member_grid]
            reference = np.asarray(ref_ds[:, start:stop, :], dtype=np.float32)
            raw, independent, crossed = _corrected_fields(y)
            pred_mean = y.mean(axis=(0, 1), dtype=np.float64)
            ref_mean = reference.mean(axis=0, dtype=np.float64)
            error = np.abs(pred_mean - ref_mean)
            for local, time_idx in enumerate(range(start, stop)):
                masks = {
                    "all_wettable": wettable,
                    "ref_wet": wettable & (ref_mean[local] > 0.01),
                    "wet_front": wettable & (ref_mean[local] > 0.01) & (ref_mean[local] <= 0.10),
                }
                corr_row = {"lead_index": int(time_idx), "lead_hour": float(time_hours[time_idx])}
                partial_row = dict(corr_row)
                for estimator, field in (("independent", independent[local]), ("crossed", crossed[local])):
                    std = np.sqrt(np.maximum(field, 0.0))
                    for stratum, mask in masks.items():
                        corr_row[f"{estimator}_{stratum}"] = _safe_corr(std, error[local], mask)
                        std_res = _depth_residual(std, ref_mean[local], mask)
                        err_res = _depth_residual(error[local], ref_mean[local], mask)
                        partial_row[f"{estimator}_{stratum}"] = _safe_corr(
                            std_res, err_res, mask
                        )
                corr_rows.append(corr_row)
                partial_rows.append(partial_row)
                if save_maps and time_idx in selected_times:
                    maps[f"lead_{time_idx:03d}_hour"] = np.asarray(time_hours[time_idx])
                    maps[f"lead_{time_idx:03d}_error"] = error[local].astype(np.float32)
                    maps[f"lead_{time_idx:03d}_independent_std"] = np.sqrt(
                        np.maximum(independent[local], 0.0)
                    ).astype(np.float32)
                    maps[f"lead_{time_idx:03d}_crossed_std"] = np.sqrt(
                        np.maximum(crossed[local], 0.0)
                    ).astype(np.float32)
            for name, field in (("raw", raw), ("independent", independent), ("crossed", crossed)):
                sums[name] += float(field[:, wettable].sum(dtype=np.float64))
            counts += int((stop - start) * wettable.sum())
        if save_maps:
            maps["geometry_raw"] = np.asarray(handle["geometry_raw"], dtype=np.float32)
            maps["wettable_mask"] = wettable
            np.savez_compressed(map_dir / f"{event_id}_legacy_estimator_remap.npz", **maps)
    row = {
        "family_id": event_id,
        "label": LABEL,
        "m": int(member_grid.shape[0]),
        "k": int(member_grid.shape[1]),
        "n_time": n_time,
        "n_wettable": int(wettable.sum()),
        "variance_epistemic_raw_mean": sums["raw"] / max(counts, 1),
        "variance_epistemic_independent_mean": sums["independent"] / max(counts, 1),
        "variance_epistemic_crossed_mean": sums["crossed"] / max(counts, 1),
    }
    for estimator in ("independent", "crossed"):
        for stratum in ("all_wettable", "ref_wet", "wet_front"):
            row[f"corr_{estimator}_{stratum}"] = float(
                np.mean([item[f"{estimator}_{stratum}"] for item in corr_rows])
            )
            row[f"partial_corr_{estimator}_{stratum}"] = float(
                np.mean([item[f"{estimator}_{stratum}"] for item in partial_rows])
            )
    return row, corr_rows, partial_rows


def main() -> int:
    args = _parse_args()
    artifact_dir = Path(args.artifacts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir = output_dir / "maps"
    map_dir.mkdir(exist_ok=True)
    paths = sorted(artifact_dir.glob("*.h5"))
    if args.max_events is not None:
        paths = paths[: int(args.max_events)]
    if not paths:
        raise FileNotFoundError(f"no HDF5 artifacts found under {artifact_dir}")
    rows = []
    lead_rows = []
    partial_rows = []
    for index, path in enumerate(paths):
        row, event_leads, event_partials = _summarize_event(
            path,
            time_chunk=int(args.time_chunk),
            map_hours=args.map_hours,
            save_maps=index < int(args.map_events),
            map_dir=map_dir,
        )
        rows.append(row)
        lead_rows.extend({"family_id": row["family_id"], **item} for item in event_leads)
        partial_rows.extend({"family_id": row["family_id"], **item} for item in event_partials)
        print(f"[{index + 1}/{len(paths)}] {row['family_id']}", flush=True)
    scalar_keys = [
        key for key, value in rows[0].items() if isinstance(value, (int, float)) and key not in {"m", "k", "n_time", "n_wettable"}
    ]
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in scalar_keys}
    payload = {
        "label": LABEL,
        "source_artifacts": str(artifact_dir),
        "estimator_design": "crossed_common_random_numbers",
        "n_families": len(rows),
        "aggregate": aggregate,
        "per_family": rows,
    }
    with (output_dir / "legacy_estimator_remap.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    for name, data in (("per_family", rows), ("per_lead", lead_rows), ("per_lead_partial", partial_rows)):
        with (output_dir / f"legacy_estimator_remap_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
