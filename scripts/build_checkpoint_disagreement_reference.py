"""Build checkpoint-disagreement monitoring references from FGN HDF artifacts.

This script is intended to run on Rivanna, next to the large calibration/test
HDF files. It reads each event one at a time, computes compact uncertainty
descriptors, and writes a refreshed Monitoring Bundle JSON. Only the final
bundle and optional JSONL descriptor rows need to be copied back to the lab PC.
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


EXTENT_THRESHOLD_M = 0.05
HIGH_CHECKPOINT_SHARE = 0.50
MIN_TOTAL_SPREAD_M = 0.05
REFERENCE_HEURISTIC_KEYS = {
    "uncertainty_to_signal_ratio",
    "iqr_area_product",
    "peak_area_weighted_iqr_wd_m",
    "peak_area_weighted_central_90_wd_m",
    "peak_area_weighted_total_ensemble_spread_wd_m",
    "peak_area_weighted_between_checkpoint_spread_wd_m",
    "peak_area_weighted_between_checkpoint_variance_share",
    "peak_high_checkpoint_disagreement_area_fraction_wettable",
}


def _percentile_payload(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot compute percentiles from an empty array.")
    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _decode_strings(raw: np.ndarray) -> list[str]:
    values: list[str] = []
    for item in raw.tolist():
        if isinstance(item, bytes):
            values.append(item.decode("utf-8"))
        else:
            values.append(str(item))
    return values


def _paths_from_globs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern))
    return sorted(dict.fromkeys(paths))


def _load_area(path: Path | None, n_cells: int) -> np.ndarray:
    if path is None:
        return np.ones(n_cells, dtype=np.float64)
    static = np.load(path)
    arr = np.asarray(static, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == n_cells and arr.shape[1] > 1:
        area = arr[:, 1]
    elif arr.ndim == 1 and arr.shape[0] == n_cells:
        area = arr
    else:
        raise ValueError(f"Static area file {path} has incompatible shape {arr.shape}; expected {n_cells} cells.")
    area = np.where(np.isfinite(area) & (area > 0.0), area, 0.0)
    if float(area.sum()) <= 0.0:
        raise ValueError(f"Static area file {path} contains no positive finite cell areas.")
    return area


def _peak_value_and_time(values: np.ndarray, lead_hours: np.ndarray) -> tuple[float, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0 or float(np.nanmax(arr)) <= 0.0:
        return 0.0, None
    idx = int(np.nanargmax(arr))
    return float(arr[idx]), float(lead_hours[idx])


def _decomposition_by_time(
    members: np.ndarray,
    member_model_id: list[str],
    area_m2: np.ndarray,
    wettable_area_m2: float,
    lead_hours: np.ndarray,
) -> dict[str, float | int | str | None]:
    groups = sorted(set(member_model_id))
    if len(groups) < 2 or len(member_model_id) != members.shape[0]:
        return {"checkpoint_disagreement_available": 0}

    group_means: list[np.ndarray] = []
    group_vars: list[np.ndarray] = []
    for group_id in groups:
        mask = np.array([item == group_id for item in member_model_id], dtype=bool)
        group = members[mask]
        if group.shape[0] < 1:
            continue
        group_means.append(group.mean(axis=0))
        group_vars.append(group.var(axis=0, ddof=0))
    if len(group_means) < 2:
        return {"checkpoint_disagreement_available": 0}

    total = members.var(axis=0, ddof=0)
    between = np.stack(group_means, axis=0).var(axis=0, ddof=0)
    within = np.stack(group_vars, axis=0).mean(axis=0)
    denom = between + within
    share = np.divide(
        between,
        denom,
        out=np.zeros_like(between, dtype=np.float64),
        where=denom > 1.0e-12,
    )

    def area_weight(values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) @ area_m2) / wettable_area_m2

    total_by_time = area_weight(total)
    between_by_time = area_weight(between)
    within_by_time = area_weight(within)
    share_by_time = area_weight(share)
    high = (share >= HIGH_CHECKPOINT_SHARE) & (np.sqrt(np.maximum(total, 0.0)) >= MIN_TOTAL_SPREAD_M)
    high_fraction_by_time = area_weight(high.astype(np.float64))
    peak_between, peak_between_lead = _peak_value_and_time(between_by_time, lead_hours)
    peak_high_fraction, peak_high_lead = _peak_value_and_time(high_fraction_by_time, lead_hours)

    return {
        "checkpoint_disagreement_available": 1,
        "n_checkpoint_groups": len(groups),
        "peak_area_weighted_total_ensemble_variance_wd_m2": float(total_by_time.max()) if total_by_time.size else 0.0,
        "peak_area_weighted_total_ensemble_spread_wd_m": (
            float(np.sqrt(max(float(total_by_time.max()), 0.0))) if total_by_time.size else 0.0
        ),
        "peak_area_weighted_between_checkpoint_variance_wd_m2": peak_between,
        "peak_area_weighted_between_checkpoint_spread_wd_m": float(np.sqrt(max(peak_between, 0.0))),
        "peak_area_weighted_within_checkpoint_variance_wd_m2": (
            float(within_by_time.max()) if within_by_time.size else 0.0
        ),
        "peak_area_weighted_within_checkpoint_spread_wd_m": (
            float(np.sqrt(max(float(within_by_time.max()), 0.0))) if within_by_time.size else 0.0
        ),
        "peak_area_weighted_between_checkpoint_variance_share": (
            float(share_by_time.max()) if share_by_time.size else 0.0
        ),
        "peak_between_checkpoint_disagreement_lead_hours": peak_between_lead,
        "peak_high_checkpoint_disagreement_area_fraction_wettable": peak_high_fraction,
        "peak_high_checkpoint_disagreement_lead_hours": peak_high_lead,
    }


def _summary_descriptors(path: Path, area_full: np.ndarray | None) -> dict[str, float | int | str | None]:
    with h5py.File(path, "r") as handle:
        members = np.clip(np.asarray(handle["pred_members_wd"], dtype=np.float64), 0.0, None)
        n_members, n_time, n_cells = members.shape
        lead_hours = np.asarray(handle["time_hours"], dtype=np.float64)
        wettable = np.asarray(handle["wettable_mask"], dtype=bool)
        member_model_id = _decode_strings(np.asarray(handle["member_model_id"]))
        hydrograph_id = str(handle.attrs.get("hydrograph_id", path.stem))

    area = _load_area(None, n_cells) if area_full is None else area_full
    if area.shape[0] != n_cells:
        raise ValueError(f"Cell-area array length {area.shape[0]} does not match HDF n_cells {n_cells} for {path}.")
    area_eval = area[wettable]
    members_eval = members[:, :, wettable]
    wettable_area_m2 = float(area_eval.sum())
    if wettable_area_m2 <= 0.0:
        raise ValueError(f"No positive wettable area for {path}.")

    mean = members_eval.mean(axis=0)
    q05, q25, q75, q95 = np.quantile(members_eval, [0.05, 0.25, 0.75, 0.95], axis=0)
    iqr = q75 - q25
    central90 = q95 - q05
    area_weighted_iqr_by_time = (iqr @ area_eval) / wettable_area_m2
    area_weighted_central90_by_time = (central90 @ area_eval) / wettable_area_m2
    max_mean_wd = float(mean.max())
    peak_iqr = float(area_weighted_iqr_by_time.max()) if area_weighted_iqr_by_time.size else 0.0

    extent_prob = (members_eval > EXTENT_THRESHOLD_M).mean(axis=0)
    expected_area_fraction_by_time = (extent_prob @ area_eval) / wettable_area_m2
    peak_area_fraction, peak_area_lead = _peak_value_and_time(expected_area_fraction_by_time, lead_hours)

    row: dict[str, float | int | str | None] = {
        "source_path": str(path),
        "hydrograph_id": hydrograph_id,
        "n_members": int(n_members),
        "n_time": int(n_time),
        "n_cells": int(n_cells),
        "n_wettable_cells": int(wettable.sum()),
        "wettable_area_m2": wettable_area_m2,
        "wettable_area_km2": wettable_area_m2 / 1_000_000.0,
        "max_mean_wd_m": max_mean_wd,
        "peak_expected_flooded_area_fraction_wettable_gt_0.05m": peak_area_fraction,
        "peak_expected_flooded_area_lead_hours_gt_0.05m": peak_area_lead,
        "peak_area_weighted_iqr_wd_m": peak_iqr,
        "peak_area_weighted_central_90_wd_m": (
            float(area_weighted_central90_by_time.max()) if area_weighted_central90_by_time.size else 0.0
        ),
        "iqr_area_product": peak_iqr * peak_area_fraction,
        "uncertainty_to_signal_ratio": (peak_iqr / max_mean_wd) if max_mean_wd > 1.0e-9 else None,
    }
    row.update(_decomposition_by_time(members_eval, member_model_id, area_eval, wettable_area_m2, lead_hours))
    return row


def _reference_percentiles(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    keys = sorted(
        key
        for key in {key for row in rows for key, value in row.items() if isinstance(value, (int, float))}
        if key in REFERENCE_HEURISTIC_KEYS
    )
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if isinstance(row.get(key), (int, float))], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            result[key] = _percentile_payload(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf-glob", action="append", required=True, help="HDF glob. Can be repeated.")
    parser.add_argument("--static-npy", type=Path, help="Optional static.npy with cell area in column 1.")
    parser.add_argument("--bundle-in", type=Path, required=True, help="Existing Monitoring Bundle JSON.")
    parser.add_argument("--bundle-out", type=Path, required=True, help="Refreshed Monitoring Bundle JSON.")
    parser.add_argument("--jsonl-out", type=Path, help="Optional per-event descriptor JSONL output.")
    parser.add_argument("--bundle-id", help="Optional replacement bundle ID.")
    args = parser.parse_args()

    paths = _paths_from_globs(args.hdf_glob)
    if not paths:
        raise SystemExit("No HDF files matched --hdf-glob.")

    area_full: np.ndarray | None = None
    if args.static_npy is not None:
        with h5py.File(paths[0], "r") as handle:
            n_cells = int(handle.attrs.get("n_cells", handle["pred_members_wd"].shape[2]))
        area_full = _load_area(args.static_npy, n_cells)

    rows: list[dict[str, object]] = []
    for idx, path in enumerate(paths, start=1):
        print(f"[{idx:03d}/{len(paths):03d}] {path}", flush=True)
        rows.append(_summary_descriptors(path, area_full))

    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl_out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    reference = _reference_percentiles(rows)
    bundle = json.loads(args.bundle_in.read_text(encoding="utf-8"))
    existing = bundle.get("heuristic_reference_percentiles") or {}
    existing.update(reference)
    bundle["heuristic_reference_percentiles"] = existing
    if args.bundle_id:
        bundle["bundle_id"] = args.bundle_id
    bundle["monitoring_bundle_schema_version"] = max(2, int(bundle.get("monitoring_bundle_schema_version", 1)))
    ref_pop = dict(bundle.get("reference_population") or {})
    ref_pop["n_reference_fgn_disagreement"] = len(rows)
    bundle["reference_population"] = ref_pop
    provenance = dict(bundle.get("provenance") or {})
    provenance["checkpoint_disagreement_reference"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hdf_globs": list(args.hdf_glob),
        "n_hdf_files": len(paths),
        "static_npy": str(args.static_npy) if args.static_npy else None,
        "descriptor_keys": sorted(reference),
        "high_checkpoint_share_threshold": HIGH_CHECKPOINT_SHARE,
        "min_total_spread_m": MIN_TOTAL_SPREAD_M,
        "note": "Computed from pred_members_wd in calibration artifacts on Rivanna; large HDF files remain on Rivanna.",
    }
    bundle["provenance"] = provenance

    args.bundle_out.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_out.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.bundle_out}")
    print(f"Events: {len(rows)}")
    print("Reference keys:")
    for key in sorted(reference):
        print(f"  {key}")


if __name__ == "__main__":
    main()
