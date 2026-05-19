"""Build a Phase 1 monitoring bundle from reference forcing and HEC-RAS events."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from neuralop.flood.serving.forcing import ForcingInput
from neuralop.flood.serving.forcing import parse_forcing_csv
from neuralop.flood.serving.model_bundle import load_model_bundle
from neuralop.flood.serving.monitoring import build_monitoring_bundle_from_forcings


HECRAS_DEPTH_DATASET = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
    "2D Flow Areas/Portsmouth/Cell Hydraulic Depth"
)
HECRAS_AREA_DATASET = "Geometry/2D Flow Areas/Portsmouth/Cells Surface Area"
HECRAS_CELL_POINTS_DATASET = "Geometry/2D Flow Areas/Cell Points"


def load_clean_boundary_table_forcings(
    *,
    stage_table: Path,
    precipitation_table: Path,
    dt_seconds: int,
    skip_before_timestep: int,
    n_history: int,
    max_forecast_steps: int,
) -> list[ForcingInput]:
    """Load clean-family stage/precipitation tables into canonical forcings.

    The coastal training data stores one hydrograph per column: header row is
    event ID, subsequent rows are values at the configured model cadence.
    """
    stage_ids, stage_matrix = _read_clean_table(stage_table)
    precip_ids, precip_matrix = _read_clean_table(precipitation_table)
    if stage_ids != precip_ids:
        raise ValueError(
            "Clean boundary tables must contain identical event IDs in the same order. "
            f"Stage starts with {stage_ids[:3]}, precipitation starts with {precip_ids[:3]}."
        )
    if stage_matrix.shape != precip_matrix.shape:
        raise ValueError(
            "Clean boundary tables must have matching value shapes. "
            f"Got stage={stage_matrix.shape}, precipitation={precip_matrix.shape}."
        )
    available_steps = int(stage_matrix.shape[0]) - int(skip_before_timestep) - int(n_history)
    if available_steps < 1:
        raise ValueError(
            f"Clean boundary tables are too short: got {stage_matrix.shape[0]} rows, "
            f"need at least {int(skip_before_timestep) + int(n_history) + 1}."
        )
    forecast_steps = min(available_steps, int(max_forecast_steps))
    forcings: list[ForcingInput] = []
    for col_idx, event_id in enumerate(stage_ids):
        stage = stage_matrix[:, col_idx].astype(np.float32, copy=False)
        precip = precip_matrix[:, col_idx].astype(np.float32, copy=False)
        text = _canonical_forcing_text(event_id, stage, precip, int(dt_seconds))
        forcings.append(
            ForcingInput(
                stage=stage,
                precipitation=precip,
                dt_seconds=int(dt_seconds),
                forecast_steps=int(forecast_steps),
                input_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                source_name=f"{event_id}.clean_boundary",
            )
        )
    return forcings


def _read_clean_table(path: Path) -> tuple[list[str], np.ndarray]:
    rows = path.expanduser().read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError(f"Clean boundary table is empty: {path}")
    event_ids = [item.strip() for item in rows[0].split()]
    if not event_ids:
        raise ValueError(f"Clean boundary table has no event IDs: {path}")
    values: list[list[float]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if not row.strip():
            continue
        try:
            parsed = [float(item) for item in row.split()]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in {path} at row {row_number}.") from exc
        if len(parsed) != len(event_ids):
            raise ValueError(
                f"Row {row_number} in {path} has {len(parsed)} values; expected {len(event_ids)}."
            )
        values.append(parsed)
    if not values:
        raise ValueError(f"Clean boundary table has no value rows: {path}")
    matrix = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Clean boundary table contains NaN or infinite values: {path}")
    return event_ids, matrix


def _canonical_forcing_text(event_id: str, stage: np.ndarray, precipitation: np.ndarray, dt_seconds: int) -> str:
    lines = ["time_seconds,stage,precipitation"]
    for idx, (stage_value, precip_value) in enumerate(zip(stage, precipitation, strict=True)):
        if not math.isfinite(float(stage_value)) or not math.isfinite(float(precip_value)):
            raise ValueError(f"Non-finite forcing value for {event_id} at row {idx}.")
        lines.append(f"{idx * int(dt_seconds)},{float(stage_value):.9g},{float(precip_value):.9g}")
    return "\n".join(lines) + "\n"


def _forcing_paths_from_globs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    return paths


def build_hecras_forecast_reference(
    *,
    hdf_paths: Sequence[Path],
    start_timestep: int,
    forecast_steps: int,
    dt_seconds: int,
    thresholds_m: Sequence[float] = (0.05, 0.30),
    rows_out: Path | None = None,
) -> dict[str, object]:
    """Stream HEC-RAS HDF outputs into compact post-run monitoring percentiles.

    The Coastal FGN serving bundle uses the first ``N`` HEC-RAS hydraulic-depth
    columns, where ``N`` is the HDF ``Cell Points`` count. The same ordering is
    validated by the serving bundle static area tensor, so these descriptors are
    directly comparable with web-run forecast descriptors.
    """
    rows: list[dict[str, float | int | str]] = []
    failures: list[dict[str, str]] = []
    for path in hdf_paths:
        try:
            rows.append(
                hecras_descriptor_for_hdf(
                    path,
                    start_timestep=start_timestep,
                    forecast_steps=forecast_steps,
                    dt_seconds=dt_seconds,
                    thresholds_m=thresholds_m,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised by integration smoke on Rivanna
            failures.append({"path": str(path), "error": str(exc)})
    if not rows:
        raise ValueError("No HEC-RAS descriptors were produced.")

    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
    )
    percentiles: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray(
            [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))],
            dtype=np.float64,
        )
        if values.size:
            percentiles[key] = _percentile_payload(values)

    if rows_out is not None:
        rows_out.expanduser().parent.mkdir(parents=True, exist_ok=True)
        with rows_out.expanduser().open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_hdf_paths": len(hdf_paths),
        "n_descriptors": len(rows),
        "n_failures": len(failures),
        "thresholds_m": [float(threshold) for threshold in thresholds_m],
        "forecast_descriptor_percentiles": percentiles,
        "failures": failures[:50],
        "provenance": {
            "start_timestep": int(start_timestep),
            "forecast_steps": int(forecast_steps),
            "dt_seconds": int(dt_seconds),
            "cell_subset": (
                "first N hydraulic-depth cells, where N equals "
                "Geometry/2D Flow Areas/Cell Points"
            ),
        },
    }


def hecras_descriptor_for_hdf(
    path: Path,
    *,
    start_timestep: int,
    forecast_steps: int,
    dt_seconds: int,
    thresholds_m: Sequence[float],
) -> dict[str, float | int | str]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("HEC-RAS monitoring scan requires h5py.") from exc

    with h5py.File(path.expanduser(), "r") as handle:
        n_cells = int(handle[HECRAS_CELL_POINTS_DATASET].shape[0])
        cell_area_m2 = np.asarray(handle[HECRAS_AREA_DATASET][:n_cells], dtype=np.float64)
        depth_dataset = handle[HECRAS_DEPTH_DATASET]
        end_timestep = min(int(depth_dataset.shape[0]), int(start_timestep) + int(forecast_steps))
        if end_timestep <= int(start_timestep):
            raise ValueError(
                f"HEC-RAS output has insufficient timesteps: {path} "
                f"shape={depth_dataset.shape}, start_timestep={start_timestep}."
            )
        wd_m = np.asarray(depth_dataset[int(start_timestep) : end_timestep, :n_cells], dtype=np.float32)

    wd_m = np.maximum(wd_m, 0.0)
    wettable_area_m2 = float(np.sum(cell_area_m2))
    if wettable_area_m2 <= 0:
        raise ValueError(f"HEC-RAS output has non-positive wettable area: {path}")
    lead_hours = np.arange(wd_m.shape[0], dtype=np.float64) * float(dt_seconds) / 3600.0
    row: dict[str, float | int | str] = {
        "source_hdf": str(path),
        "n_time": int(wd_m.shape[0]),
        "n_cells": int(wd_m.shape[1]),
        "wettable_area_m2": wettable_area_m2,
        "wettable_area_km2": wettable_area_m2 / 1_000_000.0,
        "calibrated_max_mean_wd_m": float(np.max(wd_m)),
        "hecras_mean_wd_overall_m": float(np.mean(wd_m)),
    }
    for threshold in thresholds_m:
        row.update(_threshold_hecras_descriptors(wd_m, cell_area_m2, lead_hours, threshold))
    return row


def _threshold_hecras_descriptors(
    wd_m: np.ndarray,
    cell_area_m2: np.ndarray,
    lead_hours: np.ndarray,
    threshold_m: float,
) -> dict[str, float]:
    suffix = _threshold_suffix(float(threshold_m))
    flooded = wd_m > float(threshold_m)
    area_by_time_m2 = flooded.astype(np.float64) @ cell_area_m2
    peak_idx = int(np.argmax(area_by_time_m2)) if area_by_time_m2.size else 0
    peak_area_m2 = float(area_by_time_m2[peak_idx]) if area_by_time_m2.size else 0.0
    wettable_area_m2 = float(np.sum(cell_area_m2))
    peak_fraction = peak_area_m2 / wettable_area_m2 if wettable_area_m2 > 0 else 0.0
    lead = float(lead_hours[peak_idx]) if lead_hours.size else 0.0
    descriptors = {
        f"peak_expected_area_fraction_wettable_gt_{suffix}m": peak_fraction,
        f"peak_expected_area_km2_gt_{suffix}m": peak_area_m2 / 1_000_000.0,
        f"peak_high_confidence_area_fraction_wettable_gt_{suffix}m": peak_fraction,
        f"peak_expected_area_lead_hours_gt_{suffix}m": lead,
    }
    if abs(float(threshold_m) - 0.05) < 1.0e-9:
        descriptors.update(
            {
                "peak_expected_flooded_area_fraction_wettable_gt_0.05m": peak_fraction,
                "peak_expected_flooded_area_km2_gt_0.05m": peak_area_m2 / 1_000_000.0,
                "peak_expected_flooded_area_lead_hours_gt_0.05m": lead,
            }
        )
    return descriptors


def _threshold_suffix(threshold: float) -> str:
    text = f"{float(threshold):g}"
    return text if "." in text else f"{text}.0"


def _percentile_payload(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _paths_from_list_file(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.expanduser().read_text().splitlines() if line.strip()]


def _build_heuristic_reference_percentiles(
    run_summary_paths: list[str],
) -> dict[str, dict[str, float]] | None:
    if not run_summary_paths:
        return None
    heuristic_rows: list[dict[str, float]] = []
    for raw_path in run_summary_paths:
        summary = json.loads(Path(raw_path).expanduser().read_text(encoding="utf-8"))
        row: dict[str, float] = {}
        if "uncertainty_to_signal_ratio" in summary:
            row["uncertainty_to_signal_ratio"] = float(summary["uncertainty_to_signal_ratio"])
        iqr = summary.get("peak_area_weighted_iqr_wd_m")
        area_frac = summary.get("peak_expected_flooded_area_fraction_wettable_gt_0.05m")
        if iqr is not None and area_frac is not None:
            row["iqr_area_product"] = float(iqr) * float(area_frac)
        cal_shift = summary.get("max_abs_calibration_shift_percentage_points_by_threshold")
        if cal_shift is not None:
            row["max_abs_calibration_shift_pp"] = float(cal_shift)
        if row:
            heuristic_rows.append(row)
    if not heuristic_rows:
        return None
    all_keys = sorted({k for r in heuristic_rows for k in r})
    result: dict[str, dict[str, float]] = {}
    for key in all_keys:
        vals = np.asarray([r[key] for r in heuristic_rows if key in r], dtype=np.float64)
        if vals.size:
            result[key] = _percentile_payload(vals)
    return result or None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create an FGN serving monitoring bundle from train/test reference forcings."
    )
    parser.add_argument("--model-bundle", required=True, help="Path to coastal_fgn_bundle.json.")
    parser.add_argument("--out", required=True, help="Output monitoring bundle JSON path.")
    parser.add_argument("--bundle-id", default="coastal-fgn-monitoring-v1")
    parser.add_argument(
        "--forcing-glob",
        action="append",
        default=[],
        help="Glob for train/test forcing CSVs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--clean-boundary-pair",
        action="append",
        default=[],
        metavar="STAGE_TABLE,PRECIPITATION_TABLE",
        help=(
            "Clean-family boundary table pair. Each table has event IDs as columns. "
            "Pass once per split, for example train and test."
        ),
    )
    parser.add_argument("--hecras-root", default=None, help="Optional HEC-RAS reference-output root for provenance.")
    parser.add_argument(
        "--hecras-hdf-glob",
        action="append",
        default=[],
        help="Glob for HEC-RAS HDF reference outputs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--hecras-hdf-list",
        action="append",
        default=[],
        help="Text file containing one HEC-RAS HDF path per line. Can be passed multiple times.",
    )
    parser.add_argument(
        "--hecras-scan-limit",
        type=int,
        default=0,
        help="Optional maximum number of HEC-RAS HDF files to scan, for smoke builds.",
    )
    parser.add_argument("--hecras-start-timestep", type=int, default=None)
    parser.add_argument("--hecras-forecast-steps", type=int, default=None)
    parser.add_argument("--hecras-threshold-m", type=float, action="append", default=[0.05, 0.30])
    parser.add_argument("--hecras-rows-out", default=None, help="Optional JSONL rows output for HEC-RAS descriptors.")
    parser.add_argument(
        "--transform-spec",
        default=None,
        help="JSON file mapping descriptor names to transform functions (log1p, logit_bounded, identity).",
    )
    parser.add_argument(
        "--covariance-exclude",
        action="append",
        default=[],
        help="Descriptor names to exclude from covariance computation. Can be passed multiple times.",
    )
    parser.add_argument(
        "--heuristic-reference-runs",
        action="append",
        default=[],
        help="Paths to historical FGN calibrated run summary JSONs for reference-derived post-run thresholds.",
    )
    args = parser.parse_args(argv)

    model_bundle = load_model_bundle(args.model_bundle)
    paths = _forcing_paths_from_globs(args.forcing_glob)
    forcings = [parse_forcing_csv(path, bundle=model_bundle) for path in paths]
    clean_pairs: list[tuple[str, str]] = []
    for raw_pair in args.clean_boundary_pair:
        parts = [part.strip() for part in raw_pair.split(",", maxsplit=1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise SystemExit("--clean-boundary-pair must be STAGE_TABLE,PRECIPITATION_TABLE.")
        clean_pairs.append((parts[0], parts[1]))
        forcings.extend(
            load_clean_boundary_table_forcings(
                stage_table=Path(parts[0]),
                precipitation_table=Path(parts[1]),
                dt_seconds=int(model_bundle.dt_seconds),
                skip_before_timestep=int(model_bundle.skip_before_timestep),
                n_history=int(model_bundle.n_history),
                max_forecast_steps=int(model_bundle.max_forecast_steps),
            )
        )
    if not forcings:
        raise SystemExit("No reference forcings found. Use --forcing-glob or --clean-boundary-pair.")

    hdf_paths = _forcing_paths_from_globs(args.hecras_hdf_glob)
    for raw_list in args.hecras_hdf_list:
        hdf_paths.extend(_paths_from_list_file(Path(raw_list)))
    if args.hecras_scan_limit and args.hecras_scan_limit > 0:
        hdf_paths = hdf_paths[: int(args.hecras_scan_limit)]

    transform_spec: dict[str, str] | None = None
    if args.transform_spec:
        transform_spec = json.loads(Path(args.transform_spec).expanduser().read_text(encoding="utf-8"))

    heuristic_ref_pcts = _build_heuristic_reference_percentiles(args.heuristic_reference_runs)

    payload = build_monitoring_bundle_from_forcings(
        bundle_id=args.bundle_id,
        forcings=forcings,
        provenance={
            "model_bundle_id": model_bundle.bundle_id,
            "model_bundle_path": str(Path(args.model_bundle).expanduser()),
            "forcing_globs": list(args.forcing_glob),
            "clean_boundary_pairs": clean_pairs,
            "n_reference_forcings": len(forcings),
            "hecras_root": args.hecras_root,
            "hecras_hdf_globs": list(args.hecras_hdf_glob),
            "hecras_hdf_lists": list(args.hecras_hdf_list),
        },
        transform_spec=transform_spec,
        covariance_exclude_descriptors=args.covariance_exclude or None,
        heuristic_reference_percentiles=heuristic_ref_pcts,
    )
    if hdf_paths:
        hecras_reference = build_hecras_forecast_reference(
            hdf_paths=hdf_paths,
            start_timestep=(
                int(args.hecras_start_timestep)
                if args.hecras_start_timestep is not None
                else int(model_bundle.skip_before_timestep) + int(model_bundle.n_history)
            ),
            forecast_steps=(
                int(args.hecras_forecast_steps)
                if args.hecras_forecast_steps is not None
                else int(model_bundle.max_forecast_steps)
            ),
            dt_seconds=int(model_bundle.dt_seconds),
            thresholds_m=args.hecras_threshold_m,
            rows_out=Path(args.hecras_rows_out) if args.hecras_rows_out else None,
        )
        payload["forecast_descriptor_percentiles"] = hecras_reference["forecast_descriptor_percentiles"]
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            provenance = {}
            payload["provenance"] = provenance
        provenance.update(
            {
                "hecras_scan": hecras_reference["provenance"],
                "hecras_scan_n_hdf_paths": hecras_reference["n_hdf_paths"],
                "hecras_scan_n_descriptors": hecras_reference["n_descriptors"],
                "hecras_scan_n_failures": hecras_reference["n_failures"],
                "hecras_scan_thresholds_m": hecras_reference["thresholds_m"],
                "hecras_scan_created_at": hecras_reference["created_at"],
                "hecras_scan_failures": hecras_reference["failures"],
            }
        )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote monitoring bundle: {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
