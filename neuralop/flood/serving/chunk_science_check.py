"""Compare production chunk sizes using scientific forecast metrics.

This is a non-production validation utility. It runs the same inference seam as
the Celery worker and writes a compact JSON report that can be used before
raising ``FGN_MEMBER_CHUNK_SIZE`` on the lab deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from neuralop.flood.serving.forcing import parse_forcing_csv
from neuralop.flood.serving.inference import ProductionFGNInferenceService
from neuralop.flood.serving.model_bundle import load_model_bundle
from neuralop.flood.serving.products import ForecastProductBuilder
from neuralop.flood.serving.run_spec import RunSpec


SCALAR_KEYS = (
    "max_mean_wd_m",
    "mean_wd_overall_m",
    "mean_spread_wd_m",
    "peak_expected_flooded_area_fraction_wettable_gt_0.05m",
    "peak_expected_flooded_area_km2_gt_0.05m",
    "peak_expected_flooded_area_lead_hours_gt_0.05m",
    "peak_area_weighted_iqr_wd_m",
    "peak_area_weighted_central_90_wd_m",
    "uncertainty_to_signal_ratio",
    "peak_p95_wd_lead_hours",
)

SERIES_KEYS = (
    "peak_mean_wd_by_time_m",
    "p95_wd_peak_by_time_m",
    "expected_flooded_area_fraction_wettable_by_time_gt_0.05m",
    "area_weighted_iqr_wd_m_by_time",
    "area_weighted_central_90_wd_m_by_time",
)


def _parse_chunk_sizes(raw: str) -> list[int]:
    chunks = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not chunks:
        raise ValueError("At least one chunk size is required.")
    if any(chunk < 1 for chunk in chunks):
        raise ValueError("Chunk sizes must be positive integers.")
    return chunks


def _max_abs(candidate: Any, baseline: Any) -> float | None:
    if candidate is None or baseline is None:
        return None if candidate == baseline else float("inf")
    cand = np.asarray(candidate, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    if cand.shape != base.shape:
        return float("inf")
    return float(np.nanmax(np.abs(cand - base)))


def _mean_abs(candidate: Any, baseline: Any) -> float | None:
    if candidate is None or baseline is None:
        return None if candidate == baseline else float("inf")
    cand = np.asarray(candidate, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    if cand.shape != base.shape:
        return float("inf")
    return float(np.nanmean(np.abs(cand - base)))


def _scientific_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {key: summary.get(key) for key in SCALAR_KEYS}
    metrics.update({key: summary.get(key) for key in SERIES_KEYS})
    for threshold in ("0.05", "0.3", "0.5"):
        item = (summary.get("exceedance_by_threshold_m") or {}).get(threshold)
        if not item:
            continue
        suffix = threshold.replace(".", "p")
        metrics[f"peak_exceedance_area_fraction_gt_{suffix}"] = item.get(
            "peak_expected_area_fraction_wettable"
        )
        metrics[f"peak_high_confidence_area_fraction_gt_{suffix}"] = item.get(
            "peak_high_confidence_area_fraction_wettable"
        )
        metrics[f"mean_probability_gt_{suffix}"] = item.get("mean_probability")
    return metrics


def _metric_deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    deltas: dict[str, dict[str, float | None]] = {}
    for key, candidate_value in candidate.items():
        baseline_value = baseline.get(key)
        deltas[key] = {
            "max_abs": _max_abs(candidate_value, baseline_value),
            "mean_abs": _mean_abs(candidate_value, baseline_value),
        }
    return deltas


def _cuda_peak_mb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)


def run_check(
    *,
    bundle_path: Path,
    forcing_csv_path: Path,
    chunk_sizes: list[int],
    forecast_steps: int | None,
    output_json: Path,
    device: str,
    seed: int,
) -> dict[str, Any]:
    import torch

    bundle = load_model_bundle(bundle_path)
    forcing = parse_forcing_csv(
        forcing_csv_path.read_text(encoding="utf-8"),
        bundle=bundle,
        requested_forecast_steps=forecast_steps,
    )
    spec = RunSpec.new(
        user_id="chunk-science-check",
        bundle_id=bundle.bundle_id,
        input_hash=forcing.input_hash,
        forecast_steps=forcing.forecast_steps,
        seed=seed,
    )
    builder = ForecastProductBuilder()
    baseline_members: np.ndarray | None = None
    baseline_metrics: dict[str, Any] | None = None
    results: dict[str, Any] = {}
    for chunk_size in chunk_sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        service = ProductionFGNInferenceService(
            bundle,
            device=device,
            member_chunk_size=chunk_size,
            inference_dtype="fp32",
        )
        start = perf_counter()
        forecast = service.run(spec, forcing)
        seconds = perf_counter() - start
        members = np.asarray(forecast.members_wd, dtype=np.float32)
        metrics = _scientific_metrics(builder.build_summary(forecast, label=f"chunk_{chunk_size}"))
        if baseline_members is None:
            baseline_members = members
            baseline_metrics = metrics
            comparison = {
                "max_abs_wd_m": 0.0,
                "mean_abs_wd_m": 0.0,
                "scientific_metric_deltas": {},
            }
        else:
            comparison = {
                "max_abs_wd_m": _max_abs(members, baseline_members),
                "mean_abs_wd_m": _mean_abs(members, baseline_members),
                "scientific_metric_deltas": _metric_deltas(metrics, baseline_metrics or {}),
            }
        results[str(chunk_size)] = {
            "chunk_size": chunk_size,
            "seconds": seconds,
            "cuda_peak_mb": _cuda_peak_mb(),
            "shape": list(members.shape),
            "against_baseline": comparison,
            "metrics": metrics,
        }
        print(
            json.dumps(
                {
                    "chunk_size": chunk_size,
                    "seconds": seconds,
                    "cuda_peak_mb": results[str(chunk_size)]["cuda_peak_mb"],
                    "against_baseline": comparison,
                },
                indent=2,
            ),
            flush=True,
        )
    payload = {
        "bundle_id": bundle.bundle_id,
        "forcing_csv": str(forcing_csv_path),
        "forecast_steps": forcing.forecast_steps,
        "seed": seed,
        "baseline_chunk_size": chunk_sizes[0],
        "results": results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate chunk-size scientific metric deltas.")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--forcing-csv", required=True, type=Path)
    parser.add_argument("--chunk-sizes", default="4,20")
    parser.add_argument("--forecast-steps", type=int, default=None)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    run_check(
        bundle_path=args.bundle,
        forcing_csv_path=args.forcing_csv,
        chunk_sizes=_parse_chunk_sizes(args.chunk_sizes),
        forecast_steps=args.forecast_steps,
        output_json=args.output_json,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
