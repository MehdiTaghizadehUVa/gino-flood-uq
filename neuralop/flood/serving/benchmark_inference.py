"""Non-production benchmark harness for FGN serving inference.

This module is intentionally CLI-friendly and importable by tests. It exercises
the same ProductionFGNInferenceService seam as the worker, but it does not write
run artifacts or touch the queue/database.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from neuralop.flood.serving.forcing import parse_forcing_csv
from neuralop.flood.serving.inference import ProductionFGNInferenceService
from neuralop.flood.serving.model_bundle import load_model_bundle
from neuralop.flood.serving.run_spec import RunSpec


@dataclass(frozen=True)
class BenchmarkResult:
    chunk_size: int
    dtype: str
    seconds: float | None = None
    max_cuda_memory_mb: float | None = None
    max_abs_diff_m: float | None = None
    max_summary_delta_pp: float | None = None
    oom: bool = False
    error: str | None = None

    @property
    def is_safe(self) -> bool:
        if self.oom or self.error:
            return False
        if self.seconds is None:
            return False
        if self.max_abs_diff_m is not None and self.max_abs_diff_m > 0.02:
            return False
        if self.max_summary_delta_pp is not None and self.max_summary_delta_pp > 1.0:
            return False
        return True


def parse_chunk_sizes(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        value = int(text)
        if value < 1:
            raise ValueError("Chunk sizes must be positive integers.")
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError("At least one chunk size is required.")
    return values


def choose_recommended_result(results: list[BenchmarkResult]) -> BenchmarkResult | None:
    safe = [result for result in results if result.is_safe]
    if not safe:
        return None
    return min(safe, key=lambda result: float(result.seconds))


def _cuda_peak_mb(torch) -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)


def _summary_delta_pp(candidate: np.ndarray, baseline: np.ndarray) -> float:
    thresholds = (0.05, 0.30, 0.50)
    max_delta = 0.0
    for threshold in thresholds:
        cand_fraction = np.mean(candidate > threshold, axis=(0, 2))
        base_fraction = np.mean(baseline > threshold, axis=(0, 2))
        max_delta = max(max_delta, float(np.max(np.abs(cand_fraction - base_fraction))) * 100.0)
    return max_delta


def run_benchmark(
    *,
    bundle_path: Path,
    forcing_csv_path: Path,
    chunk_sizes: list[int],
    dtypes: list[str],
    forecast_steps: int | None,
    device: str,
    seed: int,
) -> dict[str, object]:
    import torch

    bundle = load_model_bundle(bundle_path)
    forcing_text = forcing_csv_path.read_text(encoding="utf-8")
    forcing = parse_forcing_csv(forcing_text, bundle=bundle, requested_forecast_steps=forecast_steps)
    spec = RunSpec.new(
        user_id="benchmark",
        bundle_id=bundle.bundle_id,
        input_hash=forcing.input_hash,
        forecast_steps=forcing.forecast_steps,
        seed=seed,
    )
    baseline_members = None
    results: list[BenchmarkResult] = []
    for dtype in dtypes:
        for chunk_size in chunk_sizes:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                service = ProductionFGNInferenceService(
                    bundle,
                    device=device,
                    member_chunk_size=chunk_size,
                    inference_dtype=dtype,
                )
                start = perf_counter()
                result = service.run(spec, forcing)
                seconds = perf_counter() - start
                members = np.asarray(result.members_wd, dtype=np.float32)
                if baseline_members is None and dtype == "fp32":
                    baseline_members = members
                max_abs_diff = None
                max_summary_delta = None
                if baseline_members is not None:
                    max_abs_diff = float(np.max(np.abs(members - baseline_members)))
                    max_summary_delta = _summary_delta_pp(members, baseline_members)
                results.append(
                    BenchmarkResult(
                        chunk_size=chunk_size,
                        dtype=dtype,
                        seconds=seconds,
                        max_cuda_memory_mb=_cuda_peak_mb(torch),
                        max_abs_diff_m=max_abs_diff,
                        max_summary_delta_pp=max_summary_delta,
                    )
                )
            except RuntimeError as exc:
                message = str(exc)
                is_oom = "out of memory" in message.lower()
                if is_oom and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                results.append(
                    BenchmarkResult(
                        chunk_size=chunk_size,
                        dtype=dtype,
                        oom=is_oom,
                        error=message[:500],
                    )
                )
    recommended = choose_recommended_result(results)
    return {
        "bundle_id": bundle.bundle_id,
        "forecast_steps": forcing.forecast_steps,
        "results": [asdict(result) for result in results],
        "recommended": asdict(recommended) if recommended is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark FGN serving inference chunk sizes and dtypes.")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--forcing-csv", required=True, type=Path)
    parser.add_argument("--chunk-sizes", default="4,8,10,16,20")
    parser.add_argument("--dtypes", default="fp32")
    parser.add_argument("--forecast-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = run_benchmark(
        bundle_path=args.bundle,
        forcing_csv_path=args.forcing_csv,
        chunk_sizes=parse_chunk_sizes(args.chunk_sizes),
        dtypes=[x.strip() for x in args.dtypes.split(",") if x.strip()],
        forecast_steps=args.forecast_steps,
        device=args.device,
        seed=args.seed,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
