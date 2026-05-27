from __future__ import annotations

from neuralop.flood.serving.benchmark_inference import (
    BenchmarkResult,
    choose_recommended_result,
    parse_chunk_sizes,
)


def test_parse_chunk_sizes_accepts_ordered_unique_values():
    assert parse_chunk_sizes("4,8,8,20") == [4, 8, 20]


def test_benchmark_recommends_fastest_safe_result():
    results = [
        BenchmarkResult(chunk_size=4, dtype="fp32", seconds=40.0, max_abs_diff_m=0.0, max_summary_delta_pp=0.0),
        BenchmarkResult(chunk_size=8, dtype="fp32", seconds=24.0, max_abs_diff_m=0.001, max_summary_delta_pp=0.1),
        BenchmarkResult(chunk_size=20, dtype="fp32", seconds=20.0, oom=True),
        BenchmarkResult(chunk_size=16, dtype="bf16", seconds=12.0, max_abs_diff_m=0.04, max_summary_delta_pp=0.2),
    ]

    chosen = choose_recommended_result(results)

    assert chosen is not None
    assert chosen.chunk_size == 8
    assert chosen.dtype == "fp32"
