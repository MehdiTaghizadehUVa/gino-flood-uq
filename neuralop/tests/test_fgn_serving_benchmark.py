from __future__ import annotations

import pytest

from neuralop.flood.serving.benchmark_inference import (
    BenchmarkResult,
    choose_recommended_result,
    parse_chunk_sizes,
)
from neuralop.flood.serving.chunk_science_check import _metric_deltas, _parse_chunk_sizes


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


def test_chunk_science_check_reports_scalar_and_series_deltas():
    assert _parse_chunk_sizes("4,20") == [4, 20]

    deltas = _metric_deltas(
        {
            "peak_area_fraction": 0.51,
            "area_fraction_by_time": [0.1, 0.2, 0.3],
        },
        {
            "peak_area_fraction": 0.50,
            "area_fraction_by_time": [0.1, 0.25, 0.29],
        },
    )

    assert deltas["peak_area_fraction"]["max_abs"] == pytest.approx(0.01)
    assert deltas["area_fraction_by_time"]["max_abs"] == pytest.approx(0.05)
    assert deltas["area_fraction_by_time"]["mean_abs"] == pytest.approx(0.02)
