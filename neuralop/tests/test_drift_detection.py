"""Tests for Phase 2 population drift detection algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from neuralop.flood.serving.drift import DriftDetector, DriftTestConfig, DriftTestResult


def _stable_stream(n: int = 100, seed: int = 42) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    return [{"x": float(rng.normal(0, 1)), "y": float(rng.normal(0, 1))} for _ in range(n)]


def _shifted_stream(n: int = 100, shift_at: int = 50, delta: float = 2.0, seed: int = 42) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    stream = []
    for i in range(n):
        mu = delta if i >= shift_at else 0.0
        stream.append({"x": float(rng.normal(mu, 1)), "y": float(rng.normal(0, 1))})
    return stream


class TestCUSUM:
    def test_detects_mean_shift(self):
        stream = _shifted_stream(n=100, shift_at=30, delta=2.0)
        config = DriftTestConfig(window_size=100, cusum_allowance=0.5, cusum_threshold=5.0)
        detector = DriftDetector(config)

        results = detector.cusum_test(stream, target={"x": 0.0, "y": 0.0}, scale={"x": 1.0, "y": 1.0})

        x_results = [r for r in results if r.descriptor_name == "x"]
        assert len(x_results) == 1
        assert x_results[0].drift_detected is True

    def test_stable_stream_no_alarm(self):
        stream = _stable_stream(n=100)
        config = DriftTestConfig(window_size=100, cusum_allowance=0.5, cusum_threshold=5.0)
        detector = DriftDetector(config)

        results = detector.cusum_test(stream, target={"x": 0.0, "y": 0.0}, scale={"x": 1.0, "y": 1.0})

        for r in results:
            assert r.drift_detected is False

    def test_recomputed_from_log_is_deterministic(self):
        stream = _shifted_stream(n=80, shift_at=40, delta=1.5)
        config = DriftTestConfig(window_size=80)
        detector = DriftDetector(config)

        results1 = detector.cusum_test(stream, target={"x": 0.0, "y": 0.0}, scale={"x": 1.0, "y": 1.0})
        results2 = detector.cusum_test(stream, target={"x": 0.0, "y": 0.0}, scale={"x": 1.0, "y": 1.0})

        for r1, r2 in zip(results1, results2):
            assert r1.drift_detected == r2.drift_detected
            assert r1.test_statistic == pytest.approx(r2.test_statistic)


class TestWelchFDR:
    def test_detects_drift(self):
        stream = _shifted_stream(n=100, shift_at=50, delta=2.0)
        config = DriftTestConfig(window_size=100, fdr_alpha=0.05)
        detector = DriftDetector(config)

        results = detector.welch_fdr_test(stream)

        x_results = [r for r in results if r.descriptor_name == "x"]
        assert len(x_results) == 1
        assert x_results[0].drift_detected is True

    def test_fdr_controls_false_positives(self):
        rng = np.random.default_rng(99)
        many_descriptors = [
            {f"d{i}": float(rng.normal(0, 1)) for i in range(16)}
            for _ in range(100)
        ]
        config = DriftTestConfig(window_size=100, fdr_alpha=0.05)
        detector = DriftDetector(config)

        results = detector.welch_fdr_test(many_descriptors)

        detected = [r for r in results if r.drift_detected]
        assert len(detected) <= 2


class TestEnergyDistance:
    def test_detects_distribution_change(self):
        rng = np.random.default_rng(42)
        reference = [{"x": float(rng.normal(0, 1)), "y": float(rng.normal(0, 1))} for _ in range(60)]
        recent = [{"x": float(rng.normal(2, 1)), "y": float(rng.normal(0, 1))} for _ in range(60)]
        config = DriftTestConfig(energy_permutations=200)
        detector = DriftDetector(config)

        result = detector.energy_distance_test(reference, recent)

        assert result.drift_detected is True
        assert result.test_statistic > 0

    def test_same_distribution_no_alarm(self):
        rng = np.random.default_rng(42)
        ref = [{"x": float(rng.normal(0, 1)), "y": float(rng.normal(0, 1))} for _ in range(60)]
        rng2 = np.random.default_rng(99)
        recent = [{"x": float(rng2.normal(0, 1)), "y": float(rng2.normal(0, 1))} for _ in range(60)]
        config = DriftTestConfig(energy_permutations=200)
        detector = DriftDetector(config)

        result = detector.energy_distance_test(ref, recent)

        assert result.drift_detected is False


class TestPersistenceFilter:
    def test_requires_consecutive_days(self):
        from datetime import datetime, timezone

        config = DriftTestConfig(persistence_days=2)
        detector = DriftDetector(config)

        day1_results = [
            DriftTestResult(
                test_id="t1", test_type="cusum", descriptor_name="x",
                drift_detected=True, test_statistic=6.0, threshold=5.0,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        ]
        day2_results = [
            DriftTestResult(
                test_id="t2", test_type="cusum", descriptor_name="x",
                drift_detected=True, test_statistic=7.0, threshold=5.0,
                created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            )
        ]

        filtered1 = detector.apply_persistence_filter([day1_results])
        assert len(filtered1) == 0

        filtered2 = detector.apply_persistence_filter([day1_results, day2_results])
        x_persistent = [r for r in filtered2 if r.descriptor_name == "x"]
        assert len(x_persistent) >= 1
