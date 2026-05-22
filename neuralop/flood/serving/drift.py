"""Population-level drift detection for FGN monitoring descriptor streams.

Framework-independent. Consumes descriptor log entries, produces drift test
results for admin-level visibility. Not used in per-run candidate scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class DriftTestConfig:
    window_size: int = 50
    cusum_allowance: float = 0.5
    cusum_threshold: float = 5.0
    energy_permutations: int = 200
    fdr_alpha: float = 0.05
    persistence_days: int = 2


@dataclass(frozen=True)
class DriftTestResult:
    test_id: str
    test_type: str
    descriptor_name: str | None
    drift_detected: bool
    test_statistic: float
    threshold: float
    monitoring_bundle_id: str | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    n_observations: int | None = None
    details: dict[str, object] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DriftDetector:
    def __init__(self, config: DriftTestConfig | None = None) -> None:
        self.config = config or DriftTestConfig()

    def cusum_test(
        self,
        stream: list[dict[str, float]],
        target: dict[str, float],
        scale: dict[str, float],
    ) -> list[DriftTestResult]:
        results: list[DriftTestResult] = []
        if not stream:
            return results

        descriptors = sorted(stream[0].keys())
        for desc in descriptors:
            t = target.get(desc, 0.0)
            s = scale.get(desc, 1.0)
            if s <= 0:
                s = 1.0

            s_upper = 0.0
            s_lower = 0.0
            max_upper = 0.0
            max_lower = 0.0

            for entry in stream:
                z = (entry.get(desc, 0.0) - t) / s
                s_upper = max(0.0, s_upper + z - self.config.cusum_allowance)
                s_lower = max(0.0, s_lower - z - self.config.cusum_allowance)
                max_upper = max(max_upper, s_upper)
                max_lower = max(max_lower, s_lower)

            stat = max(max_upper, max_lower)
            results.append(DriftTestResult(
                test_id=f"cusum_{desc}",
                test_type="cusum",
                descriptor_name=desc,
                drift_detected=stat > self.config.cusum_threshold,
                test_statistic=stat,
                threshold=self.config.cusum_threshold,
            ))
        return results

    def welch_fdr_test(
        self,
        stream: list[dict[str, float]],
    ) -> list[DriftTestResult]:
        import numpy as np

        if len(stream) < 4:
            return []

        n = len(stream)
        half = n // 2
        ref_half = stream[:half]
        recent_half = stream[half:]

        descriptors = sorted(stream[0].keys())
        raw_results: list[tuple[float, str, float]] = []

        for desc in descriptors:
            ref_vals = np.array([e.get(desc, 0.0) for e in ref_half], dtype=np.float64)
            rec_vals = np.array([e.get(desc, 0.0) for e in recent_half], dtype=np.float64)

            n1, n2 = len(ref_vals), len(rec_vals)
            m1, m2 = np.mean(ref_vals), np.mean(rec_vals)
            v1, v2 = np.var(ref_vals, ddof=1), np.var(rec_vals, ddof=1)

            if v1 == 0 and v2 == 0:
                raw_results.append((1.0, desc, 0.0))
                continue

            se = math.sqrt(v1 / n1 + v2 / n2) if (v1 / n1 + v2 / n2) > 0 else 1e-10
            t_stat = (m1 - m2) / se

            num = (v1 / n1 + v2 / n2) ** 2
            d1 = (v1 / n1) ** 2 / max(n1 - 1, 1)
            d2 = (v2 / n2) ** 2 / max(n2 - 1, 1)
            df = num / (d1 + d2) if (d1 + d2) > 0 else 1.0

            from scipy.stats import t as t_dist
            p_value = float(2 * t_dist.sf(abs(t_stat), df))
            raw_results.append((p_value, desc, abs(t_stat)))

        raw_results.sort(key=lambda x: x[0])
        m = len(raw_results)

        results: list[DriftTestResult] = []
        for rank, (p, desc, t_stat) in enumerate(raw_results, 1):
            bh_threshold = (rank / m) * self.config.fdr_alpha
            results.append(DriftTestResult(
                test_id=f"welch_{desc}",
                test_type="welch_fdr",
                descriptor_name=desc,
                drift_detected=p <= bh_threshold,
                test_statistic=t_stat,
                threshold=bh_threshold,
                details={"p_value": p, "bh_rank": rank},
            ))
        return results

    def energy_distance_test(
        self,
        reference: list[dict[str, float]],
        recent: list[dict[str, float]],
    ) -> DriftTestResult:
        import numpy as np

        desc_names = sorted(reference[0].keys())
        R = np.array([[e[d] for d in desc_names] for e in reference], dtype=np.float64)
        W = np.array([[e[d] for d in desc_names] for e in recent], dtype=np.float64)

        observed_e = _energy_stat(R, W)

        combined = np.vstack([R, W])
        n_r = len(R)
        rng = np.random.default_rng(42)
        count_ge = 0
        for _ in range(self.config.energy_permutations):
            perm = rng.permutation(len(combined))
            perm_R = combined[perm[:n_r]]
            perm_W = combined[perm[n_r:]]
            perm_e = _energy_stat(perm_R, perm_W)
            if perm_e >= observed_e:
                count_ge += 1

        p_value = (count_ge + 1) / (self.config.energy_permutations + 1)

        return DriftTestResult(
            test_id="energy_distance",
            test_type="energy_distance",
            descriptor_name=None,
            drift_detected=p_value < 0.01,
            test_statistic=observed_e,
            threshold=0.01,
            details={"p_value": p_value},
        )

    def apply_persistence_filter(
        self,
        daily_results: list[list[DriftTestResult]],
    ) -> list[DriftTestResult]:
        if len(daily_results) < self.config.persistence_days:
            return []

        consecutive_window = daily_results[-self.config.persistence_days:]
        persistent: list[DriftTestResult] = []

        all_descriptors: set[str | None] = set()
        for day in consecutive_window:
            for r in day:
                if r.drift_detected:
                    all_descriptors.add(r.descriptor_name)

        for desc in all_descriptors:
            days_detected = 0
            latest_result: DriftTestResult | None = None
            for day in consecutive_window:
                day_detected = any(
                    r.drift_detected and r.descriptor_name == desc for r in day
                )
                if day_detected:
                    days_detected += 1
                    for r in day:
                        if r.drift_detected and r.descriptor_name == desc:
                            latest_result = r

            if days_detected >= self.config.persistence_days and latest_result is not None:
                persistent.append(latest_result)

        return persistent


def _energy_stat(R, W):
    import numpy as np
    from scipy.spatial.distance import cdist

    cross = cdist(R, W, metric="euclidean")
    intra_r = cdist(R, R, metric="euclidean")
    intra_w = cdist(W, W, metric="euclidean")

    return float(
        2 * np.mean(cross) - np.mean(intra_r) - np.mean(intra_w)
    )
