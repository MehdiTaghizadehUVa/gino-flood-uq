"""Periodic drift test executor — recomputes from descriptor log.

Intended to run daily alongside cleanup-backup. Loads recent descriptor log
entries, runs CUSUM / Welch+FDR / energy distance tests, persists results,
and applies the persistence filter before surfacing warnings.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from neuralop.flood.serving.drift import DriftDetector, DriftTestConfig, DriftTestResult
from neuralop.flood.serving.monitoring import MonitoringBundle, MonitoringPhase

logger = logging.getLogger(__name__)


def run_drift_tests(
    *,
    database_url: str,
    monitoring_bundle_path: str | None = None,
    config: DriftTestConfig | None = None,
) -> list[DriftTestResult]:
    config = config or DriftTestConfig()
    detector = DriftDetector(config)

    bundle: MonitoringBundle | None = None
    if monitoring_bundle_path and os.path.exists(monitoring_bundle_path):
        bundle = MonitoringBundle.load(monitoring_bundle_path)

    try:
        import sqlalchemy as sa
    except ImportError:
        logger.warning("SQLAlchemy not available; skipping drift tests.")
        return []

    engine = sa.create_engine(database_url, future=True)
    metadata = sa.MetaData()

    descriptor_logs = sa.Table(
        "fgn_monitoring_descriptor_log", metadata,
        autoload_with=engine,
    )

    drift_results_table = sa.Table(
        "fgn_drift_test_results", metadata,
        sa.Column("test_id", sa.String, primary_key=True),
        sa.Column("test_type", sa.String, nullable=False, index=True),
        sa.Column("descriptor_name", sa.String, nullable=True),
        sa.Column("drift_detected", sa.Boolean, nullable=False),
        sa.Column("test_statistic", sa.Float, nullable=False),
        sa.Column("threshold", sa.Float, nullable=False),
        sa.Column("details_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
        extend_existing=True,
    )
    metadata.create_all(engine, tables=[drift_results_table])

    with engine.begin() as conn:
        rows = conn.execute(
            sa.select(descriptor_logs)
            .where(descriptor_logs.c.phase == MonitoringPhase.PRE_RUN.value)
            .order_by(descriptor_logs.c.created_at.asc())
            .limit(config.window_size)
        ).all()

    if len(rows) < 10:
        logger.info("Not enough descriptor log entries (%d) for drift testing.", len(rows))
        return []

    stream: list[dict[str, float]] = []
    for row in rows:
        row_map = getattr(row, "_mapping", row)
        descs = json.loads(row_map["descriptors_json"])
        numeric = {k: float(v) for k, v in descs.items() if isinstance(v, (int, float)) and v is not None}
        stream.append(numeric)

    all_results: list[DriftTestResult] = []

    target: dict[str, float] = {}
    scale: dict[str, float] = {}
    if bundle and bundle.descriptor_percentiles:
        for desc, pcts in bundle.descriptor_percentiles.items():
            target[desc] = pcts.get("p50", 0.0)
            p05 = pcts.get("p05", 0.0)
            p95 = pcts.get("p95", 1.0)
            scale[desc] = max(p95 - p05, 0.01)

    cusum_results = detector.cusum_test(stream, target=target, scale=scale)
    all_results.extend(cusum_results)

    welch_results = detector.welch_fdr_test(stream)
    all_results.extend(welch_results)

    if bundle and bundle.forcing_covariance:
        ref_names = bundle.forcing_covariance.get("descriptor_names", [])
        ref_mean = bundle.forcing_covariance.get("mean_vector", [])
        if ref_names and len(stream) >= 20:
            reference_vecs = [{n: m for n, m in zip(ref_names, ref_mean)}] * 30
            recent_vecs = stream[-30:]
            filtered_recent = [{k: v for k, v in e.items() if k in ref_names} for e in recent_vecs]
            if all(len(e) == len(ref_names) for e in filtered_recent):
                energy_result = detector.energy_distance_test(reference_vecs, filtered_recent)
                all_results.append(energy_result)

    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        for r in all_results:
            conn.execute(
                drift_results_table.insert().values(
                    test_id=f"{r.test_id}_{now.strftime('%Y%m%d')}",
                    test_type=r.test_type,
                    descriptor_name=r.descriptor_name,
                    drift_detected=r.drift_detected,
                    test_statistic=r.test_statistic,
                    threshold=r.threshold,
                    details_json=json.dumps(r.details) if r.details else None,
                    created_at=now,
                )
            )

    detected = [r for r in all_results if r.drift_detected]
    if detected:
        logger.warning(
            "Drift detected in %d descriptor(s): %s",
            len(detected),
            ", ".join(r.descriptor_name or "multivariate" for r in detected),
        )

    return all_results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run periodic drift tests")
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--bundle-path", default=os.environ.get("FGN_MONITORING_BUNDLE_PATH"))
    parser.add_argument("--window-size", type=int, default=50)
    args = parser.parse_args()

    config = DriftTestConfig(window_size=args.window_size)
    results = run_drift_tests(
        database_url=args.database_url,
        monitoring_bundle_path=args.bundle_path,
        config=config,
    )
    detected = [r for r in results if r.drift_detected]
    print(f"Drift tests complete: {len(results)} tests, {len(detected)} detections.")


if __name__ == "__main__":
    main()
