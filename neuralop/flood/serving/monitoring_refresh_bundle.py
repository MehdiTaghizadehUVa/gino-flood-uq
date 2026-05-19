"""Semi-automated monitoring bundle refresh CLI.

Loads the current bundle, queries SIMULATED candidates from the database,
incorporates their descriptor vectors into the reference set, and rebuilds
with updated percentiles and covariance. Output is a staging JSON file —
the operator must manually activate it. Never auto-activates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neuralop.flood.serving.monitoring import (
    MonitoringBundle,
    MonitoringPhase,
    _apply_transforms,
)


def refresh_bundle(
    *,
    current_bundle_path: str,
    database_url: str,
    output_path: str,
    min_new_events: int = 5,
) -> dict[str, object]:
    bundle = MonitoringBundle.load(current_bundle_path)

    try:
        import sqlalchemy as sa
    except ImportError as exc:
        raise RuntimeError("SQLAlchemy is required for bundle refresh.") from exc

    engine = sa.create_engine(database_url, future=True)
    metadata = sa.MetaData()

    descriptor_logs = sa.Table(
        "fgn_monitoring_descriptor_log", metadata,
        autoload_with=engine,
    )
    candidates_table = sa.Table(
        "fgn_retraining_candidates", metadata,
        autoload_with=engine,
    )

    with engine.begin() as conn:
        sim_rows = conn.execute(
            sa.select(candidates_table)
            .where(candidates_table.c.status == "SIMULATED")
        ).all()

    if len(sim_rows) < min_new_events:
        print(
            f"Only {len(sim_rows)} SIMULATED candidates found "
            f"(need {min_new_events}). Aborting.",
            file=sys.stderr,
        )
        return {}

    sim_run_ids = set()
    for row in sim_rows:
        row_map = getattr(row, "_mapping", row)
        sim_run_ids.add(row_map["run_id"])

    with engine.begin() as conn:
        log_rows = conn.execute(
            sa.select(descriptor_logs)
            .where(descriptor_logs.c.phase == MonitoringPhase.PRE_RUN.value)
            .where(descriptor_logs.c.run_id.in_(list(sim_run_ids)))
        ).all()

    new_descriptors: list[dict[str, float]] = []
    for row in log_rows:
        row_map = getattr(row, "_mapping", row)
        descs = json.loads(row_map["descriptors_json"])
        numeric = {k: float(v) for k, v in descs.items() if isinstance(v, (int, float))}
        if numeric:
            new_descriptors.append(numeric)

    existing_percentiles = bundle.descriptor_percentiles
    existing_keys = sorted(existing_percentiles.keys())

    all_keys = sorted(set(existing_keys) | {k for d in new_descriptors for k in d})
    numeric_keys = [k for k in all_keys if k in existing_keys]

    reference_synthetic: list[dict[str, float]] = []
    for key in numeric_keys:
        pcts = existing_percentiles[key]
        ref_pop = bundle.reference_population.get("n_reference_forcing", 100)
        rng = np.random.default_rng(hash(key) % (2**31))
        p01 = pcts.get("p01", pcts.get("min", 0))
        p50 = pcts.get("p50", 0)
        p99 = pcts.get("p99", pcts.get("max", 1))
        scale = max((p99 - p01) / 4.0, 0.01)
        for i in range(ref_pop):
            if len(reference_synthetic) <= i:
                reference_synthetic.append({})
            reference_synthetic[i][key] = float(rng.normal(p50, scale))

    combined = reference_synthetic + new_descriptors

    updated_percentiles: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = np.array([d.get(key, 0.0) for d in combined if key in d], dtype=np.float64)
        if len(values) == 0:
            continue
        updated_percentiles[key] = {
            "min": float(np.min(values)),
            "p01": float(np.percentile(values, 1)),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }

    covariance_data = None
    if bundle.transform_spec:
        exclude = set(bundle.covariance_exclude_descriptors)
        cov_keys = [k for k in numeric_keys if k not in exclude]
        if len(cov_keys) >= 2 and len(combined) > len(cov_keys):
            transformed_rows = []
            for d in combined:
                t = _apply_transforms({k: d.get(k, 0.0) for k in cov_keys}, bundle.transform_spec)
                transformed_rows.append([t[k] for k in cov_keys])
            X = np.array(transformed_rows, dtype=np.float64)
            mean_vec = np.mean(X, axis=0)
            cov_mat = np.cov(X, rowvar=False)
            p = len(cov_keys)
            alpha = 0.01 * np.trace(cov_mat) / p
            cov_reg = cov_mat + alpha * np.eye(p)
            inv_cov = np.linalg.inv(cov_reg)

            dists = []
            for row in X:
                diff = row - mean_vec
                d2 = float(diff @ inv_cov @ diff)
                dists.append(math.sqrt(max(0, d2)))
            dist_pcts = np.percentile(dists, [50, 75, 90, 95, 99])

            covariance_data = {
                "descriptor_names": cov_keys,
                "mean_vector": mean_vec.tolist(),
                "inv_covariance_matrix": inv_cov.tolist(),
                "n_reference": len(combined),
                "regularization_alpha": float(alpha),
                "empirical_distance_percentiles": {
                    "p50": float(dist_pcts[0]),
                    "p75": float(dist_pcts[1]),
                    "p90": float(dist_pcts[2]),
                    "p95": float(dist_pcts[3]),
                    "p99": float(dist_pcts[4]),
                },
            }

    new_bundle: dict[str, object] = {
        "bundle_id": f"{bundle.bundle_id}-refreshed-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "monitoring_bundle_schema_version": max(bundle.monitoring_bundle_schema_version, 2),
        "descriptor_schema_version": bundle.descriptor_schema_version,
        "descriptor_percentiles": updated_percentiles,
        "forecast_descriptor_percentiles": dict(bundle.forecast_descriptor_percentiles),
        "transform_spec": dict(bundle.transform_spec),
        "reference_population": {
            "n_reference_forcing": len(combined),
            "n_new_simulated": len(new_descriptors),
            "n_original_reference": len(reference_synthetic),
        },
        "covariance_exclude_descriptors": list(bundle.covariance_exclude_descriptors),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_from_commit": None,
        "provenance": {
            "refresh_source_bundle": bundle.bundle_id,
            "simulated_candidate_run_ids": sorted(sim_run_ids),
        },
        "thresholds": {
            "warning_percentile_low": bundle.warning_percentile_low,
            "warning_percentile_high": bundle.warning_percentile_high,
            "candidate_percentile_low": bundle.candidate_percentile_low,
            "candidate_percentile_high": bundle.candidate_percentile_high,
            "candidate_score_threshold": bundle.candidate_score_threshold,
            "control_sample_modulus": bundle.control_sample_modulus,
        },
    }
    if covariance_data:
        new_bundle["forcing_covariance"] = covariance_data
    if bundle.heuristic_reference_percentiles:
        new_bundle["heuristic_reference_percentiles"] = dict(bundle.heuristic_reference_percentiles)

    output = Path(output_path)
    output.write_text(json.dumps(new_bundle, indent=2, sort_keys=True))

    print(f"Staging bundle written to {output}")
    print(f"  Events incorporated: {len(new_descriptors)} new SIMULATED candidates")
    print(f"  Total reference population: {len(combined)}")
    print(f"  Descriptors tracked: {len(numeric_keys)}")
    if covariance_data:
        print(f"  Covariance computed over {len(covariance_data['descriptor_names'])} descriptors")
    print()
    print("To activate: update FGN_MONITORING_BUNDLE_PATH and restart containers.")

    return new_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh monitoring bundle with SIMULATED candidates",
    )
    parser.add_argument("--current-bundle", required=True, help="Path to current monitoring bundle JSON")
    parser.add_argument("--database-url", required=True, help="Database connection URL")
    parser.add_argument("--output", required=True, help="Path for staging bundle output")
    parser.add_argument("--min-new-events", type=int, default=5, help="Minimum SIMULATED candidates required")
    args = parser.parse_args()

    refresh_bundle(
        current_bundle_path=args.current_bundle,
        database_url=args.database_url,
        output_path=args.output,
        min_new_events=args.min_new_events,
    )


if __name__ == "__main__":
    main()
