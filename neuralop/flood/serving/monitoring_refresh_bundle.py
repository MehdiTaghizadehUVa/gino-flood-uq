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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neuralop.flood.serving.forcing import parse_forcing_csv
from neuralop.flood.serving.model_bundle import load_model_bundle
from neuralop.flood.serving.monitoring import (
    MonitoringBundle,
    MonitoringPhase,
    _compute_covariance_block,
    build_forcing_descriptors,
)
from neuralop.flood.serving.monitoring_build_bundle import (
    _forcing_paths_from_globs,
    load_clean_boundary_table_forcings,
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

    reference_rows = _load_reference_descriptors_from_provenance(bundle)
    if not reference_rows:
        reference_rows = _synthetic_reference_from_percentiles(bundle)

    existing_keys = sorted(bundle.descriptor_percentiles.keys())
    all_keys = sorted(set(existing_keys) | {k for d in new_descriptors for k in d} | {k for d in reference_rows for k in d})
    numeric_keys = [k for k in all_keys if k in existing_keys or any(k in row for row in reference_rows)]

    combined = reference_rows + new_descriptors

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

    covariance_data = _compute_covariance_block(
        rows=[{k: d.get(k) for k in numeric_keys} for d in combined],
        keys=numeric_keys,
        transform_spec=dict(bundle.transform_spec),
        exclude=list(bundle.covariance_exclude_descriptors),
    )

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
            "n_original_reference": len(reference_rows),
        },
        "covariance_exclude_descriptors": list(bundle.covariance_exclude_descriptors),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_from_commit": _current_git_commit(),
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


def _load_reference_descriptors_from_provenance(bundle: MonitoringBundle) -> list[dict[str, float]]:
    provenance = bundle.provenance
    model_bundle_path = provenance.get("model_bundle_path")
    if not isinstance(model_bundle_path, str) or not model_bundle_path:
        return []
    try:
        model_bundle = load_model_bundle(model_bundle_path)
    except Exception:
        return []

    descriptors: list[dict[str, float]] = []
    forcing_globs = provenance.get("forcing_globs", [])
    if isinstance(forcing_globs, list):
        for path in _forcing_paths_from_globs([str(item) for item in forcing_globs]):
            try:
                forcing = parse_forcing_csv(path, bundle=model_bundle)
            except Exception:
                continue
            descriptors.append(_numeric_descriptor_row(build_forcing_descriptors(forcing)))

    clean_pairs = provenance.get("clean_boundary_pairs", [])
    if isinstance(clean_pairs, list):
        for pair in clean_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                forcings = load_clean_boundary_table_forcings(
                    stage_table=Path(str(pair[0])),
                    precipitation_table=Path(str(pair[1])),
                    dt_seconds=int(model_bundle.dt_seconds),
                    skip_before_timestep=int(model_bundle.skip_before_timestep),
                    n_history=int(model_bundle.n_history),
                    max_forecast_steps=int(model_bundle.max_forecast_steps),
                )
            except Exception:
                continue
            descriptors.extend(_numeric_descriptor_row(build_forcing_descriptors(forcing)) for forcing in forcings)
    return descriptors


def _synthetic_reference_from_percentiles(bundle: MonitoringBundle) -> list[dict[str, float]]:
    existing_percentiles = bundle.descriptor_percentiles
    existing_keys = sorted(existing_percentiles.keys())
    ref_pop = int(bundle.reference_population.get("n_reference_forcing", 100) or 100)
    reference_synthetic: list[dict[str, float]] = []
    for key in existing_keys:
        pcts = existing_percentiles[key]
        rng = np.random.default_rng(abs(hash(key)) % (2**31))
        p01 = pcts.get("p01", pcts.get("min", 0))
        p50 = pcts.get("p50", 0)
        p99 = pcts.get("p99", pcts.get("max", 1))
        scale = max((p99 - p01) / 4.0, 0.01)
        for i in range(ref_pop):
            if len(reference_synthetic) <= i:
                reference_synthetic.append({})
            reference_synthetic[i][key] = float(rng.normal(p50, scale))
    return reference_synthetic


def _numeric_descriptor_row(row: dict[str, object]) -> dict[str, float]:
    return {key: float(value) for key, value in row.items() if isinstance(value, (int, float)) and math.isfinite(float(value))}


def _current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    sha = completed.stdout.strip()
    return sha or None


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
