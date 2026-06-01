# ADR 0007: Result Cache For Repeated Runs

## Status

Accepted

## Context

Full 60-member coastal FGN runs are expensive enough that users may submit the
same scientific event more than once, sometimes across accounts or with
formatting-only CSV differences. Re-running inference for those duplicates
consumes GPU time without adding scientific value.

At the same time, runs are user-owned audit records. A repeated submission must
not expose another user's run ID, label, uploaded filename, or ownership
context.

## Decision

Add an exact-match server-side Result Cache behind the serving orchestration
seam. The cache key uses canonical parsed stage and precipitation values,
timestep, forecast steps, model bundle ID, calibration identity, initial
condition library identity, seed, ensemble budget, output options, and product
schema version.

The cache key deliberately excludes user ID, label, upload filename, raw CSV
formatting, monitoring bundle ID, and GPU chunk size.

On a completed cache hit, the system creates a new private Run for the
submitting user, writes that user's uploaded `forcing.csv` and run manifest,
materializes cached scientific artifacts, records `cache_manifest.json`, and
transitions the Run to `COMPLETED`.

If a matching source run is still active, duplicate submissions transition to
`WAITING_FOR_CACHE` and are materialized when the producer completes. If the
producer fails or is canceled, one waiter is promoted to a normal queued GPU run.

Cache artifacts live under `FGN_CACHE_ROOT`, defaulting to
`FGN_ARTIFACT_ROOT/_result_cache`. User run deletion and normal retention remove
only that user's Run artifacts; they do not purge shared cache packages.

Monitoring remains per-Run. Pre-run screening is performed for every submitter,
and post-run monitoring is attached to each materialized Run. Candidate capture
is deduplicated by scientific input/model/monitoring identity to avoid noisy
duplicate retraining candidates.

## Consequences

Repeated submissions can complete quickly without GPU rollout while preserving
private run ownership and auditability.

Cache invalidation is controlled by product schema version and scientific
identity fields. Changes to model bundle, calibration, initial condition
library, forecast horizon, thresholds, HDF5/GIF options, or ensemble budget
produce a different cache key.

The cache is exact-match only. It does not perform similarity search or reuse
nearby events.
