# 0004. Phase 2 Drift Monitoring — Multivariate Scoring, Population Drift, and Error Tracking

## Status

Accepted

## Context

Phase 1 drift screening (ADR 0003) treats each descriptor independently and uses hard-coded post-run heuristic thresholds. It cannot detect jointly unusual forcings where each marginal is individually in-range, gradual distributional shifts across the submission population, or systematic FGN prediction errors revealed by HEC-RAS ground truth. These gaps reduce the retraining signal's quality and leave admins without population-level visibility.

## Decision

Extend the monitoring system additively with six capabilities:

1. **Regularized Mahalanobis distance** with empirical reference distance percentiles (not chi-square critical values) and descriptor transforms for skewed hydrologic data. Covariance is precomputed in the Monitoring Bundle. Absent in old bundles = no-op.
2. **Reference-derived post-run thresholds** stored as `heuristic_reference_percentiles`, replacing hard-coded magic numbers. These thresholds include area-weighted uncertainty width, uncertainty-to-signal ratio, calibration shift, and checkpoint-disagreement metrics. Backwards-compatible fallback to Phase 1 thresholds remains available for diagnostics; ADR 0005 makes Phase 2 reference-derived thresholds, not legacy fallbacks, eligible for new candidate decisions.
3. **Population-level drift detection** via stateless CUSUM (recomputed from Descriptor Log), Welch t-tests with Benjamini-Hochberg FDR control, and energy distance permutation tests. Persistence filter requires 2 consecutive daily detections before admin-facing warning. Drift signals are admin-level context, not per-run candidate logic.
4. **FGN-vs-HEC-RAS error tracking** on SIMULATED candidate transition with path-safe HDF ingestion validated against `FGN_HECRAS_RESULT_ROOT`.
5. **Admin monitoring trends API** with score distributions, descriptor trends, drift test results, and HEC-RAS error summaries.
6. **Semi-automated bundle refresh CLI** producing a staging file; operator must manually activate. Never auto-activates.

## Consequences

- Catches jointly unusual forcings that univariate screening misses.
- Eliminates deployment-specific magic numbers via reference-derived thresholds.
- Provides early drift warning with false-positive control (FDR + persistence).
- Closes the HEC-RAS feedback loop with path safety.
- Three new SQL tables (`fgn_monitoring_descriptor_log`, `fgn_drift_test_results`, `fgn_hecras_error_records`); existing tables unmodified.
- Monitoring Bundle format backwards-compatible with explicit schema versioning.
- De-prioritizes fragile max-depth descriptors from primary drift triggers.
- Adds checkpoint-disagreement descriptors so high structural ensemble disagreement can be tracked separately from latent/member spread.
- All Phase 2 capabilities remain research guardrails, not operational safety claims.

## Alternatives Considered

- **Isolation forests / autoencoders**: Higher detection power but opaque scoring, conflicts with transparency requirement.
- **Chi-square-only p-values for Mahalanobis**: Normality assumption invalid for skewed hydrologic descriptors; empirical percentiles are more robust.
- **Per-request drift testing**: Adds latency and statistical noise; periodic batch is more statistically sound.
- **CUSUM state persistence**: Harder to audit than recompute-from-log; chosen to prioritize reproducibility.
- **Automatic bundle activation**: Rejected for safety; monitoring bundle changes must be operator-reviewed.
