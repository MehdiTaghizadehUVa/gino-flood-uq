# Context

## Coastal FGN Serving

Coastal FGN Serving is the fixed-domain research web deployment for the trained coastal flood FGN benchmark. It accepts stage and precipitation forcings, runs a 60-member uncertainty forecast, applies calibration, and returns forecast-only products.

## Model Bundle

A Model Bundle is the versioned scientific contract for one deployed coastal FGN domain. It pins checkpoints, normalizers, static tensors, geometry, calibration artifacts, rollout constants, mesh hash, and public model metadata.

## Forcing CSV

A Forcing CSV is the user-uploaded stage and precipitation time series. V1 requires a regular 15-minute cadence, finite values, enough spin-up/history rows, and forecast horizon within the Model Bundle limit.

## Run

A Run is one submitted coastal flood scenario. It has a RunSpec, uploaded Forcing CSV, owner, status, artifacts, model bundle ID, input hash, calibration version, and reproducibility manifest.

## Calibration

Calibration means applying CRPS member-by-member water-depth calibration and isotonic exceedance probability calibration. Calibrated output is the default; raw FGN output remains available for diagnostics.

## Artifact

An Artifact is a downloadable run output such as uploaded CSV, summary JSON, map PNG, GIF animation, optional HDF5 ensemble, or reproducibility manifest. Large forecast arrays live in Artifact storage, not Postgres.

## Result Cache

The Result Cache is a shared server-side store for scientifically identical completed runs. It is keyed by canonical parsed forcing values, model/calibration/initial-condition identity, seed and ensemble settings, forecast horizon, output options, and product schema version. Cache hits create a new private Run for the submitting user and materialize scientific artifacts without exposing the source Run owner.

## Lab PC Server

The Lab PC Server is the Windows 11 RTX 4090 machine that hosts V1 through WSL2 and Docker Compose. It is intended for VPN/LAN research access, Google OAuth, internal TLS, and local data storage rooted at `FGN_DATA_ROOT`.

## Monitoring Bundle

A Monitoring Bundle is the versioned reference distribution used by drift screening. It stores descriptor percentiles, covariance blocks, descriptor transforms, provenance for the reference data, and transparent candidate-scoring thresholds.

## Monitoring Report

A Monitoring Report is the per-run screening record produced before GPU inference or after forecast post-processing. It contains descriptors, scores, flags, and retraining-candidate recommendations. It is a research guardrail, not proof of model error.

## Retraining Candidate

A Retraining Candidate is a preserved event selected for later HEC-RAS labeling or dataset review. It links a Run, input hash, Monitoring Reports, model bundle ID, monitoring bundle ID, candidate score, status, and a minimal preserved artifact package.

## Candidate Event Store

The Candidate Event Store is the durable storage area for Retraining Candidate packages. It is separate from normal 30-day run artifact retention so candidate evidence remains available for later HEC-RAS and retraining workflows.

## Reference Event

A Reference Event is a train or test scenario with known forcing descriptors and, when available, HEC-RAS output descriptors. Reference Events define the comparison envelope for drift screening.

## Initial Condition Library

An Initial Condition Library is a versioned collection of reference forcing histories and HEC-RAS water-depth history frames aligned to the deployed coastal mesh. Serving uses it to choose reproducible baseline dynamic-state histories without asking users to upload a full water-depth field.

## Forcing-Conditioned Baseline

A Forcing-Conditioned Baseline is the default initial water-depth history policy for web serving. It compares the first spin-up/history forcing rows against train/calibration reference events, blends the nearest eligible HEC-RAS WD histories, records selection provenance, and keeps dry start available only as a diagnostic mode.

## Descriptor Transform

A Descriptor Transform is a monotone function (log1p, logit_bounded, identity) applied to a descriptor value before covariance computation. Transforms handle skewed hydrologic distributions (precipitation totals, area fractions) so that Mahalanobis distance is meaningful. Raw descriptor values are always stored in reports for interpretability; transforms are internal to the scoring pipeline.

## Descriptor Log

A Descriptor Log is the time-ordered series of per-run descriptor vectors persisted after screening and post-run evaluation. It provides the population-level data that drift detection consumes. Entries use deterministic IDs for idempotency under worker retries.

## Drift Test

A Drift Test is a periodic statistical check for distributional shift across recent Descriptor Log entries. Phase 2 supports CUSUM, Welch t-test (with Benjamini-Hochberg FDR), and multivariate energy distance. Drift signals are admin-level context, not per-run candidate logic.

## HEC-RAS Error Record

A HEC-RAS Error Record captures the signed and relative prediction errors between FGN forecast descriptors and HEC-RAS simulation descriptors for a SIMULATED Retraining Candidate. Accumulated error records enable systematic model degradation detection.

## Checkpoint Disagreement

Checkpoint Disagreement is the between-checkpoint share of ensemble variance in the 3-checkpoint × latent-member serving ensemble. It separates disagreement among trained checkpoints from within-checkpoint latent variability and helps identify forecasts where structural uncertainty is unusually high.
