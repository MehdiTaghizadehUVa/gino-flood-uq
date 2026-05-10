# FGN Coastal UQ Web Deployment PRD

## Problem Statement

Researchers need a safe, reproducible way to run the trained coastal FGN model for flood inference and uncertainty quantification without manually using Rivanna scripts, checkpoints, normalizers, calibration artifacts, and visualization pipelines.

## Solution

Deploy the fixed coastal FGN benchmark as a gated research web application on the lab PC. Users authenticate with Google OAuth, acknowledge the research-only disclaimer, upload stage and precipitation forcings, submit asynchronous 60-member FGN UQ runs, and download calibrated forecast products.

## User Stories

1. As a gated research user, I want Google OAuth access, so that only approved collaborators can run GPU jobs.
2. As a researcher, I want a valid CSV template, so that I can prepare forcings without reading code.
3. As a researcher, I want server-side CSV validation, so that invalid events do not enter the GPU queue.
4. As a researcher, I want queued asynchronous runs, so that 60-member inference does not time out in the browser.
5. As a researcher, I want run status and progress, so that I know whether a run is waiting, running, completed, failed, canceled, or expired.
6. As a researcher, I want calibrated maps and uncertainty summaries, so that I can inspect expected flood response and spread.
7. As a researcher, I want raw-vs-calibrated diagnostics, so that calibration effects are transparent.
8. As a researcher, I want downloadable artifacts, so that I can reuse results.
9. As an admin, I want global and per-user run limits, so that the RTX 4090 is not oversubscribed.
10. As an admin, I want retention, pinning, and cancellation, so that local storage remains bounded.
11. As a reviewer, I want model bundle ID, input hash, git commit, calibration version, and seed policy recorded, so that outputs are auditable.

## Implementation Decisions

- V1 serves only the fixed coastal FGN domain and starts from dry/baseline WD history.
- The Model Bundle is the scientific contract for checkpoints, normalizers, geometry/static tensors, mesh hash, rollout constants, and calibration files.
- FastAPI, Celery, SQL, OAuth, and React remain adapters around serving modules; inference modules do not import web infrastructure.
- Google OAuth runs through oauth2-proxy; FastAPI enforces persisted allowlist, admin, owner, and disclaimer policy.
- Lab PC deployment uses WSL2 + Docker Compose + Caddy internal TLS + `FGN_DATA_ROOT`.
- Full HDF5 export is optional and defaults off.

## Testing Decisions

- Tests target public seams: ModelBundle, ForcingInput, RunSpec, RunOrchestrator, API routes, ArtifactStore, CalibrationAdapter, ProductBuilder, and FGNInferenceService.
- GPU-heavy behavior is covered by tiny injected production models in unit tests plus a real-bundle SLURM smoke test.
- Frontend validation is covered by a Next production build and should be extended with browser tests once WSL2/Docker is available on the lab PC.

## Out of Scope

Dynamic/WV deployment, diffusion deployment, MC-dropout deployment, training, calibration fitting, arbitrary geometry upload, hot-start WD upload, and operational/emergency decision support.
