# FGN Serving Vertical Slices

These are the GitHub issue briefs for the completed V1 serving plan. Published parent PRD: #3.

## 1. Serving Contracts And Bundle Validation (#4)

Build the fixed-domain Model Bundle, Forcing CSV, RunSpec, calibration, and product contracts.

Acceptance criteria:
- Model bundle validation rejects missing calibration/checkpoints/static assets.
- CSV template validates directly.
- RunSpec records bundle ID, input hash, thresholds, output detail, HDF5/GIF options, and seed.

## 2. Backend API, Auth, And Run Lifecycle (#5)

Expose authenticated FastAPI routes for model metadata, template/validation, run submission, polling, artifacts, cancellation, and admin controls.

Acceptance criteria:
- Unauthenticated users cannot submit or read runs.
- Disclaimer acknowledgement persists.
- Owners/admins can access runs; other users cannot.
- Only RunOrchestrator owns state transitions.

## 3. Queue, Persistence, Retention, And Artifacts (#6)

Wire Postgres-backed run/user state, Celery/Redis queueing, local artifact storage, retention cleanup, pinning, and backups.

Acceptance criteria:
- One global active GPU run and per-user queue limits are enforced.
- Artifacts are not stored in Postgres.
- Expired unpinned run artifacts are deleted while audit metadata remains.

## 4. Production FGN Inference And Calibrated Products (#7)

Run 3 checkpoint x 20 persistent-latent members through the production FGN serving adapter and produce raw/calibrated outputs.

Acceptance criteria:
- Tiny injected production tests produce deterministic 60-member output.
- Member chunking is real and configurable.
- CRPS-MBM and isotonic calibration feed summaries, PNG maps, GIF, and optional HDF5.

## 5. Research Console Frontend (#8)

Provide the lab research UI for authenticated users and admins.

Acceptance criteria:
- Users download a template, validate/upload CSV, acknowledge disclaimer, submit a run, poll status, inspect results, and download artifacts.
- Admins manage allowlist, pin/unpin, cancel, and inspect runs.
- UI states are work-focused and avoid operational/emergency claims.

## 6. Lab PC Deployment And Operations (#9)

Package and document the Windows lab PC server deployment.

Acceptance criteria:
- WSL2 + Docker Desktop + GPU validation is documented and scripted.
- Docker Compose uses `FGN_DATA_ROOT`, GPU PyTorch image, Google OAuth, Caddy internal TLS, Redis, Postgres, API, worker, frontend, cleanup, and backups.
- Real coastal bundle validates and GPU smoke is run through SLURM or lab PC Docker.
