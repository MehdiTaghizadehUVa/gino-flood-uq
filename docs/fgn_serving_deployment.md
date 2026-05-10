# Coastal FGN UQ Web Deployment

This document records the implementation contract for serving the coastal FGN
benchmark as a gated research web application.

## Scientific Contract

V1 serves one fixed coastal domain only. The deployment bundle pins the three
coastal FGN checkpoints, train-fit normalizers, static/geometric domain assets,
boundary-channel contract, and calibration artifacts. Calibrated outputs are the
default. Raw FGN outputs are retained as diagnostics.

The service is research-only and is not for emergency or operational flood
forecasting decisions.

## Deep Modules

The serving layer is intentionally split into deep modules:

- `ModelBundle`: versioned scientific deployment contract and startup validation.
- `ForcingInput`: CSV parsing, timestep validation, range checks, and input hash.
- `RunSpec`: immutable run contract used by queue, worker, artifacts, and reports.
- `FGNInferenceService`: fixed-domain FGN rollout seam.
- `CalibrationAdapter`: CRPS-MBM (member-by-member) calibration of WD members and isotonic calibration of per-cell exceedance probability maps.
- `ForecastProductBuilder`: no-ground-truth user products.
- `ArtifactStore`: artifact registry/download seam.
- `RunRepository`: run metadata/state-machine seam.
- `JobQueue`: queue/concurrency seam.
- `AccessPolicy`: owner/admin/allowlist/disclaimer decisions.
- `RunOrchestrator`: only module allowed to transition run states.

FastAPI and Celery are adapters around these modules. They should not contain
checkpoint, tensor, calibration, or HDF5 logic.

## Current Implementation Status

The implementation now includes the serving contracts, production fixed-domain
FGN rollout adapter, trusted-header auth, SQL/Celery/local adapters, admin run
controls, retention cleanup, bundle building, real-bundle validation, and
server-side forecast map artifact generation. Tests cover the public seams with
fake and toy production inference paths.

## Production Rollout Slice

`ProductionFGNInferenceService` now implements the fixed-domain no-reference FGN
rollout behind the serving seam. It:

- loads the three checkpointed models from bundle metadata or `model_config_path`,
- loads train-fit normalizers once per worker process,
- loads fixed geometry/static tensors and optional structural-dry mask from the bundle,
- validates the serving domain tensor shapes,
- builds normalized query points from normalized UTM geometry,
- validates uploaded stage/precipitation forcing through `ForcingInput`,
- starts from dry/baseline WD history with `n_history=3`,
- generates exactly `members_per_checkpoint` persistent latent samples per checkpoint,
- uses member-feedback autoregressive state updates,
- inverse-transforms WD to meters, zeroes structural-dry cells, and clips negative WD to zero,
- returns a physical-space `[60, time, cells]` `ForecastResult` for product and calibration modules.

The production bundle must provide `geometry_path` and `static_tensor_path` as
`.npy`, `.pt`, or `.pth` tensors unless tests inject `DomainAssets` directly.
`structural_dry_mask_path` is optional but recommended for coastal deployment.

## Auth And Persistence Adapters

The API supports a trusted-header auth adapter intended to sit behind Google
OAuth at the reverse proxy layer. The reverse proxy must provide an authenticated
email header such as `X-Auth-Request-Email`; the FastAPI layer still enforces the
allowlist, admin role, run ownership, and disclaimer acknowledgement.

A SQL repository adapter is available through `SqlRunRepository` for PostgreSQL
or SQLite SQLAlchemy URLs. Large arrays remain in `ArtifactStore`; only run
state, immutable specs, and audit metadata belong in SQL.

## Real Coastal Bundle

The current production candidate bundle was built from the successful 3x20 FGN
evaluation and production calibration artifacts:

- bundle: `/scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json`
- checkpoints: `/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_fgn_eval60/coast_fgn3x20_eval_currentviz_20260506_152059/checkpoints_3x20`
- calibration: `/scratch/jrj6wm/GINO_Model/neuraloperator_runs/calibration_production/coast_fgn3x20_calib_m100_prod_20260508_015758/outputs/calibration`
- mesh hash: `b65401c2214b4da9063036d703a160a68de1488b85b7ebd93624d118dbe8aa95`
- cells/static/horizon: `5904` cells, `7` static channels, `94` forecast steps.

Build command pattern:

```bash
python -m neuralop.flood.serving.bundle_builder \
  --config-path <coast_fgn3x20_eval_currentviz.yaml> \
  --output-dir <bundle-output-dir> \
  --bundle-id coastal-fgn-3x20-calibrated-v1-YYYYMMDD \
  --git-commit $(git rev-parse HEAD) \
  --calibration-coefficients-path <crps_mbm_coefficients.json> \
  --isotonic-curves-path <exceedance_isotonic_curves.json>
```

Validate command:

```bash
python -m neuralop.flood.serving.cli <bundle-output-dir>/coastal_fgn_bundle.json
```

## Operational Wiring

Docker Compose provides API, GPU worker, frontend, OAuth proxy, Redis,
Postgres, reverse proxy, and daily cleanup services. The cleanup service expires
unpinned terminal run artifacts after `FGN_RETENTION_DAYS` while preserving SQL
audit metadata. On the lab PC, Compose must run inside WSL2/Docker Desktop with
GPU support enabled.

The API exposes owner-visible run/artifact endpoints plus admin list, pin,
unpin, cancel, and runtime allowlist-management endpoints. Only `RunOrchestrator`
transitions run state. Allowlist, admin, and disclaimer state are persisted in
Postgres when `DATABASE_URL` is configured.

## Lab PC Deployment Defaults

The lab PC deployment is Windows 11 + RTX 4090 + WSL2 + Docker Desktop. Docker
Compose expects `FGN_DATA_ROOT` to point at a large disk mounted inside WSL, for
example `/mnt/d/FGNServing`. The directory must contain:

- `model_bundle/` mounted read-only at `/model_bundle`
- `artifacts/` mounted read-write at `/artifacts`
- `postgres/` mounted as the Postgres data directory
- `backups/` used for daily compressed Postgres backups

Run the setup and validation helpers from Windows PowerShell:

```powershell
deployment/fgn-serving/setup_lab_pc.ps1
deployment/fgn-serving/validate_lab_pc.ps1
```

For VPN/LAN HTTPS, Caddy uses `tls internal`. Collaborators must trust the Caddy
root CA, and Google OAuth redirect URLs must point at
`https://${FGN_SITE_HOSTNAME}/oauth2/callback`.

Before starting production Compose:

```bash
cd deployment/fgn-serving
cp .env.example .env
# edit .env: FGN_SITE_HOSTNAME, FGN_DATA_ROOT, Google OAuth secrets,
# POSTGRES_PASSWORD, OAUTH2_PROXY_COOKIE_SECRET, FGN_ALLOWED_EMAILS
docker compose build
docker compose up -d
```

Health checks:

```bash
curl -k https://${FGN_SITE_HOSTNAME}/api/health
curl -k https://${FGN_SITE_HOSTNAME}/api/model-bundle-health
```

## Remaining Production Validation

The code and real bundle validation pass. The remaining environment-dependent
validation is the full GPU serving smoke on the lab/Rivanna GPU worker:

- run `python -m neuralop.flood.serving.smoke` against the real bundle,
- verify all `60` members, calibrated summary JSON, and map PNG products are generated,
- then configure the lab server `.env` with the validated bundle mount and run `docker compose up`.

The frontend is a Next.js research UI that supports the full research workflow:
CSV template download with client-side validation, allowlisted submission with
optional GIF animation and HDF5 artifacts, run-queue polling, and a per-run
result dashboard. The dashboard renders peak/spread/arrival summary cards, a
calibrated-vs-raw toggle, server-rendered map products with a time slider, an
inline hydrograph (stage and precipitation) parsed from the uploaded forcing
CSV, exceedance-probability tables, and an artifact download list.
