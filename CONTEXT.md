# Context

## Coastal FGN Serving

Coastal FGN Serving is the fixed-domain research web deployment for the trained coastal flood FGN benchmark. It accepts stage and precipitation forcings, runs a 60-member uncertainty forecast, applies calibration, and returns forecast-only products.

## Model Bundle

A Model Bundle is the versioned scientific contract for one deployed coastal FGN domain. It pins checkpoints, normalizers, static tensors, geometry, calibration artifacts, rollout constants, mesh hash, and public model metadata.

## Forcing CSV

A Forcing CSV is the user-uploaded stage and precipitation time series. V1 requires a regular 20-minute cadence, finite values, enough spin-up/history rows, and forecast horizon within the Model Bundle limit.

## Run

A Run is one submitted coastal flood scenario. It has a RunSpec, uploaded Forcing CSV, owner, status, artifacts, model bundle ID, input hash, calibration version, and reproducibility manifest.

## Calibration

Calibration means applying CRPS member-by-member water-depth calibration and isotonic exceedance probability calibration. Calibrated output is the default; raw FGN output remains available for diagnostics.

## Artifact

An Artifact is a downloadable run output such as uploaded CSV, summary JSON, map PNG, GIF animation, optional HDF5 ensemble, or reproducibility manifest. Large forecast arrays live in Artifact storage, not Postgres.

## Lab PC Server

The Lab PC Server is the Windows 11 RTX 4090 machine that hosts V1 through WSL2 and Docker Compose. It is intended for VPN/LAN research access, Google OAuth, internal TLS, and local data storage rooted at `FGN_DATA_ROOT`.
