# FGN Lab Deployment Automation

This document describes the automated deployment workflow for the Coastal FGN
UQ research server on the lab PC.

## Release Flow

The live deployment branch is `main`.

1. A pull request updates serving code, frontend code, deployment scripts, or
   model-bundle contracts.
2. `FGN Serving CI` runs targeted serving tests, frontend build, Compose
   validation, and secret checks.
3. After `main` passes CI, `FGN Serving Images` builds immutable GHCR images:
   - `ghcr.io/mehditaghizadehuva/gino-flood-uq/fgn-serving-python:<commit-sha>`
   - `ghcr.io/mehditaghizadehuva/gino-flood-uq/fgn-serving-frontend:<commit-sha>`
4. `FGN Serving Deploy Lab` runs on the lab-PC self-hosted runner and deploys
   the exact SHA-tagged images.
5. Post-deploy smoke checks must pass. If they fail, image-level rollback is
   attempted automatically.

The moving `:main` image tags are published for convenience, but automated
deployments use commit-SHA tags.

## Lab PC Runner Setup

Install the GitHub Actions runner inside WSL Ubuntu on the lab PC, not inside a
project container. In GitHub, open:

```text
Repository settings -> Actions -> Runners -> New self-hosted runner
```

Choose Linux x64 and follow GitHub's installation commands. Configure these
labels when registering the runner:

```text
self-hosted
fgn-lab
linux
gpu
```

The runner user must be able to run:

```bash
docker info
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The runner also needs GHCR pull access. The deployment workflow logs into GHCR
with `GITHUB_TOKEN`; no production OAuth or database secrets are stored in
GitHub Actions.

## Lab Secrets And Storage

Production secrets stay only in:

```bash
deployment/fgn-serving/.env
```

Do not commit `.env`. The deployment expects:

- `FGN_DATA_ROOT`
- `FGN_POSTGRES_DATA_ROOT`
- Google OAuth values
- Postgres password
- allowlist/admin emails
- model and monitoring bundle paths

Runtime state stays outside Git under `FGN_DATA_ROOT`:

```text
model_bundle/
artifacts/
backups/
deployments/
```

## Manual Commands

Validate the deployment automation without changing running services:

```bash
deployment/fgn-serving/scripts/validate_automation.sh
```

Run smoke checks against the current lab deployment:

```bash
deployment/fgn-serving/scripts/smoke_lab.sh
```

Deploy the images currently named by environment variables:

```bash
export FGN_API_IMAGE=ghcr.io/mehditaghizadehuva/gino-flood-uq/fgn-serving-python:<sha>
export FGN_WORKER_IMAGE="$FGN_API_IMAGE"
export FGN_CLEANUP_IMAGE="$FGN_API_IMAGE"
export FGN_FRONTEND_IMAGE=ghcr.io/mehditaghizadehuva/gino-flood-uq/fgn-serving-frontend:<sha>
export FGN_DEPLOY_COMMIT=<sha>
deployment/fgn-serving/scripts/deploy_lab.sh
```

Rollback to the previous deployment record:

```bash
deployment/fgn-serving/scripts/rollback_lab.sh
```

## Local Development Builds

Production Compose pulls images. For local debugging builds, use the override:

```bash
cd deployment/fgn-serving
docker compose -f docker-compose.yml -f docker-compose.local-build.yml build
docker compose -f docker-compose.yml -f docker-compose.local-build.yml up -d
```

## Deployment Records

Deployments write records under:

```text
${FGN_DATA_ROOT}/deployments/
```

Important files:

- `current.json`: current healthy deployment.
- `previous.json`: rollback target.
- `deployment_<timestamp>.json`: historical successful deployment records.
- `rollback_<timestamp>.json`: rollback attempts.

Each record includes the commit SHA, image tags, timestamp, status, and message.

## Failure Handling

`deploy_lab.sh` performs a pre-deploy Postgres backup when Postgres is already
running. If post-deploy smoke checks fail, it marks the pending deployment as
failed and calls `rollback_lab.sh` when a previous deployment record exists.

Rollback is image-level only. It does not roll back Postgres data, model bundle
files, uploaded CSVs, generated run artifacts, or monitoring candidate packages.
