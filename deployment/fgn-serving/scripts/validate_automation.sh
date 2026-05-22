#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${DEPLOY_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

bash -n deployment/fgn-serving/scripts/*.sh
deployment/fgn-serving/scripts/check_no_serving_secrets.sh

export FGN_SITE_HOSTNAME="${FGN_SITE_HOSTNAME:-fgn-lab.example.edu}"
export FGN_DATA_ROOT="${FGN_DATA_ROOT:-/tmp/fgn-serving-data}"
export FGN_POSTGRES_DATA_ROOT="${FGN_POSTGRES_DATA_ROOT:-/tmp/fgn-serving-postgres}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-test-postgres-password}"
export DATABASE_URL="${DATABASE_URL:-postgresql+psycopg://fgn_serving:test-postgres-password@postgres:5432/fgn_serving}"
export GOOGLE_CLIENT_ID="${GOOGLE_CLIENT_ID:-test-client-id}"
export GOOGLE_CLIENT_SECRET="${GOOGLE_CLIENT_SECRET:-test-client-secret}"
export OAUTH2_PROXY_COOKIE_SECRET="${OAUTH2_PROXY_COOKIE_SECRET:-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=}"
export FGN_ALLOWED_EMAILS="${FGN_ALLOWED_EMAILS:-user@example.edu}"
export FGN_ADMIN_EMAILS="${FGN_ADMIN_EMAILS:-admin@example.edu}"
export FGN_API_IMAGE="${FGN_API_IMAGE:-ghcr.io/mehditaghizadehuva/gino-flood-uq/fgn-serving-python:test}"
export FGN_WORKER_IMAGE="${FGN_WORKER_IMAGE:-ghcr.io/mehditaghizadehuva/gino-flood-uq/fgn-serving-python:test}"
export FGN_CLEANUP_IMAGE="${FGN_CLEANUP_IMAGE:-ghcr.io/mehditaghizadehuva/gino-flood-uq/fgn-serving-python:test}"
export FGN_FRONTEND_IMAGE="${FGN_FRONTEND_IMAGE:-ghcr.io/mehditaghizadehuva/gino-flood-uq/fgn-serving-frontend:test}"

docker compose -f deployment/fgn-serving/docker-compose.yml config >/tmp/fgn-serving-compose.yml
docker compose -f deployment/fgn-serving/docker-compose.yml -f deployment/fgn-serving/docker-compose.local-build.yml config >/tmp/fgn-serving-compose-local-build.yml

deployment/fgn-serving/scripts/deploy_lab.sh --dry-run
deployment/fgn-serving/scripts/smoke_lab.sh --dry-run

printf 'FGN serving automation validation passed.\n'
