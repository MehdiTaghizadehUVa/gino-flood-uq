#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=lib_deploy.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_deploy.sh"

DRY_RUN=0
SKIP_PULL=0
SKIP_BACKUP=0
NO_ROLLBACK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-pull)
      SKIP_PULL=1
      shift
      ;;
    --skip-backup)
      SKIP_BACKUP=1
      shift
      ;;
    --no-rollback)
      NO_ROLLBACK=1
      shift
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

load_lab_env
require_var FGN_DATA_ROOT
require_var FGN_POSTGRES_DATA_ROOT
require_var POSTGRES_PASSWORD
require_var FGN_SITE_HOSTNAME
require_var FGN_API_IMAGE
require_var FGN_WORKER_IMAGE
require_var FGN_CLEANUP_IMAGE
require_var FGN_FRONTEND_IMAGE

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: validated env and would deploy ${FGN_DEPLOY_COMMIT:-unknown commit}."
  exit 0
fi

root="$(deployment_root)"
mkdir -p "${root}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
pending="${root}/pending_${stamp}.json"
current="${root}/current.json"
previous="${root}/previous.json"

write_json_record "${pending}" "pending" "Deployment started."
if [[ -f "${current}" ]]; then
  cp "${current}" "${previous}"
fi

command -v docker >/dev/null || die "docker is not available."
docker info >/dev/null
if command -v nvidia-smi >/dev/null; then
  nvidia-smi >/dev/null || die "nvidia-smi failed."
else
  log "nvidia-smi is not on PATH; CUDA will still be checked inside worker smoke."
fi

if [[ "${SKIP_BACKUP}" != "1" ]]; then
  backup_dir="${FGN_DATA_ROOT}/backups/postgres"
  mkdir -p "${backup_dir}"
  backup_file="${backup_dir}/predeploy_${stamp}.sql.gz"
  if [[ -n "$(compose ps --status running -q postgres 2>/dev/null || true)" ]]; then
    log "writing pre-deploy Postgres backup: ${backup_file}"
    compose exec -T postgres sh -c 'PGPASSWORD="$POSTGRES_PASSWORD" pg_dump -U fgn_serving -d fgn_serving' | gzip -9 > "${backup_file}.tmp"
    mv "${backup_file}.tmp" "${backup_file}"
  else
    log "postgres is not running; skipping pre-deploy backup. Set REQUIRE_PREDEPLOY_BACKUP=1 to make this fatal."
    if [[ "${REQUIRE_PREDEPLOY_BACKUP:-0}" == "1" ]]; then
      die "pre-deploy backup required but postgres is not running."
    fi
  fi
fi

set +e
deploy_rc=0
if [[ "${SKIP_PULL}" != "1" ]]; then
  compose pull api worker-gpu cleanup-backup frontend
  deploy_rc=$?
fi
if [[ "${deploy_rc}" == "0" ]]; then
  compose up -d
  deploy_rc=$?
fi
if [[ "${deploy_rc}" == "0" ]]; then
  "${SCRIPT_DIR}/smoke_lab.sh"
  deploy_rc=$?
fi
set -e

if [[ "${deploy_rc}" != "0" ]]; then
  write_json_record "${pending}" "failed" "Post-deploy smoke checks failed."
  if [[ "${NO_ROLLBACK}" != "1" && -f "${previous}" ]]; then
    log "deployment failed; attempting image-level rollback."
    "${SCRIPT_DIR}/rollback_lab.sh" || true
  fi
  exit "${deploy_rc}"
fi

write_json_record "${current}" "healthy" "Deployment completed."
cp "${current}" "${root}/deployment_${stamp}.json"
persist_image_pins
rm -f "${pending}"
log "deployment completed."
