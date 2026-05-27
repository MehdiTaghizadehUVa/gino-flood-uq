#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=lib_deploy.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_deploy.sh"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

load_lab_env
root="$(deployment_root)"
previous="${root}/previous.json"
current="${root}/current.json"

export_images_from_record "${previous}"
require_var FGN_API_IMAGE
require_var FGN_WORKER_IMAGE
require_var FGN_CLEANUP_IMAGE
require_var FGN_FRONTEND_IMAGE

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: would roll back to images recorded in ${previous}."
  exit 0
fi

log "rolling back services to previous deployment images."
compose pull api worker-gpu cleanup-backup frontend
compose up -d
"${SCRIPT_DIR}/smoke_lab.sh"
cp "${previous}" "${current}"
persist_image_pins
write_json_record "${root}/rollback_$(date -u +%Y%m%dT%H%M%SZ).json" "rolled_back" "Rollback completed."
log "rollback completed."
