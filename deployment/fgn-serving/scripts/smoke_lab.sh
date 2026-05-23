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
require_var FGN_SITE_HOSTNAME
require_var FGN_DATA_ROOT

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: would smoke-test HTTPS, Compose services, worker, GPU, and disk space."
  exit 0
fi

https_check() {
  local url="$1" max_attempts="${2:-10}" delay="${3:-5}" timeout="${4:-15}"
  local attempt=0
  until curl -fsSk --max-time "${timeout}" "${url}" >/dev/null 2>&1; do
    attempt=$(( attempt + 1 ))
    if (( attempt >= max_attempts )); then
      curl -fsSk --max-time "${timeout}" "${url}" >/dev/null
    fi
    log "HTTPS not ready (attempt ${attempt}/${max_attempts}): ${url} — retrying in ${delay}s"
    sleep "${delay}"
  done
}

https_check "https://${FGN_SITE_HOSTNAME}/api/health" 12 5 15
https_check "https://${FGN_SITE_HOSTNAME}/api/model-bundle-health" 8 5 30
https_check "https://${FGN_SITE_HOSTNAME}/" 8 5 15

compose ps --status running api worker-gpu frontend redis postgres proxy >/dev/null
compose exec -T redis redis-cli ping | grep -q PONG
compose exec -T postgres pg_isready -U fgn_serving -d fgn_serving >/dev/null
compose exec -T worker-gpu celery -A neuralop.flood.serving.celery_app inspect ping --timeout=10 >/dev/null
compose exec -T worker-gpu python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in worker-gpu.")
print(torch.cuda.get_device_name(0))
PY

min_free_gb="${FGN_MIN_FREE_GB:-20}"
free_kb="$(df -Pk "${FGN_DATA_ROOT}" | awk 'NR == 2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
if (( free_gb < min_free_gb )); then
  die "Only ${free_gb} GiB free under ${FGN_DATA_ROOT}; need at least ${min_free_gb} GiB."
fi

log "smoke checks passed."
