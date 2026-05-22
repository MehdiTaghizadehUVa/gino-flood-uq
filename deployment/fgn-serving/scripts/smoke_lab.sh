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

curl -fsSk --max-time 15 "https://${FGN_SITE_HOSTNAME}/api/health" >/dev/null
curl -fsSk --max-time 30 "https://${FGN_SITE_HOSTNAME}/api/model-bundle-health" >/dev/null
curl -fsSk --max-time 15 "https://${FGN_SITE_HOSTNAME}/" >/dev/null

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
