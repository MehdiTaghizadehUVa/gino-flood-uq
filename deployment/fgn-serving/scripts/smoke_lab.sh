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

http_expect() {
  local url="$1" expected_statuses="$2" body_marker="${3:-}" location_pattern="${4:-}"
  local max_attempts="${5:-10}" delay="${6:-5}" timeout="${7:-15}"
  local attempt=0 response_dir status
  response_dir="$(mktemp -d)"

  while true; do
    status="$(curl -sSk --max-time "${timeout}" \
      -D "${response_dir}/headers" \
      -o "${response_dir}/body" \
      -w '%{http_code}' "${url}" || true)"
    if [[ ",${expected_statuses}," == *",${status},"* ]] \
      && { [[ -z "${body_marker}" ]] || grep -Fq "${body_marker}" "${response_dir}/body"; } \
      && { [[ -z "${location_pattern}" ]] || grep -Eiq "^location: .*${location_pattern}" "${response_dir}/headers"; }; then
      rm -rf "${response_dir}"
      return 0
    fi

    attempt=$(( attempt + 1 ))
    if (( attempt >= max_attempts )); then
      cat "${response_dir}/headers" >&2 || true
      cat "${response_dir}/body" >&2 || true
      rm -rf "${response_dir}"
      die "Unexpected response from ${url}: HTTP ${status}; expected ${expected_statuses}."
    fi
    log "HTTP route not ready (attempt ${attempt}/${max_attempts}): ${url}; retrying in ${delay}s"
    sleep "${delay}"
  done
}

api_json_check() {
  local path="$1" max_attempts="${2:-10}" delay="${3:-5}"
  local attempt=0
  until compose exec -T api python - "${path}" <<'PY' >/dev/null; do
import json
import sys
import urllib.request

path = sys.argv[1]
response = urllib.request.urlopen(f"http://127.0.0.1:8000{path}", timeout=15)
payload = json.loads(response.read().decode())
if response.status != 200:
    raise SystemExit(f"{path} returned HTTP {response.status}")
if path == "/api/health":
    if payload.get("status") != "ok" or payload.get("production_inference_ready") is not True:
        raise SystemExit(f"{path} not ready: {payload}")
elif path == "/api/model-bundle-health":
    required = ("bundle_id", "total_members", "max_forecast_steps", "initial_condition")
    missing = [key for key in required if key not in payload]
    if missing:
        raise SystemExit(f"{path} missing required keys: {missing}")
PY
    attempt=$(( attempt + 1 ))
    if (( attempt >= max_attempts )); then
      compose exec -T api python - "${path}" <<'PY'
import json
import sys
import urllib.request

path = sys.argv[1]
response = urllib.request.urlopen(f"http://127.0.0.1:8000{path}", timeout=15)
payload = json.loads(response.read().decode())
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(f"{path} failed readiness checks")
PY
    fi
    log "API JSON health not ready (attempt ${attempt}/${max_attempts}): ${path} — retrying in ${delay}s"
    sleep "${delay}"
  done
}

log "checking internal API health payload."
api_json_check "/api/health" 12 5
log "checking internal model-bundle health payload."
api_json_check "/api/model-bundle-health" 8 5
log "checking public product site and brand marker."
http_expect "https://${FGN_SITE_HOSTNAME}/" "200" \
  "Know where flooding may go." "" 8 5 15
log "checking public marketing media."
http_expect "https://${FGN_SITE_HOSTNAME}/marketing/hero-poster.jpg" "200" "" "" 8 5 15
log "checking public guest console and protected private routes."
http_expect "https://${FGN_SITE_HOSTNAME}/demo" "200" \
  "Sign in to launch analysis" "" 8 5 15
http_expect "https://${FGN_SITE_HOSTNAME}/api/forcing-template" "200" "" "" 8 5 15
http_expect "https://${FGN_SITE_HOSTNAME}/demo/runs/not-a-public-run" "302" "" "/oauth2/sign_in" 8 5 15
http_expect "https://${FGN_SITE_HOSTNAME}/admin" "302" "" "/oauth2/sign_in" 8 5 15
log "checking protected API boundary."
http_expect "https://${FGN_SITE_HOSTNAME}/api/health" "302,401" "" "" 8 5 15
http_expect "https://${FGN_SITE_HOSTNAME}/api/runs" "302,401" "" "" 8 5 15

log "checking Compose service state."
compose ps --status running api worker-gpu frontend redis postgres proxy >/dev/null
log "validating Caddy proxy configuration."
compose exec -T proxy caddy validate --config /etc/caddy/Caddyfile >/dev/null 2>&1
log "checking Redis."
compose exec -T redis redis-cli ping | grep -q PONG
log "checking Postgres."
compose exec -T postgres pg_isready -U fgn_serving -d fgn_serving >/dev/null
log "checking Celery worker heartbeat."
compose exec -T worker-gpu celery -A neuralop.flood.serving.celery_app inspect ping --timeout=10 >/dev/null
log "checking CUDA visibility in worker."
compose exec -T worker-gpu python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in worker-gpu.")
print(torch.cuda.get_device_name(0))
PY

min_free_gb="${FGN_MIN_FREE_GB:-20}"
log "checking data-root free disk space."
free_kb="$(df -Pk "${FGN_DATA_ROOT}" | awk 'NR == 2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
if (( free_gb < min_free_gb )); then
  die "Only ${free_gb} GiB free under ${FGN_DATA_ROOT}; need at least ${min_free_gb} GiB."
fi

log "smoke checks passed."
