#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DEPLOY_DIR}/../.." && pwd)"

log() {
  printf '[fgn-deploy] %s\n' "$*" >&2
}

die() {
  printf '[fgn-deploy] ERROR: %s\n' "$*" >&2
  exit 1
}

load_lab_env() {
  local env_file="${ENV_FILE:-${DEPLOY_DIR}/.env}"
  if [[ -f "${env_file}" ]]; then
    local preserved_deploy_commit="${FGN_DEPLOY_COMMIT:-}"
    local preserved_api_image="${FGN_API_IMAGE:-}"
    local preserved_worker_image="${FGN_WORKER_IMAGE:-}"
    local preserved_cleanup_image="${FGN_CLEANUP_IMAGE:-}"
    local preserved_frontend_image="${FGN_FRONTEND_IMAGE:-}"

    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a

    # GitHub Actions injects immutable image tags for deployments. Preserve those
    # over local .env pins so automation can deploy a newly-built commit without
    # rewriting the lab secrets/config file.
    [[ -n "${preserved_deploy_commit}" ]] && export FGN_DEPLOY_COMMIT="${preserved_deploy_commit}"
    [[ -n "${preserved_api_image}" ]] && export FGN_API_IMAGE="${preserved_api_image}"
    [[ -n "${preserved_worker_image}" ]] && export FGN_WORKER_IMAGE="${preserved_worker_image}"
    [[ -n "${preserved_cleanup_image}" ]] && export FGN_CLEANUP_IMAGE="${preserved_cleanup_image}"
    [[ -n "${preserved_frontend_image}" ]] && export FGN_FRONTEND_IMAGE="${preserved_frontend_image}"
  fi
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    die "${name} is required."
  fi
}

deployment_root() {
  require_var FGN_DATA_ROOT
  printf '%s\n' "${FGN_DEPLOYMENT_RECORD_ROOT:-${FGN_DATA_ROOT}/deployments}"
}

compose() {
  local env_file="${ENV_FILE:-${DEPLOY_DIR}/.env}"
  (cd "${DEPLOY_DIR}" && FGN_ENV_FILE="${env_file}" docker compose -f docker-compose.yml "$@")
}

write_json_record() {
  local path="$1"
  local status="$2"
  local message="${3:-}"
  mkdir -p "$(dirname "${path}")"
  python3 - "$path" "$status" "$message" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

path, status, message = sys.argv[1:4]
images = {
    "api": os.environ.get("FGN_API_IMAGE"),
    "worker": os.environ.get("FGN_WORKER_IMAGE"),
    "cleanup": os.environ.get("FGN_CLEANUP_IMAGE"),
    "frontend": os.environ.get("FGN_FRONTEND_IMAGE"),
}

digests = {}
for name, image in images.items():
    if not image:
        digests[name] = None
        continue
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = json.loads(proc.stdout.strip() or "[]")
        digests[name] = values[0] if values else None
    except Exception:
        digests[name] = None

record = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "message": message,
    "commit": os.environ.get("FGN_DEPLOY_COMMIT"),
    "images": images,
    "image_digests": digests,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

export_images_from_record() {
  local path="$1"
  [[ -f "${path}" ]] || die "Deployment record not found: ${path}"
  eval "$(
    python3 - "$path" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
images = data.get("images") or {}
mapping = {
    "FGN_API_IMAGE": images.get("api"),
    "FGN_WORKER_IMAGE": images.get("worker"),
    "FGN_CLEANUP_IMAGE": images.get("cleanup"),
    "FGN_FRONTEND_IMAGE": images.get("frontend"),
    "FGN_DEPLOY_COMMIT": data.get("commit"),
}
for key, value in mapping.items():
    if value:
        print(f"export {key}={shlex.quote(str(value))}")
PY
  )"
}
