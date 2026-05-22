#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

tracked_secret_paths="$(git ls-files | grep -E '(^|/)\.env$|(^|/)id_(rsa|ed25519)(\.pub)?$|\.pem$|\.key$' || true)"
if [[ -n "${tracked_secret_paths}" ]]; then
  printf 'Tracked secret-like paths are not allowed:\n%s\n' "${tracked_secret_paths}" >&2
  exit 1
fi

if git grep -n -E 'GOCSPX-|AIza[0-9A-Za-z_-]{20,}|-----BEGIN (RSA|OPENSSH|PRIVATE) KEY-----' -- . ':!*.md' ':!deployment/fgn-serving/.env.example'; then
  printf 'Potential production secret found in tracked files.\n' >&2
  exit 1
fi

printf 'No tracked serving secrets detected.\n'
