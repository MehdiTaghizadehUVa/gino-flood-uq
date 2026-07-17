#!/usr/bin/env sh
set -eu

base_url="${1:-http://127.0.0.1:3000}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

request() {
  path="$1"
  output="$2"
  curl --silent --show-error --max-time 15 \
    --output "${output}" \
    --write-out '%{http_code}' \
    "${base_url}${path}"
}

marketing_status="$(request / "${tmp_dir}/marketing.html")"
test "${marketing_status}" = "200"
grep -q 'data-flooduq-marketing' "${tmp_dir}/marketing.html"
grep -q 'Know where flooding may go.' "${tmp_dir}/marketing.html"

demo_status="$(request /demo "${tmp_dir}/demo.html")"
test "${demo_status}" = "200"
grep -q 'data-flooduq-demo' "${tmp_dir}/demo.html"

redirect_headers="${tmp_dir}/redirect.headers"
redirect_status="$(curl --silent --show-error --max-time 15 \
  --output /dev/null \
  --dump-header "${redirect_headers}" \
  --write-out '%{http_code}' \
  "${base_url}/runs/smoke-run-id")"
test "${redirect_status}" = "308"
grep -qi '^location: /demo/runs/smoke-run-id' "${redirect_headers}"

queue_redirect_headers="${tmp_dir}/queue-redirect.headers"
queue_redirect_status="$(curl --silent --show-error --max-time 15 \
  --output /dev/null \
  --dump-header "${queue_redirect_headers}" \
  --write-out '%{http_code}' \
  "${base_url}/runs")"
test "${queue_redirect_status}" = "308"
grep -Fqi 'location: /demo?workspace=runs' "${queue_redirect_headers}"

poster_status="$(request /marketing/hero-poster.jpg "${tmp_dir}/hero-poster.jpg")"
test "${poster_status}" = "200"
test -s "${tmp_dir}/hero-poster.jpg"

robots_status="$(request /robots.txt "${tmp_dir}/robots.txt")"
test "${robots_status}" = "200"
grep -q 'Disallow: /demo' "${tmp_dir}/robots.txt"
grep -q 'Disallow: /api' "${tmp_dir}/robots.txt"

printf 'Frontend route contract passed.\n'
