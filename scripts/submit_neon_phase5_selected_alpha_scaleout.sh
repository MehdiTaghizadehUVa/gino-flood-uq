#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 PHASE5_ROOT ALPHA_SWEEP_DECISION.json [EVIDENCE_ROOT]" >&2
  exit 2
fi

SOURCE_ROOT=$(realpath -s "$1")
DECISION=$(realpath -s "$2")
EVIDENCE_ROOT=$(realpath -s "${3:-${SOURCE_ROOT}/phase_s_selected_alpha}")
REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}

cd "${REPO}"
HEAD=$(git rev-parse HEAD)
test -z "$(git status --porcelain)" || {
  echo "Refusing selected-alpha scale-out from a dirty repository" >&2
  exit 2
}
test -s "${SOURCE_ROOT}/PROTOCOL.json"
test -s "${SOURCE_ROOT}/PROTOCOL.json.sha256"
test -s "${SOURCE_ROOT}/PHASE_S_COMPLETE.json"
test -s "${SOURCE_ROOT}/PHASE_S_COMPLETE.json.sha256"
test -s "${DECISION}"
test -s "${DECISION}.sha256"
test ! -e "${EVIDENCE_ROOT}/PHASE_S_SUBMITTED.json" || {
  echo "Selected-alpha Phase S already submitted: ${EVIDENCE_ROOT}" >&2
  exit 2
}

module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"
apptainer exec --bind /scratch,/home "${CONTAINER}" python - \
  "${DECISION}" "${SOURCE_ROOT}/PHASE_S_COMPLETE.json" \
  "${SOURCE_ROOT}/PROTOCOL.json" "${HEAD}" <<'PY'
import hashlib
import json
import pathlib
import sys

from neuralop.flood.eval.neon_phase5 import verify_checksummed_artifact

decision_path = pathlib.Path(sys.argv[1])
complete_path = pathlib.Path(sys.argv[2])
protocol_path = pathlib.Path(sys.argv[3])
head = sys.argv[4]
verify_checksummed_artifact(decision_path)
verify_checksummed_artifact(complete_path)
verify_checksummed_artifact(protocol_path)
decision = json.loads(decision_path.read_text())
complete = json.loads(complete_path.read_text())
protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
if decision.get("schema_version") != "neon_phase_s_alpha_sweep_v1":
    raise ValueError("invalid selected-alpha decision schema")
if decision.get("decision") != "alpha_selected":
    raise ValueError("alpha sweep did not select a noninferior run")
if decision.get("evaluation_events_used_for_selection") is not False:
    raise ValueError("selected alpha used evaluation events")
if decision.get("phase_s_complete") != str(complete_path):
    raise ValueError("selected alpha does not reference this Phase-S evidence cycle")
if decision.get("phase_s_complete_sha256") != hashlib.sha256(complete_path.read_bytes()).hexdigest():
    raise ValueError("selected alpha Phase-S completion checksum differs")
if decision.get("analysis_git_head") != head or complete.get("git_head") != head:
    raise ValueError("selected alpha Git HEAD differs from the current repository")
if decision.get("protocol_sha256") != protocol_sha:
    raise ValueError("selected alpha protocol differs from the current protocol")
PY

mkdir -p "${EVIDENCE_ROOT}"
env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
  NEON_PHASE5_PROTOCOL_ROOT="${SOURCE_ROOT}" \
  bash scripts/submit_neon_phase5_scaleout.sh "${EVIDENCE_ROOT}" "${DECISION}"
