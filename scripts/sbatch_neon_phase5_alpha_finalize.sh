#!/bin/bash
#SBATCH --account=uqgroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail

: "${NEON_EXPECTED_HEAD:?Set NEON_EXPECTED_HEAD}"
: "${NEON_PHASE5_ROOT:?Set NEON_PHASE5_ROOT}"
REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
SUBMISSION=${NEON_PHASE5_ROOT}/ALPHA_SWEEP_SUBMITTED.json

cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}"
test -z "$(git status --porcelain)" || {
  echo "Alpha-sweep finalizer refuses a dirty repository" >&2
  exit 2
}
module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"
mapfile -t REPORTS < <(python3 - <<'PY' "${SUBMISSION}"
import json, sys
for directory in json.load(open(sys.argv[1]))["run_dirs"]:
    print(directory + "/phase5_eval/RESULT.json")
PY
)
ARGS=()
for REPORT in "${REPORTS[@]}"; do ARGS+=(--report "${REPORT}"); done
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_phase5_alpha_sweep_finalize.py \
  --phase-s-complete "${NEON_PHASE5_ROOT}/PHASE_S_COMPLETE.json" \
  "${ARGS[@]}" --expected-head "${NEON_EXPECTED_HEAD}" \
  --output "${NEON_PHASE5_ROOT}/ALPHA_SWEEP_DECISION.json"
