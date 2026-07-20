#!/bin/bash
#SBATCH --account=uqgroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail

: "${NEON_EXPECTED_HEAD:?Set NEON_EXPECTED_HEAD}"
: "${NEON_PHASE5_ROOT:?Set NEON_PHASE5_ROOT}"
: "${NEON_PILOT_GATE_MODE:?Set NEON_PILOT_GATE_MODE}"
: "${NEON_PILOT_DIRS:?Set colon-separated NEON_PILOT_DIRS}"
: "${NEON_PILOT_GATE_OUT:?Set NEON_PILOT_GATE_OUT}"

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}"
test -z "$(git status --porcelain)" || { echo "Refusing dirty repository" >&2; exit 2; }
module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"

IFS=: read -r -a PILOT_DIRS <<< "${NEON_PILOT_DIRS}"
RESULT_ARGS=()
for directory in "${PILOT_DIRS[@]}"; do
  RESULT_ARGS+=(--pilot-result "${directory}/phase5_eval/RESULT.json")
done
OOD_ARGS=()
if [[ -n "${NEON_P2_OOD_DIRS:-}" ]]; then
  IFS=: read -r -a OOD_DIRS <<< "${NEON_P2_OOD_DIRS}"
  if [[ "${#OOD_DIRS[@]}" -ne "${#PILOT_DIRS[@]}" ]]; then
    echo "P2 gate requires one OOD evidence directory per pilot directory" >&2
    exit 2
  fi
  for directory in "${OOD_DIRS[@]}"; do
    OOD_ARGS+=(--p2-ood-result "${directory}/ranking.json")
  done
  : "${NEON_BASELINE_OOD_REPORT:?Set NEON_BASELINE_OOD_REPORT for P2}"
  OOD_ARGS+=(--baseline-ood "${NEON_BASELINE_OOD_REPORT}")
fi

apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_phase5_pilot_gate.py \
  --mode "${NEON_PILOT_GATE_MODE}" \
  --protocol "${NEON_PHASE5_ROOT}/PROTOCOL.json" \
  --gp1-decision "${NEON_PHASE5_ROOT}/gp1_replicated/DECISION.json" \
  --direct "${NEON_PHASE5_ROOT}/d1p_rpf_last/RESULT.json" \
  --baseline-cancellation "${NEON_PHASE5_ROOT}/d3_cancellation/RESULT.json" \
  "${RESULT_ARGS[@]}" \
  "${OOD_ARGS[@]}" \
  --expected-head "${NEON_EXPECTED_HEAD}" \
  --output-dir "${NEON_PILOT_GATE_OUT}"
