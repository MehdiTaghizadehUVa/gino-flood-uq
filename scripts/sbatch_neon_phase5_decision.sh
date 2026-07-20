#!/bin/bash
#SBATCH --account=uqgroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail

: "${NEON_PHASE5_GATE:?Set NEON_PHASE5_GATE to gd0 or gp1}"
: "${NEON_EXPECTED_HEAD:?Set NEON_EXPECTED_HEAD}"
: "${NEON_PHASE5_ROOT:?Set NEON_PHASE5_ROOT}"

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}"
test -z "$(git status --porcelain)"

module purge
module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"
slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"
RUN=(apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER}" python scripts/neon_phase5_decision.py)

case "${NEON_PHASE5_GATE}" in
  gd0)
    OUT=${NEON_PHASE5_ROOT}/gd0
    "${RUN[@]}" \
      --gate gd0 \
      --protocol "${NEON_PHASE5_ROOT}/PROTOCOL.json" \
      --expected-head "${NEON_EXPECTED_HEAD}" \
      --output-dir "${OUT}" \
      --geometry "${NEON_PHASE5_ROOT}/d2_geometry/RESULT.json" \
      --cancellation "${NEON_PHASE5_ROOT}/d3_cancellation/RESULT.json" \
      --direct-data "${NEON_PHASE5_ROOT}/d1p_data_last/RESULT.json" \
      --direct-rpf-last "${NEON_PHASE5_ROOT}/d1p_rpf_last/RESULT.json" \
      --direct-rpf-full "${NEON_PHASE5_ROOT}/d1p_rpf_full/RESULT.json"
    ;;
  gp1)
    P1A_DIR_SPEC=${NEON_P1A_DIRS:-${NEON_P1A_DIR:-}}
    : "${P1A_DIR_SPEC:?Set NEON_P1A_DIR or colon-separated NEON_P1A_DIRS for GP1}"
    IFS=: read -r -a P1A_DIR_ARRAY <<< "${P1A_DIR_SPEC}"
    P1A_RESULTS=()
    DIRECT_RESULTS=()
    for P1A_DIR_ITEM in "${P1A_DIR_ARRAY[@]}"; do
      P1A_RESULTS+=("${P1A_DIR_ITEM}/phase5_eval/RESULT.json")
      DIRECT_RESULTS+=("${P1A_DIR_ITEM}/direct_dirichlet/RESULT.json")
    done
    OUT=${NEON_PHASE5_GATE_OUT:-${NEON_PHASE5_ROOT}/gp1}
    "${RUN[@]}" \
      --gate gp1 \
      --protocol "${NEON_PHASE5_ROOT}/PROTOCOL.json" \
      --expected-head "${NEON_EXPECTED_HEAD}" \
      --output-dir "${OUT}" \
      --cancellation "${NEON_PHASE5_ROOT}/d3_cancellation/RESULT.json" \
      --direct-rpf-last "${NEON_PHASE5_ROOT}/d1p_rpf_last/RESULT.json" \
      --p1a "${P1A_RESULTS[@]}" \
      --dirichlet-direct "${DIRECT_RESULTS[@]}"
    ;;
  *)
    echo "Unsupported NEON_PHASE5_GATE=${NEON_PHASE5_GATE}" >&2
    exit 2
    ;;
esac

printf '%s\n' "${SLURM_JOB_ID:-manual}" > "${OUT}/job_id.txt"
