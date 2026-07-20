#!/bin/bash
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
set -euo pipefail

: "${NEON_PHASE5_TASK:?Set NEON_PHASE5_TASK}"
: "${NEON_EXPECTED_HEAD:?Set NEON_EXPECTED_HEAD}"
: "${NEON_PHASE5_ROOT:?Set NEON_PHASE5_ROOT}"
: "${NEON_CONFIG:?Set NEON_CONFIG}"
: "${NEON_BUNDLE:?Set NEON_BUNDLE}"
: "${NEON_B3_DIR:?Set NEON_B3_DIR}"
: "${NEON_CACHE_DIR:?Set NEON_CACHE_DIR}"

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
PROTOCOL=${NEON_PHASE5_ROOT}/PROTOCOL.json
PHASE5_PREFLIGHT=${NEON_PHASE5_ROOT}/PREFLIGHT.json
B3_CHECKPOINT=${NEON_B3_DIR}/neon_stage2_best.pt
B3_HISTORY=${NEON_B3_DIR}/history.json
B3_PREFLIGHT=${NEON_B3_DIR}/preflight.json
PLAN_ONLY=${NEON_PLAN_ONLY:-0}

cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}" || {
  echo "Phase-5 repository HEAD changed after preflight" >&2
  exit 2
}
test -z "$(git status --porcelain)" || {
  echo "Phase-5 jobs refuse a dirty repository" >&2
  exit 2
}
test -s "${PROTOCOL}" && test -s "${PROTOCOL}.sha256"
test -s "${PHASE5_PREFLIGHT}" && test -s "${PHASE5_PREFLIGHT}.sha256"

module purge
module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"
slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"

COMMON=(
  --config "${NEON_CONFIG}"
  --bundle "${NEON_BUNDLE}"
  --checkpoint "${B3_CHECKPOINT}"
  --history "${B3_HISTORY}"
  --preflight "${B3_PREFLIGHT}"
  --phase5-preflight "${PHASE5_PREFLIGHT}"
  --cache-dir "${NEON_CACHE_DIR}"
  --protocol "${PROTOCOL}"
  --expected-head "${NEON_EXPECTED_HEAD}"
)
if [[ "${PLAN_ONLY}" == 1 ]]; then
  EXTRA=(--plan-only --device cpu)
  APPTAINER_GPU_ARGS=()
else
  EXTRA=(--device cuda:0)
  APPTAINER_GPU_ARGS=(--nv)
fi

run_python() {
  apptainer exec "${APPTAINER_GPU_ARGS[@]}" ${APPTAINER_BIND_ARGS} "${CONTAINER}" python "$@"
}

case "${NEON_PHASE5_TASK}" in
  geometry)
    OUT=${NEON_PHASE5_ROOT}/d2_geometry
    run_python scripts/neon_phase5_geometry.py "${COMMON[@]}" \
      --output-dir "${OUT}" "${EXTRA[@]}"
    ;;
  cancellation)
    OUT=${NEON_PHASE5_ROOT}/d3_cancellation
    run_python scripts/neon_phase5_cancellation.py "${COMMON[@]}" \
      --output-dir "${OUT}" "${EXTRA[@]}"
    ;;
  direct_data_last)
    OUT=${NEON_PHASE5_ROOT}/d1p_data_last
    run_python scripts/neon_phase5_direct.py "${COMMON[@]}" \
      --output-dir "${OUT}" --mode data --law probit --head last "${EXTRA[@]}"
    ;;
  direct_rpf_last)
    OUT=${NEON_PHASE5_ROOT}/d1p_rpf_last
    run_python scripts/neon_phase5_direct.py "${COMMON[@]}" \
      --output-dir "${OUT}" --mode rpf --law probit --head last "${EXTRA[@]}"
    ;;
  direct_rpf_full)
    OUT=${NEON_PHASE5_ROOT}/d1p_rpf_full
    run_python scripts/neon_phase5_direct.py "${COMMON[@]}" \
      --output-dir "${OUT}" --mode rpf --law probit --head full "${EXTRA[@]}"
    ;;
  p1a_eval|direct_dirichlet)
    : "${NEON_P1A_DIR:?Set NEON_P1A_DIR for ${NEON_PHASE5_TASK}}"
    P1_COMMON=(
      --config "${NEON_CONFIG}"
      --bundle "${NEON_BUNDLE}"
      --checkpoint "${NEON_P1A_DIR}/neon_stage2_best.pt"
      --history "${NEON_P1A_DIR}/history.json"
      --preflight "${NEON_P1A_DIR}/preflight.json"
      --phase5-preflight "${PHASE5_PREFLIGHT}"
      --cache-dir "${NEON_CACHE_DIR}"
      --protocol "${PROTOCOL}"
      --expected-head "${NEON_EXPECTED_HEAD}"
    )
    if [[ "${NEON_PHASE5_TASK}" == p1a_eval ]]; then
      OUT=${NEON_P1A_DIR}/phase5_eval
      run_python scripts/neon_phase5_p1a_eval.py "${P1_COMMON[@]}" \
        --geometry-tensors "${NEON_PHASE5_ROOT}/d2_geometry/GEOMETRY.pt" \
        --output-dir "${OUT}" "${EXTRA[@]}"
    else
      OUT=${NEON_P1A_DIR}/direct_dirichlet
      run_python scripts/neon_phase5_direct.py "${P1_COMMON[@]}" \
        --output-dir "${OUT}" \
        --mode rpf --law dirichlet --head last --draws 16 "${EXTRA[@]}"
    fi
    ;;
  pilot_eval)
    : "${NEON_PILOT_DIR:?Set NEON_PILOT_DIR for pilot_eval}"
    OUT=${NEON_PILOT_DIR}/phase5_eval
    run_python scripts/neon_phase5_p1a_eval.py \
      --config "${NEON_CONFIG}" \
      --bundle "${NEON_BUNDLE}" \
      --checkpoint "${NEON_PILOT_DIR}/neon_stage2_best.pt" \
      --history "${NEON_PILOT_DIR}/history.json" \
      --preflight "${NEON_PILOT_DIR}/preflight.json" \
      --phase5-preflight "${PHASE5_PREFLIGHT}" \
      --cache-dir "${NEON_CACHE_DIR}" \
      --protocol "${PROTOCOL}" \
      --expected-head "${NEON_EXPECTED_HEAD}" \
      --geometry-tensors "${NEON_PHASE5_ROOT}/d2_geometry/GEOMETRY.pt" \
      --output-dir "${OUT}" "${EXTRA[@]}"
    ;;
  baseline_eval)
    # B3 ID evidence uses the exact same 50 validation families, cached
    # aleatory bank, physical transform, and nested estimator as every pilot.
    # Keeping it in the Phase-5 evaluator makes subsequent OOD ranking paired
    # to the governing B3 checkpoint rather than an older generic artifact.
    OUT=${NEON_PHASE5_ROOT}/d5_b3_id
    run_python scripts/neon_phase5_p1a_eval.py "${COMMON[@]}" \
      --geometry-tensors "${NEON_PHASE5_ROOT}/d2_geometry/GEOMETRY.pt" \
      --output-dir "${OUT}" "${EXTRA[@]}"
    ;;
  *)
    echo "Unsupported NEON_PHASE5_TASK=${NEON_PHASE5_TASK}" >&2
    exit 2
    ;;
esac

if [[ "${PLAN_ONLY}" != 1 ]]; then
  printf '%s\n' "${SLURM_JOB_ID:-manual}" > "${OUT}/job_id.txt"
fi
