#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/replicated/phase5_gate.json" >&2
  exit 2
fi
G1_REPORT=$(realpath -s "$1")
REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
RUN_ROOT=${NEON_SCALEOUT_ROOT:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4/scaleout_b3}
CACHE_DIR=${NEON_CACHE_DIR:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4/feature_cache_v3}
SCALEOUT_RUNG=${NEON_SCALEOUT_RUNG:?Set NEON_SCALEOUT_RUNG from the replicated Phase-5 gate}
PRIOR_SEED=${NEON_PRIOR_SEED:-20260703}
PROTOCOL_SHA256=${NEON_PROTOCOL_SHA256:?Set NEON_PROTOCOL_SHA256 from the governing Phase-5 protocol}
GOVERNING_GATE_SHA256=${NEON_GOVERNING_GATE_SHA256:?Set NEON_GOVERNING_GATE_SHA256 from the accepted Phase-5 gate}
SBATCH_SCRIPT=${REPO}/scripts/sbatch_neon_ablation_grid.sh
PLAN=${RUN_ROOT}/scaleout_plan.json

[[ -s "${G1_REPORT}" ]] || { echo "Missing G1 report: ${G1_REPORT}" >&2; exit 2; }
[[ -s "${G1_REPORT}.sha256" ]] || { echo "Missing gate checksum: ${G1_REPORT}.sha256" >&2; exit 2; }
[[ -f "${CONTAINER}" ]] || { echo "Missing container: ${CONTAINER}" >&2; exit 2; }
[[ -x "${SBATCH_SCRIPT}" ]] || { echo "Missing sbatch script: ${SBATCH_SCRIPT}" >&2; exit 2; }
HEAD=$(git -C "${REPO}" rev-parse HEAD)
test -z "$(git -C "${REPO}" status --porcelain)" || {
  echo "Refusing to preflight scale-out from a dirty tree" >&2
  exit 2
}
if [[ -s "${RUN_ROOT}/job_id.txt" ]]; then
  echo "Scale-out was already submitted: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"
module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}"
PREPARE_SEED_ARGS=()
if [[ -n "${NEON_DIRICHLET_PARTICLE_SEED:-}" ]]; then
  PREPARE_SEED_ARGS+=(
    --dirichlet-particle-seed "${NEON_DIRICHLET_PARTICLE_SEED}"
  )
fi
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python "${REPO}/scripts/neon_scaleout_preflight.py" prepare \
  --g1-report "${G1_REPORT}" --run-root "${RUN_ROOT}" \
  --cache-dir "${CACHE_DIR}" --expected-head "${HEAD}" \
  --ladder-rung "${SCALEOUT_RUNG}" --prior-seed "${PRIOR_SEED}" \
  --protocol-sha256 "${PROTOCOL_SHA256}" \
  --governing-gate-sha256 "${GOVERNING_GATE_SHA256}" \
  "${PREPARE_SEED_ARGS[@]}" >/dev/null
test -s "${PLAN}"

if [[ "${NEON_SUBMIT_DRY_RUN:-0}" == 1 ]]; then
  printf 'Validated scale-out only: head=%s plan=%s\n' "${HEAD}" "${PLAN}"
  exit 0
fi
EXPORTS="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_SCALEOUT_PLAN=${PLAN},NEON_SCALEOUT_ROOT=${RUN_ROOT},NEON_CACHE_DIR=${CACHE_DIR},NEON_SCALEOUT_RUNG=${SCALEOUT_RUNG},NEON_PRIOR_SEED=${PRIOR_SEED},NEON_PROTOCOL_SHA256=${PROTOCOL_SHA256},NEON_GOVERNING_GATE_SHA256=${GOVERNING_GATE_SHA256}"
if [[ -n "${NEON_DIRICHLET_PARTICLE_SEED:-}" ]]; then
  EXPORTS+=",NEON_DIRICHLET_PARTICLE_SEED=${NEON_DIRICHLET_PARTICLE_SEED}"
fi
JOB_ID=$(sbatch --parsable \
  --output="${RUN_ROOT}/slurm-%A_%a.out" \
  --error="${RUN_ROOT}/slurm-%A_%a.err" \
  --export="${EXPORTS}" \
  "${SBATCH_SCRIPT}")
printf '%s\n' "${JOB_ID}" > "${RUN_ROOT}/job_id.txt"
for replicate in 0 1 2 3 4; do
  task_offset=$((replicate * 5))
  index=0
  for n_train in 25 50 100 250 400; do
    printf '%s_%s\n' "${JOB_ID}" "$((task_offset + index))" \
      > "${RUN_ROOT}/rep${replicate}/n${n_train}/job_id.txt"
    index=$((index + 1))
  done
done
printf 'Submitted replicated N-sweep: job=%s head=%s plan=%s\n' "${JOB_ID}" "${HEAD}" "${PLAN}"
