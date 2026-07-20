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

cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}"
test -z "$(git status --porcelain)" || {
  echo "Phase-S finalizer refuses a dirty repository" >&2
  exit 2
}
module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}"
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_phase_s_finalize.py \
  --submission "${NEON_PHASE5_ROOT}/PHASE_S_SUBMITTED.json" \
  --expected-head "${NEON_EXPECTED_HEAD}" \
  --output "${NEON_PHASE5_ROOT}/PHASE_S_COMPLETE.json" >/dev/null
printf '%s\n' "${SLURM_JOB_ID:-manual}" > "${NEON_PHASE5_ROOT}/phase_s_finalize_job_id.txt"
