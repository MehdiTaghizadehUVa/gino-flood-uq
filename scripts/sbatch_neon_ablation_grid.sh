#!/bin/bash
#SBATCH --job-name=neon_scaleout
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-24%5
#SBATCH --output=/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/scaleout_%A_%a.out
#SBATCH --error=/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/scaleout_%A_%a.err
set -euo pipefail

: "${NEON_EXPECTED_HEAD:?Missing preflight Git HEAD}"
: "${NEON_SCALEOUT_PLAN:?Missing scale-out plan}"
REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_v4_integrated}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
N_VALUES=(25 50 100 250 400)
N_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
export NEON_SUBSET_REPLICATE=$((SLURM_ARRAY_TASK_ID / 5))
export NEON_N_TRAIN=${N_VALUES[$N_INDEX]}
export NEON_LADDER_RUNG=B3
export NEON_OUT_DIR=${NEON_SCALEOUT_ROOT}/rep${NEON_SUBSET_REPLICATE}/n${NEON_N_TRAIN}
export NEON_CACHE_DIR

cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}"
test -z "$(git status --porcelain)" || {
  echo "Refusing to launch scale-out from a dirty tree" >&2
  exit 2
}
module purge
module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"
slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python "${REPO}/scripts/neon_scaleout_preflight.py" validate-task \
  --plan "${NEON_SCALEOUT_PLAN}" --task-id "${SLURM_ARRAY_TASK_ID}" \
  --expected-head "${NEON_EXPECTED_HEAD}" >/dev/null
printf '%s\n' "${NEON_EXPECTED_HEAD}" > "${NEON_OUT_DIR}/git_head.txt"
printf '%s\n' "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" > "${NEON_OUT_DIR}/slurm_task_id.txt"
export APPTAINERENV_NEON_LADDER_RUNG="${NEON_LADDER_RUNG}"
export APPTAINERENV_NEON_N_TRAIN="${NEON_N_TRAIN}"
export APPTAINERENV_NEON_SUBSET_REPLICATE="${NEON_SUBSET_REPLICATE}"
export APPTAINERENV_NEON_OUT_DIR="${NEON_OUT_DIR}"
export APPTAINERENV_NEON_CACHE_DIR="${NEON_CACHE_DIR}"
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" \
  python scripts/neon_stage2_tr_train.py
