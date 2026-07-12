#!/bin/bash
#SBATCH --job-name=neon_repair
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_slurm_%j.out
set -euo pipefail

: "${NEON_LADDER_RUNG:?Set NEON_LADDER_RUNG to B0, B1a, B1b, B2, B3, B4, or B5}"
REPO=/home/jrj6wm/GINO_Model/neuraloperator_clean_mcdropout
CONTAINER=/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif
RUN_ROOT=/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4
export NEON_N_TRAIN=${NEON_N_TRAIN:-450}
export NEON_OUT_DIR=${NEON_OUT_DIR:-${RUN_ROOT}/${NEON_LADDER_RUNG,,}_n${NEON_N_TRAIN}}
export NEON_CACHE_DIR=${NEON_CACHE_DIR:-${RUN_ROOT}/feature_cache_v3}

cd "${REPO}"
test -z "$(git status --porcelain)" || { echo "Refusing to launch from a dirty tree" >&2; exit 2; }
mkdir -p "${NEON_OUT_DIR}"
git rev-parse HEAD > "${NEON_OUT_DIR}/git_head.txt"
git status --porcelain > "${NEON_OUT_DIR}/git_status.txt"
printf '%s\n' "${NEON_LADDER_RUNG}" > "${NEON_OUT_DIR}/ladder_rung.txt"

module purge
module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"
slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
export APPTAINERENV_NEON_LADDER_RUNG="${NEON_LADDER_RUNG}"
export APPTAINERENV_NEON_N_TRAIN="${NEON_N_TRAIN}"
export APPTAINERENV_NEON_OUT_DIR="${NEON_OUT_DIR}"
export APPTAINERENV_NEON_CACHE_DIR="${NEON_CACHE_DIR}"
if [[ -n "${NEON_DE_SPREAD_MULTIPLIER:-}" ]]; then
  export APPTAINERENV_NEON_DE_SPREAD_MULTIPLIER="${NEON_DE_SPREAD_MULTIPLIER}"
fi
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" \
  python scripts/neon_stage2_tr_train.py
