#!/bin/bash
#SBATCH --job-name=neon_tr450
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/slurm_tr_%j.out
set -euo pipefail
REPO=/home/jrj6wm/GINO_Model/neuraloperator_clean_mcdropout
CONTAINER=/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif
mkdir -p /scratch/jrj6wm/GINO_Model/neon_stage2_full_train
cd "${REPO}"
module purge; module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"; slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
export APPTAINERENV_NEON_N_TRAIN="${NEON_N_TRAIN:-450}"
export APPTAINERENV_NEON_OUT_DIR="${NEON_OUT_DIR:-}"
export APPTAINERENV_NEON_CACHE_DIR="${NEON_CACHE_DIR:-}"
export APPTAINERENV_NEON_PRIOR_SCALE="${NEON_PRIOR_SCALE:-}"
export APPTAINERENV_NEON_D_E="${NEON_D_E:-}"
export APPTAINERENV_NEON_EPOCHS="${NEON_EPOCHS:-}"
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" python scripts/neon_stage2_tr_train.py
