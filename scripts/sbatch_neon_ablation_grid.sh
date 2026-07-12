#!/bin/bash
#SBATCH --job-name=neon_ablation
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-24
#SBATCH --output=/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/slurm_abl_%A_%a.out
set -euo pipefail
N_VALUES=(25 50 100 250 400)
N_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
export NEON_SUBSET_REPLICATE=$((SLURM_ARRAY_TASK_ID / 5))
export NEON_N_TRAIN=${N_VALUES[$N_INDEX]}
REPO=/home/jrj6wm/GINO_Model/neuraloperator_clean_mcdropout
CONTAINER=/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif
cd "${REPO}"
module purge; module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"; slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
export APPTAINERENV_NEON_N_TRAIN="${NEON_N_TRAIN}"
export APPTAINERENV_NEON_SUBSET_REPLICATE="${NEON_SUBSET_REPLICATE}"
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" python scripts/neon_stage2_tr_train.py
