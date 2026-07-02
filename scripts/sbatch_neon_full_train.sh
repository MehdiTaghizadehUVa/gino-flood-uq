#!/bin/bash
#SBATCH --job-name=neon_full_train
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/slurm_%j.out
set -euo pipefail
REPO=/home/jrj6wm/GINO_Model/neuraloperator_clean_mcdropout
CONTAINER=/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif
mkdir -p /scratch/jrj6wm/GINO_Model/neon_stage2_full_train
cd "${REPO}"
module purge; module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"; slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" python scripts/neon_stage2_full_train.py
