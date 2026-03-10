#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -J gino_gauss_wv
#SBATCH -o logs/out/gino_gauss_wv-%j.out
#SBATCH -e logs/err/gino_gauss_wv-%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/lib/common.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  source "${SCRIPT_DIR}/slurm/lib/common.sh"
else
  CANONICAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  source "$(cd "${CANONICAL_DIR}/.." && pwd)/lib/common.sh"
  SCRIPT_DIR="$(slurm_resolve_scripts_root "${CANONICAL_DIR}")"
fi
slurm_load_apptainer
cd "${SCRIPT_DIR}"
mkdir -p logs/out logs/err

PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_no_physics_git_main}"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/train_gino_flood_train_rollout_animation_WV.py"
TRAIN_CONFIG="${PROJECT_DIR}/config/flood/wv/gino_pluvial_flood_config_WV_depth_only_gaussian.yaml"
CONTAINER_PATH="/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif"
DATA_ROOT="/scratch/$USER/Data_Generation_UQ/Results/M40"

SEED=123
RUN_TAG_BASE="wv_m40_gaussianNLL_ep150_lr1e-4_gr0.1_h64_wd1e-4"
WANDB_GROUP="${RUN_TAG_BASE}_job${SLURM_JOB_ID}"
WANDB_NAME="${WANDB_GROUP}_seed${SEED}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_DIR}/scripts/checkpoints_WV_depth_only_gaussian}"
CKPT_DIR="${CKPT_ROOT}/${WANDB_GROUP}"
mkdir -p "${CKPT_DIR}"

slurm_configure_host_ca

for key_path in "${PROJECT_DIR}/config/wandb_api_key.txt" "/scratch/$USER/Data_Generation_UQ/GINO_Model/neuraloperator_no_physics/config/wandb_api_key.txt"; do
  if [[ -f "${key_path}" ]]; then
    export APPTAINERENV_WANDB_API_KEY="$(head -n 1 "${key_path}" | tr -d "\r")"
    break
  fi
done

echo "Training script: ${TRAIN_SCRIPT}"
echo "Config:          ${TRAIN_CONFIG}"
echo "Data root:       ${DATA_ROOT}"
echo "Seed:            ${SEED}"
echo "Checkpoint dir:  ${CKPT_DIR}"
echo "W&B group:       ${WANDB_GROUP}"
echo "W&B name:        ${WANDB_NAME}"

apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${TRAIN_SCRIPT}" \
  --config_path "${TRAIN_CONFIG}" \
  --data.root "${DATA_ROOT}" \
  --rollout_data.root "${DATA_ROOT}" \
  --distributed.seed "${SEED}" \
  --wandb.log true \
  --wandb.group "${WANDB_GROUP}" \
  --wandb.name "${WANDB_NAME}" \
  --checkpoint.save_dir "${CKPT_DIR}"
