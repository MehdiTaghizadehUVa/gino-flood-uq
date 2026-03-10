#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 72:00:00
#SBATCH -J ddofs_wv_e4_ddp2
#SBATCH --array=0-3
#SBATCH -o logs/out/ddofs_wv_e4_ddp2-%A_%a.out
#SBATCH -e logs/err/ddofs_wv_e4_ddp2-%A_%a.err

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

PROJECT_DIR="/home/$USER/GINO_Model/neuraloperator_no_physics"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/train_diffusion_forecaster_WV.py"
TRAIN_CONFIG="${PROJECT_DIR}/config/flood/wv/gino_pluvial_flood_config_WV_depth_only_diffusion.yaml"
CONTAINER_PATH="/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif"
DATA_ROOT="/scratch/$USER/Data_Generation_UQ/Results/M40"

ENSEMBLE_ID="${SLURM_ARRAY_TASK_ID}"
WORLD_SIZE=2
BASE_SEED_DEFAULT=$((100000 + SLURM_ARRAY_JOB_ID * 10))
BASE_SEED="${BASE_SEED:-${BASE_SEED_DEFAULT}}"
SEED=$((BASE_SEED + ENSEMBLE_ID))
SHORT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"

RUN_TAG="ddofs_wv_m40_depth_e4_ddp2_${SHORT_SHA}"
RUN_GROUP="${RUN_TAG}_job${SLURM_ARRAY_JOB_ID}_sbase${BASE_SEED}"
WANDB_NAME="${RUN_TAG}_ens${ENSEMBLE_ID}_seed${SEED}_ws${WORLD_SIZE}"
CKPT_ROOT="/home/$USER/GINO_Model/neuraloperator_no_physics/scripts/checkpoints_WV_depth_only_diffusion"
CKPT_DIR="${CKPT_ROOT}/${RUN_GROUP}/ens${ENSEMBLE_ID}"
mkdir -p "${CKPT_DIR}"

slurm_configure_host_ca

for key_path in "${PROJECT_DIR}/config/wandb_api_key.txt" "/scratch/$USER/Data_Generation_UQ/GINO_Model/neuraloperator_no_physics/config/wandb_api_key.txt"; do
  if [[ -f "${key_path}" ]]; then
    export APPTAINERENV_WANDB_API_KEY="$(head -n 1 "${key_path}" | tr -d "\r")"
    break
  fi
done

export OMP_NUM_THREADS=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((13000 + SLURM_JOB_ID % 20000))

echo "Training script: ${TRAIN_SCRIPT}"
echo "Config:          ${TRAIN_CONFIG}"
echo "Data root:       ${DATA_ROOT}"
echo "Git commit:      $(git -C "${PROJECT_DIR}" rev-parse HEAD)"
echo "Ensemble ID:     ${ENSEMBLE_ID}"
echo "Base seed:       ${BASE_SEED}"
echo "Seed:            ${SEED}"
echo "World size:      ${WORLD_SIZE}"
echo "Checkpoint dir:  ${CKPT_DIR}"
echo "W&B group:       ${RUN_GROUP}"
echo "W&B name:        ${WANDB_NAME}"

apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" \
  torchrun --standalone --nnodes=1 --nproc_per_node=${WORLD_SIZE} "${TRAIN_SCRIPT}" \
  --config_path "${TRAIN_CONFIG}" \
  --data.root "${DATA_ROOT}" \
  --rollout_data.root "${DATA_ROOT}" \
  --distributed.use_distributed true \
  --distributed.seed "${SEED}" \
  --checkpoint.save_dir "${CKPT_DIR}" \
  --wandb.log true \
  --wandb.group "${RUN_GROUP}" \
  --wandb.name "${WANDB_NAME}"
