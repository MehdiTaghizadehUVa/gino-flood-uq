#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 72:00:00
#SBATCH -J ddofs_wv_e4_ddpMN
#SBATCH --array=0-3
#SBATCH -o logs/out/ddofs_wv_e4_ddpMN-%A_%a.out
#SBATCH -e logs/err/ddofs_wv_e4_ddpMN-%A_%a.err

# Multi-node template for diffusion DDP training.
# Tune partition/constraints for your cluster before production usage.

set -euo pipefail
module purge
module load apptainer

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
cd "${SCRIPT_DIR}"
mkdir -p logs/out logs/err

PROJECT_DIR="/home/$USER/GINO_Model/neuraloperator_no_physics"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/train_diffusion_forecaster_WV.py"
TRAIN_CONFIG="${PROJECT_DIR}/config/gino_pluvial_flood_config_WV_depth_only_diffusion.yaml"
CONTAINER_PATH="/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif"
DATA_ROOT="/scratch/$USER/Data_Generation_UQ/Results/M40"

ENSEMBLE_ID="${SLURM_ARRAY_TASK_ID}"
GPUS_PER_NODE=2
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
BASE_SEED_DEFAULT=$((200000 + SLURM_ARRAY_JOB_ID * 10))
BASE_SEED="${BASE_SEED:-${BASE_SEED_DEFAULT}}"
SEED=$((BASE_SEED + ENSEMBLE_ID))
SHORT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"

RUN_TAG="ddofs_wv_m40_depth_e4_ddpMN_${SHORT_SHA}"
RUN_GROUP="${RUN_TAG}_job${SLURM_ARRAY_JOB_ID}_sbase${BASE_SEED}"
WANDB_NAME="${RUN_TAG}_ens${ENSEMBLE_ID}_seed${SEED}_ws${WORLD_SIZE}"
CKPT_ROOT="/home/$USER/GINO_Model/neuraloperator_no_physics/scripts/checkpoints_WV_depth_only_diffusion"
CKPT_DIR="${CKPT_ROOT}/${RUN_GROUP}/ens${ENSEMBLE_ID}"
mkdir -p "${CKPT_DIR}"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT=$((14000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT

HOST_CA_BUNDLE=""
for cand in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt; do
  if [[ -f "$cand" ]]; then
    HOST_CA_BUNDLE="$cand"
    break
  fi
done
APPTAINER_BIND_ARGS="--nv"
if [[ -n "${HOST_CA_BUNDLE}" ]]; then
  APPTAINER_BIND_ARGS="--nv --bind ${HOST_CA_BUNDLE}:/host_ca_bundle.crt:ro"
  export APPTAINERENV_SSL_CERT_FILE=/host_ca_bundle.crt
  export APPTAINERENV_REQUESTS_CA_BUNDLE=/host_ca_bundle.crt
fi

for key_path in "${PROJECT_DIR}/config/wandb_api_key.txt" "/scratch/$USER/Data_Generation_UQ/GINO_Model/neuraloperator_no_physics/config/wandb_api_key.txt"; do
  if [[ -f "${key_path}" ]]; then
    export APPTAINERENV_WANDB_API_KEY="$(head -n 1 "${key_path}" | tr -d "\r")"
    break
  fi
done

export OMP_NUM_THREADS=8

echo "Training script: ${TRAIN_SCRIPT}"
echo "Config:          ${TRAIN_CONFIG}"
echo "Data root:       ${DATA_ROOT}"
echo "Git commit:      $(git -C "${PROJECT_DIR}" rev-parse HEAD)"
echo "Ensemble ID:     ${ENSEMBLE_ID}"
echo "Seed:            ${SEED}"
echo "World size:      ${WORLD_SIZE}"
echo "Master endpoint: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Checkpoint dir:  ${CKPT_DIR}"
echo "W&B group:       ${RUN_GROUP}"
echo "W&B name:        ${WANDB_NAME}"

srun --ntasks=${SLURM_NNODES} apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" \
  torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  "${TRAIN_SCRIPT}" \
  --config_path "${TRAIN_CONFIG}" \
  --data.root "${DATA_ROOT}" \
  --rollout_data.root "${DATA_ROOT}" \
  --distributed.use_distributed true \
  --distributed.seed "${SEED}" \
  --checkpoint.save_dir "${CKPT_DIR}" \
  --wandb.log true \
  --wandb.group "${RUN_GROUP}" \
  --wandb.name "${WANDB_NAME}"
