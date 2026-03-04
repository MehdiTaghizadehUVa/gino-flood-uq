#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -J gino_gauss_e4
#SBATCH --array=0-3
#SBATCH -o logs/out/gino_gauss_e4-%A_%a.out
#SBATCH -e logs/err/gino_gauss_e4-%A_%a.err

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
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/train_gino_flood_train_rollout_animation_WV.py"
TRAIN_CONFIG="${PROJECT_DIR}/config/gino_pluvial_flood_config_WV_depth_only_gaussian.yaml"
CONTAINER_PATH="/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif"
DATA_ROOT="/scratch/$USER/Data_Generation_UQ/Results/M40"

ENSEMBLE_ID="${SLURM_ARRAY_TASK_ID}"
BASE_SEED=123
SEED=$((BASE_SEED + ENSEMBLE_ID))
SHORT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"

# Keep model/training config identical to latest full Gaussian run; vary only seed/member ID.
RUN_TAG_BASE="wv_m40_gaussianNLL_ep150_lr1e-4_gr0.1_h64_wd1e-4_${SHORT_SHA}"
WANDB_GROUP="${RUN_TAG_BASE}_job${SLURM_ARRAY_JOB_ID}"
WANDB_NAME="${RUN_TAG_BASE}_ens${ENSEMBLE_ID}_seed${SEED}"
CKPT_ROOT="${PROJECT_DIR}/scripts/checkpoints_WV_depth_only_gaussian"
CKPT_DIR="${CKPT_ROOT}/${WANDB_GROUP}/ens${ENSEMBLE_ID}"
mkdir -p "${CKPT_DIR}"

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

echo "Training script: ${TRAIN_SCRIPT}"
echo "Config:          ${TRAIN_CONFIG}"
echo "Data root:       ${DATA_ROOT}"
echo "Git commit:      $(git -C "${PROJECT_DIR}" rev-parse HEAD)"
echo "Ensemble ID:     ${ENSEMBLE_ID}"
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
