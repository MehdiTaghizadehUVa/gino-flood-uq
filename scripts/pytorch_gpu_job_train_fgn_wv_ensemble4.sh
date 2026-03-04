#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -J gino_fgn_e4
#SBATCH --array=0-3
#SBATCH -o logs/out/gino_fgn_e4-%A_%a.out
#SBATCH -e logs/err/gino_fgn_e4-%A_%a.err

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
TRAIN_CONFIG="${PROJECT_DIR}/config/gino_pluvial_flood_config_WV_depth_only.yaml"
CONTAINER_PATH="/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif"
DATA_ROOT="/scratch/$USER/Data_Generation_UQ/Results/M40"

ENSEMBLE_ID="${SLURM_ARRAY_TASK_ID}"
BASE_SEED=123
SEED=$((BASE_SEED + ENSEMBLE_ID))
SHORT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"

# Keep model/training config + overrides identical to latest full FGN run.
RUN_GROUP="wdonly_ep300_lr2e-4_gr0.1_h64_wd5e-4_${SHORT_SHA}_job${SLURM_ARRAY_JOB_ID}"
CKPT_ROOT="${PROJECT_DIR}/scripts/checkpoints_WV_depth_only_300ep"
RUN_TAG="${RUN_GROUP}_ens${ENSEMBLE_ID}"
CKPT_DIR="${CKPT_ROOT}/${RUN_TAG}"
mkdir -p "${CKPT_DIR}"
WANDB_GROUP="${RUN_GROUP}"
WANDB_NAME="${RUN_GROUP}_ens${ENSEMBLE_ID}_seed${SEED}"

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
echo "Ensemble id:     ${ENSEMBLE_ID}"
echo "Seed:            ${SEED}"
echo "Checkpoint dir:  ${CKPT_DIR}"
echo "W&B group:       ${WANDB_GROUP}"
echo "W&B name:        ${WANDB_NAME}"
echo "Hyperparameters: n_epochs=300, lr=2e-4, gno_radius=0.1, hidden=64, weight_decay=5e-4"

apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${TRAIN_SCRIPT}" \
  --config_path "${TRAIN_CONFIG}" \
  --data.root "${DATA_ROOT}" \
  --rollout_data.root "${DATA_ROOT}" \
  --distributed.seed "${SEED}" \
  --wandb.log true \
  --wandb.group "${WANDB_GROUP}" \
  --wandb.name "${WANDB_NAME}" \
  --opt.n_epochs 300 \
  --opt.learning_rate 0.00020 \
  --gino.gno_radius 0.1 \
  --gino.fno_hidden_channels 64 \
  --opt.weight_decay 0.00050 \
  --checkpoint.save_dir "${CKPT_DIR}"
