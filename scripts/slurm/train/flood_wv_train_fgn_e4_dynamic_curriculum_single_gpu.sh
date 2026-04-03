#!/bin/bash
#SBATCH -A uqgroup_plus
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -J wv_fgn_dyn_e4_s1
#SBATCH --array=0-3
#SBATCH -o runtime/logs/out/wv_fgn_dyn_e4_s1-%A_%a.out
#SBATCH -e runtime/logs/err/wv_fgn_dyn_e4_s1-%A_%a.err

set -euo pipefail

PROJECT_DIR_DEFAULT="/home/$USER/GINO_Model/neuraloperator_no_physics_git_main"
PROJECT_DIR="${PROJECT_DIR:-${PROJECT_DIR_DEFAULT}}"
if [[ -f "${PROJECT_DIR}/scripts/slurm/lib/common.sh" ]]; then
  source "${PROJECT_DIR}/scripts/slurm/lib/common.sh"
  SCRIPT_DIR="${PROJECT_DIR}/scripts"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/lib/common.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  source "${SCRIPT_DIR}/slurm/lib/common.sh"
else
  CANONICAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  source "$(cd "${CANONICAL_DIR}/.." && pwd)/lib/common.sh"
  SCRIPT_DIR="$(slurm_resolve_scripts_root "${CANONICAL_DIR}")"
fi
slurm_load_apptainer
cd "${SCRIPT_DIR}"
mkdir -p runtime/logs/out runtime/logs/err runtime/checkpoints

TRAIN_SCRIPT="${PROJECT_DIR}/scripts/flood_wv_train_operator.py"
TRAIN_CONFIG="${TRAIN_CONFIG:?TRAIN_CONFIG must point to the rendered 250-epoch WV config}"
CONTAINER_PATH="/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/train}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${DATA_ROOT}}"
NORMALIZER_ROOT="${NORMALIZER_ROOT:-${DATA_ROOT}}"
CLEAN_BOUNDARY_ROOT="${CLEAN_BOUNDARY_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/metadata}"
CLEAN_BOUNDARY_FILE="${CLEAN_BOUNDARY_FILE:-Hydrographs_Train_Clean.txt}"
TRAIN_TXT_NAME="${TRAIN_TXT_NAME:-train.txt}"
WORLD_SIZE=1
ENSEMBLE_ID="${SLURM_ARRAY_TASK_ID}"
BASE_SEED_DEFAULT=$((100000 + SLURM_ARRAY_JOB_ID * 10))
BASE_SEED="${BASE_SEED:-${BASE_SEED_DEFAULT}}"
SEED=$((BASE_SEED + ENSEMBLE_ID))
SHORT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
RUN_GROUP_PREFIX="${RUN_GROUP_PREFIX:-wdonly_fgn_ws1_dynamic250_ar150s2e25}"
RUN_GROUP="${RUN_GROUP_PREFIX}_${SHORT_SHA}_job${SLURM_ARRAY_JOB_ID}_sbase${BASE_SEED}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_DIR}/scripts/runtime/checkpoints_WV_depth_only_250ep_dynamic_ensemble_single_gpu}"
RUN_TAG="${RUN_GROUP}_ens${ENSEMBLE_ID}"
CKPT_DIR="${CKPT_ROOT}/${RUN_TAG}"
mkdir -p "${CKPT_DIR}"
WANDB_GROUP="${RUN_GROUP}"
WANDB_NAME="${RUN_GROUP}_ens${ENSEMBLE_ID}_seed${SEED}_ws${WORLD_SIZE}"
WANDB_LOG="${WANDB_LOG:-true}"

slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
export OMP_NUM_THREADS=8

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
else
  WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/.config/wandb_api_key.txt}"
  if [[ -f "${WANDB_KEY_FILE}" ]]; then
    IFS= read -r APPTAINERENV_WANDB_API_KEY < "${WANDB_KEY_FILE}"
    export APPTAINERENV_WANDB_API_KEY
  fi
fi

slurm_assert_container_gpus "${CONTAINER_PATH}" "${WORLD_SIZE}"

echo "Training script: ${TRAIN_SCRIPT}"
echo "Config:          ${TRAIN_CONFIG}"
echo "Data root:       ${DATA_ROOT}"
echo "Rollout root:    ${ROLLOUT_ROOT}"
echo "Boundary source: clean_family"
echo "Train txt:       ${TRAIN_TXT_NAME}"
echo "Clean boundary:  ${CLEAN_BOUNDARY_ROOT}/${CLEAN_BOUNDARY_FILE}"
echo "Git commit:      $(git -C "${PROJECT_DIR}" rev-parse HEAD)"
echo "Ensemble id:     ${ENSEMBLE_ID}"
echo "Base seed:       ${BASE_SEED}"
echo "Seed:            ${SEED}"
echo "World size:      ${WORLD_SIZE}"
echo "Checkpoint dir:  ${CKPT_DIR}"
echo "W&B logging:     ${WANDB_LOG}"
echo "W&B group:       ${WANDB_GROUP}"
echo "W&B name:        ${WANDB_NAME}"
echo "Requested AR curriculum: start_epoch=150, start_steps=2, step_every=25 epochs, max_steps=5"
echo "Training verification disabled for this launcher to avoid the deterministic CUDA bicubic backward check aborting before training starts."

CLI_ARGS=(
  --config_path "${TRAIN_CONFIG}"
  --data.root "${DATA_ROOT}"
  --rollout_data.root "${ROLLOUT_ROOT}"
  --data.normalizer_root "${NORMALIZER_ROOT}"
  --data.train_txt "${TRAIN_TXT_NAME}"
  --distributed.use_distributed false
  --distributed.seed "${SEED}"
  --wandb.log "${WANDB_LOG}"
  --wandb.group "${WANDB_GROUP}"
  --wandb.name "${WANDB_NAME}"
  --checkpoint.save_dir "${CKPT_DIR}"
  --data.boundary_source clean_family
  --data.clean_boundary_root "${CLEAN_BOUNDARY_ROOT}"
  --data.clean_boundary_file "${CLEAN_BOUNDARY_FILE}"
  --rollout_data.boundary_source clean_family
  --rollout_data.clean_boundary_root "${CLEAN_BOUNDARY_ROOT}"
  --rollout_data.clean_boundary_file "${CLEAN_BOUNDARY_FILE}"
  --verify_training false
)

apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" \
  python3 "${TRAIN_SCRIPT}" "${CLI_ARGS[@]}"
