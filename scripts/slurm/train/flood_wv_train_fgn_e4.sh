#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -J gino_fgn_e4
#SBATCH --array=0-3
#SBATCH -o runtime/logs/out/gino_fgn_e4-%A_%a.out
#SBATCH -e runtime/logs/err/gino_fgn_e4-%A_%a.err

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
mkdir -p runtime/logs/out runtime/logs/err runtime/checkpoints

PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_no_physics_git_main}"
export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/flood_wv_train_operator.py"
TRAIN_CONFIG="${PROJECT_DIR}/config/flood/wv/gino_pluvial_flood_config_WV_depth_only.yaml"
CONTAINER_PATH="/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/Data_Generation_UQ/Results/M40}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${DATA_ROOT}}"
NORMALIZER_ROOT="${NORMALIZER_ROOT:-${DATA_ROOT}}"
BOUNDARY_SOURCE="${BOUNDARY_SOURCE:-member_hdf}"
ROLLOUT_BOUNDARY_SOURCE="${ROLLOUT_BOUNDARY_SOURCE:-${BOUNDARY_SOURCE}}"
CLEAN_BOUNDARY_ROOT="${CLEAN_BOUNDARY_ROOT:-}"
CLEAN_BOUNDARY_FILE="${CLEAN_BOUNDARY_FILE:-}"
ROLLOUT_CLEAN_BOUNDARY_ROOT="${ROLLOUT_CLEAN_BOUNDARY_ROOT:-${CLEAN_BOUNDARY_ROOT}}"
ROLLOUT_CLEAN_BOUNDARY_FILE="${ROLLOUT_CLEAN_BOUNDARY_FILE:-${CLEAN_BOUNDARY_FILE}}"

ENSEMBLE_ID="${SLURM_ARRAY_TASK_ID}"
# Always use fresh seed blocks per submitted array job by default.
# Optional override: sbatch --export=ALL,BASE_SEED=<value> ...
BASE_SEED_DEFAULT=$((100000 + SLURM_ARRAY_JOB_ID * 10))
BASE_SEED="${BASE_SEED:-${BASE_SEED_DEFAULT}}"
SEED=$((BASE_SEED + ENSEMBLE_ID))
SHORT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"

# Keep model/training config + overrides identical to latest full FGN run.
RUN_GROUP="wdonly_ep300_lr2e-4_gr0.1_h64_wd5e-4_${SHORT_SHA}_job${SLURM_ARRAY_JOB_ID}_sbase${BASE_SEED}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_DIR}/scripts/runtime/checkpoints_WV_depth_only_300ep}"
RUN_TAG="${RUN_GROUP}_ens${ENSEMBLE_ID}"
CKPT_DIR="${CKPT_ROOT}/${RUN_TAG}"
mkdir -p "${CKPT_DIR}"
WANDB_GROUP="${RUN_GROUP}"
WANDB_NAME="${RUN_GROUP}_ens${ENSEMBLE_ID}_seed${SEED}"

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
echo "Rollout root:    ${ROLLOUT_ROOT}"
echo "Boundary source: ${BOUNDARY_SOURCE}"
if [[ "${BOUNDARY_SOURCE}" == "clean_family" ]]; then
  echo "Clean boundary:  ${CLEAN_BOUNDARY_ROOT}/${CLEAN_BOUNDARY_FILE}"
fi
echo "Git commit:      $(git -C "${PROJECT_DIR}" rev-parse HEAD)"
echo "Ensemble id:     ${ENSEMBLE_ID}"
echo "Base seed:       ${BASE_SEED}"
echo "Seed:            ${SEED}"
echo "Checkpoint dir:  ${CKPT_DIR}"
echo "W&B group:       ${WANDB_GROUP}"
echo "W&B name:        ${WANDB_NAME}"
echo "Hyperparameters: n_epochs=300, lr=2e-4, gno_radius=0.1, hidden=64, weight_decay=5e-4"

CLI_ARGS=(
  --config_path "${TRAIN_CONFIG}"
  --data.root "${DATA_ROOT}"
  --rollout_data.root "${ROLLOUT_ROOT}"
  --data.normalizer_root "${NORMALIZER_ROOT}"
  --distributed.seed "${SEED}"
  --wandb.log true
  --wandb.group "${WANDB_GROUP}"
  --wandb.name "${WANDB_NAME}"
  --opt.n_epochs 300
  --opt.learning_rate 0.00020
  --gino.gno_radius 0.1
  --gino.fno_hidden_channels 64
  --opt.weight_decay 0.00050
  --checkpoint.save_dir "${CKPT_DIR}"
)

if [[ "${BOUNDARY_SOURCE}" == "clean_family" ]]; then
  if [[ -z "${CLEAN_BOUNDARY_ROOT}" || -z "${CLEAN_BOUNDARY_FILE}" ]]; then
    echo "ERROR: clean_family requires CLEAN_BOUNDARY_ROOT and CLEAN_BOUNDARY_FILE" >&2
    exit 2
  fi
  CLI_ARGS+=(
    --data.boundary_source clean_family
    --data.clean_boundary_root "${CLEAN_BOUNDARY_ROOT}"
    --data.clean_boundary_file "${CLEAN_BOUNDARY_FILE}"
  )
fi

if [[ "${ROLLOUT_BOUNDARY_SOURCE}" == "clean_family" ]]; then
  if [[ -z "${ROLLOUT_CLEAN_BOUNDARY_ROOT}" || -z "${ROLLOUT_CLEAN_BOUNDARY_FILE}" ]]; then
    echo "ERROR: rollout clean_family requires ROLLOUT_CLEAN_BOUNDARY_ROOT and ROLLOUT_CLEAN_BOUNDARY_FILE" >&2
    exit 2
  fi
  CLI_ARGS+=(
    --rollout_data.boundary_source clean_family
    --rollout_data.clean_boundary_root "${ROLLOUT_CLEAN_BOUNDARY_ROOT}"
    --rollout_data.clean_boundary_file "${ROLLOUT_CLEAN_BOUNDARY_FILE}"
  )
fi

apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${TRAIN_SCRIPT}" "${CLI_ARGS[@]}"
