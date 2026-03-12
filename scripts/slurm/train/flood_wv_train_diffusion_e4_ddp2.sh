#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 72:00:00
#SBATCH -J ddofs_wv_e4_ddp2
#SBATCH --array=0-3
#SBATCH -o runtime/logs/out/ddofs_wv_e4_ddp2-%A_%a.out
#SBATCH -e runtime/logs/err/ddofs_wv_e4_ddp2-%A_%a.err

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
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/flood_wv_train_diffusion.py"
TRAIN_CONFIG="${PROJECT_DIR}/config/flood/wv/gino_pluvial_flood_config_WV_depth_only_diffusion.yaml"
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
WORLD_SIZE=2
BASE_SEED_DEFAULT=$((100000 + SLURM_ARRAY_JOB_ID * 10))
BASE_SEED="${BASE_SEED:-${BASE_SEED_DEFAULT}}"
SEED=$((BASE_SEED + ENSEMBLE_ID))
SHORT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"

RUN_TAG="ddofs_wv_m40_depth_e4_ddp2_${SHORT_SHA}"
RUN_GROUP="${RUN_TAG}_job${SLURM_ARRAY_JOB_ID}_sbase${BASE_SEED}"
WANDB_NAME="${RUN_TAG}_ens${ENSEMBLE_ID}_seed${SEED}_ws${WORLD_SIZE}"
LEARNING_RATE_OVERRIDE="${LEARNING_RATE_OVERRIDE:-}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
N_SAMPLES_MAX_OVERRIDE="${N_SAMPLES_MAX_OVERRIDE:-}"
MAX_VAL_BATCHES_OVERRIDE="${MAX_VAL_BATCHES_OVERRIDE:-}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_DIR}/scripts/runtime/checkpoints_WV_depth_only_diffusion}"
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

slurm_assert_container_gpus "${CONTAINER_PATH}" "${WORLD_SIZE}"

echo "Training script: ${TRAIN_SCRIPT}"
echo "Config:          ${TRAIN_CONFIG}"
echo "Data root:       ${DATA_ROOT}"
echo "Rollout root:    ${ROLLOUT_ROOT}"
echo "Boundary source: ${BOUNDARY_SOURCE}"
if [[ "${BOUNDARY_SOURCE}" == "clean_family" ]]; then
  echo "Clean boundary:  ${CLEAN_BOUNDARY_ROOT}/${CLEAN_BOUNDARY_FILE}"
fi
echo "Git commit:      $(git -C "${PROJECT_DIR}" rev-parse HEAD)"
echo "Ensemble ID:     ${ENSEMBLE_ID}"
echo "Base seed:       ${BASE_SEED}"
echo "Seed:            ${SEED}"
echo "World size:      ${WORLD_SIZE}"
echo "Checkpoint dir:  ${CKPT_DIR}"
echo "W&B group:       ${RUN_GROUP}"
echo "W&B name:        ${WANDB_NAME}"
if [[ -n "${LEARNING_RATE_OVERRIDE}" ]]; then
  echo "Learning rate:   ${LEARNING_RATE_OVERRIDE}"
fi
if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
  echo "Epoch override:  ${EPOCHS_OVERRIDE}"
fi
if [[ -n "${N_SAMPLES_MAX_OVERRIDE}" ]]; then
  echo "Sample cap:      ${N_SAMPLES_MAX_OVERRIDE}"
fi
if [[ -n "${MAX_VAL_BATCHES_OVERRIDE}" ]]; then
  echo "Max val batches: ${MAX_VAL_BATCHES_OVERRIDE}"
fi

CLI_ARGS=(
  --config_path "${TRAIN_CONFIG}"
  --data.root "${DATA_ROOT}"
  --rollout_data.root "${ROLLOUT_ROOT}"
  --data.normalizer_root "${NORMALIZER_ROOT}"
  --distributed.use_distributed true
  --distributed.seed "${SEED}"
  --checkpoint.save_dir "${CKPT_DIR}"
  --wandb.log true
  --wandb.group "${RUN_GROUP}"
  --wandb.name "${WANDB_NAME}"
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

if [[ -n "${BATCH_SIZE_OVERRIDE:-}" ]]; then
  CLI_ARGS+=(--data.batch_size "${BATCH_SIZE_OVERRIDE}")
fi

if [[ -n "${LEARNING_RATE_OVERRIDE}" ]]; then
  CLI_ARGS+=(--opt.learning_rate "${LEARNING_RATE_OVERRIDE}")
fi

if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
  CLI_ARGS+=(--opt.n_epochs "${EPOCHS_OVERRIDE}")
fi

if [[ -n "${N_SAMPLES_MAX_OVERRIDE}" ]]; then
  CLI_ARGS+=(--data.n_samples_max "${N_SAMPLES_MAX_OVERRIDE}")
fi

if [[ -n "${MAX_VAL_BATCHES_OVERRIDE}" ]]; then
  CLI_ARGS+=(--max_val_batches "${MAX_VAL_BATCHES_OVERRIDE}")
fi

apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" \
  torchrun --standalone --nnodes=1 --nproc_per_node=${WORLD_SIZE} "${TRAIN_SCRIPT}" "${CLI_ARGS[@]}"
