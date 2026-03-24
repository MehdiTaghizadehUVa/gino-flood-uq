#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J coastal_fgn_gs30
#SBATCH --array=0-95%8
#SBATCH -o runtime/logs/out/coastal_fgn_gs30-%A_%a.out
#SBATCH -e runtime/logs/err/coastal_fgn_gs30-%A_%a.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_no_physics_git_main}"
CANONICAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
COMMON_SH=""
for candidate in \
  "${PROJECT_DIR}/scripts/slurm/lib/common.sh" \
  "${SLURM_SUBMIT_DIR:-}/slurm/lib/common.sh" \
  "${SLURM_SUBMIT_DIR:-}/scripts/slurm/lib/common.sh" \
  "$(cd "${CANONICAL_DIR}/.." && pwd)/lib/common.sh"
do
  if [[ -n "${candidate}" && -f "${candidate}" ]]; then
    COMMON_SH="${candidate}"
    break
  fi
done

if [[ -z "${COMMON_SH}" ]]; then
  echo "ERROR: Unable to locate scripts/slurm/lib/common.sh" >&2
  echo "Checked PROJECT_DIR=${PROJECT_DIR}, SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}, CANONICAL_DIR=${CANONICAL_DIR}" >&2
  exit 1
fi

source "${COMMON_SH}"
if [[ -d "${PROJECT_DIR}/scripts" ]]; then
  SCRIPT_DIR="${PROJECT_DIR}/scripts"
else
  SCRIPT_DIR="$(slurm_resolve_scripts_root "${CANONICAL_DIR}")"
fi
slurm_load_apptainer
cd "${SCRIPT_DIR}"
mkdir -p runtime/logs/out runtime/logs/err runtime/checkpoints

TRAIN_SCRIPT="${PROJECT_DIR}/scripts/flood_wv_train_operator.py"
BASE_CONFIG="${BASE_CONFIG:-${PROJECT_DIR}/config/flood/coastal/gino_pluvial_flood_config_coastal_depth_only_fgn_grid.yaml}"
CONTAINER_PATH="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
DATA_ROOT="${DATA_ROOT:-/scratch/jrj6wm/uncertainty_floodmodel_linux/results/coastal/Coastal_Flood_coastal_v1_5k_train_prod_t2_w64_20260318_233556/train}"
CLEAN_BOUNDARY_ROOT="${CLEAN_BOUNDARY_ROOT:-/scratch/jrj6wm/uncertainty_floodmodel_linux/synthetic/coastal/dynamic_v1_5k}"
N_EPOCHS_OVERRIDE="${N_EPOCHS_OVERRIDE:-30}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-}"
N_SAMPLES_MAX_OVERRIDE="${N_SAMPLES_MAX_OVERRIDE:-}"
FIXED_SEED="${FIXED_SEED:-123}"
WANDB_LOG="${WANDB_LOG:-true}"
GRID_INDEX="${SLURM_ARRAY_TASK_ID}"

export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"

SHORT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
SEARCH_ROOT="${SEARCH_ROOT:-${PROJECT_DIR}/scripts/runtime/coastal_fgn_grid_30ep_${SHORT_SHA}_job${SLURM_ARRAY_JOB_ID}}"
CONFIG_ROOT="${SEARCH_ROOT}/configs"
CHECKPOINT_ROOT="${SEARCH_ROOT}/checkpoints"
mkdir -p "${CONFIG_ROOT}" "${CHECKPOINT_ROOT}"

slurm_configure_host_ca
slurm_assert_container_gpus "${CONTAINER_PATH}" 1

for key_path in "${PROJECT_DIR}/config/wandb_api_key.txt" "/scratch/$USER/Data_Generation_UQ/GINO_Model/neuraloperator_no_physics/config/wandb_api_key.txt"; do
  if [[ -f "${key_path}" ]]; then
    export APPTAINERENV_WANDB_API_KEY="$(head -n 1 "${key_path}" | tr -d "\r")"
    break
  fi
done

eval "$({
  apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python -m neuralop.flood.cli.coastal_fgn_gridsearch \
    describe --index "${GRID_INDEX}" --format shell
})"

RUN_TAG="${GS_RUN_TAG}"
WANDB_GROUP="coastal_fgn_grid_30ep_${SHORT_SHA}"
WANDB_NAME="${RUN_TAG}"
CKPT_DIR="${CHECKPOINT_ROOT}/${RUN_TAG}"
SHARED_NORMALIZER_ROOT="${SHARED_NORMALIZER_ROOT:-${SEARCH_ROOT}/shared_normalizers}"
NORMALIZER_ROOT="${SHARED_NORMALIZER_ROOT}"
SHARED_NORMALIZER_PATH="${NORMALIZER_ROOT}/normalizers_depth_only.pt"
CONFIG_PATH="${CONFIG_ROOT}/${RUN_TAG}.yaml"
SUMMARY_PATH="${CKPT_DIR}/run_summary.json"
TRAIN_LOG_PATH="${CKPT_DIR}/training.log"
mkdir -p "${CKPT_DIR}" "${NORMALIZER_ROOT}"

RUN_STATUS="failed"
finalize() {
  set +e
  apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python -m neuralop.flood.cli.coastal_fgn_gridsearch \
    summarize \
    --index "${GRID_INDEX}" \
    --summary-path "${SUMMARY_PATH}" \
    --log-path "${TRAIN_LOG_PATH}" \
    --config-path "${CONFIG_PATH}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --status "${RUN_STATUS}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}" \
    --array-task-id "${SLURM_ARRAY_TASK_ID:-}" \
    --git-sha "${SHORT_SHA}" \
    --run-tag "${RUN_TAG}"
}
trap finalize EXIT

RENDER_ARGS=(
  render-config
  --index "${GRID_INDEX}"
  --base-config "${BASE_CONFIG}"
  --output-config "${CONFIG_PATH}"
  --checkpoint-dir "${CKPT_DIR}"
  --normalizer-root "${NORMALIZER_ROOT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-name "${WANDB_NAME}"
  --data-root "${DATA_ROOT}"
  --clean-boundary-root "${CLEAN_BOUNDARY_ROOT}"
  --n-epochs "${N_EPOCHS_OVERRIDE}"
  --seed "${FIXED_SEED}"
  --deterministic false
  --wandb-log "${WANDB_LOG}"
)
if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
  RENDER_ARGS+=(--batch-size "${BATCH_SIZE_OVERRIDE}")
fi
if [[ -n "${N_SAMPLES_MAX_OVERRIDE}" ]]; then
  RENDER_ARGS+=(--n-samples-max "${N_SAMPLES_MAX_OVERRIDE}")
fi
apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python -m neuralop.flood.cli.coastal_fgn_gridsearch "${RENDER_ARGS[@]}"

echo "Training script: ${TRAIN_SCRIPT}"
echo "Base config:     ${BASE_CONFIG}"
echo "Run config:      ${CONFIG_PATH}"
echo "Data root:       ${DATA_ROOT}"
echo "Clean boundary:  ${CLEAN_BOUNDARY_ROOT}"
echo "Git commit:      $(git -C "${PROJECT_DIR}" rev-parse HEAD)"
echo "Grid index:      ${GRID_INDEX}"
echo "Run tag:         ${RUN_TAG}"
echo "Seed:            ${FIXED_SEED}"
echo "Batch size:      ${BATCH_SIZE_OVERRIDE:-<base config>}"
echo "Checkpoint dir:  ${CKPT_DIR}"
echo "Normalizer path: ${SHARED_NORMALIZER_PATH}"
echo "Summary path:    ${SUMMARY_PATH}"
echo "Hyperparameters: lr=${GS_LR}, wd=${GS_WD}, radius=${GS_RADIUS}, hidden=${GS_HIDDEN}, noise_dim=${GS_NOISE_DIM}, epochs=${N_EPOCHS_OVERRIDE}"
echo "Sample cap:      ${N_SAMPLES_MAX_OVERRIDE:-<full dataset>}"

if [[ ! -f "${SHARED_NORMALIZER_PATH}" ]]; then
  echo "ERROR: Shared normalizer not found at ${SHARED_NORMALIZER_PATH}. Precompute it before launching the array." >&2
  exit 1
fi

apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${TRAIN_SCRIPT}" --config_path "${CONFIG_PATH}"
RUN_STATUS="completed"
