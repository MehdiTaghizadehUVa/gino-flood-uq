#!/bin/bash
set -euo pipefail

PROJECT_DIR_DEFAULT="/home/$USER/GINO_Model/neuraloperator_no_physics_git_main"
PROJECT_DIR="${PROJECT_DIR:-${PROJECT_DIR_DEFAULT}}"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
TRAIN_SLURM_DIR="${SCRIPTS_DIR}/slurm/train"
EVAL_SLURM_DIR="${SCRIPTS_DIR}/slurm/eval"

RUN_GROUP_DEFAULT="wv_fgn_masked_primary_$(date +%Y%m%d_%H%M%S)_$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
RUN_GROUP="${RUN_GROUP:-${RUN_GROUP_DEFAULT}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/scripts/runtime/checkpoints_WV_depth_only_300ep_masked_primary/${RUN_GROUP}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${RUN_ROOT}/artifacts}"
CANARY_DIR="${CANARY_DIR:-${RUN_ROOT}/canary}"
TRAIN_DIR="${TRAIN_DIR:-${RUN_ROOT}/train}"
EVAL_DIR="${EVAL_DIR:-${RUN_ROOT}/eval_outputs}"

DATA_ROOT="${DATA_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/train}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/test}"
SEED="${SEED:-123}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_DIR}/config/flood/wv/gino_pluvial_flood_config_WV_depth_only_masked_primary.yaml}"
PREP_TIME="${PREP_TIME:-02:00:00}"
CANARY_TIME="${CANARY_TIME:-02:00:00}"
TRAIN_TIME="${TRAIN_TIME:-72:00:00}"
EVAL_TIME="${EVAL_TIME:-24:00:00}"

mkdir -p "${RUN_ROOT}" "${ARTIFACT_DIR}" "${CANARY_DIR}" "${TRAIN_DIR}" "${EVAL_DIR}"

submit_job() {
  local desc="$1"
  shift
  local output
  output="$($@)"
  echo "${output}" >&2
  local job_id
  job_id="$(echo "${output}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    echo "ERROR: Failed to parse job id for ${desc}" >&2
    exit 1
  fi
  printf '%s' "${job_id}"
}

echo "Project dir:   ${PROJECT_DIR}"
echo "Run group:     ${RUN_GROUP}"
echo "Run root:      ${RUN_ROOT}"
echo "Artifact dir:  ${ARTIFACT_DIR}"
echo "Canary dir:    ${CANARY_DIR}"
echo "Train dir:     ${TRAIN_DIR}"
echo "Eval dir:      ${EVAL_DIR}"
echo "Data root:     ${DATA_ROOT}"
echo "Rollout root:  ${ROLLOUT_ROOT}"
echo "Seed:          ${SEED}"
echo "Config:        ${TRAIN_CONFIG}"
echo "Prep time:     ${PREP_TIME}"
echo "Canary time:   ${CANARY_TIME}"
echo "Train time:    ${TRAIN_TIME}"
echo "Eval time:     ${EVAL_TIME}"

PREP_JOB_ID="$(submit_job prep sbatch \
  --time=${PREP_TIME} \
  --export=ALL,PROJECT_DIR=${PROJECT_DIR},TRAIN_CONFIG=${TRAIN_CONFIG},DATA_ROOT=${DATA_ROOT},ARTIFACT_DIR=${ARTIFACT_DIR},SEED=${SEED},RUN_GROUP=${RUN_GROUP} \
  "${TRAIN_SLURM_DIR}/flood_wv_prepare_fgn_masked_primary.sh")"

CANARY_JOB_ID="$(submit_job canary sbatch \
  --dependency=afterok:${PREP_JOB_ID} \
  --time=${CANARY_TIME} \
  --job-name=wv_fgn_masked_canary \
  --export=ALL,PROJECT_DIR=${PROJECT_DIR},TRAIN_CONFIG=${TRAIN_CONFIG},DATA_ROOT=${DATA_ROOT},ROLLOUT_ROOT=${ROLLOUT_ROOT},ARTIFACT_DIR=${ARTIFACT_DIR},CHECKPOINT_DIR=${CANARY_DIR},SEED=${SEED},RUN_GROUP=${RUN_GROUP},MODE=canary,WANDB_LOG=false \
  "${TRAIN_SLURM_DIR}/flood_wv_train_fgn_masked_primary.sh")"

TRAIN_JOB_ID="$(submit_job train sbatch \
  --dependency=afterok:${CANARY_JOB_ID} \
  --time=${TRAIN_TIME} \
  --job-name=wv_fgn_masked_full \
  --export=ALL,PROJECT_DIR=${PROJECT_DIR},TRAIN_CONFIG=${TRAIN_CONFIG},DATA_ROOT=${DATA_ROOT},ROLLOUT_ROOT=${ROLLOUT_ROOT},ARTIFACT_DIR=${ARTIFACT_DIR},CHECKPOINT_DIR=${TRAIN_DIR},SEED=${SEED},RUN_GROUP=${RUN_GROUP},MODE=full,WANDB_LOG=true \
  "${TRAIN_SLURM_DIR}/flood_wv_train_fgn_masked_primary.sh")"

EVAL_JOB_ID="$(submit_job eval sbatch \
  --dependency=afterok:${TRAIN_JOB_ID} \
  --time=${EVAL_TIME} \
  --job-name=wv_fgn_masked_eval \
  --export=ALL,PROJECT_DIR=${PROJECT_DIR},EVAL_CONFIG=${TRAIN_CONFIG},TRAIN_ROOT=${DATA_ROOT},TEST_ROOT=${ROLLOUT_ROOT},ARTIFACT_DIR=${ARTIFACT_DIR},CHECKPOINT_ROOT=${TRAIN_DIR},OUT_DIR=${EVAL_DIR} \
  "${EVAL_SLURM_DIR}/flood_wv_eval_operator_masked_primary.sh")"

cat <<OUT
prep_job_id=${PREP_JOB_ID}
canary_job_id=${CANARY_JOB_ID}
train_job_id=${TRAIN_JOB_ID}
eval_job_id=${EVAL_JOB_ID}
run_root=${RUN_ROOT}
artifact_dir=${ARTIFACT_DIR}
canary_dir=${CANARY_DIR}
train_dir=${TRAIN_DIR}
eval_dir=${EVAL_DIR}
OUT
