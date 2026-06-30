#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
TRAIN_SLURM="${PROJECT_DIR}/scripts/slurm/train/flood_wv_train_mcdropout.sh"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_DIR}/config/flood/wv/gino_pluvial_flood_config_WV_depth_only_mcdropout.yaml}"
SHORT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
RUN_GROUP="${RUN_GROUP:-wv_mcdropout_legacy_h128_bs128_r48_lr5e4_${SHORT_SHA}_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/GINO_Model/neuraloperator_runs/wv_mcdropout/${RUN_GROUP}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${RUN_ROOT}/artifacts}"
SMOKE_ARTIFACT_ROOT="${SMOKE_ARTIFACT_ROOT:-${RUN_ROOT}/smoke/artifacts}"
SMOKE_CKPT_DIR="${SMOKE_CKPT_DIR:-${RUN_ROOT}/smoke/checkpoints}"
TRAIN_DIR="${TRAIN_DIR:-${RUN_ROOT}/checkpoints/train}"
NORMALIZER_ROOT="${NORMALIZER_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/train}"
NORMALIZER_FILE="${NORMALIZER_FILE:-normalizers_depth_only.pt}"
STRUCTURAL_DRY_POLICY="${STRUCTURAL_DRY_POLICY:-legacy_full_domain}"
BASE_SEED="${BASE_SEED:-26505001}"
SMOKE_TIME="${SMOKE_TIME:-02:00:00}"
TRAIN_TIME="${TRAIN_TIME:-3-00:00:00}"
SUBMIT_PRODUCTION="${SUBMIT_PRODUCTION:-1}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
PROD_DEPENDENCY="${PROD_DEPENDENCY:-}"
PROD_INITIAL_RESUME_FROM_DIR="${PROD_INITIAL_RESUME_FROM_DIR:-}"
PROD_EXCLUDE_NODES="${PROD_EXCLUDE_NODES:-}"
EPOCH_TARGETS="${EPOCH_TARGETS:-100 130 160 180 200}"
SMOKE_N_SAMPLES_MAX="${SMOKE_N_SAMPLES_MAX:-32}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
SMOKE_MC_SAMPLES="${SMOKE_MC_SAMPLES:-4}"
PROD_BATCH_SIZE="${PROD_BATCH_SIZE:-128}"
PROD_GRAD_ACCUM_STEPS="${PROD_GRAD_ACCUM_STEPS:-1}"
PROD_MC_SAMPLES="${PROD_MC_SAMPLES:-32}"

if [[ "${STRUCTURAL_DRY_POLICY}" != "legacy_full_domain" && "${STRUCTURAL_DRY_POLICY}" != "masked_primary" ]]; then
  echo "ERROR: unsupported STRUCTURAL_DRY_POLICY=${STRUCTURAL_DRY_POLICY}" >&2
  exit 2
fi
if [[ ! -f "${NORMALIZER_ROOT}/${NORMALIZER_FILE}" ]]; then
  echo "ERROR: dynamic train normalizer artifact not found: ${NORMALIZER_ROOT}/${NORMALIZER_FILE}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${ARTIFACT_ROOT}" "${SMOKE_ARTIFACT_ROOT}" "${SMOKE_CKPT_DIR}" "${TRAIN_DIR}"

submit_job() {
  local desc="$1"
  shift
  local output
  output="$($@)"
  echo "${output}" >&2
  local job_id
  job_id="$(echo "${output}" | awk "/Submitted batch job/ {print \$4}" | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    echo "ERROR: failed to parse job id for ${desc}" >&2
    exit 1
  fi
  printf "%s" "${job_id}"
}

cat <<INFO
Project dir:           ${PROJECT_DIR}
Config:                ${TRAIN_CONFIG}
Run group:             ${RUN_GROUP}
Run root:              ${RUN_ROOT}
Artifact root:         ${ARTIFACT_ROOT}
Smoke artifact root:   ${SMOKE_ARTIFACT_ROOT}
Normalizer root:       ${NORMALIZER_ROOT}
Normalizer file:       ${NORMALIZER_FILE}
Smoke checkpoint dir:  ${SMOKE_CKPT_DIR}
Train checkpoint dir:  ${TRAIN_DIR}
Structural dry policy: ${STRUCTURAL_DRY_POLICY}
Base seed:             ${BASE_SEED}
Epoch targets:         ${EPOCH_TARGETS}
Production batch size: ${PROD_BATCH_SIZE}
Production grad accum: ${PROD_GRAD_ACCUM_STEPS}
Production MC samples: ${PROD_MC_SAMPLES}
Submit production:     ${SUBMIT_PRODUCTION}
Skip smoke:            ${SKIP_SMOKE}
Prod dependency:       ${PROD_DEPENDENCY:-<none>}
Initial resume dir:    ${PROD_INITIAL_RESUME_FROM_DIR:-<none>}
Exclude nodes:         ${PROD_EXCLUDE_NODES:-<none>}
INFO

SMOKE_TRAIN_JOB_ID=""
if [[ "${SKIP_SMOKE}" != "1" ]]; then
  SMOKE_TRAIN_JOB_ID="$(submit_job smoke_train sbatch \
    --time=${SMOKE_TIME} \
    --job-name=wv_mcd_smtr \
    --export=ALL,PROJECT_DIR=${PROJECT_DIR},TRAIN_CONFIG=${TRAIN_CONFIG},RUN_ROOT=${RUN_ROOT},ARTIFACT_ROOT=${SMOKE_ARTIFACT_ROOT},SMOKE_ARTIFACT_ROOT=${SMOKE_ARTIFACT_ROOT},NORMALIZER_ROOT=${NORMALIZER_ROOT},NORMALIZER_FILE=${NORMALIZER_FILE},NORMALIZER_FORCE_LOAD=true,CKPT_DIR=${SMOKE_CKPT_DIR},BASE_SEED=${BASE_SEED},RUN_GROUP=${RUN_GROUP}_smoke,STRUCTURAL_DRY_POLICY=${STRUCTURAL_DRY_POLICY},WV_MCD_SMOKE=1,WANDB_LOG=false,N_SAMPLES_MAX=${SMOKE_N_SAMPLES_MAX},BATCH_SIZE=${SMOKE_BATCH_SIZE},MC_SAMPLES=${SMOKE_MC_SAMPLES} \
    "${TRAIN_SLURM}")"
fi

TRAIN_JOB_IDS=()
if [[ "${SUBMIT_PRODUCTION}" == "1" ]]; then
  prev_job="${PROD_DEPENDENCY:-${SMOKE_TRAIN_JOB_ID}}"
  part_idx=0
  for target_epoch in ${EPOCH_TARGETS}; do
    part_idx=$((part_idx + 1))
    part_label="p$(printf %02d ${part_idx})_e${target_epoch}"
    export_args="ALL,PROJECT_DIR=${PROJECT_DIR},TRAIN_CONFIG=${TRAIN_CONFIG},RUN_ROOT=${RUN_ROOT},ARTIFACT_ROOT=${ARTIFACT_ROOT},NORMALIZER_ROOT=${NORMALIZER_ROOT},NORMALIZER_FILE=${NORMALIZER_FILE},NORMALIZER_FORCE_LOAD=true,CKPT_DIR=${TRAIN_DIR},BASE_SEED=${BASE_SEED},RUN_GROUP=${RUN_GROUP},STRUCTURAL_DRY_POLICY=${STRUCTURAL_DRY_POLICY},N_EPOCHS=${target_epoch},BATCH_SIZE=${PROD_BATCH_SIZE},GRAD_ACCUM_STEPS=${PROD_GRAD_ACCUM_STEPS},MC_SAMPLES=${PROD_MC_SAMPLES},WANDB_LOG=true,WANDB_NAME=${RUN_GROUP}_${part_label}"
    resume_from_dir=""
    if [[ ${part_idx} -eq 1 && -n "${PROD_INITIAL_RESUME_FROM_DIR}" ]]; then
      resume_from_dir="${PROD_INITIAL_RESUME_FROM_DIR}"
    elif [[ ${part_idx} -gt 1 ]]; then
      resume_from_dir="${TRAIN_DIR}"
    fi
    if [[ -n "${resume_from_dir}" ]]; then
      export_args="${export_args},RESUME_FROM_DIR=${resume_from_dir}"
    fi
    dep_args=()
    if [[ -n "${prev_job}" ]]; then
      dep_args+=(--dependency=afterok:${prev_job})
    fi
    exclude_args=()
    if [[ -n "${PROD_EXCLUDE_NODES}" ]]; then
      exclude_args+=(--exclude="${PROD_EXCLUDE_NODES}")
    fi
    jid="$(submit_job train_${part_label} sbatch \
      "${dep_args[@]}" \
      "${exclude_args[@]}" \
      --time=${TRAIN_TIME} \
      --job-name=wv_mcd_lg_p$(printf %02d ${part_idx}) \
      --export=${export_args} \
      "${TRAIN_SLURM}")"
    TRAIN_JOB_IDS+=("${jid}")
    prev_job="${jid}"
  done
fi

cat <<OUT
smoke_train_job_id=${SMOKE_TRAIN_JOB_ID}
train_job_ids=${TRAIN_JOB_IDS[*]:-}
run_root=${RUN_ROOT}
artifact_root=${ARTIFACT_ROOT}
normalizer_root=${NORMALIZER_ROOT}
smoke_checkpoint_dir=${SMOKE_CKPT_DIR}
train_checkpoint_dir=${TRAIN_DIR}
OUT
