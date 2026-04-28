#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
TRAIN_SLURM="${PROJECT_DIR}/scripts/slurm/train/flood_coastal_train_mcdropout.sh"
EVAL_SLURM="${PROJECT_DIR}/scripts/slurm/eval/flood_coastal_eval_mcdropout.sh"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_DIR}/config/flood/coastal/gino_coastal_depth_only_mcdropout.yaml}"
SHORT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
RUN_GROUP="${RUN_GROUP:-coastal_mcdropout_legacy_h128_bs128_r48_lr5e4_${SHORT_SHA}_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/GINO_Model/neuraloperator_runs/coastal_mcdropout/${RUN_GROUP}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${RUN_ROOT}/artifacts}"
SMOKE_ARTIFACT_ROOT="${SMOKE_ARTIFACT_ROOT:-${RUN_ROOT}/smoke/artifacts}"
SMOKE_CKPT_DIR="${SMOKE_CKPT_DIR:-${RUN_ROOT}/smoke/checkpoints}"
TRAIN_DIR="${TRAIN_DIR:-${RUN_ROOT}/checkpoints/train}"
SMOKE_EVAL_DIR="${SMOKE_EVAL_DIR:-${RUN_ROOT}/smoke/eval_outputs}"
PROD_EVAL_DIR="${PROD_EVAL_DIR:-${RUN_ROOT}/eval_outputs}"
FAIR_NORMALIZER_ROOT="${FAIR_NORMALIZER_ROOT:-/sfs/gpfs/tardis/home/jrj6wm/GINO_Model/neuraloperator_no_physics_git_main/scripts/runtime/coastal_full_train_wvlike250/coastal_trial0035_wvlike_bs128_r48_det_a100_lr5e4_20260406_172705_3af9ccf/artifacts}"
NORMALIZER_FILE="${NORMALIZER_FILE:-normalizers_depth_only.pt}"
BASE_SEED="${BASE_SEED:-123}"
STRUCTURAL_DRY_POLICY="${STRUCTURAL_DRY_POLICY:-legacy_full_domain}"
SMOKE_TIME="${SMOKE_TIME:-02:00:00}"
SMOKE_EVAL_TIME="${SMOKE_EVAL_TIME:-02:00:00}"
TRAIN_TIME="${TRAIN_TIME:-2-18:00:00}"
EVAL_TIME="${EVAL_TIME:-24:00:00}"
SUBMIT_PRODUCTION="${SUBMIT_PRODUCTION:-1}"
SUBMIT_PROD_EVAL="${SUBMIT_PROD_EVAL:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
PROD_DEPENDENCY="${PROD_DEPENDENCY:-}"
EPOCH_TARGETS="${EPOCH_TARGETS:-75 150 175 200 215 225 240 250}"
SMOKE_N_SAMPLES_MAX="${SMOKE_N_SAMPLES_MAX:-32}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
SMOKE_MC_SAMPLES="${SMOKE_MC_SAMPLES:-4}"
PROD_BATCH_SIZE="${PROD_BATCH_SIZE:-128}"
PROD_MC_SAMPLES="${PROD_MC_SAMPLES:-32}"

if [[ "${STRUCTURAL_DRY_POLICY}" != "legacy_full_domain" ]]; then
  echo "ERROR: this benchmark chain is intended to run with STRUCTURAL_DRY_POLICY=legacy_full_domain; got ${STRUCTURAL_DRY_POLICY}" >&2
  exit 2
fi
if [[ ! -f "${FAIR_NORMALIZER_ROOT}/${NORMALIZER_FILE}" ]]; then
  echo "ERROR: fair train normalizer artifact not found: ${FAIR_NORMALIZER_ROOT}/${NORMALIZER_FILE}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${ARTIFACT_ROOT}" "${SMOKE_ARTIFACT_ROOT}" "${SMOKE_CKPT_DIR}" "${TRAIN_DIR}" "${SMOKE_EVAL_DIR}" "${PROD_EVAL_DIR}"

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
Fair normalizer root:  ${FAIR_NORMALIZER_ROOT}
Normalizer file:       ${NORMALIZER_FILE}
Smoke checkpoint dir:  ${SMOKE_CKPT_DIR}
Train checkpoint dir:  ${TRAIN_DIR}
Structural dry policy: ${STRUCTURAL_DRY_POLICY}
Base seed:             ${BASE_SEED}
Epoch targets:         ${EPOCH_TARGETS}
Production batch size: ${PROD_BATCH_SIZE}
Production MC samples: ${PROD_MC_SAMPLES}
Submit production:     ${SUBMIT_PRODUCTION}
Submit production eval:${SUBMIT_PROD_EVAL}
Skip smoke:            ${SKIP_SMOKE}
Prod dependency:       ${PROD_DEPENDENCY:-<none>}
INFO

SMOKE_TRAIN_JOB_ID=""
SMOKE_EVAL_JOB_ID=""
if [[ "${SKIP_SMOKE}" != "1" ]]; then
  SMOKE_TRAIN_JOB_ID="$(submit_job smoke_train sbatch \
    --time=${SMOKE_TIME} \
    --job-name=coast_mcd_smtr \
    --export=ALL,PROJECT_DIR=${PROJECT_DIR},TRAIN_CONFIG=${TRAIN_CONFIG},RUN_ROOT=${RUN_ROOT},ARTIFACT_ROOT=${SMOKE_ARTIFACT_ROOT},NORMALIZER_ROOT=${SMOKE_ARTIFACT_ROOT},NORMALIZER_FILE=${NORMALIZER_FILE},NORMALIZER_FORCE_LOAD=false,CKPT_DIR=${SMOKE_CKPT_DIR},BASE_SEED=${BASE_SEED},RUN_GROUP=${RUN_GROUP}_smoke,STRUCTURAL_DRY_POLICY=${STRUCTURAL_DRY_POLICY},COASTAL_MCD_SMOKE=1,WANDB_LOG=false,N_SAMPLES_MAX=${SMOKE_N_SAMPLES_MAX},BATCH_SIZE=${SMOKE_BATCH_SIZE},MC_SAMPLES=${SMOKE_MC_SAMPLES} \
    "${TRAIN_SLURM}")"

  SMOKE_EVAL_JOB_ID="$(submit_job smoke_eval sbatch \
    --dependency=afterok:${SMOKE_TRAIN_JOB_ID} \
    --time=${SMOKE_EVAL_TIME} \
    --job-name=coast_mcd_smev \
    --export=ALL,PROJECT_DIR=${PROJECT_DIR},EVAL_CONFIG=${TRAIN_CONFIG},RUN_ROOT=${RUN_ROOT},ARTIFACT_ROOT=${SMOKE_ARTIFACT_ROOT},NORMALIZER_ROOT=${SMOKE_ARTIFACT_ROOT},NORMALIZER_FILE=${NORMALIZER_FILE},NORMALIZER_FORCE_LOAD=false,CHECKPOINT_ROOT=${SMOKE_CKPT_DIR},OUT_DIR=${SMOKE_EVAL_DIR},STRUCTURAL_DRY_POLICY=${STRUCTURAL_DRY_POLICY},COASTAL_MCD_EVAL_SMOKE=1,MC_SAMPLES=${SMOKE_MC_SAMPLES} \
    "${EVAL_SLURM}")"
fi

TRAIN_JOB_IDS=()
PROD_EVAL_JOB_ID=""
if [[ "${SUBMIT_PRODUCTION}" == "1" ]]; then
  prev_job="${PROD_DEPENDENCY:-${SMOKE_EVAL_JOB_ID}}"
  part_idx=0
  for target_epoch in ${EPOCH_TARGETS}; do
    part_idx=$((part_idx + 1))
    part_label="p$(printf %02d ${part_idx})_e${target_epoch}"
    export_args="ALL,PROJECT_DIR=${PROJECT_DIR},TRAIN_CONFIG=${TRAIN_CONFIG},RUN_ROOT=${RUN_ROOT},ARTIFACT_ROOT=${ARTIFACT_ROOT},NORMALIZER_ROOT=${FAIR_NORMALIZER_ROOT},NORMALIZER_FILE=${NORMALIZER_FILE},NORMALIZER_FORCE_LOAD=true,CKPT_DIR=${TRAIN_DIR},BASE_SEED=${BASE_SEED},RUN_GROUP=${RUN_GROUP},STRUCTURAL_DRY_POLICY=${STRUCTURAL_DRY_POLICY},N_EPOCHS=${target_epoch},BATCH_SIZE=${PROD_BATCH_SIZE},MC_SAMPLES=${PROD_MC_SAMPLES},WANDB_LOG=true,WANDB_NAME=${RUN_GROUP}_${part_label}"
    if [[ ${part_idx} -gt 1 ]]; then
      export_args="${export_args},RESUME_FROM_DIR=${TRAIN_DIR}"
    fi
    dep_args=()
    if [[ -n "${prev_job}" ]]; then
      dep_args+=(--dependency=afterok:${prev_job})
    fi
    jid="$(submit_job train_${part_label} sbatch \
      "${dep_args[@]}" \
      --time=${TRAIN_TIME} \
      --job-name=coast_mcd_lg_p$(printf %02d ${part_idx}) \
      --export=${export_args} \
      "${TRAIN_SLURM}")"
    TRAIN_JOB_IDS+=("${jid}")
    prev_job="${jid}"
  done
  if [[ "${SUBMIT_PROD_EVAL}" == "1" ]]; then
    PROD_EVAL_JOB_ID="$(submit_job prod_eval sbatch \
      --dependency=afterok:${prev_job} \
      --time=${EVAL_TIME} \
      --job-name=coast_mcd_eval \
      --export=ALL,PROJECT_DIR=${PROJECT_DIR},EVAL_CONFIG=${TRAIN_CONFIG},RUN_ROOT=${RUN_ROOT},ARTIFACT_ROOT=${ARTIFACT_ROOT},NORMALIZER_ROOT=${FAIR_NORMALIZER_ROOT},NORMALIZER_FILE=${NORMALIZER_FILE},NORMALIZER_FORCE_LOAD=true,CHECKPOINT_ROOT=${TRAIN_DIR},OUT_DIR=${PROD_EVAL_DIR},STRUCTURAL_DRY_POLICY=${STRUCTURAL_DRY_POLICY},MC_SAMPLES=${PROD_MC_SAMPLES} \
      "${EVAL_SLURM}")"
  fi
else
  cat <<MSG
Production chain was not submitted. To submit after inspecting smoke:
  SKIP_SMOKE=1 SUBMIT_PRODUCTION=1 RUN_GROUP=${RUN_GROUP} RUN_ROOT=${RUN_ROOT} FAIR_NORMALIZER_ROOT=${FAIR_NORMALIZER_ROOT} BASE_SEED=${BASE_SEED} ${PROJECT_DIR}/scripts/slurm/train/submit_coastal_mcdropout_chain.sh
MSG
fi

cat <<OUT
smoke_train_job_id=${SMOKE_TRAIN_JOB_ID}
smoke_eval_job_id=${SMOKE_EVAL_JOB_ID}
train_job_ids=${TRAIN_JOB_IDS[*]:-}
prod_eval_job_id=${PROD_EVAL_JOB_ID}
run_root=${RUN_ROOT}
artifact_root=${ARTIFACT_ROOT}
fair_normalizer_root=${FAIR_NORMALIZER_ROOT}
smoke_checkpoint_dir=${SMOKE_CKPT_DIR}
train_checkpoint_dir=${TRAIN_DIR}
smoke_eval_dir=${SMOKE_EVAL_DIR}
prod_eval_dir=${PROD_EVAL_DIR}
OUT
