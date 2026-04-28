#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -J coast_mcd_tr
#SBATCH --array=0-0
#SBATCH -o /scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_mcdropout/slurm/coast_mcd_tr-%A_%a.out
#SBATCH -e /scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_mcdropout/slurm/coast_mcd_tr-%A_%a.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-}"
if [[ -f "${PROJECT_DIR}/scripts/slurm/lib/common.sh" ]]; then
  source "${PROJECT_DIR}/scripts/slurm/lib/common.sh"
  SCRIPT_DIR="${PROJECT_DIR}/scripts"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/lib/common.sh" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  source "${PROJECT_DIR}/scripts/slurm/lib/common.sh"
  SCRIPT_DIR="${PROJECT_DIR}/scripts"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/lib/common.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  source "${SCRIPT_DIR}/slurm/lib/common.sh"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  CANONICAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  source "$(cd "${CANONICAL_DIR}/.." && pwd)/lib/common.sh"
  SCRIPT_DIR="$(slurm_resolve_scripts_root "${CANONICAL_DIR}")"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
slurm_load_apptainer
cd "${SCRIPT_DIR}"

RUN_ROOT="${RUN_ROOT:-/scratch/$USER/GINO_Model/neuraloperator_runs/coastal_mcdropout}"
COASTAL_MCD_SMOKE="${COASTAL_MCD_SMOKE:-0}"
if [[ "${COASTAL_MCD_SMOKE}" == "1" && -z "${ARTIFACT_ROOT:-}" ]]; then
  ARTIFACT_ROOT="${RUN_ROOT}/smoke_artifacts/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
else
  ARTIFACT_ROOT="${ARTIFACT_ROOT:-${RUN_ROOT}/artifacts}"
fi
CKPT_ROOT="${CKPT_ROOT:-${RUN_ROOT}/checkpoints}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT:-${RUN_ROOT}/slurm}"
mkdir -p "${ARTIFACT_ROOT}" "${CKPT_ROOT}" "${LOG_ROOT}" "${SLURM_LOG_ROOT}"

export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_DIR}/scripts/flood_wv_train_operator.py}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_DIR}/config/flood/coastal/gino_coastal_depth_only_mcdropout.yaml}"
CONTAINER_PATH="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"

TRAIN_ROOT="${TRAIN_ROOT:-/scratch/$USER/uncertainty_floodmodel_linux/results/coastal/Coastal_Flood_coastal_v1_5k_train_prod_t2_w64_20260318_233556/train}"
TEST_ROOT="${TEST_ROOT:-/scratch/$USER/uncertainty_floodmodel_linux/results/coastal/Coastal_Flood_coastal_v1_5k_test_prod_t2_w16_20260414/test}"
NORMALIZER_ROOT="${NORMALIZER_ROOT:-${ARTIFACT_ROOT}}"
NORMALIZER_FILE="${NORMALIZER_FILE:-normalizers_depth_only.pt}"
NORMALIZER_FORCE_LOAD="${NORMALIZER_FORCE_LOAD:-true}"
STRUCTURAL_DRY_POLICY="${STRUCTURAL_DRY_POLICY:-legacy_full_domain}"
STRUCTURAL_DRY_MASK="${STRUCTURAL_DRY_MASK:-${ARTIFACT_ROOT}/structural_dry_mask_exact_zero.pt}"

ENSEMBLE_ID="${SLURM_ARRAY_TASK_ID:-0}"
BASE_SEED_DEFAULT=$((123000 + ${SLURM_ARRAY_JOB_ID:-0} * 10))
BASE_SEED="${BASE_SEED:-${BASE_SEED_DEFAULT}}"
SEED=$((BASE_SEED + ENSEMBLE_ID))
SHORT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
RUN_TAG="${RUN_TAG:-coastal_mcdropout_${SHORT_SHA}}"
RUN_GROUP="${RUN_GROUP:-${RUN_TAG}_job${SLURM_ARRAY_JOB_ID:-manual}_sbase${BASE_SEED}}"
WANDB_NAME="${WANDB_NAME:-${RUN_GROUP}_ens${ENSEMBLE_ID}_seed${SEED}}"
CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/${RUN_GROUP}/ens${ENSEMBLE_ID}}"
RESUME_FROM_DIR="${RESUME_FROM_DIR:-}"
mkdir -p "${CKPT_DIR}"

slurm_configure_host_ca
slurm_assert_container_gpus "${CONTAINER_PATH}" 1
for key_path in "${PROJECT_DIR}/config/wandb_api_key.txt" "${HOME}/.config/wandb_api_key.txt" "/scratch/$USER/Data_Generation_UQ/GINO_Model/neuraloperator_no_physics/config/wandb_api_key.txt"; do
  if [[ -f "${key_path}" ]]; then
    export APPTAINERENV_WANDB_API_KEY="$(head -n 1 "${key_path}" | tr -d "\r")"
    break
  fi
done

TRAIN_TXT="${TRAIN_TXT:-train.txt}"
WANDB_LOG="${WANDB_LOG:-}"
N_SAMPLES_MAX="${N_SAMPLES_MAX:-}"
N_EPOCHS="${N_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
MC_SAMPLES="${MC_SAMPLES:-}"

if [[ "${COASTAL_MCD_SMOKE}" == "1" ]]; then
  SMOKE_SPLIT_DIR="${RUN_ROOT}/smoke_splits"
  mkdir -p "${SMOKE_SPLIT_DIR}"
  SMOKE_TRAIN_TXT="${SMOKE_TRAIN_TXT:-${SMOKE_SPLIT_DIR}/train_smoke_${SLURM_JOB_ID:-manual}_${ENSEMBLE_ID}.txt}"
  if [[ ! -f "${SMOKE_TRAIN_TXT}" ]]; then
    head -n "${SMOKE_RUN_IDS:-8}" "${TRAIN_ROOT}/train.txt" > "${SMOKE_TRAIN_TXT}"
  fi
  TRAIN_TXT="${SMOKE_TRAIN_TXT}"
  N_SAMPLES_MAX="${N_SAMPLES_MAX:-32}"
  N_EPOCHS="${N_EPOCHS:-1}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  MC_SAMPLES="${MC_SAMPLES:-4}"
  WANDB_LOG="${WANDB_LOG:-false}"
fi
WANDB_LOG="${WANDB_LOG:-true}"

CLI_ARGS=(
  --config_path "${TRAIN_CONFIG}"
  --data.root "${TRAIN_ROOT}"
  --data.train_root "${TRAIN_ROOT}"
  --data.train_txt "${TRAIN_TXT}"
  --rollout_data.root "${TEST_ROOT}"
  --data.normalizer_root "${NORMALIZER_ROOT}"
  --data.normalizer_path "${NORMALIZER_FILE}"
  --data.force_load_normalizers "${NORMALIZER_FORCE_LOAD}"
  --structural_dry.policy "${STRUCTURAL_DRY_POLICY}"
  --checkpoint.save_dir "${CKPT_DIR}"
  --log_file "${LOG_ROOT}/${RUN_GROUP}_ens${ENSEMBLE_ID}.log"
  --distributed.seed "${SEED}"
  --wandb.log "${WANDB_LOG}"
  --wandb.group "${RUN_GROUP}"
  --wandb.name "${WANDB_NAME}"
)

if [[ "${STRUCTURAL_DRY_POLICY}" == "masked_primary" ]]; then
  CLI_ARGS+=(
    --structural_dry.mask_path "${STRUCTURAL_DRY_MASK}"
    --structural_dry.canonical_data_root "${TRAIN_ROOT}"
    --structural_dry.canonical_train_txt "${TRAIN_TXT}"
  )
fi

[[ -n "${RESUME_FROM_DIR}" ]] && CLI_ARGS+=(--checkpoint.resume_from_dir "${RESUME_FROM_DIR}")
[[ -n "${N_SAMPLES_MAX}" ]] && CLI_ARGS+=(--data.n_samples_max "${N_SAMPLES_MAX}")
[[ -n "${N_EPOCHS}" ]] && CLI_ARGS+=(--opt.n_epochs "${N_EPOCHS}")
[[ -n "${BATCH_SIZE}" ]] && CLI_ARGS+=(--data.batch_size "${BATCH_SIZE}")
[[ -n "${MC_SAMPLES}" ]] && CLI_ARGS+=(--uq.mc_samples "${MC_SAMPLES}" --rollout.n_ensemble_samples "${MC_SAMPLES}")

cat <<INFO
Training script:       ${TRAIN_SCRIPT}
Config:                ${TRAIN_CONFIG}
Train root:            ${TRAIN_ROOT}
Train txt:             ${TRAIN_TXT}
Test root:             ${TEST_ROOT}
Artifact root:         ${ARTIFACT_ROOT}
Normalizer:            ${NORMALIZER_ROOT}/${NORMALIZER_FILE}
Normalizer force load: ${NORMALIZER_FORCE_LOAD}
Structural dry policy: ${STRUCTURAL_DRY_POLICY}
Structural dry mask:   ${STRUCTURAL_DRY_MASK}
Checkpoint dir:        ${CKPT_DIR}
Resume from dir:       ${RESUME_FROM_DIR:-<none>}
Git commit:            $(git -C "${PROJECT_DIR}" rev-parse HEAD)
Seed:                  ${SEED}
MC samples:            ${MC_SAMPLES:-32}
W&B group/name/log:    ${RUN_GROUP} / ${WANDB_NAME} / ${WANDB_LOG}
INFO

apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${TRAIN_SCRIPT}" "${CLI_ARGS[@]}"
