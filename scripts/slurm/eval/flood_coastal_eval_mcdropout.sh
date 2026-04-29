#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J coast_mcd_ev
#SBATCH -o /scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_mcdropout/slurm/coast_mcd_ev-%j.out
#SBATCH -e /scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_mcdropout/slurm/coast_mcd_ev-%j.err

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
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${RUN_ROOT}/artifacts}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/eval_outputs}"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT:-${RUN_ROOT}/slurm}"
EVAL_CONFIG_ROOT="${EVAL_CONFIG_ROOT:-${RUN_ROOT}/eval_configs}"
mkdir -p "${ARTIFACT_ROOT}" "${EVAL_ROOT}" "${SLURM_LOG_ROOT}" "${EVAL_CONFIG_ROOT}"

export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${PROJECT_DIR}/scripts/flood_wv_eval_operator.py}"
EVAL_CONFIG="${EVAL_CONFIG:-${PROJECT_DIR}/config/flood/coastal/gino_coastal_depth_only_mcdropout.yaml}"
CONTAINER_PATH="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
TRAIN_ROOT="${TRAIN_ROOT:-/scratch/$USER/uncertainty_floodmodel_linux/results/coastal/Coastal_Flood_coastal_v1_5k_train_prod_t2_w64_20260318_233556/train}"
TEST_ROOT="${TEST_ROOT:-/scratch/$USER/uncertainty_floodmodel_linux/results/coastal/Coastal_Flood_coastal_v1_5k_test_prod_t2_w16_20260414/test}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_ROOT}/checkpoints}"
CHECKPOINT_EVAL_NAME="${CHECKPOINT_EVAL_NAME:-best_model}"
NORMALIZER_ROOT="${NORMALIZER_ROOT:-${ARTIFACT_ROOT}}"
NORMALIZER_FILE="${NORMALIZER_FILE:-normalizers_depth_only.pt}"
NORMALIZER_PATH="${NORMALIZER_PATH:-${NORMALIZER_ROOT}/${NORMALIZER_FILE}}"
NORMALIZER_FORCE_LOAD="${NORMALIZER_FORCE_LOAD:-true}"
STRUCTURAL_DRY_POLICY="${STRUCTURAL_DRY_POLICY:-legacy_full_domain}"
STRUCTURAL_DRY_MASK="${STRUCTURAL_DRY_MASK:-${ARTIFACT_ROOT}/structural_dry_mask_exact_zero.pt}"
JOB_NAME="${JOB_NAME:-coastal_mcdropout_eval}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)}"

MC_SAMPLES="${MC_SAMPLES:-}"
N_SAMPLES_MAX="${N_SAMPLES_MAX:-}"
TEST_TXT_NAME="${TEST_TXT_NAME:-}"
if [[ "${COASTAL_MCD_EVAL_SMOKE:-0}" == "1" ]]; then
  MC_SAMPLES="${MC_SAMPLES:-4}"
  N_SAMPLES_MAX="${N_SAMPLES_MAX:-4}"
  SMOKE_SPLIT_DIR="${RUN_ROOT}/smoke_splits"
  mkdir -p "${SMOKE_SPLIT_DIR}"
  TEST_TXT_NAME="${TEST_TXT_NAME:-test_smoke_${SLURM_JOB_ID:-manual}.txt}"
fi
TEST_TXT_NAME="${TEST_TXT_NAME:-test.txt}"
MC_SAMPLES="${MC_SAMPLES:-32}"

slurm_configure_host_ca
slurm_assert_container_gpus "${CONTAINER_PATH}" 1

if [[ ! -f "${NORMALIZER_PATH}" ]]; then
  echo "ERROR: training normalizer artifact not found: ${NORMALIZER_PATH}" >&2
  exit 2
fi
if [[ "${STRUCTURAL_DRY_POLICY}" == "masked_primary" && ! -f "${STRUCTURAL_DRY_MASK}" ]]; then
  echo "ERROR: structural-dry artifact not found: ${STRUCTURAL_DRY_MASK}" >&2
  exit 2
fi
BASE_TEST_TXT="${TEST_ROOT}/test.txt"
if [[ ! -f "${BASE_TEST_TXT}" ]]; then
  find "${TEST_ROOT}" -maxdepth 1 -type f -name "*.hdf" -printf "%f\n" | sed "s/\.hdf$//" | sort > "${BASE_TEST_TXT}"
fi
if [[ ! -s "${BASE_TEST_TXT}" ]]; then
  echo "ERROR: Missing/empty coastal test split: ${BASE_TEST_TXT}" >&2
  exit 1
fi
if [[ "${COASTAL_MCD_EVAL_SMOKE:-0}" == "1" ]]; then
  head -n "${SMOKE_RUN_IDS:-4}" "${BASE_TEST_TXT}" > "${TEST_ROOT}/${TEST_TXT_NAME}"
fi
if [[ ! -s "${TEST_ROOT}/${TEST_TXT_NAME}" ]]; then
  echo "ERROR: Missing/empty eval split: ${TEST_ROOT}/${TEST_TXT_NAME}" >&2
  exit 1
fi

RENDERED_CONFIG="${EVAL_CONFIG_RENDERED:-${EVAL_CONFIG_ROOT}/${JOB_NAME}_${SLURM_JOB_ID:-manual}.yaml}"
sed \
  -e "s/Stage_Hydrographs_Train_Clean.txt/Stage_Hydrographs_Test_Clean.txt/g" \
  -e "s/Precipitation_Train_Clean.txt/Precipitation_Test_Clean.txt/g" \
  "${EVAL_CONFIG}" > "${RENDERED_CONFIG}"
mkdir -p "${OUT_DIR}"

CLI_ARGS=(
  --config_path "${RENDERED_CONFIG}"
  --eval_log_file "${OUT_DIR}/eval_mcdropout.log"
  --checkpoint.save_dir "${CHECKPOINT_ROOT}"
  --checkpoint.eval_name "${CHECKPOINT_EVAL_NAME}"
  --data.root "${TEST_ROOT}"
  --data.train_txt "${TEST_TXT_NAME}"
  --data.write_train_txt false
  --data.normalizer_root "${NORMALIZER_ROOT}"
  --data.normalizer_path "${NORMALIZER_PATH}"
  --data.force_load_normalizers "${NORMALIZER_FORCE_LOAD}"
  --structural_dry.policy "${STRUCTURAL_DRY_POLICY}"
  --rollout_data.root "${TEST_ROOT}"
  --rollout_data.test_txt "${TEST_TXT_NAME}"
  --rollout.out_dir "${OUT_DIR}"
  --rollout.n_ensemble_samples "${MC_SAMPLES}"
  --rollout.n_ensemble_samples_per_model null
  --uq.mc_samples "${MC_SAMPLES}"
  --wandb.log false
  --run_single_step
  --run_rollout
)
if [[ "${STRUCTURAL_DRY_POLICY}" == "masked_primary" ]]; then
  CLI_ARGS+=(
    --structural_dry.mask_path "${STRUCTURAL_DRY_MASK}"
    --structural_dry.canonical_data_root "${TRAIN_ROOT}"
    --structural_dry.canonical_train_txt "train.txt"
  )
fi

[[ -n "${N_SAMPLES_MAX}" ]] && CLI_ARGS+=(--data.n_samples_max "${N_SAMPLES_MAX}")

cat <<INFO
Eval script:          ${EVAL_SCRIPT}
Source config:        ${EVAL_CONFIG}
Rendered config:      ${RENDERED_CONFIG}
Checkpoint root:      ${CHECKPOINT_ROOT}
Checkpoint eval name: ${CHECKPOINT_EVAL_NAME}
Train root:           ${TRAIN_ROOT}
Test root:            ${TEST_ROOT}
Eval split:           ${TEST_ROOT}/${TEST_TXT_NAME}
Train normalizer:     ${NORMALIZER_PATH}
Normalizer force load:${NORMALIZER_FORCE_LOAD}
Structural dry policy:${STRUCTURAL_DRY_POLICY}
Structural dry mask:  ${STRUCTURAL_DRY_MASK}
MC samples:           ${MC_SAMPLES}
Output dir:           ${OUT_DIR}
Git commit:           $(git -C "${PROJECT_DIR}" rev-parse HEAD)
INFO

apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${EVAL_SCRIPT}" "${CLI_ARGS[@]}"
