#!/bin/bash
set -euo pipefail

PROJECT_DIR_DEFAULT="/home/$USER/GINO_Model/neuraloperator_no_physics_git_main"
PROJECT_DIR="${PROJECT_DIR:-${PROJECT_DIR_DEFAULT}}"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
TRAIN_SLURM_DIR="${SCRIPTS_DIR}/slurm/train"
BASE_CONFIG="${BASE_CONFIG:-${PROJECT_DIR}/config/flood/wv/gino_pluvial_flood_config_WV_depth_only.yaml}"
ACCOUNT="${ACCOUNT:-uqgroup_plus}"

RUN_GROUP_DEFAULT="wdonly_fgn_ws1_dynamic250_sharednorm_$(date +%Y%m%d_%H%M%S)_$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
RUN_GROUP="${RUN_GROUP:-${RUN_GROUP_DEFAULT}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/scripts/runtime/checkpoints_WV_depth_only_250ep_dynamic_ensemble_single_gpu/${RUN_GROUP}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${RUN_ROOT}/artifacts}"
MEMBER_ROOT="${MEMBER_ROOT:-${RUN_ROOT}/members}"
TRAIN_CONFIG_RENDERED="${TRAIN_CONFIG_RENDERED:-${RUN_ROOT}/config.yaml}"

DATA_ROOT="${DATA_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/train}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${DATA_ROOT}}"
CLEAN_BOUNDARY_ROOT="${CLEAN_BOUNDARY_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/metadata}"
CLEAN_BOUNDARY_FILE="${CLEAN_BOUNDARY_FILE:-Hydrographs_Train_Clean.txt}"
TRAIN_TXT_NAME="${TRAIN_TXT_NAME:-train.txt}"

SPLIT_SEED="${SPLIT_SEED:-123}"
BASE_MODEL_SEED="${BASE_MODEL_SEED:-123}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-4}"
ARRAY_RANGE="${ARRAY_RANGE:-0-$((ENSEMBLE_SIZE - 1))}"
PREP_TIME="${PREP_TIME:-04:00:00}"
TRAIN_TIME="${TRAIN_TIME:-72:00:00}"
OVERWRITE_NORMALIZERS="${OVERWRITE_NORMALIZERS:-false}"
WANDB_LOG="${WANDB_LOG:-true}"
VERIFY_TRAINING="${VERIFY_TRAINING:-true}"
EPOCHS="${EPOCHS:-250}"
N_SAMPLES_MAX="${N_SAMPLES_MAX:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
RUN_GROUP_PREFIX="${RUN_GROUP_PREFIX:-${RUN_GROUP}}"

mkdir -p "${RUN_ROOT}" "${ARTIFACT_DIR}" "${MEMBER_ROOT}"

export BASE_CONFIG TRAIN_CONFIG_RENDERED DATA_ROOT ROLLOUT_ROOT ARTIFACT_DIR CLEAN_BOUNDARY_ROOT CLEAN_BOUNDARY_FILE TRAIN_TXT_NAME SPLIT_SEED BASE_MODEL_SEED EPOCHS N_SAMPLES_MAX BATCH_SIZE MEMBER_ROOT VERIFY_TRAINING
python3 - <<'PY'
import os
from pathlib import Path
import yaml

base_config = Path(os.environ['BASE_CONFIG'])
output_path = Path(os.environ['TRAIN_CONFIG_RENDERED'])
payload = yaml.safe_load(base_config.read_text())
if not isinstance(payload, dict):
    raise SystemExit('Base config must be a mapping')
config = payload.get('flood', payload)

config.setdefault('distributed', {})['use_distributed'] = False
config['distributed']['seed'] = int(os.environ['BASE_MODEL_SEED'])
config.setdefault('checkpoint', {})['resume_from_dir'] = None
config.setdefault('checkpoint', {})['save_dir'] = str(Path(os.environ['MEMBER_ROOT']).resolve())
config.setdefault('data', {})['root'] = os.environ['DATA_ROOT']
config['data']['boundary_source'] = 'clean_family'
config['data']['clean_boundary_root'] = os.environ['CLEAN_BOUNDARY_ROOT']
config['data']['clean_boundary_file'] = os.environ['CLEAN_BOUNDARY_FILE']
config['data']['normalizer_root'] = str(Path(os.environ['ARTIFACT_DIR']).resolve())
config['data']['normalizer_path'] = 'normalizers_depth_only.pt'
config['data']['force_load_normalizers'] = False
config['data']['split_seed'] = int(os.environ['SPLIT_SEED'])
config['data']['train_txt'] = os.environ['TRAIN_TXT_NAME']
config['data']['batch_size'] = int(os.environ['BATCH_SIZE'])
if os.environ['N_SAMPLES_MAX']:
    config['data']['n_samples_max'] = int(os.environ['N_SAMPLES_MAX'])
else:
    config['data']['n_samples_max'] = None
config.setdefault('rollout_data', {})['root'] = os.environ['ROLLOUT_ROOT']
config['rollout_data']['boundary_source'] = 'clean_family'
config['rollout_data']['clean_boundary_root'] = os.environ['CLEAN_BOUNDARY_ROOT']
config['rollout_data']['clean_boundary_file'] = os.environ['CLEAN_BOUNDARY_FILE']
config.setdefault('opt', {})['n_epochs'] = int(os.environ['EPOCHS'])
config['opt']['ar_finetune_start_epoch'] = 150
config['opt']['ar_curriculum_start_steps'] = 2
config['opt']['ar_curriculum_epochs_per_step'] = 25
config['opt']['ar_rollout_steps'] = 5
config['verify_training'] = os.environ.get('VERIFY_TRAINING', 'true').lower() == 'true'

wrapped = {'flood': config}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(yaml.safe_dump(wrapped, default_flow_style=False))
PY

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

echo "Project dir:      ${PROJECT_DIR}"
echo "Base config:      ${BASE_CONFIG}"
echo "Rendered config:  ${TRAIN_CONFIG_RENDERED}"
echo "Run root:         ${RUN_ROOT}"
echo "Artifact dir:     ${ARTIFACT_DIR}"
echo "Member root:      ${MEMBER_ROOT}"
echo "Data root:        ${DATA_ROOT}"
echo "Rollout root:     ${ROLLOUT_ROOT}"
echo "Boundary root:    ${CLEAN_BOUNDARY_ROOT}"
echo "Boundary file:    ${CLEAN_BOUNDARY_FILE}"
echo "Split seed:       ${SPLIT_SEED}"
echo "Base model seed:  ${BASE_MODEL_SEED}"
echo "Epochs:           ${EPOCHS}"
echo "n_samples_max:    ${N_SAMPLES_MAX:-full}"
echo "Array range:      ${ARRAY_RANGE}"

PREP_JOB_ID="$(submit_job prep sbatch \
  --account="${ACCOUNT}" \
  --job-name=wv_norm_prep \
  --time="${PREP_TIME}" \
  --export=ALL,PROJECT_DIR=${PROJECT_DIR},TRAIN_CONFIG=${TRAIN_CONFIG_RENDERED},ARTIFACT_DIR=${ARTIFACT_DIR},DATA_ROOT=${DATA_ROOT},SPLIT_SEED=${SPLIT_SEED},OVERWRITE_NORMALIZERS=${OVERWRITE_NORMALIZERS} \
  "${TRAIN_SLURM_DIR}/flood_wv_prepare_normalizers.sh")"

TRAIN_JOB_ID="$(submit_job train sbatch \
  --account="${ACCOUNT}" \
  --job-name=wv_fgn_dyn_e4_s1 \
  --array="${ARRAY_RANGE}" \
  --time="${TRAIN_TIME}" \
  --dependency=afterok:${PREP_JOB_ID} \
  --export=ALL,PROJECT_DIR=${PROJECT_DIR},TRAIN_CONFIG=${TRAIN_CONFIG_RENDERED},DATA_ROOT=${DATA_ROOT},ROLLOUT_ROOT=${ROLLOUT_ROOT},NORMALIZER_ROOT=${ARTIFACT_DIR},SPLIT_SEED=${SPLIT_SEED},CLEAN_BOUNDARY_ROOT=${CLEAN_BOUNDARY_ROOT},CLEAN_BOUNDARY_FILE=${CLEAN_BOUNDARY_FILE},TRAIN_TXT_NAME=${TRAIN_TXT_NAME},CKPT_ROOT=${MEMBER_ROOT},BASE_SEED=${BASE_MODEL_SEED},RUN_GROUP_PREFIX=${RUN_GROUP_PREFIX},WANDB_LOG=${WANDB_LOG},VERIFY_TRAINING=${VERIFY_TRAINING} \
  "${TRAIN_SLURM_DIR}/flood_wv_train_fgn_e4_dynamic_curriculum_single_gpu.sh")"

cat <<OUT
prep_job_id=${PREP_JOB_ID}
train_job_id=${TRAIN_JOB_ID}
run_root=${RUN_ROOT}
artifact_dir=${ARTIFACT_DIR}
member_root=${MEMBER_ROOT}
config_path=${TRAIN_CONFIG_RENDERED}
OUT
