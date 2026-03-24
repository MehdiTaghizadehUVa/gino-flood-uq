#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -J gino_fgn_masked_primary
#SBATCH -o runtime/logs/out/gino_fgn_masked_primary-%j.out
#SBATCH -e runtime/logs/err/gino_fgn_masked_primary-%j.err

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
mkdir -p runtime/logs/out runtime/logs/err

export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
CONTAINER_PATH="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/flood_wv_train_operator.py"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_DIR}/config/flood/wv/gino_pluvial_flood_config_WV_depth_only_masked_primary.yaml}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/train}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/test}"
ARTIFACT_DIR="${ARTIFACT_DIR:?ARTIFACT_DIR must be set}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set}"
MODE="${MODE:-full}"
SEED="${SEED:-123}"
RUN_GROUP="${RUN_GROUP:-wv_masked_primary_manual}"
WANDB_GROUP="${WANDB_GROUP:-${RUN_GROUP}}"
WANDB_NAME_DEFAULT="${RUN_GROUP}_${MODE}_seed${SEED}"
WANDB_NAME="${WANDB_NAME:-${WANDB_NAME_DEFAULT}}"
WANDB_LOG_DEFAULT="true"
if [[ "${MODE}" == "canary" ]]; then
  WANDB_LOG_DEFAULT="false"
fi
WANDB_LOG="${WANDB_LOG:-${WANDB_LOG_DEFAULT}}"
NORMALIZER_PATH="${NORMALIZER_PATH:-${ARTIFACT_DIR}/normalizers_depth_only_masked_primary.pt}"
MASK_PATH="${MASK_PATH:-${ARTIFACT_DIR}/structural_dry_mask_exact_zero.pt}"
SUMMARY_PATH="${ARTIFACT_DIR}/prepare_flood_training_artifacts_summary.json"

if [[ ! -f "${NORMALIZER_PATH}" ]]; then
  echo "ERROR: normalizer artifact not found: ${NORMALIZER_PATH}" >&2
  exit 2
fi
if [[ ! -f "${MASK_PATH}" ]]; then
  echo "ERROR: structural-dry artifact not found: ${MASK_PATH}" >&2
  exit 2
fi

slurm_configure_host_ca
slurm_assert_container_gpus "${CONTAINER_PATH}" 1
mkdir -p "${CHECKPOINT_DIR}"

CLI_ARGS=(
  --config_path "${TRAIN_CONFIG}"
  --data.root "${DATA_ROOT}"
  --rollout_data.root "${ROLLOUT_ROOT}"
  --data.normalizer_root "${ARTIFACT_DIR}"
  --data.normalizer_path "${NORMALIZER_PATH}"
  --structural_dry.mask_path "${MASK_PATH}"
  --distributed.seed "${SEED}"
  --deterministic false
  --verify_training false
  --rollout.run_after_training false
  --checkpoint.save_dir "${CHECKPOINT_DIR}"
  --wandb.log "${WANDB_LOG}"
  --wandb.group "${WANDB_GROUP}"
  --wandb.name "${WANDB_NAME}"
)

if [[ "${MODE}" == "canary" ]]; then
  CLI_ARGS+=(
    --opt.n_epochs 1
    --data.n_samples_max 4096
    --data.force_load_normalizers true
  )
fi

echo "Train script:     ${TRAIN_SCRIPT}"
echo "Config:           ${TRAIN_CONFIG}"
echo "Mode:             ${MODE}"
echo "Data root:        ${DATA_ROOT}"
echo "Rollout root:     ${ROLLOUT_ROOT}"
echo "Artifact dir:     ${ARTIFACT_DIR}"
echo "Checkpoint dir:   ${CHECKPOINT_DIR}"
echo "Normalizers:      ${NORMALIZER_PATH}"
echo "Dry mask:         ${MASK_PATH}"
echo "Prep summary:     ${SUMMARY_PATH}"
echo "Seed:             ${SEED}"
echo "W&B log:          ${WANDB_LOG}"
echo "W&B group:        ${WANDB_GROUP}"
echo "W&B name:         ${WANDB_NAME}"
echo "Git commit:       $(git -C "${PROJECT_DIR}" rev-parse HEAD)"

apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${TRAIN_SCRIPT}" "${CLI_ARGS[@]}"
