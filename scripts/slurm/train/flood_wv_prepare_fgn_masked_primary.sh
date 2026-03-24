#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -J wv_mask_prep
#SBATCH -o runtime/logs/out/wv_mask_prep-%j.out
#SBATCH -e runtime/logs/err/wv_mask_prep-%j.err

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
PREP_SCRIPT="${PROJECT_DIR}/neuralop/flood/cli/prepare_flood_training_artifacts.py"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_DIR}/config/flood/wv/gino_pluvial_flood_config_WV_depth_only_masked_primary.yaml}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/train}"
SEED="${SEED:-123}"
ARTIFACT_DIR="${ARTIFACT_DIR:?ARTIFACT_DIR must be set}"
PREP_OVERWRITE="${PREP_OVERWRITE:-0}"

slurm_configure_host_ca
slurm_assert_container_gpus "${CONTAINER_PATH}" 1

CLI_ARGS=(
  "${PREP_SCRIPT}"
  --config_path "${TRAIN_CONFIG}"
  --artifact_root "${ARTIFACT_DIR}"
  --data_root "${DATA_ROOT}"
  --seed "${SEED}"
)
if [[ "${PREP_OVERWRITE}" == "1" || "${PREP_OVERWRITE}" == "true" ]]; then
  CLI_ARGS+=(--overwrite)
fi

echo "Prep script:   ${PREP_SCRIPT}"
echo "Config:        ${TRAIN_CONFIG}"
echo "Data root:     ${DATA_ROOT}"
echo "Artifact dir:  ${ARTIFACT_DIR}"
echo "Seed:          ${SEED}"
echo "Overwrite:     ${PREP_OVERWRITE}"
echo "Git commit:    $(git -C "${PROJECT_DIR}" rev-parse HEAD)"

apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python "${CLI_ARGS[@]}"
