#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J ddofs_wv_ev
#SBATCH -o runtime/logs/out/ddofs_wv_ev-%j.out
#SBATCH -e runtime/logs/err/ddofs_wv_ev-%j.err

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
mkdir -p runtime/logs/out runtime/logs/err runtime/eval_outputs

PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_no_physics_git_main}"
export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${PROJECT_DIR}/scripts/flood_wv_eval_diffusion.py}"
EVAL_CONFIG="${EVAL_CONFIG:-${PROJECT_DIR}/config/flood/wv/gino_pluvial_flood_config_WV_depth_only_diffusion.yaml}"
CONTAINER_PATH="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
TEST_ROOT="${TEST_ROOT:-/scratch/$USER/Data_Generation_UQ/Results_Test/M40}"
TRAIN_ROOT="${TRAIN_ROOT:-/scratch/$USER/Data_Generation_UQ/Results/M40}"
TRAIN_NORMALIZER="${TRAIN_NORMALIZER:-${TRAIN_ROOT}/normalizers_depth_only.pt}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_DIR}/scripts/runtime/checkpoints_WV_depth_only_diffusion}"
JOB_NAME="${JOB_NAME:-ddofs_wv_ev}"
OUT_DIR_DEFAULT="${PROJECT_DIR}/scripts/runtime/eval_outputs/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${OUT_DIR_DEFAULT}}"

slurm_configure_host_ca
slurm_prepare_geo_venv "${CONTAINER_PATH}"

echo "Eval script:      ${EVAL_SCRIPT}"
echo "Config:           ${EVAL_CONFIG}"
echo "Checkpoint root:  ${CHECKPOINT_ROOT}"
echo "Test data root:   ${TEST_ROOT}"
echo "Train normalizer: ${TRAIN_NORMALIZER}"
echo "Output dir:       ${OUT_DIR}"
echo "Job label:        ${JOB_NAME}"
echo "Host:             $(hostname)"

apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" \
  python "${EVAL_SCRIPT}" \
  --config_path "${EVAL_CONFIG}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --data.root "${TEST_ROOT}" \
  --data.normalizer_path "${TRAIN_NORMALIZER}" \
  --rollout_data.root "${TEST_ROOT}" \
  --rollout.out_dir "${OUT_DIR}"
