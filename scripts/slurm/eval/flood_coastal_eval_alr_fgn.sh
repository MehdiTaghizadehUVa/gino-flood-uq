#!/bin/bash
#SBATCH --job-name=coast_alr_eval
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/jrj6wm/GINO_Model/alr_fgno_pilot/slurm-%A_%a.out
#SBATCH --error=/scratch/jrj6wm/GINO_Model/alr_fgno_pilot/slurm-%A_%a.err
set -euo pipefail

: "${ALR_EVAL_RUN_DIR:?ALR_EVAL_RUN_DIR is required}"
: "${ALR_EVAL_CONFIG_DIR:?ALR_EVAL_CONFIG_DIR is required}"
: "${ALR_EVAL_EVENT_LIST:?ALR_EVAL_EVENT_LIST is required}"
PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_clean_mcdropout}"
CONTAINER="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
EVENT_ID="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${ALR_EVAL_EVENT_LIST}")"
CONFIG_PATH="${ALR_EVAL_CONFIG_DIR}/${EVENT_ID}.yaml"
EVENT_DIR="${ALR_EVAL_RUN_DIR}/events/${EVENT_ID}"
mkdir -p "${EVENT_DIR}"
test -s "${CONFIG_PATH}"

cd "${PROJECT_DIR}"
git diff --quiet && git diff --cached --quiet || {
  echo "Refusing to evaluate with modified tracked files." >&2
  exit 2
}
module purge
module load apptainer
source "${PROJECT_DIR}/scripts/slurm/lib/common.sh"
slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}"
slurm_assert_container_gpus "${CONTAINER}" 1

echo "ALR-FGNO event=${EVENT_ID} config=${CONFIG_PATH} git=$(git rev-parse HEAD)"
apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER}" \
  "${PROJECT_DIR}/scripts/flood_wv_eval_operator.py" \
  --config_path "${CONFIG_PATH}" \
  --skip_single_step \
  --run_rollout \
  --eval_log_file "${EVENT_DIR}/evaluation.log"
