#!/bin/bash
#SBATCH --job-name=coast_alr_fgno
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jrj6wm/GINO_Model/alr_fgno_pilot/slurm-%j.out
#SBATCH --error=/scratch/jrj6wm/GINO_Model/alr_fgno_pilot/slurm-%j.err
set -euo pipefail

: "${ALR_CONFIG_PATH:?ALR_CONFIG_PATH must point to a pre-rendered config}"
PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_clean_mcdropout}"
CONTAINER="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
RUN_DIR="${ALR_RUN_DIR:-$(dirname "${ALR_CONFIG_PATH}")}"

cd "${PROJECT_DIR}"
git diff --quiet && git diff --cached --quiet || {
  echo "Refusing to train with modified tracked files." >&2
  exit 2
}
test -f "${ALR_CONFIG_PATH}"
test -f "${CONTAINER}"
mkdir -p "${RUN_DIR}"
# A queued job runs from the live working tree, so it executes whatever HEAD is
# when it STARTS -- not the commit it was submitted at.  Any commit landing
# during a long Priority wait silently changes the experiment.  Export
# ALR_EXPECTED_COMMIT at submission to pin it.
if [ -n "${ALR_EXPECTED_COMMIT:-}" ]; then
  ACTUAL_COMMIT="$(git rev-parse HEAD)"
  case "${ACTUAL_COMMIT}" in
    "${ALR_EXPECTED_COMMIT}"*) : ;;
    *)
      echo "FATAL: expected commit ${ALR_EXPECTED_COMMIT} but HEAD is ${ACTUAL_COMMIT}." >&2
      echo "The repository moved while this job was queued; refusing to run." >&2
      exit 78
      ;;
  esac
fi
git rev-parse HEAD > "${RUN_DIR}/git_head.txt"
git status --short > "${RUN_DIR}/git_status_at_launch.txt"
cp "${ALR_CONFIG_PATH}" "${RUN_DIR}/effective_submission_config.yaml"

module purge
module load apptainer
source "${PROJECT_DIR}/scripts/slurm/lib/common.sh"
slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}"
slurm_assert_container_gpus "${CONTAINER}" 1

apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER}" \
  "${PROJECT_DIR}/scripts/flood_wv_train_operator.py" \
  --config_path "${ALR_CONFIG_PATH}"
