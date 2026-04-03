#!/bin/bash
#SBATCH -A uqgroup_plus
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -J wv_norm_prep
#SBATCH -o runtime/logs/out/wv_norm_prep-%A.out
#SBATCH -e runtime/logs/err/wv_norm_prep-%A.err

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
mkdir -p runtime/logs/out runtime/logs/err runtime/checkpoints

CONTAINER_PATH="/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif"
TRAIN_CONFIG="${TRAIN_CONFIG:?TRAIN_CONFIG must point to the rendered config}"
ARTIFACT_DIR="${ARTIFACT_DIR:?ARTIFACT_DIR must point to the run-scoped artifacts dir}"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT must be set}"
SPLIT_SEED="${SPLIT_SEED:-123}"
OVERWRITE_NORMALIZERS="${OVERWRITE_NORMALIZERS:-false}"

slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
export OMP_NUM_THREADS=8

echo "Preparing shared normalizers"
echo "Config:        ${TRAIN_CONFIG}"
echo "Artifact dir:  ${ARTIFACT_DIR}"
echo "Data root:     ${DATA_ROOT}"
echo "Split seed:    ${SPLIT_SEED}"
echo "Overwrite:     ${OVERWRITE_NORMALIZERS}"
echo "Git commit:    $(git -C "${PROJECT_DIR}" rev-parse HEAD)"

CLI_ARGS=(
  -m neuralop.flood.cli.prepare_flood_normalizers
  --config-path "${TRAIN_CONFIG}"
  --artifact-root "${ARTIFACT_DIR}"
  --data-root "${DATA_ROOT}"
  --split-seed "${SPLIT_SEED}"
)
if [[ "${OVERWRITE_NORMALIZERS}" == "true" ]]; then
  CLI_ARGS+=(--overwrite)
fi

apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" \
  python3 "${CLI_ARGS[@]}"
