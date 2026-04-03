#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -J coastal_hpo_norm
#SBATCH -o runtime/logs/out/coastal_hpo_norm-%j.out
#SBATCH -e runtime/logs/err/coastal_hpo_norm-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_no_physics_git_main}"
CANONICAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
COMMON_SH=""
for candidate in \
  "${PROJECT_DIR}/scripts/slurm/lib/common.sh" \
  "${SLURM_SUBMIT_DIR:-}/slurm/lib/common.sh" \
  "${SLURM_SUBMIT_DIR:-}/scripts/slurm/lib/common.sh" \
  "$(cd "${CANONICAL_DIR}/.." && pwd)/lib/common.sh"
do
  if [[ -n "${candidate}" && -f "${candidate}" ]]; then
    COMMON_SH="${candidate}"
    break
  fi
done
if [[ -z "${COMMON_SH}" ]]; then
  echo "ERROR: Unable to locate scripts/slurm/lib/common.sh" >&2
  exit 1
fi
source "${COMMON_SH}"
slurm_load_apptainer
slurm_configure_host_ca
SCRIPT_DIR="${PROJECT_DIR}/scripts"
cd "${SCRIPT_DIR}"
mkdir -p runtime/logs/out runtime/logs/err

CONTAINER_PATH="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
STUDY_ROOT="${STUDY_ROOT:?STUDY_ROOT is required}"
OVERWRITE_NORMALIZERS="${OVERWRITE_NORMALIZERS:-false}"
export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
slurm_assert_container_gpus "${CONTAINER_PATH}" 1

ARGS=(
  precompute-study-normalizers
  --study-root "${STUDY_ROOT}"
)
if [[ "${OVERWRITE_NORMALIZERS}" == "true" ]]; then
  ARGS+=(--overwrite)
fi
apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python -m neuralop.flood.cli.coastal_fgn_hpo "${ARGS[@]}"
