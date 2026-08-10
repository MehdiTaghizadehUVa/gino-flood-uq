#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {smoke|pilot50|residual50_epoch1|n150|full450} [--prepare-only]" >&2
  exit 2
fi
RUN_KIND="$1"
[[ " smoke pilot50 residual50_epoch1 n150 full450 " == *" ${RUN_KIND} "* ]] || {
  echo "Unknown run kind: ${RUN_KIND}" >&2
  exit 2
}
PREPARE_ONLY=false
if [[ ${2:-} == "--prepare-only" ]]; then PREPARE_ONLY=true; fi

PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_clean_mcdropout}"
BASE_CONFIG="${BASE_CONFIG:-${PROJECT_DIR}/config/flood/coastal/gino_pluvial_flood_config_coastal_alr_fgn_pilot.yaml}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/GINO_Model/alr_fgno_pilot}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FULL_SHA="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
SHA="${FULL_SHA:0:7}"
RUN_DIR="${RUN_ROOT}/${RUN_KIND}_${STAMP}_${SHA}"
CONFIG_PATH="${RUN_DIR}/config/${RUN_KIND}.yaml"
mkdir -p "${RUN_DIR}/config"

# Config generation happens before sbatch. The renderer contains a PyYAML
# compatibility fallback for Rivanna's older system installation.
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/render_alr_fgno_pilot_config.py" \
  --base "${BASE_CONFIG}" \
  --output "${CONFIG_PATH}" \
  --run-dir "${RUN_DIR}" \
  --run-kind "${RUN_KIND}"
test -s "${CONFIG_PATH}"

printf 'Run directory: %s\nConfig: %s\nGit: %s\n' "${RUN_DIR}" "${CONFIG_PATH}" "${SHA}"
if ${PREPARE_ONLY}; then
  exit 0
fi

WALLTIME="24:00:00"
if [[ "${RUN_KIND}" == "residual50_epoch1" ]]; then WALLTIME="10:00:00"; fi
JOB_ID="$(sbatch --parsable \
  --time="${WALLTIME}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ALR_CONFIG_PATH="${CONFIG_PATH}",ALR_RUN_DIR="${RUN_DIR}",ALR_EXPECTED_COMMIT="${FULL_SHA}" \
  "${PROJECT_DIR}/scripts/slurm/train/flood_coastal_train_alr_fgn.sh")"
printf '%s\n' "${JOB_ID}" > "${RUN_DIR}/job_id.txt"
echo "Submitted ${JOB_ID}"
