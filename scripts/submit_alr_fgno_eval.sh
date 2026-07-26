#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 {heldout|historical} CHECKPOINT_DIR [--prepare-only]" >&2
  exit 2
fi
DATASET="$1"
CHECKPOINT_DIR="$2"
PREPARE_ONLY=false
[[ ${3:-} == "--prepare-only" ]] && PREPARE_ONLY=true
[[ " heldout historical " == *" ${DATASET} "* ]] || {
  echo "Dataset must be heldout or historical." >&2
  exit 2
}

PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_clean_mcdropout}"
BASE_CONFIG="${BASE_CONFIG:-${PROJECT_DIR}/config/flood/coastal/gino_pluvial_flood_config_coastal_alr_fgn_pilot.yaml}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/GINO_Model/alr_fgno_pilot/evaluation}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
RUN_DIR="${RUN_ROOT}/${DATASET}_${STAMP}_${SHA}"
CONFIG_DIR="${RUN_DIR}/config"
EVENT_LIST="${RUN_DIR}/event_ids.txt"
mkdir -p "${CONFIG_DIR}" "${RUN_DIR}/slurm" "${RUN_DIR}/forecast_artifacts/${DATASET}"

if [[ ${DATASET} == heldout ]]; then
  seq -f 'TE%06g' 1 50 > "${EVENT_LIST}"
else
  HIST_ROOT=/scratch/jrj6wm/uncertainty_floodmodel/results/coastal/portsmouth/Coastal_Flood_historical_extreme_events_15min_20260625/test
  find "${HIST_ROOT}" -maxdepth 1 -name 'Flood_coastal_HIST_*_sim00.hdf' -printf '%f\n' \
    | sed -e 's/_sim00\.hdf$//' | sort > "${EVENT_LIST}"
fi

while IFS= read -r EVENT_ID; do
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/render_alr_fgno_eval_config.py" \
    --base "${BASE_CONFIG}" \
    --output "${CONFIG_DIR}/${EVENT_ID}.yaml" \
    --run-dir "${RUN_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --dataset "${DATASET}" \
    --event-id "${EVENT_ID}" >/dev/null
done < "${EVENT_LIST}"

N_EVENTS="$(wc -l < "${EVENT_LIST}")"
test "${N_EVENTS}" -gt 0
git -C "${PROJECT_DIR}" rev-parse HEAD > "${RUN_DIR}/git_head.txt"
printf 'Dataset: %s\nCheckpoint: %s\nEvents: %s\nRun: %s\n' \
  "${DATASET}" "${CHECKPOINT_DIR}" "${N_EVENTS}" "${RUN_DIR}"
if ${PREPARE_ONLY}; then
  exit 0
fi

JOB_ID="$(sbatch --parsable \
  --array="0-$((N_EVENTS - 1))%5" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ALR_EVAL_RUN_DIR="${RUN_DIR}",ALR_EVAL_CONFIG_DIR="${CONFIG_DIR}",ALR_EVAL_EVENT_LIST="${EVENT_LIST}" \
  "${PROJECT_DIR}/scripts/slurm/eval/flood_coastal_eval_alr_fgn.sh")"
printf '%s\n' "${JOB_ID}" > "${RUN_DIR}/job_id.txt"
echo "Submitted ${JOB_ID}"
