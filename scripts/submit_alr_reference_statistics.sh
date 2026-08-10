#!/bin/bash
set -euo pipefail

PREPARE_ONLY=false
if [[ ${1:-} == "--prepare-only" ]]; then
  PREPARE_ONLY=true
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--prepare-only]" >&2
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_clean_mcdropout}"
RUN_ROOT="${ALR_REFERENCE_ROOT:-/scratch/$USER/GINO_Model/alr_fgno_pilot/residual_pilot}"
WORKER="${PROJECT_DIR}/scripts/slurm/data/flood_build_alr_reference_statistics.sh"
BUILDER="${PROJECT_DIR}/scripts/build_reference_dispersion_table.py"
BASE_CONFIG="${BASE_CONFIG:-${PROJECT_DIR}/config/flood/coastal/gino_pluvial_flood_config_coastal_alr_fgn_pilot.yaml}"
CONTAINER="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
NUM_SHARDS="${ALR_REFERENCE_SHARDS:-20}"
FULL_SHA="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"

test -x "${WORKER}"
test -s "${BUILDER}"
test -s "${BASE_CONFIG}"
[[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ALR_REFERENCE_SHARDS must be a positive integer" >&2
  exit 2
}

# Validate both the estimator and the configured HDF layout before sbatch.
module purge
module load apptainer
apptainer exec --bind /scratch:/scratch,/home:/home "${CONTAINER}" \
  python "${BUILDER}" selftest
apptainer exec --bind /scratch:/scratch,/home:/home "${CONTAINER}" \
  python - "${BASE_CONFIG}" <<'PY'
import sys
from pathlib import Path
from scripts.build_reference_dispersion_table import resolve_hdf_paths

path = Path(sys.argv[1])
resolved = resolve_hdf_paths(path)
if not resolved.get("wd"):
    raise SystemExit(f"{path}: resolved water-depth HDF path is empty")
print(f"validated HDF water-depth path: {resolved['wd']}")
PY

printf 'Reference root: %s\nGit: %s\nShards: %s\n' \
  "${RUN_ROOT}" "${FULL_SHA}" "${NUM_SHARDS}"
if ${PREPARE_ONLY}; then
  exit 0
fi

mkdir -p "${RUN_ROOT}/shards" "${RUN_ROOT}/logs"
SHARD_JOB="$(sbatch --parsable \
  --array="0-$((NUM_SHARDS - 1))" \
  --output="${RUN_ROOT}/logs/shard_%A_%a.out" \
  --error="${RUN_ROOT}/logs/shard_%A_%a.err" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONTAINER_PATH="${CONTAINER}",ALR_REFERENCE_MODE=shard,ALR_REFERENCE_ROOT="${RUN_ROOT}",ALR_REFERENCE_SHARDS="${NUM_SHARDS}",ALR_EXPECTED_COMMIT="${FULL_SHA}" \
  "${WORKER}")"

MERGE_JOB="$(sbatch --parsable \
  --dependency="afterok:${SHARD_JOB}" \
  --mem=32G \
  --time=02:00:00 \
  --output="${RUN_ROOT}/logs/merge_%j.out" \
  --error="${RUN_ROOT}/logs/merge_%j.err" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONTAINER_PATH="${CONTAINER}",ALR_REFERENCE_MODE=merge,ALR_REFERENCE_ROOT="${RUN_ROOT}",ALR_REFERENCE_SHARDS="${NUM_SHARDS}",ALR_EXPECTED_COMMIT="${FULL_SHA}" \
  "${WORKER}")"

printf '%s\n' "${SHARD_JOB}" > "${RUN_ROOT}/shard_job_id.txt"
printf '%s\n' "${MERGE_JOB}" > "${RUN_ROOT}/merge_job_id.txt"
printf '%s\n' "${FULL_SHA}" > "${RUN_ROOT}/expected_commit.txt"
printf 'Submitted shards: %s\nSubmitted merge: %s\n' "${SHARD_JOB}" "${MERGE_JOB}"
