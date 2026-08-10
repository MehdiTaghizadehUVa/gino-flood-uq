#!/bin/bash
#SBATCH --account=uqgroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=alr_ref_stats
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_clean_mcdropout}"
CONTAINER="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
: "${ALR_REFERENCE_MODE:?ALR_REFERENCE_MODE must be shard or merge}"
: "${ALR_REFERENCE_ROOT:?ALR_REFERENCE_ROOT is required}"
: "${ALR_EXPECTED_COMMIT:?ALR_EXPECTED_COMMIT is required}"

cd "${PROJECT_DIR}"
test "$(git rev-parse HEAD)" = "${ALR_EXPECTED_COMMIT}" || {
  echo "Repository HEAD moved after submission; refusing to build mixed-provenance data." >&2
  exit 78
}
module purge
module load apptainer

if [[ "${ALR_REFERENCE_MODE}" == "shard" ]]; then
  : "${SLURM_ARRAY_TASK_ID:?Shard mode requires a Slurm array task ID}"
  apptainer exec --bind /scratch:/scratch,/home:/home "${CONTAINER}" \
    python "${PROJECT_DIR}/scripts/build_reference_dispersion_table.py" shard \
      --shard "${SLURM_ARRAY_TASK_ID}" \
      --num-shards "${ALR_REFERENCE_SHARDS:-20}" \
      --output "${ALR_REFERENCE_ROOT}/shards"
elif [[ "${ALR_REFERENCE_MODE}" == "merge" ]]; then
  apptainer exec --bind /scratch:/scratch,/home:/home "${CONTAINER}" \
    python "${PROJECT_DIR}/scripts/build_reference_dispersion_table.py" merge \
      --shards "${ALR_REFERENCE_ROOT}/shards" \
      --output "${ALR_REFERENCE_ROOT}/reference_statistics_train.pt"
else
  echo "Unknown ALR_REFERENCE_MODE=${ALR_REFERENCE_MODE}" >&2
  exit 2
fi
