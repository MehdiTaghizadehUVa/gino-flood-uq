#!/bin/bash
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-249%10
set -euo pipefail

: "${NEON_EXPECTED_HEAD:?Set NEON_EXPECTED_HEAD}"
: "${NEON_LEGACY_PLAN:?Set NEON_LEGACY_PLAN}"

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}"
test -z "$(git status --porcelain)" || {
  echo "Refusing legacy export from a dirty repository" >&2
  exit 2
}
module purge
module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"
slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"

TASK_ROW=$(apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_legacy_sweep_preflight.py validate-task \
  --plan "${NEON_LEGACY_PLAN}" --expected-head "${NEON_EXPECTED_HEAD}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" --format tsv)
IFS=$'\t' read -r N_TRAIN FAMILY_INDEX CHECKPOINT OUTPUT_DIR <<<"${TASK_ROW}"
CONFIG=$(apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' "${NEON_LEGACY_PLAN}"
import json, sys
print(json.load(open(sys.argv[1]))["config"])
PY
)
BUNDLE=$(apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' "${NEON_LEGACY_PLAN}"
import json, sys
print(json.load(open(sys.argv[1]))["bundle"])
PY
)
CACHE_ROOT=$(apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' "${NEON_LEGACY_PLAN}"
import json, pathlib, sys
p=json.load(open(sys.argv[1])); print(pathlib.Path(p["output_root"]) / "cache_k8")
PY
)
mkdir -p "${OUTPUT_DIR}"

apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" \
  python -m neuralop.flood.cli.eval_neon_stage2 \
  --config "${CONFIG}" --stage2-checkpoint "${CHECKPOINT}" \
  --stage1-bundle "${BUNDLE}" --output-dir "${OUTPUT_DIR}" \
  --families val --m-eval 16 --k-eval 8 --seed 0 \
  --cache-dir "${CACHE_ROOT}" --k-chunk 8 --variance-maps 0 \
  --compare-base --write-artifacts --family-index "${FAMILY_INDEX}" \
  --shard-only --resume --expected-families 50

printf '%s\n' "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  > "${OUTPUT_DIR}/task_${FAMILY_INDEX}.job_id"
