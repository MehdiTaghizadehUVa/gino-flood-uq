#!/bin/bash
#SBATCH --account=uqgroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
set -euo pipefail

: "${NEON_EXPECTED_HEAD:?Set NEON_EXPECTED_HEAD}"
: "${NEON_LEGACY_PLAN:?Set NEON_LEGACY_PLAN}"

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}"
test -z "$(git status --porcelain)"
module purge
module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"
slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"

apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_legacy_sweep_preflight.py validate-final \
  --plan "${NEON_LEGACY_PLAN}" --expected-head "${NEON_EXPECTED_HEAD}" >/dev/null
readarray -t PLAN_VALUES < <(apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' "${NEON_LEGACY_PLAN}"
import json, sys
p=json.load(open(sys.argv[1])); print(p["config"]); print(p["bundle"]); print(p["output_root"]); print(p["source_root"])
PY
)
CONFIG=${PLAN_VALUES[0]}
BUNDLE=${PLAN_VALUES[1]}
OUTPUT_ROOT=${PLAN_VALUES[2]}
SOURCE_ROOT=${PLAN_VALUES[3]}

for N_TRAIN in 25 50 100 250 400; do
  CHECKPOINT="${SOURCE_ROOT}/tr_n${N_TRAIN}/neon_stage2_best.pt"
  OUTPUT_DIR="${OUTPUT_ROOT}/n${N_TRAIN}/output"
  ARTIFACT_DIR="${OUTPUT_DIR}/artifacts"
  test "$(find "${ARTIFACT_DIR}" -maxdepth 1 -type f -name '*.h5' | wc -l)" -eq 50
  apptainer exec --bind /scratch,/home "${CONTAINER}" \
    python -m neuralop.flood.cli.eval_neon_stage2 \
    --config "${CONFIG}" --stage2-checkpoint "${CHECKPOINT}" \
    --stage1-bundle "${BUNDLE}" --output-dir "${OUTPUT_DIR}" \
    --families val --m-eval 16 --k-eval 8 --seed 0 \
    --variance-maps 0 --compare-base --write-artifacts \
    --merge-only --expected-families 50
  apptainer exec --bind /scratch,/home "${CONTAINER}" \
    python scripts/neon_legacy_estimator_remap.py \
    --artifacts "${ARTIFACT_DIR}" \
    --output-dir "${OUTPUT_ROOT}/n${N_TRAIN}/remap" --map-events 0
done

apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_legacy_sweep_analysis.py \
  --run-root "${OUTPUT_ROOT}" --source-root "${SOURCE_ROOT}" \
  --output-prefix "${OUTPUT_ROOT}/legacy_n_sweep_preliminary"
printf '%s\n' "${SLURM_JOB_ID:-manual}" > "${OUTPUT_ROOT}/finalize_job_id.txt"
