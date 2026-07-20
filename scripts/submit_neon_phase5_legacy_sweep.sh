#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi
# Preserve Rivanna's container-visible /scratch path. Resolving host symlinks
# rewrites it to /sfs/weka/scratch, which is not mounted in the container.
OUTPUT_ROOT=$(realpath -ms "$1")
REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
CONFIG=${NEON_CONFIG:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/config/coast_fgn_neon_tr450.yaml}
BUNDLE=${NEON_BUNDLE:-/scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json}
SOURCE_ROOT=${NEON_LEGACY_SOURCE_ROOT:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train}
PLAN=${OUTPUT_ROOT}/legacy_export_plan.json

cd "${REPO}"
HEAD=$(git rev-parse HEAD)
test -z "$(git status --porcelain)" || {
  echo "Refusing legacy sweep submission from a dirty repository" >&2
  exit 2
}
[[ -f "${CONTAINER}" ]] || { echo "Missing container: ${CONTAINER}" >&2; exit 2; }
if [[ -s "${OUTPUT_ROOT}/SUBMITTED.json" ]]; then
  echo "Legacy sweep already submitted: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_legacy_sweep_preflight.py prepare \
  --config "${CONFIG}" --bundle "${BUNDLE}" \
  --source-root "${SOURCE_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --expected-head "${HEAD}" --plan "${PLAN}" >/dev/null

# Resolve all five legacy checkpoint/config combinations before the scheduler
# is mutated. Rivanna's system Python/PyYAML is deliberately not used here.
for N_TRAIN in 25 50 100 250 400; do
  apptainer exec --bind /scratch,/home "${CONTAINER}" \
    python -m neuralop.flood.cli.eval_neon_stage2 \
    --config "${CONFIG}" \
    --stage2-checkpoint "${SOURCE_ROOT}/tr_n${N_TRAIN}/neon_stage2_best.pt" \
    --stage1-bundle "${BUNDLE}" --output-dir "${OUTPUT_ROOT}/n${N_TRAIN}/output" \
    --families val --m-eval 16 --k-eval 8 --seed 0 --variance-maps 0 \
    --compare-base --write-artifacts --family-index 0 --shard-only --dry-run >/dev/null
done
if [[ "${NEON_SUBMIT_DRY_RUN:-0}" == 1 ]]; then
  printf 'Validated preliminary legacy N-sweep: %s\n' "${PLAN}"
  exit 0
fi

EXPORTS="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_LEGACY_PLAN=${PLAN}"
ARRAY_JOB=$(sbatch --parsable \
  --output="${OUTPUT_ROOT}/export_%A_%a.out" \
  --error="${OUTPUT_ROOT}/export_%A_%a.err" \
  --export="${EXPORTS}" scripts/sbatch_neon_phase5_legacy_export.sh)
FINAL_JOB=$(sbatch --parsable --job-name=neon_p5_d4final \
  --output="${OUTPUT_ROOT}/finalize_%j.out" \
  --error="${OUTPUT_ROOT}/finalize_%j.err" \
  --dependency="afterok:${ARRAY_JOB}" --export="${EXPORTS}" \
  scripts/sbatch_neon_phase5_legacy_finalize.sh)
apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' \
  "${OUTPUT_ROOT}" "${HEAD}" "${ARRAY_JOB}" "${FINAL_JOB}" "${PLAN}"
import hashlib, json, pathlib, sys
root, head, array_job, final_job, plan = sys.argv[1:]
payload = {"schema_version": "neon_phase5_legacy_submission_v1", "git_head": head,
           "array_job_id": array_job, "finalize_job_id": final_job, "plan": plan}
path = pathlib.Path(root) / "SUBMITTED.json"
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
path.write_bytes(encoded)
path.with_suffix(path.suffix + ".sha256").write_text(
    f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n")
PY
printf 'Submitted D4 legacy export=%s finalize=%s\n' "${ARRAY_JOB}" "${FINAL_JOB}"
