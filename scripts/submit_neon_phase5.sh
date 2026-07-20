#!/usr/bin/env bash
set -euo pipefail

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
B3_DIR=${NEON_B3_DIR:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4_complete_resume_20260714/b3_n450_zero_init_6631946}
CONFIG=${NEON_CONFIG:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/config/coast_fgn_neon_tr450.yaml}
BUNDLE=${NEON_BUNDLE:-/scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json}
CACHE_DIR=${NEON_CACHE_DIR:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4/feature_cache_v3}
HISTORICAL_CONFIG=${NEON_HISTORICAL_CONFIG:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4/eval_b3_ood_historical/config/historical_13_single_reference.yaml}

cd "${REPO}"
[[ -d .git || -f .git ]] || { echo "Not a git worktree: ${REPO}" >&2; exit 2; }
HEAD=$(git rev-parse HEAD)
test -z "$(git status --porcelain)" || {
  echo "Refusing Phase-5 submission from a dirty tree" >&2
  exit 2
}
ROOT=${NEON_PHASE5_ROOT:-/scratch/jrj6wm/GINO_Model/neon_stage2_phase5/diagnose_then_pilot_${HEAD:0:7}}
P1A_DIR=${NEON_P1A_DIR:-${ROOT}/p1a_seed0_training}
if [[ -s "${ROOT}/SUBMITTED.json" ]]; then
  echo "Phase-5 root was already submitted: ${ROOT}" >&2
  exit 2
fi
mkdir -p "${ROOT}"

module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"
SOURCE_G1_DECISION=/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4_complete_resume_20260714/g1_stop_loss_6631946/DECISION.txt
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_phase5_quarantine.py \
  --run-root /scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4 \
  --decision "${SOURCE_G1_DECISION}" \
  --output "${ROOT}/QUARANTINE.json" --expected-head "${HEAD}"
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_phase5_preflight.py \
  --config "${CONFIG}" --bundle "${BUNDLE}" \
  --checkpoint "${B3_DIR}/neon_stage2_best.pt" \
  --history "${B3_DIR}/history.json" --preflight "${B3_DIR}/preflight.json" \
  --cache-dir "${CACHE_DIR}" --output-dir "${ROOT}" --expected-head "${HEAD}"

BASE_EXPORT="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_PHASE5_ROOT=${ROOT},NEON_CONFIG=${CONFIG},NEON_BUNDLE=${BUNDLE},NEON_B3_DIR=${B3_DIR},NEON_CACHE_DIR=${CACHE_DIR}"
TASK_SCRIPT=${REPO}/scripts/sbatch_neon_phase5_task.sh
DECISION_SCRIPT=${REPO}/scripts/sbatch_neon_phase5_decision.sh

# Validate every GPU command before any sbatch call. These paths intentionally
# do not need to exist yet because plan-only exits before loading checkpoints.
for task in geometry cancellation direct_data_last direct_rpf_last direct_rpf_full baseline_eval; do
  env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
    NEON_EXPECTED_HEAD="${HEAD}" NEON_PHASE5_ROOT="${ROOT}" \
    NEON_CONFIG="${CONFIG}" NEON_BUNDLE="${BUNDLE}" NEON_B3_DIR="${B3_DIR}" \
    NEON_CACHE_DIR="${CACHE_DIR}" NEON_PHASE5_TASK="${task}" NEON_PLAN_ONLY=1 \
    bash "${TASK_SCRIPT}"
done
# D5 uses the already validated historical single-reference configuration.
# Resolve the complete CLI before the first scheduler mutation; its ID metrics
# are produced later by the dependent baseline_eval job.
test -s "${HISTORICAL_CONFIG}"
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python -m neuralop.flood.cli.eval_neon_stage2 \
  --config "${HISTORICAL_CONFIG}" \
  --stage2-checkpoint "${B3_DIR}/neon_stage2_best.pt" \
  --stage1-bundle "${BUNDLE}" --output-dir "${ROOT}/d5_b3_ood/output" \
  --families all --m-eval 16 --k-eval 50 --rollout-length -1 \
  --thresholds 0.1 0.3 0.5 --seed 0 \
  --cache-dir "${ROOT}/d5_b3_ood/cache_k50" --k-chunk 8 \
  --compare-base --impact-metrics --variance-maps 0 \
  --expected-families 13 --allow-single-reference --dry-run >/dev/null
for task in p1a_eval direct_dirichlet; do
  env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
    NEON_EXPECTED_HEAD="${HEAD}" NEON_PHASE5_ROOT="${ROOT}" \
    NEON_CONFIG="${CONFIG}" NEON_BUNDLE="${BUNDLE}" NEON_B3_DIR="${B3_DIR}" \
    NEON_CACHE_DIR="${CACHE_DIR}" NEON_P1A_DIR="${P1A_DIR}" \
    NEON_PHASE5_TASK="${task}" NEON_PLAN_ONLY=1 bash "${TASK_SCRIPT}"
done
env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
  NEON_RUN_ROOT="${ROOT}" NEON_OUT_DIR="${P1A_DIR}" NEON_CACHE_DIR="${CACHE_DIR}" \
  NEON_SUBMIT_DRY_RUN=1 bash scripts/submit_neon_repair_rung.sh B5
# D4 is descriptive and non-gating, but its exact five legacy checkpoints,
# 250 artifact tasks, and current config must also resolve before the first
# scheduler mutation in this submission transaction.
env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
  NEON_CONFIG="${CONFIG}" NEON_BUNDLE="${BUNDLE}" NEON_SUBMIT_DRY_RUN=1 \
  bash scripts/submit_neon_phase5_legacy_sweep.sh "${ROOT}/d4_legacy"

submit_task() {
  local task=$1 dependency=${2:-} name=$3
  local args=(--parsable --job-name="${name}" \
    --output="${ROOT}/${task}_%j.out" --error="${ROOT}/${task}_%j.err" \
    --export="${BASE_EXPORT},NEON_PHASE5_TASK=${task}")
  if [[ -n "${dependency}" ]]; then args+=(--dependency="${dependency}"); fi
  sbatch "${args[@]}" "${TASK_SCRIPT}"
}

D2=$(submit_task geometry "" neon_p5_d2)
D5_ID=$(submit_task baseline_eval "afterok:${D2}" neon_p5_d5id)
D3=$(submit_task cancellation "afterok:${D2}" neon_p5_d3)
DATA=$(submit_task direct_data_last "afterok:${D3}" neon_p5_data)
RPF_LAST=$(submit_task direct_rpf_last "afterok:${D3}" neon_p5_rpf_l)
RPF_FULL=$(submit_task direct_rpf_full "afterok:${D3}" neon_p5_rpf_f)
GD0=$(sbatch --parsable --job-name=neon_p5_gd0 \
  --output="${ROOT}/gd0_%j.out" --error="${ROOT}/gd0_%j.err" \
  --dependency="afterok:${DATA}:${RPF_LAST}:${RPF_FULL}" \
  --export="${BASE_EXPORT},NEON_PHASE5_GATE=gd0" "${DECISION_SCRIPT}")

# P1a is mandatory for every scientific GD0 outcome. Those outcomes all emit a
# successful decision job, so afterok preserves the contract while failing
# closed when GD0 itself has an infrastructure or analysis failure.
env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
  NEON_RUN_ROOT="${ROOT}" NEON_OUT_DIR="${P1A_DIR}" NEON_CACHE_DIR="${CACHE_DIR}" \
  NEON_SBATCH_DEPENDENCY="afterok:${GD0}" \
  bash scripts/submit_neon_repair_rung.sh B5 >/dev/null
P1A_TRAIN=$(<"${P1A_DIR}/job_id.txt")
P1A_EXPORT="${BASE_EXPORT},NEON_P1A_DIR=${P1A_DIR}"
P1A_EVAL=$(sbatch --parsable --job-name=neon_p5_p1ae \
  --output="${ROOT}/p1a_eval_%j.out" --error="${ROOT}/p1a_eval_%j.err" \
  --dependency="afterok:${P1A_TRAIN}" \
  --export="${P1A_EXPORT},NEON_PHASE5_TASK=p1a_eval" "${TASK_SCRIPT}")
DIR_DIRECT=$(sbatch --parsable --job-name=neon_p5_p1ad \
  --output="${ROOT}/p1a_direct_%j.out" --error="${ROOT}/p1a_direct_%j.err" \
  --dependency="afterok:${P1A_TRAIN}" \
  --export="${P1A_EXPORT},NEON_PHASE5_TASK=direct_dirichlet" "${TASK_SCRIPT}")
GP1=$(sbatch --parsable --job-name=neon_p5_gp1 \
  --output="${ROOT}/gp1_%j.out" --error="${ROOT}/gp1_%j.err" \
  --dependency="afterok:${GD0}:${P1A_EVAL}:${DIR_DIRECT}" \
  --export="${P1A_EXPORT},NEON_PHASE5_GATE=gp1" \
  "${DECISION_SCRIPT}")

# D5 is descriptive and intentionally independent of the mechanism gates.
# The OOD array may begin only after the paired B3 ID report is complete.
env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" NEON_BUNDLE="${BUNDLE}" \
  NEON_HISTORICAL_CONFIG="${HISTORICAL_CONFIG}" \
  NEON_SBATCH_DEPENDENCY="afterok:${D5_ID}" \
  bash scripts/submit_neon_phase5_ood_evidence.sh \
  "${ROOT}" "${B3_DIR}" "${ROOT}/d5_b3_id/RESULT.json" "${ROOT}/d5_b3_ood" \
  > "${ROOT}/d5_ood_submit.log"
D5_OOD_FINAL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["finalize_job_id"])' \
  "${ROOT}/d5_b3_ood/SUBMITTED.json")

# D4 runs independently of GD0/GP1 and cannot alter either decision.
env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
  NEON_CONFIG="${CONFIG}" NEON_BUNDLE="${BUNDLE}" \
  bash scripts/submit_neon_phase5_legacy_sweep.sh "${ROOT}/d4_legacy" \
  > "${ROOT}/d4_legacy_submit.log"
D4_FINAL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["finalize_job_id"])' \
  "${ROOT}/d4_legacy/SUBMITTED.json")

apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' "${ROOT}" "${HEAD}" "${D2}" "${D3}" "${DATA}" "${RPF_LAST}" "${RPF_FULL}" "${GD0}" "${P1A_TRAIN}" "${P1A_EVAL}" "${DIR_DIRECT}" "${GP1}" "${D4_FINAL}" "${D5_ID}" "${D5_OOD_FINAL}"
import pathlib, sys
from neuralop.flood.eval.neon_phase5 import write_checksummed_artifact
keys = ["d2", "d3", "direct_data", "direct_rpf_last", "direct_rpf_full", "gd0", "p1a_train", "p1a_eval", "dirichlet_direct", "gp1", "d4_legacy_final", "d5_b3_id", "d5_b3_ood_final"]
root, head, *values = sys.argv[1:]
payload = {"schema_version": "neon_phase5_submission_v1", "git_head": head,
           "root": root, "jobs": dict(zip(keys, values))}
path = pathlib.Path(root) / "SUBMITTED.json"
write_checksummed_artifact(path, payload)
print(path)
PY
printf 'Phase-5 chain submitted under %s; GP1 terminal job %s\n' "${ROOT}" "${GP1}"
