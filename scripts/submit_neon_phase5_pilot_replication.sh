#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /scratch/.../neon_phase5_root" >&2
  exit 2
fi
ROOT=$1
REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
CONFIG=${NEON_CONFIG:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/config/coast_fgn_neon_tr450.yaml}
BUNDLE=${NEON_BUNDLE:-/scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json}
B3_DIR=${NEON_B3_DIR:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4_complete_resume_20260714/b3_n450_zero_init_6631946}
CACHE_DIR=${NEON_CACHE_DIR:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4/feature_cache_v3}

cd "${REPO}"
HEAD=$(git rev-parse HEAD)
test -z "$(git status --porcelain)" || { echo "Refusing dirty-tree submission" >&2; exit 2; }
SCREEN=${ROOT}/pilot_screen/DECISION.json
test -s "${SCREEN}" && test -s "${SCREEN}.sha256"
test -s "${ROOT}/CONDITIONAL_PILOT_SUBMITTED.json"
test -s "${ROOT}/CONDITIONAL_PILOT_SUBMITTED.json.sha256"
test ! -e "${ROOT}/PILOT_REPLICATION_SUBMITTED.json" || {
  echo "Pilot replication already submitted under ${ROOT}" >&2
  exit 2
}

module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"
read -r RUNG SCREEN_DIR < <(apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' \
  "${SCREEN}" "${ROOT}/CONDITIONAL_PILOT_SUBMITTED.json" "${HEAD}" \
  "${ROOT}/PROTOCOL.json" "${ROOT}/gp1_replicated/DECISION.json"
import pathlib, sys
from neuralop.flood.eval.neon_phase5 import (
    validate_pilot_rungs_for_gp1_decision,
    verify_checksummed_artifact,
    verify_phase5_decision_artifact,
)
screen_path = pathlib.Path(sys.argv[1])
submission_path = pathlib.Path(sys.argv[2])
head = sys.argv[3]
protocol_sha = verify_checksummed_artifact(sys.argv[4])
verify_checksummed_artifact(submission_path)
screen = verify_phase5_decision_artifact(
    screen_path, expected_head=head, protocol_sha256=protocol_sha
)
gp1 = verify_phase5_decision_artifact(
    sys.argv[5], expected_head=head, protocol_sha256=protocol_sha
)
submission = json.loads(submission_path.read_text())
assert submission["git_head"] == head
assert screen["gate"] == "PILOT_SCREEN"
assert screen["decision"] == "pilot_screen_passed"
rung = screen["selected_rung"]
validate_pilot_rungs_for_gp1_decision(gp1["decision"], {rung})
matches = [path for path in submission["pilot_dirs"] if pathlib.Path(path).name.startswith(rung.lower())]
assert len(matches) == 1, (rung, matches)
print(rung, matches[0])
PY
)

TASK_SCRIPT=${REPO}/scripts/sbatch_neon_phase5_task.sh
GATE_SCRIPT=${REPO}/scripts/sbatch_neon_phase5_pilot_gate.sh
BASE_EXPORT="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_PHASE5_ROOT=${ROOT},NEON_CONFIG=${CONFIG},NEON_BUNDLE=${BUNDLE},NEON_B3_DIR=${B3_DIR},NEON_CACHE_DIR=${CACHE_DIR}"
SEEDS=(20260704 20260705)
DIRS=("${SCREEN_DIR}")
TRAIN_JOBS=()
EVAL_JOBS=()
OOD_DIRS=()
OOD_FINAL_JOBS=()
if [[ "${RUNG}" == P2 ]]; then
  OOD_DIRS+=("${SCREEN_DIR}/ood_evidence")
fi

# Validate all commands before the first scheduler mutation.
for SEED in "${SEEDS[@]}"; do
  DIR=${ROOT}/${RUNG,,}_accept_seed${SEED}
  DIRS+=("${DIR}")
  env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
    NEON_RUN_ROOT="${ROOT}" NEON_OUT_DIR="${DIR}" NEON_CACHE_DIR="${CACHE_DIR}" \
    NEON_PRIOR_SEED="${SEED}" NEON_TRAIN_SEED=0 NEON_SUBMIT_DRY_RUN=1 \
    bash scripts/submit_neon_repair_rung.sh "${RUNG}" >/dev/null
  env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
    NEON_EXPECTED_HEAD="${HEAD}" NEON_PHASE5_ROOT="${ROOT}" \
    NEON_CONFIG="${CONFIG}" NEON_BUNDLE="${BUNDLE}" NEON_B3_DIR="${B3_DIR}" \
    NEON_CACHE_DIR="${CACHE_DIR}" NEON_PILOT_DIR="${DIR}" \
    NEON_PHASE5_TASK=pilot_eval NEON_PLAN_ONLY=1 bash "${TASK_SCRIPT}"
done

for SEED in "${SEEDS[@]}"; do
  DIR=${ROOT}/${RUNG,,}_accept_seed${SEED}
  env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
    NEON_RUN_ROOT="${ROOT}" NEON_OUT_DIR="${DIR}" NEON_CACHE_DIR="${CACHE_DIR}" \
    NEON_PRIOR_SEED="${SEED}" NEON_TRAIN_SEED=0 \
    bash scripts/submit_neon_repair_rung.sh "${RUNG}" >/dev/null
  TRAIN=$(<"${DIR}/job_id.txt")
  EVAL=$(sbatch --parsable --job-name="neon_${RUNG,,}_a" \
    --output="${DIR}/phase5_eval_%j.out" --error="${DIR}/phase5_eval_%j.err" \
    --dependency="afterok:${TRAIN}" \
    --export="${BASE_EXPORT},NEON_PILOT_DIR=${DIR},NEON_PHASE5_TASK=pilot_eval" \
    "${TASK_SCRIPT}")
  TRAIN_JOBS+=("${TRAIN}")
  EVAL_JOBS+=("${EVAL}")
  if [[ "${RUNG}" == P2 ]]; then
    OOD_DIR=${DIR}/ood_evidence
    env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
      NEON_BUNDLE="${BUNDLE}" NEON_HISTORICAL_CONFIG="${NEON_HISTORICAL_CONFIG:-}" \
      NEON_SBATCH_DEPENDENCY="afterok:${EVAL}" \
      bash scripts/submit_neon_phase5_ood_evidence.sh \
      "${ROOT}" "${DIR}" "${DIR}/phase5_eval/RESULT.json" "${OOD_DIR}" \
      > "${DIR}/ood_submit.log"
    OOD_DIRS+=("${OOD_DIR}")
    OOD_FINAL_JOBS+=(
      "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["finalize_job_id"])' \
        "${OOD_DIR}/SUBMITTED.json")"
    )
  fi
done

if [[ "${RUNG}" == P2 ]]; then
  EVAL_SPEC=$(IFS=:; echo "${OOD_FINAL_JOBS[*]}")
  OOD_SPEC=$(IFS=:; echo "${OOD_DIRS[*]}")
  P2_EXPORT=",NEON_P2_OOD_DIRS=${OOD_SPEC},NEON_BASELINE_OOD_REPORT=${ROOT}/d5_b3_ood/ranking.json"
else
  EVAL_SPEC=$(IFS=:; echo "${EVAL_JOBS[*]}")
  OOD_SPEC=""
  P2_EXPORT=""
fi
DIR_SPEC=$(IFS=:; echo "${DIRS[*]}")
ACCEPT=$(sbatch --parsable --job-name=neon_p5_pacc \
  --output="${ROOT}/pilot_acceptance_%j.out" --error="${ROOT}/pilot_acceptance_%j.err" \
  --dependency="afterok:${EVAL_SPEC}" \
  --export="${BASE_EXPORT},NEON_PILOT_GATE_MODE=accept,NEON_PILOT_DIRS=${DIR_SPEC}${P2_EXPORT},NEON_PILOT_GATE_OUT=${ROOT}/pilot_acceptance" \
  "${GATE_SCRIPT}")

apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' \
  "${ROOT}" "${HEAD}" "${RUNG}" "${ACCEPT}" "${DIR_SPEC}" \
  "${TRAIN_JOBS[*]}" "${EVAL_JOBS[*]}" "${OOD_SPEC}" "${OOD_FINAL_JOBS[*]}"
import pathlib, sys
from neuralop.flood.eval.neon_phase5 import write_checksummed_artifact
root, head, rung, accept, dirs, trains, evals, ood_dirs, ood_finals = sys.argv[1:]
payload = {
    "schema_version": "neon_phase5_pilot_replication_submission_v1",
    "git_head": head,
    "ladder_rung": rung,
    "pilot_dirs": dirs.split(":"),
    "train_jobs": trains.split(),
    "eval_jobs": evals.split(),
    "ood_dirs": ood_dirs.split(":") if ood_dirs else [],
    "ood_finalize_jobs": ood_finals.split(),
    "acceptance_job": accept,
}
write_checksummed_artifact(
    pathlib.Path(root) / "PILOT_REPLICATION_SUBMITTED.json", payload
)
PY
printf 'Submitted %s acceptance replications; terminal gate job %s\n' "${RUNG}" "${ACCEPT}"
