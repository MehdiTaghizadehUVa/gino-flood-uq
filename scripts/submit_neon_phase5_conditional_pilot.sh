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
DECISION=${NEON_GP1_DECISION:-${ROOT}/gp1_replicated/DECISION.json}

cd "${REPO}"
HEAD=$(git rev-parse HEAD)
test -z "$(git status --porcelain)" || { echo "Refusing dirty-tree submission" >&2; exit 2; }
test -s "${DECISION}" && test -s "${DECISION}.sha256"
test ! -e "${ROOT}/CONDITIONAL_PILOT_SUBMITTED.json" || {
  echo "Conditional Phase-5 pilot already submitted under ${ROOT}" >&2
  exit 2
}
module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"
VERDICT=$(apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' \
  "${DECISION}" "${ROOT}/PROTOCOL.json" "${HEAD}"
import pathlib, sys
from neuralop.flood.eval.neon_phase5 import (
    verify_checksummed_artifact,
    verify_phase5_decision_artifact,
)
path = pathlib.Path(sys.argv[1])
protocol_sha = verify_checksummed_artifact(sys.argv[2])
payload = verify_phase5_decision_artifact(
    path, expected_head=sys.argv[3], protocol_sha256=protocol_sha
)
assert payload["gate"] == "GP1"
assert payload["verdict_status"] in {"acceptance_replicated", "replicated_inconsistent"}
assert int(payload["p1a_seed_count"]) >= 3
print(payload["decision"])
PY
)

case "${VERDICT}" in
  contraction_confirmation)
    # Phase S is allowed only after the replicated pilot gate. The scale-out
    # adapter validates this GP1 artifact and uses B5 rather than legacy B3.
    env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
      NEON_CACHE_DIR="${CACHE_DIR}" \
      bash scripts/submit_neon_phase5_scaleout.sh "${ROOT}" "${DECISION}" \
      > "${ROOT}/phase_s_submit.log"
    JOB=$(apptainer exec --bind /scratch,/home "${CONTAINER}" python -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["final_audit_job"])' \
      "${ROOT}/PHASE_S_SUBMITTED.json")
    RUNGS=B5
    ;;
  continuous_amortization_failure|indeterminate)
    RUNGS="P1B_A P1B_B P1B_C"
    ;;
  shared_subspace_pathology)
    RUNGS=P2
    ;;
  *)
    echo "Unsupported replicated GP1 verdict: ${VERDICT}" >&2
    exit 2
    ;;
esac

TASK_SCRIPT=${REPO}/scripts/sbatch_neon_phase5_task.sh
PILOT_GATE_SCRIPT=${REPO}/scripts/sbatch_neon_phase5_pilot_gate.sh
BASE_EXPORT="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_PHASE5_ROOT=${ROOT},NEON_CONFIG=${CONFIG},NEON_BUNDLE=${BUNDLE},NEON_B3_DIR=${B3_DIR},NEON_CACHE_DIR=${CACHE_DIR}"
TRAIN_JOBS=()
EVAL_JOBS=()
PILOT_DIRS=()
if [[ "${VERDICT}" != contraction_confirmation ]]; then
  # Screen one predeclared seed per arm before paying for acceptance replicates.
  for RUNG in ${RUNGS}; do
    TAG=${RUNG,,}
    DIR=${ROOT}/${TAG}_screen_seed20260703
    PILOT_DIRS+=("${DIR}")
    env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
      NEON_RUN_ROOT="${ROOT}" NEON_OUT_DIR="${DIR}" NEON_CACHE_DIR="${CACHE_DIR}" \
      NEON_PRIOR_SEED=20260703 NEON_TRAIN_SEED=0 NEON_SUBMIT_DRY_RUN=1 \
      bash scripts/submit_neon_repair_rung.sh "${RUNG}" >/dev/null
    env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
      NEON_EXPECTED_HEAD="${HEAD}" NEON_PHASE5_ROOT="${ROOT}" \
      NEON_CONFIG="${CONFIG}" NEON_BUNDLE="${BUNDLE}" NEON_B3_DIR="${B3_DIR}" \
      NEON_CACHE_DIR="${CACHE_DIR}" NEON_PILOT_DIR="${DIR}" \
      NEON_PHASE5_TASK=pilot_eval NEON_PLAN_ONLY=1 bash "${TASK_SCRIPT}"
  done
  for RUNG in ${RUNGS}; do
    TAG=${RUNG,,}
    DIR=${ROOT}/${TAG}_screen_seed20260703
    env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
      NEON_RUN_ROOT="${ROOT}" NEON_OUT_DIR="${DIR}" NEON_CACHE_DIR="${CACHE_DIR}" \
      NEON_PRIOR_SEED=20260703 NEON_TRAIN_SEED=0 \
      bash scripts/submit_neon_repair_rung.sh "${RUNG}" >/dev/null
    TRAIN=$(<"${DIR}/job_id.txt")
    EVAL=$(sbatch --parsable --job-name="neon_${TAG}_eval" \
      --output="${DIR}/phase5_eval_%j.out" --error="${DIR}/phase5_eval_%j.err" \
      --dependency="afterok:${TRAIN}" \
      --export="${BASE_EXPORT},NEON_PILOT_DIR=${DIR},NEON_PHASE5_TASK=pilot_eval" \
      "${TASK_SCRIPT}")
    TRAIN_JOBS+=("${TRAIN}")
    EVAL_JOBS+=("${EVAL}")
  done
  EVAL_SPEC=$(IFS=:; echo "${EVAL_JOBS[*]}")
  DIR_SPEC=$(IFS=:; echo "${PILOT_DIRS[*]}")
  if [[ "${VERDICT}" == continuous_amortization_failure || "${VERDICT}" == indeterminate ]]; then
    SCREEN=$(sbatch --parsable --job-name=neon_p5_pscr \
      --output="${ROOT}/pilot_screen_%j.out" --error="${ROOT}/pilot_screen_%j.err" \
      --dependency="afterok:${EVAL_SPEC}" \
      --export="${BASE_EXPORT},NEON_PILOT_GATE_MODE=screen,NEON_PILOT_DIRS=${DIR_SPEC},NEON_PILOT_GATE_OUT=${ROOT}/pilot_screen" \
      "${PILOT_GATE_SCRIPT}")
    JOB=${SCREEN}
  else
    # P2 is admissible only with its historical ranking/risk-coverage
    # firewall. Produce all 13 single-reference shards before screening.
    DIR=${PILOT_DIRS[0]}
    EVAL=${EVAL_JOBS[0]}
    OOD_DIR=${DIR}/ood_evidence
    env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
      NEON_BUNDLE="${BUNDLE}" NEON_HISTORICAL_CONFIG="${NEON_HISTORICAL_CONFIG:-}" \
      NEON_SBATCH_DEPENDENCY="afterok:${EVAL}" \
      bash scripts/submit_neon_phase5_ood_evidence.sh \
      "${ROOT}" "${DIR}" "${DIR}/phase5_eval/RESULT.json" "${OOD_DIR}" \
      > "${DIR}/ood_submit.log"
    OOD_FINAL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["finalize_job_id"])' \
      "${OOD_DIR}/SUBMITTED.json")
    BASELINE_SUBMISSION=${ROOT}/d5_b3_ood/SUBMITTED.json
    test -s "${BASELINE_SUBMISSION}"
    BASELINE_FINAL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["finalize_job_id"])' \
      "${BASELINE_SUBMISSION}")
    SCREEN=$(sbatch --parsable --job-name=neon_p5_p2scr \
      --output="${ROOT}/p2_screen_%j.out" --error="${ROOT}/p2_screen_%j.err" \
      --dependency="afterok:${OOD_FINAL}:${BASELINE_FINAL}" \
      --export="${BASE_EXPORT},NEON_PILOT_GATE_MODE=screen,NEON_PILOT_DIRS=${DIR},NEON_P2_OOD_DIRS=${OOD_DIR},NEON_BASELINE_OOD_REPORT=${ROOT}/d5_b3_ood/ranking.json,NEON_PILOT_GATE_OUT=${ROOT}/pilot_screen" \
      "${PILOT_GATE_SCRIPT}")
    JOB=${SCREEN}
  fi
fi

apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' \
  "${ROOT}" "${HEAD}" "${DECISION}" "${VERDICT}" "${RUNGS}" "${JOB}" \
  "${TRAIN_JOBS[*]}" "${EVAL_JOBS[*]}" "${PILOT_DIRS[*]}" "${SCREEN:-}"
import json, pathlib, sys
from neuralop.flood.eval.neon_phase5 import write_checksummed_artifact
root, head, decision, verdict, rungs, terminal, trains, evals, dirs, screen = sys.argv[1:]
payload = {"schema_version": "neon_phase5_conditional_pilot_submission_v1",
           "git_head": head, "gp1_decision": decision, "verdict": verdict,
           "rungs": rungs.split(), "terminal_jobs": terminal.split(":"),
           "train_jobs": trains.split(), "eval_jobs": evals.split(),
           "pilot_dirs": dirs.split(), "pilot_screen_job": screen or None}
write_checksummed_artifact(
    pathlib.Path(root) / "CONDITIONAL_PILOT_SUBMITTED.json", payload
)
PY
printf 'Conditional Phase-5 action for %s submitted: %s\n' "${VERDICT}" "${JOB}"
