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
test -s "${ROOT}/SUBMITTED.json"
test -s "${ROOT}/SUBMITTED.json.sha256"
test -s "${ROOT}/gp1/DECISION.json"
test -s "${ROOT}/gp1/DECISION.json.sha256"
test ! -e "${ROOT}/P1A_REPLICATION_SUBMITTED.json" || {
  echo "P1a replication was already submitted under ${ROOT}" >&2
  exit 2
}

module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"
apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' \
  "${ROOT}/SUBMITTED.json" "${ROOT}/gp1/DECISION.json" \
  "${ROOT}/PROTOCOL.json" "${HEAD}"
import json, pathlib, sys
submission = pathlib.Path(sys.argv[1])
decision_path = pathlib.Path(sys.argv[2])
protocol_path = pathlib.Path(sys.argv[3])
head = sys.argv[4]
from neuralop.flood.eval.neon_phase5 import (
    verify_checksummed_artifact,
    verify_phase5_decision_artifact,
)
protocol_sha = verify_checksummed_artifact(protocol_path)
verify_checksummed_artifact(submission)
submitted = json.loads(submission.read_text())
decision = verify_phase5_decision_artifact(
    decision_path, expected_head=head, protocol_sha256=protocol_sha
)
assert submitted["git_head"] == head
assert decision["gate"] == "GP1"
assert decision["verdict_status"].startswith("provisional")
assert decision["mandatory_next"] == "replicate_P1a_on_at_least_three_support_seeds"
PY

TASK_SCRIPT=${REPO}/scripts/sbatch_neon_phase5_task.sh
DECISION_SCRIPT=${REPO}/scripts/sbatch_neon_phase5_decision.sh
BASE_EXPORT="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_PHASE5_ROOT=${ROOT},NEON_CONFIG=${CONFIG},NEON_BUNDLE=${BUNDLE},NEON_B3_DIR=${B3_DIR},NEON_CACHE_DIR=${CACHE_DIR}"
DIRS=("${ROOT}/p1a_seed0_training")
SEEDS=(124 125)

# Validate every training/evaluation command before the first sbatch.
for SEED in "${SEEDS[@]}"; do
  DIR=${ROOT}/p1a_seed${SEED}_training
  DIRS+=("${DIR}")
  env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
    NEON_RUN_ROOT="${ROOT}" NEON_OUT_DIR="${DIR}" NEON_CACHE_DIR="${CACHE_DIR}" \
    NEON_DIRICHLET_PARTICLE_SEED="${SEED}" NEON_PRIOR_SEED=20260703 \
    NEON_TRAIN_SEED=0 NEON_SUBMIT_DRY_RUN=1 \
    bash scripts/submit_neon_repair_rung.sh B5 >/dev/null
  for TASK in p1a_eval direct_dirichlet; do
    env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
      NEON_EXPECTED_HEAD="${HEAD}" NEON_PHASE5_ROOT="${ROOT}" \
      NEON_CONFIG="${CONFIG}" NEON_BUNDLE="${BUNDLE}" NEON_B3_DIR="${B3_DIR}" \
      NEON_CACHE_DIR="${CACHE_DIR}" NEON_P1A_DIR="${DIR}" \
      NEON_PHASE5_TASK="${TASK}" NEON_PLAN_ONLY=1 bash "${TASK_SCRIPT}"
  done
done

DEPENDENCIES=()
TRAIN_JOBS=()
EVAL_JOBS=()
DIRECT_JOBS=()
for SEED in "${SEEDS[@]}"; do
  DIR=${ROOT}/p1a_seed${SEED}_training
  env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
    NEON_RUN_ROOT="${ROOT}" NEON_OUT_DIR="${DIR}" NEON_CACHE_DIR="${CACHE_DIR}" \
    NEON_DIRICHLET_PARTICLE_SEED="${SEED}" NEON_PRIOR_SEED=20260703 \
    NEON_TRAIN_SEED=0 bash scripts/submit_neon_repair_rung.sh B5 >/dev/null
  TRAIN=$(<"${DIR}/job_id.txt")
  TRAIN_JOBS+=("${TRAIN}")
  P1_EXPORT="${BASE_EXPORT},NEON_P1A_DIR=${DIR}"
  EVAL=$(sbatch --parsable --job-name="neon_p5_e${SEED}" \
    --output="${ROOT}/p1a_eval_seed${SEED}_%j.out" \
    --error="${ROOT}/p1a_eval_seed${SEED}_%j.err" \
    --dependency="afterok:${TRAIN}" --export="${P1_EXPORT},NEON_PHASE5_TASK=p1a_eval" \
    "${TASK_SCRIPT}")
  DIRECT=$(sbatch --parsable --job-name="neon_p5_d${SEED}" \
    --output="${ROOT}/p1a_direct_seed${SEED}_%j.out" \
    --error="${ROOT}/p1a_direct_seed${SEED}_%j.err" \
    --dependency="afterok:${TRAIN}" --export="${P1_EXPORT},NEON_PHASE5_TASK=direct_dirichlet" \
    "${TASK_SCRIPT}")
  EVAL_JOBS+=("${EVAL}")
  DIRECT_JOBS+=("${DIRECT}")
  DEPENDENCIES+=("${EVAL}" "${DIRECT}")
done

DIR_SPEC=$(IFS=:; echo "${DIRS[*]}")
DEP_SPEC=$(IFS=:; echo "${DEPENDENCIES[*]}")
GP1=$(sbatch --parsable --job-name=neon_p5_gp1r \
  --output="${ROOT}/gp1_replicated_%j.out" --error="${ROOT}/gp1_replicated_%j.err" \
  --dependency="afterok:${DEP_SPEC}" \
  --export="${BASE_EXPORT},NEON_P1A_DIRS=${DIR_SPEC},NEON_PHASE5_GATE=gp1,NEON_PHASE5_GATE_OUT=${ROOT}/gp1_replicated" \
  "${DECISION_SCRIPT}")

apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' \
  "${ROOT}" "${HEAD}" "${GP1}" "${DIR_SPEC}" "${TRAIN_JOBS[*]}" "${EVAL_JOBS[*]}" "${DIRECT_JOBS[*]}"
import pathlib, sys
from neuralop.flood.eval.neon_phase5 import write_checksummed_artifact
root, head, gp1, dirs, trains, evals, directs = sys.argv[1:]
payload = {"schema_version": "neon_phase5_p1a_replication_submission_v1",
           "git_head": head, "p1a_dirs": dirs.split(":"), "replicated_gp1_job": gp1,
           "train_jobs": trains.split(), "eval_jobs": evals.split(),
           "direct_jobs": directs.split()}
write_checksummed_artifact(
    pathlib.Path(root) / "P1A_REPLICATION_SUBMITTED.json", payload
)
PY
printf 'Submitted two P1a support replications and replicated GP1: %s\n' "${GP1}"
