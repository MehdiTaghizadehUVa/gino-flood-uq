#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 B0|B1a|B1b|B2|B3|B4|B5|P1B_A|P1B_B|P1B_C|P2 [B4_DE_SPREAD_MULTIPLIER]" >&2
  exit 2
fi
RUNG=${1^^}
case "${RUNG}" in
  B0|B1A|B1B|B2|B3|B5|P1B_A|P1B_B|P1B_C|P2) ;;
  B4)
    : "${2:?B4 requires a DE-spread multiplier: 0.5, 1.0, or 2.0}"
    case "$2" in 0.5|1.0|2.0) ;; *) echo "Invalid B4 multiplier: $2" >&2; exit 2;; esac
    export NEON_DE_SPREAD_MULTIPLIER=$2
    ;;
  *) echo "Unsupported rung: $1" >&2; exit 2;;
esac

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
RUN_ROOT=${NEON_RUN_ROOT:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4}
N_TRAIN=${NEON_N_TRAIN:-450}
CACHE_DIR=${NEON_CACHE_DIR:-${RUN_ROOT}/feature_cache_v3}
TAG=${RUNG,,}
if [[ "${RUNG}" == B4 ]]; then TAG="b4_x${NEON_DE_SPREAD_MULTIPLIER//./p}"; fi
OUT_DIR=${NEON_OUT_DIR:-${RUN_ROOT}/${TAG}_n${N_TRAIN}}
PREFLIGHT=${OUT_DIR}/preflight.json
SBATCH_SCRIPT=${REPO}/scripts/sbatch_neon_repair_rung.sh

[[ -d "${REPO}/.git" || -f "${REPO}/.git" ]] || { echo "Not a git worktree: ${REPO}" >&2; exit 2; }
[[ -f "${CONTAINER}" ]] || { echo "Missing container: ${CONTAINER}" >&2; exit 2; }
[[ -x "${SBATCH_SCRIPT}" ]] || { echo "Missing sbatch script: ${SBATCH_SCRIPT}" >&2; exit 2; }
HEAD=$(git -C "${REPO}" rev-parse HEAD)
test -z "$(git -C "${REPO}" status --porcelain)" || {
  echo "Refusing to preflight a dirty tree" >&2
  exit 2
}

CONTINUE=${NEON_CONTINUE:-0}
STATE_PATH=${OUT_DIR}/neon_stage2_latest_state.pt
SUBMISSION_MODE=initial
EXPECTED_PREFLIGHT_SHA=
GENERATED_PREFLIGHT=${PREFLIGHT}
if [[ -s "${OUT_DIR}/job_id.txt" ]]; then
  [[ "${CONTINUE}" == 1 ]] || {
    echo "Output already has a submitted job: ${OUT_DIR}; set NEON_CONTINUE=1 only after it stops" >&2
    exit 2
  }
  SUBMISSION_MODE=continuation
  GENERATED_PREFLIGHT=${PREFLIGHT}.candidate.$$
  PREVIOUS_JOB_ID=$(<"${OUT_DIR}/job_id.txt")
  [[ -s "${STATE_PATH}" ]] || {
    echo "Cannot continue without a completed-epoch state: ${STATE_PATH}" >&2
    exit 2
  }
  [[ -s "${OUT_DIR}/git_head.txt" ]] || {
    echo "Cannot verify provenance: missing ${OUT_DIR}/git_head.txt" >&2
    exit 2
  }
  [[ "$(<"${OUT_DIR}/git_head.txt")" == "${HEAD}" ]] || {
    echo "Refusing cross-commit continuation for ${OUT_DIR}" >&2
    exit 2
  }
  [[ -s "${PREFLIGHT}" ]] || {
    echo "Cannot verify continuation config: missing ${PREFLIGHT}" >&2
    exit 2
  }
  EXPECTED_PREFLIGHT_SHA=$(sha256sum "${PREFLIGHT}" | cut -d" " -f1)
  if squeue -j "${PREVIOUS_JOB_ID}" -h 2>/dev/null | grep -q .; then
    echo "Previous job ${PREVIOUS_JOB_ID} is still active; refusing duplicate continuation" >&2
    exit 2
  fi
else
  [[ "${CONTINUE}" != 1 ]] || {
    echo "NEON_CONTINUE=1 requested, but no prior job exists in ${OUT_DIR}" >&2
    exit 2
  }
fi
mkdir -p "${OUT_DIR}"
cleanup_candidate() {
  if [[ "${GENERATED_PREFLIGHT}" != "${PREFLIGHT}" ]]; then
    rm -f "${GENERATED_PREFLIGHT}"
  fi
}
trap cleanup_candidate EXIT

module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}"
ENV_ARGS=(
  "NEON_PLAN_ONLY=1"
  "NEON_PREFLIGHT_PATH=${GENERATED_PREFLIGHT}"
  "NEON_LADDER_RUNG=${RUNG}"
  "NEON_N_TRAIN=${N_TRAIN}"
  "NEON_OUT_DIR=${OUT_DIR}"
  "NEON_CACHE_DIR=${CACHE_DIR}"
)
if [[ -n "${NEON_DE_SPREAD_MULTIPLIER:-}" ]]; then
  ENV_ARGS+=("NEON_DE_SPREAD_MULTIPLIER=${NEON_DE_SPREAD_MULTIPLIER}")
fi
if [[ -n "${NEON_DIRICHLET_PARTICLE_SEED:-}" ]]; then
  ENV_ARGS+=("NEON_DIRICHLET_PARTICLE_SEED=${NEON_DIRICHLET_PARTICLE_SEED}")
fi
if [[ -n "${NEON_PRIOR_TARGET_STD_M:-}" ]]; then
  ENV_ARGS+=("NEON_PRIOR_TARGET_STD_M=${NEON_PRIOR_TARGET_STD_M}")
fi
for NAME in NEON_PRIOR_SEED NEON_TRAIN_SEED; do
  if [[ -n "${!NAME:-}" ]]; then ENV_ARGS+=("${NAME}=${!NAME}"); fi
done
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  env "${ENV_ARGS[@]}" python "${REPO}/scripts/neon_stage2_tr_train.py"
test -s "${GENERATED_PREFLIGHT}"
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["schema_version"] == "neon_repair_preflight_v1"' \
  "${GENERATED_PREFLIGHT}"
if [[ -n "${EXPECTED_PREFLIGHT_SHA}" ]]; then
  CURRENT_PREFLIGHT_SHA=$(sha256sum "${GENERATED_PREFLIGHT}" | cut -d" " -f1)
  [[ "${CURRENT_PREFLIGHT_SHA}" == "${EXPECTED_PREFLIGHT_SHA}" ]] || {
    echo "Refusing continuation because the validated preflight config changed" >&2
    exit 2
  }
  apptainer exec --bind /scratch,/home "${CONTAINER}" \
    python -c 'import json,sys,torch
state=torch.load(sys.argv[1], map_location="cpu")
preflight=json.load(open(sys.argv[2]))
next_epoch=int(state["next_epoch"])
n_epochs=int(preflight["config"]["n_epochs"])
assert next_epoch < n_epochs, f"training already complete: next_epoch={next_epoch}, n_epochs={n_epochs}"' \
    "${STATE_PATH}" "${GENERATED_PREFLIGHT}"
fi
printf '%s\n' "${HEAD}" > "${OUT_DIR}/preflight_git_head.txt"

if [[ "${NEON_SUBMIT_DRY_RUN:-0}" == 1 ]]; then
  printf 'Validated only: rung=%s mode=%s head=%s preflight=%s\n' \
    "${RUNG}" "${SUBMISSION_MODE}" "${HEAD}" "${PREFLIGHT}"
  exit 0
fi

EXPORTS="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_LADDER_RUNG=${RUNG},NEON_N_TRAIN=${N_TRAIN},NEON_OUT_DIR=${OUT_DIR},NEON_CACHE_DIR=${CACHE_DIR},NEON_PREFLIGHT_PATH=${PREFLIGHT}"
if [[ -n "${NEON_DE_SPREAD_MULTIPLIER:-}" ]]; then
  EXPORTS+=",NEON_DE_SPREAD_MULTIPLIER=${NEON_DE_SPREAD_MULTIPLIER}"
fi
if [[ -n "${NEON_DIRICHLET_PARTICLE_SEED:-}" ]]; then
  EXPORTS+=",NEON_DIRICHLET_PARTICLE_SEED=${NEON_DIRICHLET_PARTICLE_SEED}"
fi
if [[ -n "${NEON_PRIOR_TARGET_STD_M:-}" ]]; then
  EXPORTS+=",NEON_PRIOR_TARGET_STD_M=${NEON_PRIOR_TARGET_STD_M}"
fi
for NAME in NEON_PRIOR_SEED NEON_TRAIN_SEED; do
  if [[ -n "${!NAME:-}" ]]; then EXPORTS+=",${NAME}=${!NAME}"; fi
done
SBATCH_ARGS=()
if [[ -n "${NEON_SBATCH_DEPENDENCY:-}" ]]; then
  SBATCH_ARGS+=(--dependency="${NEON_SBATCH_DEPENDENCY}")
fi
JOB_ID=$(sbatch --parsable \
  "${SBATCH_ARGS[@]}" \
  --job-name="neon_${TAG}" \
  --output="${OUT_DIR}/slurm-%j.out" \
  --error="${OUT_DIR}/slurm-%j.err" \
  --export="${EXPORTS}" \
  "${SBATCH_SCRIPT}")
printf '%s\n' "${JOB_ID}" > "${OUT_DIR}/job_id.txt"
printf '%s\n' "${JOB_ID}" >> "${OUT_DIR}/job_ids.txt"
printf '%s\t%s\t%s\t%s\n' \
  "$(date --iso-8601=seconds)" "${SUBMISSION_MODE}" "${JOB_ID}" "${HEAD}" \
  >> "${OUT_DIR}/submission_history.tsv"
printf 'Submitted %s (%s): job=%s head=%s preflight=%s\n' \
  "${RUNG}" "${SUBMISSION_MODE}" "${JOB_ID}" "${HEAD}" "${PREFLIGHT}"
