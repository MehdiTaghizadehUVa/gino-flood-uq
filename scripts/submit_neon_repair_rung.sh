#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 B0|B1a|B1b|B2|B3|B4|B5 [B4_DE_SPREAD_MULTIPLIER]" >&2
  exit 2
fi
RUNG=${1^^}
case "${RUNG}" in
  B0|B1A|B1B|B2|B3|B5) ;;
  B4)
    : "${2:?B4 requires a DE-spread multiplier: 0.5, 1.0, or 2.0}"
    case "$2" in 0.5|1.0|2.0) ;; *) echo "Invalid B4 multiplier: $2" >&2; exit 2;; esac
    export NEON_DE_SPREAD_MULTIPLIER=$2
    ;;
  *) echo "Unsupported rung: $1" >&2; exit 2;;
esac

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_v4_integrated}
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
if [[ -s "${OUT_DIR}/job_id.txt" ]]; then
  echo "Output already has a submitted job: ${OUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUT_DIR}"

module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}"
ENV_ARGS=(
  "NEON_PLAN_ONLY=1"
  "NEON_PREFLIGHT_PATH=${PREFLIGHT}"
  "NEON_LADDER_RUNG=${RUNG}"
  "NEON_N_TRAIN=${N_TRAIN}"
  "NEON_OUT_DIR=${OUT_DIR}"
  "NEON_CACHE_DIR=${CACHE_DIR}"
)
if [[ -n "${NEON_DE_SPREAD_MULTIPLIER:-}" ]]; then
  ENV_ARGS+=("NEON_DE_SPREAD_MULTIPLIER=${NEON_DE_SPREAD_MULTIPLIER}")
fi
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  env "${ENV_ARGS[@]}" python "${REPO}/scripts/neon_stage2_tr_train.py"
test -s "${PREFLIGHT}"
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["schema_version"] == "neon_repair_preflight_v1"' \
  "${PREFLIGHT}"
printf '%s\n' "${HEAD}" > "${OUT_DIR}/preflight_git_head.txt"

if [[ "${NEON_SUBMIT_DRY_RUN:-0}" == 1 ]]; then
  printf 'Validated only: rung=%s head=%s preflight=%s\n' "${RUNG}" "${HEAD}" "${PREFLIGHT}"
  exit 0
fi

EXPORTS="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_LADDER_RUNG=${RUNG},NEON_N_TRAIN=${N_TRAIN},NEON_OUT_DIR=${OUT_DIR},NEON_CACHE_DIR=${CACHE_DIR},NEON_PREFLIGHT_PATH=${PREFLIGHT}"
if [[ -n "${NEON_DE_SPREAD_MULTIPLIER:-}" ]]; then
  EXPORTS+=",NEON_DE_SPREAD_MULTIPLIER=${NEON_DE_SPREAD_MULTIPLIER}"
fi
JOB_ID=$(sbatch --parsable \
  --job-name="neon_${TAG}" \
  --output="${OUT_DIR}/slurm-%j.out" \
  --error="${OUT_DIR}/slurm-%j.err" \
  --export="${EXPORTS}" \
  "${SBATCH_SCRIPT}")
printf '%s\n' "${JOB_ID}" > "${OUT_DIR}/job_id.txt"
printf 'Submitted %s: job=%s head=%s preflight=%s\n' "${RUNG}" "${JOB_ID}" "${HEAD}" "${PREFLIGHT}"
