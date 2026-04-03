#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_no_physics_git_main}"
STUDY_SPEC="${STUDY_SPEC:-${PROJECT_DIR}/config/flood/coastal/coastal_fgn_hpo_study.yaml}"
GIT_SHA="$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_GROUP="${RUN_GROUP:-coastal_fgn_hpo_${TIMESTAMP}_${GIT_SHA}}"
STUDY_ROOT="${STUDY_ROOT:-${PROJECT_DIR}/scripts/runtime/coastal_fgn_hpo/${RUN_GROUP}}"

mkdir -p "$(dirname "${STUDY_ROOT}")"

BOOTSTRAP_JOB=$(sbatch --parsable \
  -J coastal_hpo_mgr \
  -A uqgroup \
  -p standard \
  -c 2 \
  --mem=8G \
  -t 01:00:00 \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",STUDY_SPEC="${STUDY_SPEC}",STUDY_ROOT="${STUDY_ROOT}",ACTION=bootstrap \
  "${PROJECT_DIR}/scripts/slurm/train/flood_coastal_hpo_stage_manager.sh")

cat <<REPORT
Submitted coastal FGN HPO study.
bootstrap_job=${BOOTSTRAP_JOB}
study_spec=${STUDY_SPEC}
study_root=${STUDY_ROOT}
ranking_json=${STUDY_ROOT}/ranking/final_ranking.json
ranking_csv=${STUDY_ROOT}/ranking/final_ranking.csv
jobs_json=${STUDY_ROOT}/jobs.json
REPORT
