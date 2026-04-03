#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p standard
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -J coastal_hpo_mgr
#SBATCH -o runtime/logs/out/coastal_hpo_mgr-%j.out
#SBATCH -e runtime/logs/err/coastal_hpo_mgr-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_no_physics_git_main}"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
CONTAINER_PATH="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
ACTION="${ACTION:?ACTION is required}"
STUDY_ROOT="${STUDY_ROOT:?STUDY_ROOT is required}"
STUDY_SPEC="${STUDY_SPEC:-${PROJECT_DIR}/config/flood/coastal/coastal_fgn_hpo_study.yaml}"
CURRENT_STAGE="${CURRENT_STAGE:-}"
TARGET_STAGE="${TARGET_STAGE:-}"
WAIT_FOR_JOB="${WAIT_FOR_JOB:-}"

cd "${SCRIPT_DIR}"
mkdir -p runtime/logs/out runtime/logs/err
module purge
module load apptainer

HOST_CA_BUNDLE=""
for cand in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt; do
  if [[ -f "${cand}" ]]; then
    HOST_CA_BUNDLE="${cand}"
    break
  fi
done
APPTAINER_ARGS=""
if [[ -n "${HOST_CA_BUNDLE}" ]]; then
  APPTAINER_ARGS="--bind ${HOST_CA_BUNDLE}:/host_ca_bundle.crt:ro"
  export APPTAINERENV_SSL_CERT_FILE=/host_ca_bundle.crt
  export APPTAINERENV_REQUESTS_CA_BUNDLE=/host_ca_bundle.crt
fi
export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"

run_hpo() {
  apptainer exec ${APPTAINER_ARGS} "${CONTAINER_PATH}" python -m neuralop.flood.cli.coastal_fgn_hpo "$@"
}

job_state() {
  local job_id="$1"
  sacct -j "${job_id}" --format=State -n -P 2>/dev/null | awk -F'|' 'NF {print $1; exit}'
}

record_job() {
  local kind="$1"
  local stage="$2"
  local job_id="$3"
  python3 - <<'PY' "${STUDY_ROOT}" "${kind}" "${stage}" "${job_id}"
import json
import sys
from pathlib import Path

study_root, kind, stage, job_id = sys.argv[1:]
path = Path(study_root) / 'jobs.json'
if path.exists():
    payload = json.loads(path.read_text())
else:
    payload = {'jobs': []}
payload.setdefault('jobs', []).append({'kind': kind, 'stage': stage, 'job_id': job_id})
path.write_text(json.dumps(payload, indent=2, sort_keys=True))
PY
}

submit_stage_array() {
  local stage="$1"
  eval "$(run_hpo describe-study --study-root "${STUDY_ROOT}" --stage "${stage}" --format shell)"
  local trial_count="${HPO_STAGE_TRIAL_COUNT}"
  if [[ "${trial_count}" -le 0 ]]; then
    return 1
  fi
  local upper=$((trial_count - 1))
  sbatch --parsable \
    -J "coastal_hpo_${stage}" \
    -A "${HPO_STAGE_ACCOUNT}" \
    -p "${HPO_STAGE_PARTITION}" \
    --gres="${HPO_STAGE_GRES}" \
    -c "${HPO_STAGE_CPUS}" \
    --mem="${HPO_STAGE_MEM}" \
    -t "${HPO_STAGE_WALLTIME}" \
    --array="0-${upper}%${HPO_STAGE_CONCURRENCY}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",STUDY_ROOT="${STUDY_ROOT}",TRIAL_SPEC_LIST="${HPO_STAGE_TRIAL_LIST}" \
    "${PROJECT_DIR}/scripts/slurm/train/flood_coastal_hpo_trial.sh"
}

submit_launch_stage_manager() {
  local dependency_job="$1"
  local stage="$2"
  sbatch --parsable \
    -J coastal_hpo_mgr \
    -A uqgroup \
    -p standard \
    -c 2 \
    --mem=8G \
    -t 01:00:00 \
    --dependency="afterany:${dependency_job}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",STUDY_ROOT="${STUDY_ROOT}",STUDY_SPEC="${STUDY_SPEC}",ACTION=launch-stage,TARGET_STAGE="${stage}",WAIT_FOR_JOB="${dependency_job}" \
    "$0"
}

submit_advance_manager() {
  local dependency_job="$1"
  local stage="$2"
  sbatch --parsable \
    -J coastal_hpo_mgr \
    -A uqgroup \
    -p standard \
    -c 2 \
    --mem=8G \
    -t 01:00:00 \
    --dependency="afterany:${dependency_job}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",STUDY_ROOT="${STUDY_ROOT}",STUDY_SPEC="${STUDY_SPEC}",ACTION=advance,CURRENT_STAGE="${stage}" \
    "$0"
}

export_final_ranking() {
  run_hpo export-ranking \
    --study-root "${STUDY_ROOT}" \
    --output-json "${STUDY_ROOT}/ranking/final_ranking.json" \
    --output-csv "${STUDY_ROOT}/ranking/final_ranking.csv"
}

case "${ACTION}" in
  bootstrap)
    run_hpo init-study --study-spec "${STUDY_SPEC}" --study-root "${STUDY_ROOT}"
    run_hpo suggest-trial --study-root "${STUDY_ROOT}" --stage stage_a
    eval "$(run_hpo describe-study --study-root "${STUDY_ROOT}" --format shell)"
    PRECOMP_JOB=$(sbatch --parsable \
      -J coastal_hpo_norm \
      -A "${HPO_PRECOMP_ACCOUNT}" \
      -p "${HPO_PRECOMP_PARTITION}" \
      --gres="${HPO_PRECOMP_GRES}" \
      -c "${HPO_PRECOMP_CPUS}" \
      --mem="${HPO_PRECOMP_MEM}" \
      -t "${HPO_PRECOMP_WALLTIME}" \
      --export=ALL,PROJECT_DIR="${PROJECT_DIR}",STUDY_ROOT="${STUDY_ROOT}" \
      "${PROJECT_DIR}/scripts/slurm/train/flood_coastal_hpo_precompute.sh")
    record_job precompute bootstrap "${PRECOMP_JOB}"
    LAUNCH_JOB=$(submit_launch_stage_manager "${PRECOMP_JOB}" stage_a)
    record_job manager stage_a "${LAUNCH_JOB}"
    echo "Submitted precompute job ${PRECOMP_JOB} and launch-stage manager ${LAUNCH_JOB}."
    ;;
  launch-stage)
    if [[ -z "${TARGET_STAGE}" || -z "${WAIT_FOR_JOB}" ]]; then
      echo "TARGET_STAGE and WAIT_FOR_JOB are required for ACTION=launch-stage" >&2
      exit 1
    fi
    STATE="$(job_state "${WAIT_FOR_JOB}")"
    if [[ ! "${STATE}" =~ ^COMPLETED ]]; then
      echo "Dependency job ${WAIT_FOR_JOB} finished in state ${STATE:-unknown}; not launching ${TARGET_STAGE}." >&2
      exit 1
    fi
    if ! STAGE_JOB=$(submit_stage_array "${TARGET_STAGE}"); then
      echo "No trials to submit for ${TARGET_STAGE}; exporting ranking instead."
      export_final_ranking
      exit 0
    fi
    record_job stage_array "${TARGET_STAGE}" "${STAGE_JOB}"
    ADVANCE_JOB=$(submit_advance_manager "${STAGE_JOB}" "${TARGET_STAGE}")
    record_job manager "${TARGET_STAGE}" "${ADVANCE_JOB}"
    echo "Submitted ${TARGET_STAGE} array ${STAGE_JOB} and advance manager ${ADVANCE_JOB}."
    ;;
  advance)
    if [[ -z "${CURRENT_STAGE}" ]]; then
      echo "CURRENT_STAGE is required for ACTION=advance" >&2
      exit 1
    fi
    run_hpo tell-result --study-root "${STUDY_ROOT}" --stage "${CURRENT_STAGE}" --mark-missing-failed
    case "${CURRENT_STAGE}" in
      stage_a)
        NEXT_STAGE="stage_b"
        ;;
      stage_b)
        NEXT_STAGE="stage_c"
        ;;
      stage_c)
        NEXT_STAGE=""
        ;;
      *)
        echo "Unknown CURRENT_STAGE=${CURRENT_STAGE}" >&2
        exit 1
        ;;
    esac
    if [[ -z "${NEXT_STAGE}" ]]; then
      export_final_ranking
      exit 0
    fi
    PROMOTION_JSON="$(run_hpo promote-stage --study-root "${STUDY_ROOT}" --from-stage "${CURRENT_STAGE}" --to-stage "${NEXT_STAGE}")"
    PROMOTE_COUNT="$(printf '%s' "${PROMOTION_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("trial_count", 0))')"
    if [[ "${PROMOTE_COUNT}" -le 0 ]]; then
      echo "No successful trials promoted from ${CURRENT_STAGE} to ${NEXT_STAGE}; exporting ranking."
      export_final_ranking
      exit 0
    fi
    STAGE_JOB=$(submit_stage_array "${NEXT_STAGE}")
    record_job stage_array "${NEXT_STAGE}" "${STAGE_JOB}"
    ADVANCE_JOB=$(submit_advance_manager "${STAGE_JOB}" "${NEXT_STAGE}")
    record_job manager "${NEXT_STAGE}" "${ADVANCE_JOB}"
    echo "Promoted ${PROMOTE_COUNT} trials into ${NEXT_STAGE}; submitted array ${STAGE_JOB} and manager ${ADVANCE_JOB}."
    ;;
  *)
    echo "Unknown ACTION=${ACTION}" >&2
    exit 1
    ;;
esac
