#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -J coastal_hpo_trial
#SBATCH -o runtime/logs/out/coastal_hpo_trial-%A_%a.out
#SBATCH -e runtime/logs/err/coastal_hpo_trial-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/$USER/GINO_Model/neuraloperator_no_physics_git_main}"
CANONICAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
COMMON_SH=""
for candidate in \
  "${PROJECT_DIR}/scripts/slurm/lib/common.sh" \
  "${SLURM_SUBMIT_DIR:-}/slurm/lib/common.sh" \
  "${SLURM_SUBMIT_DIR:-}/scripts/slurm/lib/common.sh" \
  "$(cd "${CANONICAL_DIR}/.." && pwd)/lib/common.sh"
do
  if [[ -n "${candidate}" && -f "${candidate}" ]]; then
    COMMON_SH="${candidate}"
    break
  fi
done
if [[ -z "${COMMON_SH}" ]]; then
  echo "ERROR: Unable to locate scripts/slurm/lib/common.sh" >&2
  exit 1
fi
source "${COMMON_SH}"
slurm_load_apptainer
slurm_configure_host_ca
SCRIPT_DIR="${PROJECT_DIR}/scripts"
cd "${SCRIPT_DIR}"
mkdir -p runtime/logs/out runtime/logs/err

CONTAINER_PATH="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_DIR}/scripts/flood_wv_train_operator.py}"
TRIAL_SPEC_LIST="${TRIAL_SPEC_LIST:-}"
TRIAL_SPEC_PATH="${TRIAL_SPEC_PATH:-}"
WANDB_LOG="${WANDB_LOG:-true}"
export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
slurm_assert_container_gpus "${CONTAINER_PATH}" 1

if [[ -z "${TRIAL_SPEC_PATH}" ]]; then
  if [[ -z "${TRIAL_SPEC_LIST}" ]]; then
    echo "ERROR: TRIAL_SPEC_PATH or TRIAL_SPEC_LIST is required" >&2
    exit 1
  fi
  TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
  LINE_NO=$((TASK_INDEX + 1))
  TRIAL_SPEC_PATH="$(sed -n "${LINE_NO}p" "${TRIAL_SPEC_LIST}")"
fi
if [[ -z "${TRIAL_SPEC_PATH}" || ! -f "${TRIAL_SPEC_PATH}" ]]; then
  echo "ERROR: Trial spec not found: ${TRIAL_SPEC_PATH:-<empty>}" >&2
  exit 1
fi

for key_path in \
  "${HOME}/.config/wandb_api_key.txt" \
  "${PROJECT_DIR}/config/wandb_api_key.txt" \
  "/scratch/$USER/Data_Generation_UQ/GINO_Model/neuraloperator_no_physics/config/wandb_api_key.txt"
do
  if [[ -f "${key_path}" ]]; then
    export APPTAINERENV_WANDB_API_KEY="$(head -n 1 "${key_path}" | tr -d '\r')"
    break
  fi
done

eval "$({
  apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python -m neuralop.flood.cli.coastal_fgn_hpo \
    describe-trial --trial-spec "${TRIAL_SPEC_PATH}" --format shell
})"

RUN_STATUS="failed"
finalize() {
  set +e
  apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python -m neuralop.flood.cli.coastal_fgn_hpo \
    summarize-trial \
    --trial-spec "${TRIAL_SPEC_PATH}" \
    --status "${RUN_STATUS}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}" \
    --array-task-id "${SLURM_ARRAY_TASK_ID:-}" \
    --git-sha "$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"

  if [[ "${WANDB_LOG}" != "false" && -n "${APPTAINERENV_WANDB_API_KEY:-}" ]]; then
    apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python - <<'PY' "${TRIAL_SPEC_PATH}"
import json
import sys
from pathlib import Path

import wandb

trial_spec = json.loads(Path(sys.argv[1]).read_text())
summary_path = Path(trial_spec["summary_path"])
config_path = Path(trial_spec["config_path"])
if not summary_path.exists():
    raise SystemExit(0)
run = wandb.init(
    project=trial_spec["wandb"].get("project"),
    entity=trial_spec["wandb"].get("entity"),
    group=trial_spec["wandb"]["group"],
    name=f"{trial_spec['wandb']['name']}_artifacts",
    job_type="hpo_artifacts",
    tags=list(trial_spec["wandb"].get("tags", [])),
    reinit=True,
)
artifact = wandb.Artifact(name=f"{trial_spec['run_tag']}-artifacts", type="hpo-trial")
if config_path.exists():
    artifact.add_file(str(config_path), name="config.yaml")
artifact.add_file(str(summary_path), name="run_summary.json")
run.log_artifact(artifact)
run.finish()
PY
  fi
}
trap finalize EXIT

apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python -m neuralop.flood.cli.coastal_fgn_hpo \
  render-trial-config --trial-spec "${TRIAL_SPEC_PATH}" >/dev/null

if [[ ! -f "${HPO_NORMALIZER_PATH}" ]]; then
  echo "ERROR: Shared normalizer not found at ${HPO_NORMALIZER_PATH}" >&2
  exit 1
fi

if [[ "${WANDB_LOG}" == "false" ]]; then
  python3 - <<'PY' "${TRIAL_SPEC_PATH}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
data.setdefault('wandb', {})['log'] = False
path.write_text(json.dumps(data, indent=2, sort_keys=True))
PY
  apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python -m neuralop.flood.cli.coastal_fgn_hpo \
    render-trial-config --trial-spec "${TRIAL_SPEC_PATH}" >/dev/null
fi

apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${TRAIN_SCRIPT}" --config_path "${HPO_CONFIG_PATH}"
RUN_STATUS="completed"
