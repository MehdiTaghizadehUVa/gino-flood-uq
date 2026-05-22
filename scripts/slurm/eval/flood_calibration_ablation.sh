#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J cal_ablate
#SBATCH -o runtime/logs/out/cal_ablate-%j.out
#SBATCH -e runtime/logs/err/cal_ablate-%j.err

set -euo pipefail

PROJECT_DIR_DEFAULT="/home/$USER/GINO_Model/neuraloperator_clean_mcdropout"
PROJECT_DIR="${PROJECT_DIR:-${PROJECT_DIR_DEFAULT}}"
if [[ -f "${PROJECT_DIR}/scripts/slurm/lib/common.sh" ]]; then
  source "${PROJECT_DIR}/scripts/slurm/lib/common.sh"
  SCRIPT_DIR="${PROJECT_DIR}/scripts"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/lib/common.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  source "${SCRIPT_DIR}/slurm/lib/common.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm/lib/common.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts"
  source "${SCRIPT_DIR}/slurm/lib/common.sh"
else
  CANONICAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  source "$(cd "${CANONICAL_DIR}/.." && pwd)/lib/common.sh"
  SCRIPT_DIR="$(slurm_resolve_scripts_root "${CANONICAL_DIR}")"
fi

slurm_load_apptainer
cd "${SCRIPT_DIR}"
mkdir -p runtime/logs/out runtime/logs/err

CONTAINER_PATH="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
slurm_configure_host_ca

export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"

CALIBRATION_RUN_DIR="${CALIBRATION_RUN_DIR:-}"
if [[ -n "${CALIBRATION_RUN_DIR}" ]]; then
  CALIBRATION_ARTIFACT_ROOT="${CALIBRATION_ARTIFACT_ROOT:-${CALIBRATION_RUN_DIR}/outputs/forecast_artifacts/calibration}"
  HELDOUT_ARTIFACT_ROOT="${HELDOUT_ARTIFACT_ROOT:-${CALIBRATION_RUN_DIR}/outputs/forecast_artifacts/test_raw}"
  CONFIG_PATH="${CONFIG_PATH:-${CALIBRATION_RUN_DIR}/config/$(basename "${CALIBRATION_RUN_DIR}").yaml}"
  OUT_DIR="${OUT_DIR:-${CALIBRATION_RUN_DIR}/outputs/calibration_ablation_$(date +%Y%m%d_%H%M%S)}"
else
  CALIBRATION_ARTIFACT_ROOT="${CALIBRATION_ARTIFACT_ROOT:?Set CALIBRATION_ARTIFACT_ROOT or CALIBRATION_RUN_DIR}"
  HELDOUT_ARTIFACT_ROOT="${HELDOUT_ARTIFACT_ROOT:?Set HELDOUT_ARTIFACT_ROOT or CALIBRATION_RUN_DIR}"
  CONFIG_PATH="${CONFIG_PATH:-}"
  OUT_DIR="${OUT_DIR:?Set OUT_DIR or CALIBRATION_RUN_DIR}"
fi

if [[ ! -d "${CALIBRATION_ARTIFACT_ROOT}" ]]; then
  echo "ERROR: CALIBRATION_ARTIFACT_ROOT does not exist: ${CALIBRATION_ARTIFACT_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${HELDOUT_ARTIFACT_ROOT}" ]]; then
  echo "ERROR: HELDOUT_ARTIFACT_ROOT does not exist: ${HELDOUT_ARTIFACT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUT_DIR}"

CLI_ARGS=(
  -m neuralop.flood.eval.calibration_ablation
  --calibration-artifacts "${CALIBRATION_ARTIFACT_ROOT}"
  --heldout-artifacts "${HELDOUT_ARTIFACT_ROOT}"
  --out-dir "${OUT_DIR}"
)

if [[ -n "${CONFIG_PATH}" && -f "${CONFIG_PATH}" ]]; then
  CLI_ARGS+=(--config "${CONFIG_PATH}")
fi
if [[ -n "${THRESHOLDS_M:-}" ]]; then
  # shellcheck disable=SC2206
  THRESHOLD_ARGS=(${THRESHOLDS_M})
  CLI_ARGS+=(--thresholds-m "${THRESHOLD_ARGS[@]}")
fi
if [[ -n "${MEAN_RMSE_WEIGHT:-}" ]]; then
  CLI_ARGS+=(--mean-rmse-weight "${MEAN_RMSE_WEIGHT}")
fi
if [[ -n "${SPREAD_RATIO_WEIGHT:-}" ]]; then
  CLI_ARGS+=(--spread-ratio-weight "${SPREAD_RATIO_WEIGHT}")
fi
if [[ -n "${TARGET_SPREAD_RATIO:-}" ]]; then
  CLI_ARGS+=(--target-spread-ratio "${TARGET_SPREAD_RATIO}")
fi
if [[ -n "${TAIL_THRESHOLD_M:-}" ]]; then
  CLI_ARGS+=(--tail-threshold-m "${TAIL_THRESHOLD_M}")
fi
if [[ -n "${TAIL_WEIGHT:-}" ]]; then
  CLI_ARGS+=(--tail-weight "${TAIL_WEIGHT}")
fi
if [[ -n "${PROGRESS_INTERVAL:-}" ]]; then
  CLI_ARGS+=(--progress-interval "${PROGRESS_INTERVAL}")
fi
if [[ "${SKIP_IMPACT_METRICS:-0}" == "1" ]]; then
  CLI_ARGS+=(--skip-impact-metrics)
fi

printf 'Running offline calibration ablation\n'
printf '  CALIBRATION_ARTIFACT_ROOT=%s\n' "${CALIBRATION_ARTIFACT_ROOT}"
printf '  HELDOUT_ARTIFACT_ROOT=%s\n' "${HELDOUT_ARTIFACT_ROOT}"
printf '  CONFIG_PATH=%s\n' "${CONFIG_PATH}"
printf '  OUT_DIR=%s\n' "${OUT_DIR}"
printf '  SKIP_IMPACT_METRICS=%s\n' "${SKIP_IMPACT_METRICS:-0}"
printf '  PROGRESS_INTERVAL=%s\n' "${PROGRESS_INTERVAL:-50}"

apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python "${CLI_ARGS[@]}"
