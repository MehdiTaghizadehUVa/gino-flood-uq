#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J cal_artviz
#SBATCH -o runtime/logs/out/cal_artviz-%j.out
#SBATCH -e runtime/logs/err/cal_artviz-%j.err

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
slurm_prepare_geo_venv "${CONTAINER_PATH}"

export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"

CALIBRATION_RUN_DIR="${CALIBRATION_RUN_DIR:-}"
if [[ -n "${CALIBRATION_RUN_DIR}" ]]; then
  ARTIFACT_ROOT="${ARTIFACT_ROOT:-${CALIBRATION_RUN_DIR}/outputs/forecast_artifacts/test_raw}"
  COEFFICIENT_PATH="${COEFFICIENT_PATH:-${CALIBRATION_RUN_DIR}/outputs/calibration/crps_mbm_coefficients.json}"
  ISOTONIC_PATH="${ISOTONIC_PATH:-${CALIBRATION_RUN_DIR}/outputs/calibration/exceedance_isotonic_curves.json}"
  VISUALIZATION_CONFIG_PATH="${VISUALIZATION_CONFIG_PATH:-${CALIBRATION_RUN_DIR}/config/$(basename "${CALIBRATION_RUN_DIR}").yaml}"
  OUT_DIR="${OUT_DIR:-${CALIBRATION_RUN_DIR}/outputs/calibrated_artifact_visuals_$(date +%Y%m%d_%H%M%S)}"
else
  ARTIFACT_ROOT="${ARTIFACT_ROOT:?Set ARTIFACT_ROOT or CALIBRATION_RUN_DIR}"
  COEFFICIENT_PATH="${COEFFICIENT_PATH:?Set COEFFICIENT_PATH or CALIBRATION_RUN_DIR}"
  OUT_DIR="${OUT_DIR:?Set OUT_DIR or CALIBRATION_RUN_DIR}"
  VISUALIZATION_CONFIG_PATH="${VISUALIZATION_CONFIG_PATH:-}"
fi

if [[ ! -d "${ARTIFACT_ROOT}" ]]; then
  echo "ERROR: ARTIFACT_ROOT does not exist: ${ARTIFACT_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${COEFFICIENT_PATH}" ]]; then
  echo "ERROR: COEFFICIENT_PATH does not exist: ${COEFFICIENT_PATH}" >&2
  exit 2
fi
mkdir -p "${OUT_DIR}"

CLI_ARGS=(
  -m neuralop.flood.eval.artifact_visualization
  --artifact_root "${ARTIFACT_ROOT}"
  --coefficient_path "${COEFFICIENT_PATH}"
  --out_dir "${OUT_DIR}"
  --write-gif
  --write-mp4
  --map-enabled
)

if [[ -n "${VISUALIZATION_CONFIG_PATH:-}" && -f "${VISUALIZATION_CONFIG_PATH}" ]]; then
  CLI_ARGS+=(--visualization_config_path "${VISUALIZATION_CONFIG_PATH}")
fi
if [[ -n "${ISOTONIC_PATH:-}" && -f "${ISOTONIC_PATH}" ]]; then
  CLI_ARGS+=(--isotonic_path "${ISOTONIC_PATH}")
fi
if [[ -n "${HYDROGRAPH_IDS:-}" ]]; then
  CLI_ARGS+=(--hydrograph_id "${HYDROGRAPH_IDS}")
fi
# Default to a small visual audit subset; set MAX_ARTIFACTS=0 or unset it in a custom script for all artifacts.
MAX_ARTIFACTS="${MAX_ARTIFACTS:-3}"
if [[ -n "${MAX_ARTIFACTS}" && "${MAX_ARTIFACTS}" != "0" ]]; then
  CLI_ARGS+=(--max_artifacts "${MAX_ARTIFACTS}")
fi
if [[ -n "${ELEVATION_PATH:-}" ]]; then
  CLI_ARGS+=(--elevation_path "${ELEVATION_PATH}")
fi
if [[ -n "${ELEVATION_DATASET:-}" ]]; then
  CLI_ARGS+=(--elevation_dataset "${ELEVATION_DATASET}")
fi
if [[ -n "${MAP_MODE:-}" ]]; then
  CLI_ARGS+=(--map-mode "${MAP_MODE}")
fi
if [[ -n "${MAP_PROVIDER:-}" ]]; then
  CLI_ARGS+=(--map-provider "${MAP_PROVIDER}")
fi
if [[ "${SHOW_WET_EDGE:-0}" == "1" ]]; then
  CLI_ARGS+=(--show-wet-edge)
else
  CLI_ARGS+=(--no-show-wet-edge)
fi

printf 'Rendering calibrated artifact visuals\n'
printf '  ARTIFACT_ROOT=%s\n' "${ARTIFACT_ROOT}"
printf '  COEFFICIENT_PATH=%s\n' "${COEFFICIENT_PATH}"
printf '  OUT_DIR=%s\n' "${OUT_DIR}"
printf '  MAX_ARTIFACTS=%s\n' "${MAX_ARTIFACTS}"

apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" python "${CLI_ARGS[@]}"
