#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J fgn_wv_eval_masked_primary
#SBATCH -o runtime/logs/out/fgn_wv_eval_masked_primary-%j.out
#SBATCH -e runtime/logs/err/fgn_wv_eval_masked_primary-%j.err

set -euo pipefail

PROJECT_DIR_DEFAULT="/home/$USER/GINO_Model/neuraloperator_no_physics_git_main"
PROJECT_DIR="${PROJECT_DIR:-${PROJECT_DIR_DEFAULT}}"
if [[ -f "${PROJECT_DIR}/scripts/slurm/lib/common.sh" ]]; then
  source "${PROJECT_DIR}/scripts/slurm/lib/common.sh"
  SCRIPT_DIR="${PROJECT_DIR}/scripts"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm/lib/common.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
  source "${SCRIPT_DIR}/slurm/lib/common.sh"
else
  CANONICAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  source "$(cd "${CANONICAL_DIR}/.." && pwd)/lib/common.sh"
  SCRIPT_DIR="$(slurm_resolve_scripts_root "${CANONICAL_DIR}")"
fi
slurm_load_apptainer
cd "${SCRIPT_DIR}"
mkdir -p runtime/logs/out runtime/logs/err

export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
CONTAINER_PATH="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${PROJECT_DIR}/scripts/flood_wv_eval_operator.py}"
EVAL_CONFIG="${EVAL_CONFIG:-${PROJECT_DIR}/config/flood/wv/gino_pluvial_flood_config_WV_depth_only_masked_primary.yaml}"
TEST_ROOT="${TEST_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/test}"
TRAIN_ROOT="${TRAIN_ROOT:-/scratch/$USER/Data_Generation_UQ_dataset/results/Dynamic_M40_v1/train}"
ARTIFACT_DIR="${ARTIFACT_DIR:?ARTIFACT_DIR must be set}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?CHECKPOINT_ROOT must be set}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set}"
EVAL_CONFIG_RENDERED="${EVAL_CONFIG_RENDERED:-${OUT_DIR}/effective_eval_config.yaml}"
NORMALIZER_PATH="${NORMALIZER_PATH:-${ARTIFACT_DIR}/normalizers_depth_only_masked_primary.pt}"
MASK_PATH="${MASK_PATH:-${ARTIFACT_DIR}/structural_dry_mask_exact_zero.pt}"
TEST_TXT_NAME="${TEST_TXT_NAME:-test.txt}"
TRAIN_STUB_NAME="${TRAIN_STUB_NAME:-train_eval_stub.txt}"
TRAIN_TXT_NAME="${TRAIN_TXT_NAME:-train.txt}"
TEST_CLEAN_BOUNDARY_ROOT="${TEST_CLEAN_BOUNDARY_ROOT:-$(cd "${TEST_ROOT}/.." && pwd)/metadata}"
TEST_CLEAN_BOUNDARY_FILE="${TEST_CLEAN_BOUNDARY_FILE:-Hydrographs_Test_Clean.txt}"
ENSEMBLE_PER_MODEL="${ENSEMBLE_PER_MODEL:-25}"
RUN_SINGLE_STEP="${RUN_SINGLE_STEP:-true}"
RUN_ROLLOUT="${RUN_ROLLOUT:-true}"

slurm_configure_host_ca
slurm_assert_container_gpus "${CONTAINER_PATH}" 1
slurm_prepare_geo_venv "${CONTAINER_PATH}"

if [[ ! -f "${NORMALIZER_PATH}" ]]; then
  echo "ERROR: normalizer artifact not found: ${NORMALIZER_PATH}" >&2
  exit 2
fi
if [[ ! -f "${MASK_PATH}" ]]; then
  echo "ERROR: structural-dry artifact not found: ${MASK_PATH}" >&2
  exit 2
fi

TEST_TXT="${TEST_ROOT}/${TEST_TXT_NAME}"
if [[ ! -f "${TEST_TXT}" ]]; then
  find "${TEST_ROOT}" -maxdepth 1 -type f -name "*.hdf" -printf "%f\n" | sed 's/\.hdf$//' | sort > "${TEST_TXT}"
fi
if [[ ! -s "${TEST_TXT}" ]]; then
  echo "ERROR: Missing/empty rollout split: ${TEST_TXT}" >&2
  exit 1
fi
TRAIN_STUB_TXT="${TEST_ROOT}/${TRAIN_STUB_NAME}"
if [[ ! -s "${TRAIN_STUB_TXT}" ]]; then
  head -n 1 "${TEST_TXT}" > "${TRAIN_STUB_TXT}"
fi

MODEL_COUNT=0
if [[ -f "${CHECKPOINT_ROOT}/best_model_state_dict.pt" || -f "${CHECKPOINT_ROOT}/model_state_dict.pt" ]]; then
  MODEL_COUNT=1
else
  for d in "${CHECKPOINT_ROOT}"/*; do
    [[ -d "$d" ]] || continue
    if [[ -f "$d/best_model_state_dict.pt" || -f "$d/model_state_dict.pt" ]]; then
      MODEL_COUNT=$((MODEL_COUNT + 1))
    fi
  done
fi
if [[ "${MODEL_COUNT}" -lt 1 ]]; then
  echo "ERROR: No operator checkpoints found under ${CHECKPOINT_ROOT}" >&2
  exit 1
fi
TOTAL_ENSEMBLE=$((ENSEMBLE_PER_MODEL * MODEL_COUNT))
mkdir -p "${OUT_DIR}"
python3 - "${EVAL_CONFIG}" "${EVAL_CONFIG_RENDERED}" "${OUT_DIR}" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
out_dir = sys.argv[3]
with source.open("r", encoding="utf-8-sig") as f:
    payload = yaml.safe_load(f)
root = payload.get("flood", payload) if isinstance(payload, dict) else payload
if not isinstance(root, dict):
    raise SystemExit(f"Unsupported eval config structure in {source}")
root.setdefault("rollout", {})["out_dir"] = out_dir
dest.parent.mkdir(parents=True, exist_ok=True)
with dest.open("w", encoding="utf-8") as f:
    yaml.safe_dump(payload, f, sort_keys=False)
PY

EVAL_ARGS=(
  --config_path "${EVAL_CONFIG_RENDERED}"
  --checkpoint.save_dir "${CHECKPOINT_ROOT}"
  --data.root "${TEST_ROOT}"
  --data.train_txt "${TRAIN_STUB_NAME}"
  --data.write_train_txt false
  --data.normalizer_path "${NORMALIZER_PATH}"
  --data.clean_boundary_root "${TEST_CLEAN_BOUNDARY_ROOT}"
  --data.clean_boundary_file "${TEST_CLEAN_BOUNDARY_FILE}"
  --structural_dry.mask_path "${MASK_PATH}"
  --structural_dry.canonical_data_root "${TRAIN_ROOT}"
  --structural_dry.canonical_train_txt "${TRAIN_TXT_NAME}"
  --rollout_data.root "${TEST_ROOT}"
  --rollout_data.clean_boundary_root "${TEST_CLEAN_BOUNDARY_ROOT}"
  --rollout_data.clean_boundary_file "${TEST_CLEAN_BOUNDARY_FILE}"
  --rollout_data.test_txt "${TEST_TXT_NAME}"
  --rollout.n_ensemble_samples "${TOTAL_ENSEMBLE}"
  --wandb.log false
)
if [[ "${RUN_SINGLE_STEP}" == "true" ]]; then
  EVAL_ARGS+=(--run_single_step)
else
  EVAL_ARGS+=(--skip_single_step)
fi
if [[ "${RUN_ROLLOUT}" == "true" ]]; then
  EVAL_ARGS+=(--run_rollout)
else
  EVAL_ARGS+=(--skip_rollout)
fi

echo "Eval script:      ${EVAL_SCRIPT}"
echo "Config:           ${EVAL_CONFIG}"
echo "Checkpoint root:  ${CHECKPOINT_ROOT}"
echo "Artifact dir:     ${ARTIFACT_DIR}"
echo "Test data root:   ${TEST_ROOT}"
echo "Train root:       ${TRAIN_ROOT}"
echo "Canonical train:  ${TRAIN_ROOT}/${TRAIN_TXT_NAME}"
echo "Test boundary:    ${TEST_CLEAN_BOUNDARY_ROOT}/${TEST_CLEAN_BOUNDARY_FILE}"
echo "Train normalizer: ${NORMALIZER_PATH}"
echo "Dry mask:         ${MASK_PATH}"
echo "Models found:     ${MODEL_COUNT}"
echo "Ens/model:        ${ENSEMBLE_PER_MODEL}"
echo "Total ensemble:   ${TOTAL_ENSEMBLE}"
echo "Output dir:       ${OUT_DIR}"
echo "Git commit:       $(git -C "${PROJECT_DIR}" rev-parse HEAD)"

apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}"
