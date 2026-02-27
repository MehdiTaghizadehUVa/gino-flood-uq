#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -J eval_uq_m40
#SBATCH -o logs/out/eval_uq_m40-%A.out
#SBATCH -e logs/err/eval_uq_m40-%A.err
# Submit from scripts/ so SCRIPT_DIR and logs resolve:  cd scripts && sbatch eval_post_training_multi_model_testM40.sh
# Partition gpu requires at least one GPU (QOSMinGRES); this requests any available GPU.

set -euo pipefail

module purge
module load apptainer

# Use submission directory when in a SLURM job so logs/ and eval_outputs/ are writable (spool dir is not)
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PROJECT_DIR="/scratch/$USER/Data_Generation_UQ/GINO_Model/neuraloperator_no_physics"
TRAIN_CONFIG="${TRAIN_PROJECT_DIR}/config/gino_pluvial_flood_config_WV_depth_only.yaml"
CONTAINER_PATH="/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif"
CHECKPOINT_ROOT="/scratch/$USER/Data_Generation_UQ/GINO_Model/neuraloperator_no_physics/scripts/checkpoints_WV_depth_only_300ep"
# CHECKPOINT_ROOT must be the directory that directly contains best_model_state_dict.pt or model_state_dict.pt
ROLLOUT_ROOT="/scratch/$USER/Data_Generation_UQ/Results_Test/M40"
TRAIN_NORMALIZER="/scratch/$USER/Data_Generation_UQ/Results/M40/normalizers_depth_only.pt"
ENSEMBLE_PER_MODEL=25
RUN_TAG="eval_multi_model_testM40_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SCRIPT_DIR}/eval_outputs/${RUN_TAG}"

cd "${SCRIPT_DIR}"
mkdir -p logs/out logs/err "${OUT_DIR}"

TEST_TXT="${ROLLOUT_ROOT}/test.txt"
if [[ ! -f "${TEST_TXT}" ]]; then
  find "${ROLLOUT_ROOT}" -maxdepth 1 -type f -name "*.hdf" -printf "%f\n" | sed s/\.hdf$// | sort > "${TEST_TXT}"
fi
if [[ ! -s "${TEST_TXT}" ]]; then
  echo "ERROR: ${TEST_TXT} is empty."
  exit 1
fi
if [[ ! -f "${TRAIN_CONFIG}" ]]; then
  echo "ERROR: Training config not found at ${TRAIN_CONFIG}"
  exit 1
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
  echo "ERROR: No checkpoint runs found under ${CHECKPOINT_ROOT}"
  exit 1
fi
TOTAL_ENSEMBLE=$((ENSEMBLE_PER_MODEL * MODEL_COUNT))

TRAIN_STUB_TXT="${ROLLOUT_ROOT}/train_eval_stub.txt"
head -n 1 "${TEST_TXT}" > "${TRAIN_STUB_TXT}"

HOST_CA_BUNDLE=""
for cand in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt; do
  if [[ -f "$cand" ]]; then
    HOST_CA_BUNDLE="$cand"
    break
  fi
done
APPTAINER_BIND_ARGS="--nv"
if [[ -n "${HOST_CA_BUNDLE}" ]]; then
  APPTAINER_BIND_ARGS="--nv --bind ${HOST_CA_BUNDLE}:/host_ca_bundle.crt:ro"
  export APPTAINERENV_SSL_CERT_FILE=/host_ca_bundle.crt
  export APPTAINERENV_REQUESTS_CA_BUNDLE=/host_ca_bundle.crt
fi

if [[ -f "${PROJECT_DIR}/config/wandb_api_key.txt" ]]; then
  export APPTAINERENV_WANDB_API_KEY="$(cat "${PROJECT_DIR}/config/wandb_api_key.txt")"
fi

echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Rollout root:    ${ROLLOUT_ROOT}"
echo "Training config: ${TRAIN_CONFIG}"
echo "Ensemble/model:  ${ENSEMBLE_PER_MODEL}"
echo "Model count:     ${MODEL_COUNT}"
echo "Total ensemble:  ${TOTAL_ENSEMBLE}"
if command -v sha256sum >/dev/null 2>&1; then
  echo "Config SHA256:   $(sha256sum "${TRAIN_CONFIG}" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  echo "Config SHA256:   $(shasum -a 256 "${TRAIN_CONFIG}" | awk '{print $1}')"
fi
echo "Output dir:      ${OUT_DIR}"

echo "Starting evaluation..."
apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${PROJECT_DIR}/scripts/evaluate_post_training_flood_WV.py" \
  --config_path "${TRAIN_CONFIG}" \
  --checkpoint.save_dir "${CHECKPOINT_ROOT}" \
  --data.root "${ROLLOUT_ROOT}" \
  --data.train_txt "train_eval_stub.txt" \
  --data.write_train_txt false \
  --data.normalizer_path "${TRAIN_NORMALIZER}" \
  --rollout_data.root "${ROLLOUT_ROOT}" \
  --rollout_data.test_txt "test.txt" \
  --rollout.out_dir "${OUT_DIR}" \
  --rollout.n_ensemble_samples "${TOTAL_ENSEMBLE}" \
  --wandb.log false \
  --run_rollout \
  --skip_single_step

echo "Evaluation finished."
