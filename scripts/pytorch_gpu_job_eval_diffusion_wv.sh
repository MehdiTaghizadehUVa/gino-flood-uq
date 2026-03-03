#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J ddofs_wv_ev
#SBATCH -o logs/out/ddofs_wv_ev-%j.out
#SBATCH -e logs/err/ddofs_wv_ev-%j.err

set -euo pipefail
module purge
module load apptainer

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
cd "${SCRIPT_DIR}"
mkdir -p logs/out logs/err

PROJECT_DIR="/home/$USER/GINO_Model/neuraloperator_no_physics"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/evaluate_diffusion_forecaster_WV.py"
EVAL_CONFIG="${PROJECT_DIR}/config/gino_pluvial_flood_config_WV_depth_only_diffusion.yaml"
CONTAINER_PATH="/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif"
TEST_ROOT="/scratch/$USER/Data_Generation_UQ/Results_Test/M40"
CHECKPOINT_ROOT="/home/$USER/GINO_Model/neuraloperator_no_physics/scripts/checkpoints_WV_depth_only_diffusion"

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

echo "Eval script:      ${EVAL_SCRIPT}"
echo "Config:           ${EVAL_CONFIG}"
echo "Checkpoint root:  ${CHECKPOINT_ROOT}"
echo "Test data root:   ${TEST_ROOT}"

apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${EVAL_SCRIPT}" \
  --config_path "${EVAL_CONFIG}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --data.root "${TEST_ROOT}" \
  --rollout_data.root "${TEST_ROOT}"
