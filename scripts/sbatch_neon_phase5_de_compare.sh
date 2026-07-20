#!/bin/bash
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
set -euo pipefail

: "${NEON_EXPECTED_HEAD:?Set NEON_EXPECTED_HEAD}"
: "${NEON_DE_CONFIG:?Set NEON_DE_CONFIG}"
: "${NEON_DE_CONFIG_SHA256:?Set NEON_DE_CONFIG_SHA256}"
: "${NEON_DE_BUNDLE:?Set NEON_DE_BUNDLE}"
: "${NEON_DE_BUNDLE_SHA256:?Set NEON_DE_BUNDLE_SHA256}"
: "${NEON_DE_STAGE2_DIR:?Set NEON_DE_STAGE2_DIR}"
: "${NEON_DE_CHECKPOINT_SHA256:?Set NEON_DE_CHECKPOINT_SHA256}"
: "${NEON_DE_OUTPUT_DIR:?Set NEON_DE_OUTPUT_DIR}"

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
CHECKPOINT=${NEON_DE_STAGE2_DIR}/neon_stage2_best.pt

cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}"
test -z "$(git status --porcelain)" || {
  echo "Phase-S deep-ensemble comparison refuses a dirty repository" >&2
  exit 2
}
test "$(sha256sum "${NEON_DE_CONFIG}" | awk '{print $1}')" = "${NEON_DE_CONFIG_SHA256}"
test "$(sha256sum "${NEON_DE_BUNDLE}" | awk '{print $1}')" = "${NEON_DE_BUNDLE_SHA256}"
test "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" = "${NEON_DE_CHECKPOINT_SHA256}"

module purge
module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"
slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
mkdir -p "${NEON_DE_OUTPUT_DIR}/cache"

apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" \
  python scripts/neon_deep_ensemble_compare.py \
  --config "${NEON_DE_CONFIG}" \
  --stage2-checkpoint "${CHECKPOINT}" \
  --stage1-bundle "${NEON_DE_BUNDLE}" \
  --output-dir "${NEON_DE_OUTPUT_DIR}" --families val --max-families 50 \
  --m-eval 16 --k-neon 50 --k-de 50 --k-chunk 8 \
  --epistemic-chunk 4 --cache-dir "${NEON_DE_OUTPUT_DIR}/cache" --seed 0

sha256sum "${NEON_DE_OUTPUT_DIR}/deep_ensemble_comparison.json" \
  > "${NEON_DE_OUTPUT_DIR}/deep_ensemble_comparison.json.sha256"
printf 'complete\n' > "${NEON_DE_OUTPUT_DIR}/COMPLETE"
printf '%s\n' "${SLURM_JOB_ID:-manual}" > "${NEON_DE_OUTPUT_DIR}/job_id.txt"
