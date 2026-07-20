#!/bin/bash
#SBATCH --account=uqgroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail

: "${NEON_EXPECTED_HEAD:?Set NEON_EXPECTED_HEAD}"
: "${NEON_SCALEOUT_ROOT:?Set NEON_SCALEOUT_ROOT}"
: "${NEON_SCALEOUT_RUNG:?Set NEON_SCALEOUT_RUNG}"
: "${NEON_PROTOCOL_SHA256:?Set NEON_PROTOCOL_SHA256}"
: "${NEON_GOVERNING_GATE_SHA256:?Set NEON_GOVERNING_GATE_SHA256}"
REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}"
test -z "$(git status --porcelain)"
module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}"
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_contraction_analysis.py "${NEON_SCALEOUT_ROOT}" \
  --ladder-rung "${NEON_SCALEOUT_RUNG}" \
  --expected-git-head "${NEON_EXPECTED_HEAD}" \
  --analysis-git-head "${NEON_EXPECTED_HEAD}" \
  --protocol-sha256 "${NEON_PROTOCOL_SHA256}" \
  --governing-gate-sha256 "${NEON_GOVERNING_GATE_SHA256}" \
  --output-prefix "${NEON_SCALEOUT_ROOT}/contraction_analysis"
