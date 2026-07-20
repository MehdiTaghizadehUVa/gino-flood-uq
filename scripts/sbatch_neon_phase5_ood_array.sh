#!/bin/bash
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail

: "${NEON_EXPECTED_HEAD:?Set NEON_EXPECTED_HEAD}"
: "${NEON_OOD_CONFIG:?Set NEON_OOD_CONFIG}"
: "${NEON_OOD_CONFIG_SHA256:?Set NEON_OOD_CONFIG_SHA256}"
: "${NEON_OOD_OUTPUT_DIR:?Set NEON_OOD_OUTPUT_DIR}"
: "${NEON_STAGE2_DIR:?Set NEON_STAGE2_DIR}"
: "${NEON_PHASE5_ROOT:?Set NEON_PHASE5_ROOT}"
: "${NEON_ID_METRICS:?Set NEON_ID_METRICS}"
: "${NEON_BUNDLE:?Set NEON_BUNDLE}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a 0-12 array}"

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
CHECKPOINT=${NEON_STAGE2_DIR}/neon_stage2_best.pt
PROTOCOL=${NEON_PHASE5_ROOT}/PROTOCOL.json

cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}"
test -z "$(git status --porcelain)" || {
  echo "Phase-5 OOD evaluation refuses a dirty repository" >&2
  exit 2
}
test "$(sha256sum "${NEON_OOD_CONFIG}" | awk '{print $1}')" = "${NEON_OOD_CONFIG_SHA256}"
test -s "${CHECKPOINT}"
test -s "${PROTOCOL}" && test -s "${PROTOCOL}.sha256"
test -s "${NEON_ID_METRICS}" && test -s "${NEON_ID_METRICS}.sha256"

module purge
module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"
slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"
export PYTHONUNBUFFERED=1

# Every shard verifies the checkpoint against the signed in-distribution
# evidence after any upstream training/evaluation dependency has completed.
apptainer exec --bind /scratch,/home "${CONTAINER}" python - \
  "${PROTOCOL}" "${NEON_ID_METRICS}" "${CHECKPOINT}" \
  "${NEON_EXPECTED_HEAD}" <<'PY'
import hashlib, json, pathlib, sys
from neuralop.flood.eval.neon_phase5 import (
    verify_checksummed_artifact,
    verify_phase5_evidence_artifact,
)

protocol = pathlib.Path(sys.argv[1])
id_metrics = pathlib.Path(sys.argv[2])
checkpoint = pathlib.Path(sys.argv[3])
head = sys.argv[4]
protocol_sha = verify_checksummed_artifact(protocol)
verify_phase5_evidence_artifact(
    id_metrics, expected_head=head, protocol_sha256=protocol_sha
)
provenance = json.loads((id_metrics.parent / "PROVENANCE.json").read_text())
checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if str(provenance.get("checkpoint_sha256", "")) != checkpoint_sha:
    raise ValueError("ID evidence and OOD checkpoint SHA-256 values differ.")
PY

mkdir -p "${NEON_OOD_OUTPUT_DIR}/output" "${NEON_OOD_OUTPUT_DIR}/cache_k50"
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" \
  python -m neuralop.flood.cli.eval_neon_stage2 \
  --config "${NEON_OOD_CONFIG}" \
  --stage2-checkpoint "${CHECKPOINT}" \
  --stage1-bundle "${NEON_BUNDLE}" \
  --output-dir "${NEON_OOD_OUTPUT_DIR}/output" \
  --families all --m-eval 16 --k-eval 50 --rollout-length -1 \
  --thresholds 0.1 0.3 0.5 --seed 0 \
  --cache-dir "${NEON_OOD_OUTPUT_DIR}/cache_k50" --k-chunk 8 \
  --compare-base --impact-metrics --variance-maps 0 \
  --family-index "${SLURM_ARRAY_TASK_ID}" --expected-families 13 \
  --allow-single-reference --resume --shard-only
