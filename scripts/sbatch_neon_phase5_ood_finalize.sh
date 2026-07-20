#!/bin/bash
#SBATCH --account=uqgroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail

: "${NEON_EXPECTED_HEAD:?Set NEON_EXPECTED_HEAD}"
: "${NEON_PHASE5_ROOT:?Set NEON_PHASE5_ROOT}"
: "${NEON_OOD_CONFIG:?Set NEON_OOD_CONFIG}"
: "${NEON_OOD_CONFIG_SHA256:?Set NEON_OOD_CONFIG_SHA256}"
: "${NEON_OOD_OUTPUT_DIR:?Set NEON_OOD_OUTPUT_DIR}"
: "${NEON_STAGE2_DIR:?Set NEON_STAGE2_DIR}"
: "${NEON_ID_METRICS:?Set NEON_ID_METRICS}"
: "${NEON_BUNDLE:?Set NEON_BUNDLE}"

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
CHECKPOINT=${NEON_STAGE2_DIR}/neon_stage2_best.pt
PROTOCOL=${NEON_PHASE5_ROOT}/PROTOCOL.json

cd "${REPO}"
test "$(git rev-parse HEAD)" = "${NEON_EXPECTED_HEAD}"
test -z "$(git status --porcelain)" || {
  echo "Phase-5 OOD finalization refuses a dirty repository" >&2
  exit 2
}
test "$(sha256sum "${NEON_OOD_CONFIG}" | awk '{print $1}')" = "${NEON_OOD_CONFIG_SHA256}"
test -s "${CHECKPOINT}"
test -s "${PROTOCOL}"
test -s "${PROTOCOL}.sha256"
test -s "${NEON_ID_METRICS}"
test -s "${NEON_ID_METRICS}.sha256"

module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"

read -r PROTOCOL_SHA CHECKPOINT_SHA < <(
  apptainer exec --bind /scratch,/home "${CONTAINER}" python - \
    "${PROTOCOL}" "${NEON_ID_METRICS}" "${CHECKPOINT}" "${NEON_EXPECTED_HEAD}" <<'PY'
import hashlib, json, pathlib, sys
from neuralop.flood.eval.neon_phase5 import (
    verify_checksummed_artifact,
    verify_phase5_evidence_artifact,
)
protocol = pathlib.Path(sys.argv[1])
iid = pathlib.Path(sys.argv[2])
checkpoint = pathlib.Path(sys.argv[3])
head = sys.argv[4]
protocol_sha = verify_checksummed_artifact(protocol)
verify_phase5_evidence_artifact(
    iid, expected_head=head, protocol_sha256=protocol_sha
)
provenance = json.loads((iid.parent / "PROVENANCE.json").read_text())
checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if str(provenance.get("checkpoint_sha256", "")) != checkpoint_sha:
    raise ValueError("ID evidence and OOD checkpoint SHA-256 values differ.")
print(protocol_sha, checkpoint_sha)
PY
)

apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python -m neuralop.flood.cli.eval_neon_stage2 \
  --config "${NEON_OOD_CONFIG}" \
  --stage2-checkpoint "${CHECKPOINT}" \
  --stage1-bundle "${NEON_BUNDLE}" \
  --output-dir "${NEON_OOD_OUTPUT_DIR}/output" \
  --families all --m-eval 16 --k-eval 50 --rollout-length -1 \
  --thresholds 0.1 0.3 0.5 --seed 0 \
  --cache-dir "${NEON_OOD_OUTPUT_DIR}/cache_k50" --k-chunk 8 \
  --compare-base --impact-metrics --variance-maps 0 \
  --expected-families 13 --allow-single-reference --merge-only

apptainer exec --bind /scratch,/home "${CONTAINER}" python - \
  "${NEON_OOD_OUTPUT_DIR}" "${NEON_EXPECTED_HEAD}" "${CHECKPOINT}" \
  "${NEON_OOD_OUTPUT_DIR}/cache_k50" "${PROTOCOL_SHA}" <<'PY'
import pathlib, sys
from neon_phase5_runtime import write_provenance
output, head, checkpoint, cache_dir, protocol_sha = sys.argv[1:]
write_provenance(
    pathlib.Path(output),
    head=head,
    checkpoint=pathlib.Path(checkpoint),
    cache_dir=pathlib.Path(cache_dir),
    protocol_sha256=protocol_sha,
)
PY

apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_ood_ranking_analysis.py \
  --ood-metrics "${NEON_OOD_OUTPUT_DIR}/output/neon_eval_metrics.json" \
  --id-metrics "${NEON_ID_METRICS}" \
  --output-prefix "${NEON_OOD_OUTPUT_DIR}/ranking" \
  --expected-ood-events 13 \
  --analysis-git-head "${NEON_EXPECTED_HEAD}" \
  --protocol-sha256 "${PROTOCOL_SHA}" \
  --stage2-checkpoint-sha256 "${CHECKPOINT_SHA}"

apptainer exec --bind /scratch,/home "${CONTAINER}" python - \
  "${NEON_OOD_OUTPUT_DIR}/ranking.json" "${NEON_EXPECTED_HEAD}" \
  "${PROTOCOL_SHA}" "${CHECKPOINT_SHA}" <<'PY'
import sys
from neuralop.flood.eval.neon_phase5 import verify_phase5_ood_ranking_artifact
verify_phase5_ood_ranking_artifact(
    sys.argv[1],
    expected_head=sys.argv[2],
    protocol_sha256=sys.argv[3],
    checkpoint_sha256=sys.argv[4],
)
PY

printf 'complete\n' > "${NEON_OOD_OUTPUT_DIR}/COMPLETE"
printf '%s\n' "${SLURM_JOB_ID:-manual}" > "${NEON_OOD_OUTPUT_DIR}/finalize_job_id.txt"
