#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 PHASE5_ROOT STAGE2_DIR ID_METRICS OUTPUT_DIR" >&2
  exit 2
fi
ROOT=$1
STAGE2_DIR=$2
ID_METRICS=$3
OUT=$4

REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
BUNDLE=${NEON_BUNDLE:-/scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json}
CONFIG=${NEON_HISTORICAL_CONFIG:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4/eval_b3_ood_historical/config/historical_13_single_reference.yaml}
DEPENDENCY=${NEON_SBATCH_DEPENDENCY:-}

cd "${REPO}"
HEAD=$(git rev-parse HEAD)
test -z "$(git status --porcelain)" || {
  echo "Refusing OOD submission from a dirty repository" >&2
  exit 2
}
test -s "${ROOT}/PROTOCOL.json" && test -s "${ROOT}/PROTOCOL.json.sha256"
test -s "${CONFIG}"
if [[ -s "${STAGE2_DIR}/neon_stage2_best.pt" ]]; then
  STAGE2_READY=1
elif [[ -n "${DEPENDENCY}" ]]; then
  STAGE2_READY=0
else
  echo "Missing Stage-2 checkpoint without an upstream dependency: ${STAGE2_DIR}" >&2
  exit 2
fi
if [[ -s "${ID_METRICS}" && -s "${ID_METRICS}.sha256" ]]; then
  ID_METRICS_READY=1
elif [[ -n "${DEPENDENCY}" ]]; then
  # The finalizer depends on the producer job and verifies the completed
  # checksummed artifact before publishing any ranking evidence.
  ID_METRICS_READY=0
else
  echo "Missing checksummed ID metrics without an upstream dependency: ${ID_METRICS}" >&2
  exit 2
fi
test ! -e "${OUT}/SUBMITTED.json" || {
  echo "OOD evidence was already submitted under ${OUT}" >&2
  exit 2
}
mkdir -p "${OUT}/logs"
CONFIG_SHA=$(sha256sum "${CONFIG}" | awk '{print $1}')

module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"

# Validate the complete scientific argument set before touching Slurm.
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python -m neuralop.flood.cli.eval_neon_stage2 \
  --config "${CONFIG}" \
  --stage2-checkpoint "${STAGE2_DIR}/neon_stage2_best.pt" \
  --stage1-bundle "${BUNDLE}" --output-dir "${OUT}/output" \
  --families all --m-eval 16 --k-eval 50 --rollout-length -1 \
  --thresholds 0.1 0.3 0.5 --seed 0 --cache-dir "${OUT}/cache_k50" \
  --k-chunk 8 --compare-base --impact-metrics --variance-maps 0 \
  --expected-families 13 --allow-single-reference --dry-run >/dev/null
if [[ "${STAGE2_READY}" == 1 && "${ID_METRICS_READY}" == 1 ]]; then
  apptainer exec --bind /scratch,/home "${CONTAINER}" python - \
    "${ROOT}/PROTOCOL.json" "${ID_METRICS}" \
    "${STAGE2_DIR}/neon_stage2_best.pt" "${HEAD}" <<'PY'
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
verify_phase5_evidence_artifact(iid, expected_head=head, protocol_sha256=protocol_sha)
provenance = json.loads((iid.parent / "PROVENANCE.json").read_text())
checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if str(provenance.get("checkpoint_sha256", "")) != checkpoint_sha:
    raise ValueError("ID evidence and OOD checkpoint SHA-256 values differ.")
PY
else
  apptainer exec --bind /scratch,/home "${CONTAINER}" python - \
    "${ROOT}/PROTOCOL.json" <<'PY'
import sys
from neuralop.flood.eval.neon_phase5 import verify_checksummed_artifact
verify_checksummed_artifact(sys.argv[1])
PY
fi

if [[ "${NEON_SUBMIT_DRY_RUN:-0}" == 1 ]]; then
  printf 'Validated OOD evidence only: stage2=%s output=%s\n' "${STAGE2_DIR}" "${OUT}"
  exit 0
fi

EXPORTS="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_PHASE5_ROOT=${ROOT},NEON_OOD_CONFIG=${CONFIG},NEON_OOD_CONFIG_SHA256=${CONFIG_SHA},NEON_OOD_OUTPUT_DIR=${OUT},NEON_STAGE2_DIR=${STAGE2_DIR},NEON_ID_METRICS=${ID_METRICS},NEON_BUNDLE=${BUNDLE}"
ARRAY_ARGS=(--parsable --job-name=neon_p5_ood --array=0-12%5 \
  --output="${OUT}/logs/%A_%a.out" --error="${OUT}/logs/%A_%a.err" \
  --export="${EXPORTS}")
if [[ -n "${DEPENDENCY}" ]]; then
  ARRAY_ARGS+=(--dependency="${DEPENDENCY}")
fi
ARRAY=$(sbatch "${ARRAY_ARGS[@]}" "${REPO}/scripts/sbatch_neon_phase5_ood_array.sh")
FINAL=$(sbatch --parsable --job-name=neon_p5_oodf \
  --output="${OUT}/logs/finalize_%j.out" --error="${OUT}/logs/finalize_%j.err" \
  --dependency="afterok:${ARRAY}" --export="${EXPORTS}" \
  "${REPO}/scripts/sbatch_neon_phase5_ood_finalize.sh")

apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' \
  "${OUT}" "${HEAD}" "${CONFIG}" "${CONFIG_SHA}" "${STAGE2_DIR}" \
  "${ID_METRICS}" "${ARRAY}" "${FINAL}" "${DEPENDENCY}"
import hashlib, json, pathlib, sys
out, head, config, config_sha, stage2, iid, array, final, dependency = sys.argv[1:]
payload = {
    "schema_version": "neon_phase5_ood_submission_v1",
    "git_head": head,
    "historical_config": config,
    "historical_config_sha256": config_sha,
    "stage2_dir": stage2,
    "id_metrics": iid,
    "array_job_id": array,
    "finalize_job_id": final,
    "upstream_dependency": dependency or None,
    "event_count": 13,
    "reference_policy": "one_observed_hindcast_trajectory_per_event",
}
path = pathlib.Path(out) / "SUBMITTED.json"
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
path.write_bytes(encoded)
digest = hashlib.sha256(encoded).hexdigest()
path.with_suffix(".json.sha256").write_text(f"{digest}  {path.name}\n")
PY
printf 'OOD evidence submitted: array=%s finalize=%s\n' "${ARRAY}" "${FINAL}"
