#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 PHASE5_ROOT STAGE2_DIR OUTPUT_DIR" >&2
  exit 2
fi
ROOT=$1
STAGE2_DIR=$2
OUT=$3
REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
CONFIG=${NEON_CONFIG:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/config/coast_fgn_neon_tr450.yaml}
BUNDLE=${NEON_BUNDLE:-/scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json}
CHECKPOINT=${STAGE2_DIR}/neon_stage2_best.pt

cd "${REPO}"
HEAD=$(git rev-parse HEAD)
test -z "$(git status --porcelain)" || { echo "Refusing DE submission from a dirty tree" >&2; exit 2; }
test -s "${ROOT}/PROTOCOL.json" && test -s "${ROOT}/PROTOCOL.json.sha256"
test -s "${CONFIG}" && test -s "${BUNDLE}" && test -s "${CHECKPOINT}"
test ! -e "${OUT}/SUBMITTED.json" || { echo "DE evidence already submitted: ${OUT}" >&2; exit 2; }
CONFIG_SHA=$(sha256sum "${CONFIG}" | awk '{print $1}')
BUNDLE_SHA=$(sha256sum "${BUNDLE}" | awk '{print $1}')
CHECKPOINT_SHA=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')

module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_deep_ensemble_compare.py \
  --config "${CONFIG}" --stage2-checkpoint "${CHECKPOINT}" \
  --stage1-bundle "${BUNDLE}" --output-dir "${OUT}" \
  --families val --max-families 50 --m-eval 16 --k-neon 50 --k-de 50 \
  --k-chunk 8 --epistemic-chunk 4 --cache-dir "${OUT}/cache" --seed 0 --dry-run \
  >/dev/null
apptainer exec --bind /scratch,/home "${CONTAINER}" python - \
  "${ROOT}/PROTOCOL.json" <<'PY'
import sys
from neuralop.flood.eval.neon_phase5 import verify_checksummed_artifact
verify_checksummed_artifact(sys.argv[1])
PY
PROTOCOL_SHA=$(sha256sum "${ROOT}/PROTOCOL.json" | awk '{print $1}')
if [[ "${NEON_SUBMIT_DRY_RUN:-0}" == 1 ]]; then
  printf 'Validated DE evidence only: stage2=%s output=%s\n' "${STAGE2_DIR}" "${OUT}"
  exit 0
fi

mkdir -p "${OUT}"
EXPORTS="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_DE_CONFIG=${CONFIG},NEON_DE_CONFIG_SHA256=${CONFIG_SHA},NEON_DE_BUNDLE=${BUNDLE},NEON_DE_BUNDLE_SHA256=${BUNDLE_SHA},NEON_DE_STAGE2_DIR=${STAGE2_DIR},NEON_DE_CHECKPOINT_SHA256=${CHECKPOINT_SHA},NEON_DE_OUTPUT_DIR=${OUT}"
JOB=$(sbatch --parsable --job-name=neon_p5_de \
  --output="${OUT}/slurm-%j.out" --error="${OUT}/slurm-%j.err" \
  --export="${EXPORTS}" scripts/sbatch_neon_phase5_de_compare.sh)

python3 - <<'PY' "${OUT}" "${HEAD}" "${CONFIG}" "${CONFIG_SHA}" "${BUNDLE}" "${BUNDLE_SHA}" "${STAGE2_DIR}" "${CHECKPOINT_SHA}" "${PROTOCOL_SHA}" "${JOB}"
import hashlib, json, pathlib, sys
out, head, config, config_sha, bundle, bundle_sha, stage2, checkpoint_sha, protocol_sha, job = sys.argv[1:]
payload = {
    "schema_version": "neon_phase5_de_submission_v1",
    "git_head": head,
    "config": config,
    "config_sha256": config_sha,
    "stage1_bundle": bundle,
    "stage1_bundle_sha256": bundle_sha,
    "stage2_dir": stage2,
    "stage2_checkpoint_sha256": checkpoint_sha,
    "protocol_sha256": protocol_sha,
    "job_id": job,
    "physical_space": True,
    "common_aleatory_latent_bank": True,
    "stage1_model_count_policy": "all_models_in_bundle; minimum_two_enforced_by_evaluator",
}
path = pathlib.Path(out) / "SUBMITTED.json"
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
path.write_bytes(encoded)
digest = hashlib.sha256(encoded).hexdigest()
path.with_suffix(".json.sha256").write_text(f"{digest}  {path.name}\n")
PY
printf 'DE evidence submitted: job=%s\n' "${JOB}"
