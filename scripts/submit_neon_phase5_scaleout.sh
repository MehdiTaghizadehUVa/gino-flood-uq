#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /scratch/.../phase5_root /path/to/accepted/DECISION.json" >&2
  exit 2
fi
ROOT=$1
GATE=$2
PROTOCOL_ROOT=${NEON_PHASE5_PROTOCOL_ROOT:-${ROOT}}
REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
CACHE_DIR=${NEON_CACHE_DIR:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4/feature_cache_v3}
CONFIG=${NEON_CONFIG:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/config/coast_fgn_neon_tr450.yaml}
BUNDLE=${NEON_BUNDLE:-/scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json}
REQUESTED_PRIOR_SEED=${NEON_PRIOR_SEED:-}
REQUESTED_DIRICHLET_SEED=${NEON_DIRICHLET_PARTICLE_SEED:-}
cd "${REPO}"
HEAD=$(git rev-parse HEAD)
test -z "$(git status --porcelain)" || { echo "Refusing dirty-tree submission" >&2; exit 2; }
test -s "${GATE}" && test -s "${GATE}.sha256"
test -s "${PROTOCOL_ROOT}/PROTOCOL.json" && \
  test -s "${PROTOCOL_ROOT}/PROTOCOL.json.sha256"
test ! -e "${ROOT}/PHASE_S_SUBMITTED.json" || {
  echo "Phase S already submitted under ${ROOT}" >&2
  exit 2
}

module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"
apptainer exec --bind /scratch,/home "${CONTAINER}" python - \
  "${PROTOCOL_ROOT}/PROTOCOL.json" <<'PY'
import sys
from neuralop.flood.eval.neon_phase5 import verify_checksummed_artifact
verify_checksummed_artifact(sys.argv[1])
PY
PROTOCOL_SHA=$(sha256sum "${PROTOCOL_ROOT}/PROTOCOL.json" | awk '{print $1}')
GATE_SHA=$(sha256sum "${GATE}" | awk '{print $1}')
RESOLVE_ARGS=(
  --gate "${GATE}"
  --expected-head "${HEAD}"
  --protocol-sha256 "${PROTOCOL_SHA}"
)
if [[ -n "${REQUESTED_PRIOR_SEED}" ]]; then
  RESOLVE_ARGS+=(--prior-seed "${REQUESTED_PRIOR_SEED}")
fi
if [[ -n "${REQUESTED_DIRICHLET_SEED}" ]]; then
  RESOLVE_ARGS+=(--dirichlet-particle-seed "${REQUESTED_DIRICHLET_SEED}")
fi
TARGET=${ROOT}/PHASE_S_TARGET.json
TARGET_TMP=${TARGET}.tmp.$$
apptainer exec --bind /scratch,/home "${CONTAINER}" \
  python scripts/neon_scaleout_preflight.py resolve-evidence "${RESOLVE_ARGS[@]}" \
  > "${TARGET_TMP}"
mv "${TARGET_TMP}" "${TARGET}"
sha256sum "${TARGET}" > "${TARGET}.sha256"
read -r RUNG PRIOR_SEED DIRICHLET_SEED STAGE2_DIR ID_METRICS < <(
  python3 - <<'PY' "${TARGET}"
import json, sys
target = json.load(open(sys.argv[1]))
print(
    target["ladder_rung"],
    target["prior_seed"],
    "NONE" if target["dirichlet_particle_seed"] is None else target["dirichlet_particle_seed"],
    target["stage2_dir"],
    target["id_metrics"],
)
PY
)
if [[ "${DIRICHLET_SEED}" == NONE ]]; then DIRICHLET_SEED=; fi
SCALE_ROOT=${ROOT}/phase_s_n_sweep_${RUNG,,}

# Validate every Phase-S command before the first scheduler mutation.
SCALE_ENV=(
  NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}"
  NEON_SCALEOUT_ROOT="${SCALE_ROOT}" NEON_CACHE_DIR="${CACHE_DIR}"
  NEON_SCALEOUT_RUNG="${RUNG}" NEON_PRIOR_SEED="${PRIOR_SEED}"
  NEON_PROTOCOL_SHA256="${PROTOCOL_SHA}"
  NEON_GOVERNING_GATE_SHA256="${GATE_SHA}"
)
if [[ -n "${DIRICHLET_SEED}" ]]; then
  SCALE_ENV+=(NEON_DIRICHLET_PARTICLE_SEED="${DIRICHLET_SEED}")
fi
env "${SCALE_ENV[@]}" NEON_SUBMIT_DRY_RUN=1 \
  bash scripts/submit_neon_ablation_grid.sh "${GATE}" >/dev/null
env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" NEON_BUNDLE="${BUNDLE}" \
  NEON_SUBMIT_DRY_RUN=1 bash scripts/submit_neon_phase5_ood_evidence.sh \
  "${PROTOCOL_ROOT}" "${STAGE2_DIR}" "${ID_METRICS}" \
  "${ROOT}/phase_s_ood_${RUNG,,}" >/dev/null
env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" NEON_CONFIG="${CONFIG}" \
  NEON_BUNDLE="${BUNDLE}" NEON_SUBMIT_DRY_RUN=1 \
  bash scripts/submit_neon_phase5_de_evidence.sh \
  "${PROTOCOL_ROOT}" "${STAGE2_DIR}" "${ROOT}/phase_s_de_${RUNG,,}" >/dev/null

env "${SCALE_ENV[@]}" \
  bash scripts/submit_neon_ablation_grid.sh "${GATE}" >/dev/null
ARRAY_JOB=$(<"${SCALE_ROOT}/job_id.txt")
ANALYSIS=$(sbatch --parsable --job-name=neon_p5_gamma \
  --output="${SCALE_ROOT}/analysis_%j.out" --error="${SCALE_ROOT}/analysis_%j.err" \
  --dependency="afterok:${ARRAY_JOB}" \
  --export="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_SCALEOUT_ROOT=${SCALE_ROOT},NEON_SCALEOUT_RUNG=${RUNG},NEON_PROTOCOL_SHA256=${PROTOCOL_SHA},NEON_GOVERNING_GATE_SHA256=${GATE_SHA}" \
  scripts/sbatch_neon_phase5_scaleout_analysis.sh)

OOD_ROOT=${ROOT}/phase_s_ood_${RUNG,,}
env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" NEON_BUNDLE="${BUNDLE}" \
  bash scripts/submit_neon_phase5_ood_evidence.sh \
  "${PROTOCOL_ROOT}" "${STAGE2_DIR}" "${ID_METRICS}" "${OOD_ROOT}" >/dev/null
OOD_ARRAY=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["array_job_id"])' "${OOD_ROOT}/SUBMITTED.json")
OOD_FINAL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["finalize_job_id"])' "${OOD_ROOT}/SUBMITTED.json")

DE_ROOT=${ROOT}/phase_s_de_${RUNG,,}
env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" NEON_CONFIG="${CONFIG}" \
  NEON_BUNDLE="${BUNDLE}" bash scripts/submit_neon_phase5_de_evidence.sh \
  "${PROTOCOL_ROOT}" "${STAGE2_DIR}" "${DE_ROOT}" >/dev/null
DE_JOB=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "${DE_ROOT}/SUBMITTED.json")

apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' \
  "${ROOT}" "${PROTOCOL_ROOT}" "${HEAD}" "${GATE}" "${TARGET}" "${RUNG}" "${PRIOR_SEED}" \
  "${DIRICHLET_SEED}" "${SCALE_ROOT}" "${ARRAY_JOB}" "${ANALYSIS}" \
  "${OOD_ROOT}" "${OOD_ARRAY}" "${OOD_FINAL}" "${DE_ROOT}" "${DE_JOB}" \
  "${GATE_SHA}"
import hashlib, json, pathlib, sys
(root, protocol_root, head, gate, target_path, rung, prior_seed, dirichlet_seed, scale_root,
 array, analysis, ood_root, ood_array, ood_final, de_root, de_job,
 governing_gate_sha) = sys.argv[1:]
target = json.loads(pathlib.Path(target_path).read_text())
payload = {
    "schema_version": "neon_phase5_scaleout_submission_v1",
    "git_head": head,
    "protocol_root": protocol_root,
    "protocol_sha256": hashlib.sha256(
        (pathlib.Path(protocol_root) / "PROTOCOL.json").read_bytes()
    ).hexdigest(),
    "governing_gate": gate,
    "governing_gate_sha256": governing_gate_sha,
    "ladder_rung": rung,
    "prior_seed": int(prior_seed),
    "dirichlet_particle_seed": (
        None if not dirichlet_seed else int(dirichlet_seed)
    ),
    "selected_evidence_target": target,
    "scaleout_root": scale_root,
    "array_job": array,
    "analysis_job": analysis,
    "ood_root": ood_root,
    "ood_array_job": ood_array,
    "ood_finalize_job": ood_final,
    "deep_ensemble_root": de_root,
    "deep_ensemble_job": de_job,
    "required_terminal_jobs": [analysis, ood_final, de_job],
}
path = pathlib.Path(root) / "PHASE_S_SUBMITTED.json"
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
path.write_bytes(encoded)
digest = hashlib.sha256(encoded).hexdigest()
path.with_suffix(".json.sha256").write_text(f"{digest}  {path.name}\n")
PY
FINAL_AUDIT=$(sbatch --parsable --job-name=neon_p5_done \
  --output="${ROOT}/phase_s_finalize_%j.out" --error="${ROOT}/phase_s_finalize_%j.err" \
  --dependency="afterok:${ANALYSIS}:${OOD_FINAL}:${DE_JOB}" \
  --export="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_PHASE5_ROOT=${ROOT}" \
  scripts/sbatch_neon_phase_s_finalize.sh)
python3 - <<'PY' "${ROOT}/PHASE_S_SUBMITTED.json" "${FINAL_AUDIT}"
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["final_audit_job"] = sys.argv[2]
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
path.write_bytes(encoded)
digest = hashlib.sha256(encoded).hexdigest()
path.with_suffix(".json.sha256").write_text(f"{digest}  {path.name}\n")
PY
printf 'Phase S submitted for %s: N-sweep=%s analysis=%s OOD=%s DE=%s audit=%s\n' \
  "${RUNG}" "${ARRAY_JOB}" "${ANALYSIS}" "${OOD_FINAL}" "${DE_JOB}" "${FINAL_AUDIT}"
