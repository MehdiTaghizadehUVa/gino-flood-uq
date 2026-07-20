#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /scratch/.../neon_phase5_root" >&2
  exit 2
fi
ROOT=$1
REPO=${NEON_REPO:-/home/jrj6wm/GINO_Model/neuraloperator_neon_phase5}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
CONFIG=${NEON_CONFIG:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/config/coast_fgn_neon_tr450.yaml}
BUNDLE=${NEON_BUNDLE:-/scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json}
B3_DIR=${NEON_B3_DIR:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4_complete_resume_20260714/b3_n450_zero_init_6631946}
CACHE_DIR=${NEON_CACHE_DIR:-/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/repair_v4/feature_cache_v3}
COMPLETE=${ROOT}/PHASE_S_COMPLETE.json
SUBMISSION=${ROOT}/ALPHA_SWEEP_SUBMITTED.json

cd "${REPO}"
HEAD=$(git rev-parse HEAD)
test -z "$(git status --porcelain)" || { echo "Refusing dirty-tree submission" >&2; exit 2; }
test -s "${COMPLETE}" && test -s "${COMPLETE}.sha256"
test -s "${ROOT}/PROTOCOL.json" && test -s "${ROOT}/PROTOCOL.json.sha256"
test -s "${ROOT}/d2_geometry/GEOMETRY.pt"
test ! -e "${SUBMISSION}" || { echo "Alpha sweep already submitted" >&2; exit 2; }
module purge
module load apptainer
export APPTAINERENV_PYTHONPATH="${REPO}:${REPO}/scripts"

apptainer exec --bind /scratch,/home "${CONTAINER}" python - \
  "${COMPLETE}" "${ROOT}/PROTOCOL.json" "${HEAD}" <<'PY'
import hashlib, json, pathlib, sys
from neuralop.flood.eval.neon_phase5 import verify_checksummed_artifact
complete_path = pathlib.Path(sys.argv[1])
protocol_path = pathlib.Path(sys.argv[2])
head = sys.argv[3]
verify_checksummed_artifact(complete_path)
verify_checksummed_artifact(protocol_path)
complete = json.loads(complete_path.read_text())
protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
if complete.get("git_head") != head:
    raise ValueError("Phase-S completion Git HEAD differs from alpha-sweep submission")
if (complete.get("source_artifacts") or {}).get("protocol_sha256") != protocol_sha:
    raise ValueError("Phase-S completion protocol differs from alpha-sweep submission")
PY

read -r RUNG PRIOR_SEED SUPPORT_SEED DE_STD < <(
  apptainer exec --bind /scratch,/home "${CONTAINER}" python - <<'PY' "${COMPLETE}"
import json, math, pathlib, sys
from neuralop.flood.eval.neon_phase5 import verify_checksummed_artifact
path = pathlib.Path(sys.argv[1]); verify_checksummed_artifact(path)
p = json.loads(path.read_text())
gate = p["amplitude_gate"]
assert gate["decision"] == "alpha_sweep" and gate["alpha_sweep_eligible"] is True
target = p["selected_evidence_target"]
variance = float(p["deep_ensemble"]["aggregate"]["deep_epistemic_variance_mean_m2"])
assert variance > 0.0
print(target["ladder_rung"], target["prior_seed"],
      "NONE" if target["dirichlet_particle_seed"] is None else target["dirichlet_particle_seed"],
      math.sqrt(variance))
PY
)
if [[ "${SUPPORT_SEED}" == NONE ]]; then SUPPORT_SEED=; fi
MULTIPLIERS=(0.5 1.0 2.0)
RUN_DIRS=()
TARGETS=()
for MULTIPLIER in "${MULTIPLIERS[@]}"; do
  TARGET=$(python3 -c 'import sys; print(format(float(sys.argv[1])*float(sys.argv[2]), ".17g"))' "${DE_STD}" "${MULTIPLIER}")
  TAG=${MULTIPLIER//./p}
  DIR=${ROOT}/phase_s_alpha_${RUNG,,}_x${TAG}_seed${PRIOR_SEED}
  RUN_DIRS+=("${DIR}")
  TARGETS+=("${TARGET}")
  ENV=(
    NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}"
    NEON_RUN_ROOT="${ROOT}" NEON_OUT_DIR="${DIR}" NEON_CACHE_DIR="${CACHE_DIR}"
    NEON_PRIOR_SEED="${PRIOR_SEED}" NEON_TRAIN_SEED=0
    NEON_PRIOR_TARGET_STD_M="${TARGET}"
  )
  if [[ -n "${SUPPORT_SEED}" ]]; then
    ENV+=(NEON_DIRICHLET_PARTICLE_SEED="${SUPPORT_SEED}")
  fi
  env "${ENV[@]}" NEON_SUBMIT_DRY_RUN=1 \
    bash scripts/submit_neon_repair_rung.sh "${RUNG}" >/dev/null
  env NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}" \
    NEON_EXPECTED_HEAD="${HEAD}" NEON_PHASE5_ROOT="${ROOT}" \
    NEON_CONFIG="${CONFIG}" NEON_BUNDLE="${BUNDLE}" NEON_B3_DIR="${B3_DIR}" \
    NEON_CACHE_DIR="${CACHE_DIR}" NEON_PILOT_DIR="${DIR}" \
    NEON_PHASE5_TASK=pilot_eval NEON_PLAN_ONLY=1 \
    bash scripts/sbatch_neon_phase5_task.sh >/dev/null
done

if [[ "${NEON_SUBMIT_DRY_RUN:-0}" == 1 ]]; then
  printf 'Validated Phase-S alpha sweep: rung=%s targets=%s\n' "${RUNG}" "${TARGETS[*]}"
  exit 0
fi

TRAIN_JOBS=()
EVAL_JOBS=()
for INDEX in "${!MULTIPLIERS[@]}"; do
  DIR=${RUN_DIRS[$INDEX]}
  TARGET=${TARGETS[$INDEX]}
  ENV=(
    NEON_REPO="${REPO}" NEON_CONTAINER="${CONTAINER}"
    NEON_RUN_ROOT="${ROOT}" NEON_OUT_DIR="${DIR}" NEON_CACHE_DIR="${CACHE_DIR}"
    NEON_PRIOR_SEED="${PRIOR_SEED}" NEON_TRAIN_SEED=0
    NEON_PRIOR_TARGET_STD_M="${TARGET}"
  )
  if [[ -n "${SUPPORT_SEED}" ]]; then
    ENV+=(NEON_DIRICHLET_PARTICLE_SEED="${SUPPORT_SEED}")
  fi
  env "${ENV[@]}" bash scripts/submit_neon_repair_rung.sh "${RUNG}" >/dev/null
  TRAIN=$(<"${DIR}/job_id.txt")
  EVAL=$(sbatch --parsable --job-name="neon_alpha_eval" \
    --output="${DIR}/phase5_eval_%j.out" --error="${DIR}/phase5_eval_%j.err" \
    --dependency="afterok:${TRAIN}" \
    --export="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_PHASE5_ROOT=${ROOT},NEON_CONFIG=${CONFIG},NEON_BUNDLE=${BUNDLE},NEON_B3_DIR=${B3_DIR},NEON_CACHE_DIR=${CACHE_DIR},NEON_PILOT_DIR=${DIR},NEON_PHASE5_TASK=pilot_eval" \
    scripts/sbatch_neon_phase5_task.sh)
  TRAIN_JOBS+=("${TRAIN}")
  EVAL_JOBS+=("${EVAL}")
done

python3 - <<'PY' "${SUBMISSION}" "${HEAD}" "${RUNG}" "${PRIOR_SEED}" \
  "${SUPPORT_SEED}" "${DE_STD}" "${RUN_DIRS[*]}" "${TARGETS[*]}" \
  "${TRAIN_JOBS[*]}" "${EVAL_JOBS[*]}"
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "neon_phase_s_alpha_submission_v1",
    "git_head": sys.argv[2], "ladder_rung": sys.argv[3],
    "prior_seed": int(sys.argv[4]),
    "dirichlet_particle_seed": None if not sys.argv[5] else int(sys.argv[5]),
    "deep_ensemble_target_std_m": float(sys.argv[6]),
    "run_dirs": sys.argv[7].split(), "target_std_m": [float(x) for x in sys.argv[8].split()],
    "train_jobs": sys.argv[9].split(), "eval_jobs": sys.argv[10].split(),
}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
path.write_bytes(encoded)
path.with_suffix(".json.sha256").write_text(
    f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n"
)
PY
DEPENDENCY=$(IFS=:; echo "${EVAL_JOBS[*]}")
FINAL=$(sbatch --parsable --job-name=neon_alpha_done \
  --output="${ROOT}/alpha_finalize_%j.out" --error="${ROOT}/alpha_finalize_%j.err" \
  --dependency="afterok:${DEPENDENCY}" \
  --export="ALL,NEON_REPO=${REPO},NEON_CONTAINER=${CONTAINER},NEON_EXPECTED_HEAD=${HEAD},NEON_PHASE5_ROOT=${ROOT}" \
  scripts/sbatch_neon_phase5_alpha_finalize.sh)
python3 - <<'PY' "${SUBMISSION}" "${FINAL}"
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1]); payload = json.loads(path.read_text())
payload["finalize_job"] = sys.argv[2]
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
path.write_bytes(encoded)
path.with_suffix(".json.sha256").write_text(
    f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n"
)
PY
printf 'Submitted conditional alpha sweep: train=%s eval=%s finalize=%s\n' \
  "${TRAIN_JOBS[*]}" "${EVAL_JOBS[*]}" "${FINAL}"
