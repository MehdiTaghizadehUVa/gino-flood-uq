#!/bin/bash
#SBATCH --job-name=fgn_serving_smoke
#SBATCH -A uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/jrj6wm/fgn_serving_smoke/slurm/fgn_serving_smoke_%j.out
#SBATCH --error=/scratch/jrj6wm/fgn_serving_smoke/slurm/fgn_serving_smoke_%j.err

# Smoke-tests the production FGN serving stack against the deployed bundle.
# Submit with:   sbatch scripts/run_serving_smoke.slurm
# Override the bundle path or output dir via env vars before sbatch:
#   FGN_SMOKE_BUNDLE=/path/to/bundle.json sbatch scripts/run_serving_smoke.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/jrj6wm/GINO_Model/neuraloperator_clean_mcdropout}"
BUNDLE="${FGN_SMOKE_BUNDLE:-/scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json}"
SMOKE_ROOT="${FGN_SMOKE_ROOT:-/scratch/jrj6wm/fgn_serving_smoke}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${SMOKE_ROOT}/${STAMP}"

mkdir -p "${OUT_DIR}"
mkdir -p "${SMOKE_ROOT}/slurm"

echo "[smoke] repo:    ${REPO_DIR}"
echo "[smoke] bundle:  ${BUNDLE}"
echo "[smoke] output:  ${OUT_DIR}"

cd "${REPO_DIR}"

module purge
module load apptainer/1.3.4

CONTAINER="${FGN_SMOKE_CONTAINER:-/share/resources/containers/apptainer/pytorch-2.9.0.sif}"
PYUSERBASE="${FGN_SMOKE_PYTHONUSERBASE:-/scratch/jrj6wm/codex_pyuser_fgn_serving}"

apptainer exec --nv \
  --bind "${REPO_DIR}":"${REPO_DIR}" \
  --bind /scratch/jrj6wm:/scratch/jrj6wm \
  --env PYTHONPATH="${REPO_DIR}" \
  --env PYTHONUSERBASE="${PYUSERBASE}" \
  "${CONTAINER}" \
  python -m neuralop.flood.serving.smoke \
    --bundle-path "${BUNDLE}" \
    --output-dir "${OUT_DIR}" \
    --forecast-steps 6 \
    --device cuda:0 \
    --member-chunk-size 4 \
    --seed 17

echo "[smoke] artifacts in ${OUT_DIR}:"
ls -la "${OUT_DIR}"

# Sanity check: the smoke runner produces these three JSON files plus map PNGs.
for required in raw_summary.json calibrated_summary.json smoke_summary.json; do
  if [[ ! -s "${OUT_DIR}/${required}" ]]; then
    echo "[smoke] FAIL: missing required artifact ${required}"
    exit 1
  fi
done

PNG_COUNT=$(find "${OUT_DIR}" -maxdepth 1 -name '*.png' | wc -l)
if [[ "${PNG_COUNT}" -lt 4 ]]; then
  echo "[smoke] FAIL: expected at least 4 map PNGs, found ${PNG_COUNT}"
  exit 1
fi

echo "[smoke] PASS: 60-member rollout, calibrated summary, and ${PNG_COUNT} map PNGs generated."
