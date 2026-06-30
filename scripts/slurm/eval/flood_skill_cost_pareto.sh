#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p standard
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 24:00:00
#SBATCH -J skill_cost
#SBATCH -o runtime/logs/out/skill_cost-%j.out
#SBATCH -e runtime/logs/err/skill_cost-%j.err

set -euo pipefail

PROJECT_DIR_DEFAULT="/home/$USER/GINO_Model/neuraloperator_clean_mcdropout"
PROJECT_DIR="${PROJECT_DIR:-${PROJECT_DIR_DEFAULT}}"
CONTAINER_PATH="${CONTAINER_PATH:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/GINO_Model/neuraloperator_runs/coastal_uq_model_comparison/outputs/skill_cost_pareto_3x20_50events_${TIMESTAMP}}"
TIMING_JSON="${TIMING_JSON:-${OUT_DIR}/skill_cost_timing_input.json}"
FORWARD_ROOT="${FORWARD_ROOT:-/scratch/$USER/GINO_Model/neuraloperator_runs/coastal_uq_model_comparison/outputs/skill_cost_forward_timing_a100_10events_20260612_180547}"
REPETITIONS="${REPETITIONS:-20}"
WORKERS="${WORKERS:-12}"
SEED="${SEED:-20260612}"
ENSEMBLE_SIZES="${ENSEMBLE_SIZES:-1 2 5 10 20 40 60}"

mkdir -p "${PROJECT_DIR}/runtime/logs/out" "${PROJECT_DIR}/runtime/logs/err" "${OUT_DIR}"
cd "${PROJECT_DIR}"

module purge
module load apptainer

export APPTAINERENV_PYTHONPATH="${PROJECT_DIR}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
export APPTAINERENV_MPLBACKEND=Agg
export APPTAINERENV_WANDB_MODE=disabled

if [[ ! -s "${TIMING_JSON}" ]]; then
  apptainer exec "${CONTAINER_PATH}" python scripts/analysis/build_skill_cost_timing_from_forward_only.py \
    --forward-root "${FORWARD_ROOT}" \
    --out "${TIMING_JSON}" \
    --sizes ${ENSEMBLE_SIZES}
fi

cat > "${OUT_DIR}/job_environment.txt" <<EOF
job_id=${SLURM_JOB_ID:-unknown}
host=$(hostname)
date=$(date --iso-8601=seconds)
project_dir=${PROJECT_DIR}
out_dir=${OUT_DIR}
timing_json=${TIMING_JSON}
forward_root=${FORWARD_ROOT}
repetitions=${REPETITIONS}
workers=${WORKERS}
seed=${SEED}
ensemble_sizes=${ENSEMBLE_SIZES}
EOF

apptainer exec "${CONTAINER_PATH}" python scripts/analysis/skill_cost_pareto.py \
  --out-dir "${OUT_DIR}" \
  --timing-json "${TIMING_JSON}" \
  --ensemble-sizes ${ENSEMBLE_SIZES} \
  --repetitions "${REPETITIONS}" \
  --workers "${WORKERS}" \
  --seed "${SEED}"

echo "Skill-cost Pareto outputs: ${OUT_DIR}"
