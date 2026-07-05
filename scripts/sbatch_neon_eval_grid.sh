#!/bin/bash
#SBATCH --job-name=neon_evalgrid
#SBATCH --account=uqgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/slurm_evalgrid_%A_%a.out
set -euo pipefail
N_VALUES=(25 50 100 250 400)
N=${N_VALUES[$SLURM_ARRAY_TASK_ID]}
REPO=/home/jrj6wm/GINO_Model/neuraloperator_clean_mcdropout
CONTAINER=/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif
BASE=/scratch/jrj6wm/GINO_Model/neon_stage2_full_train
cd "${REPO}"
module purge; module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"; slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" python -m neuralop.flood.cli.eval_neon_stage2   --config ${BASE}/config/coast_fgn_neon_tr450.yaml   --stage2-checkpoint ${BASE}/tr_n${N}/neon_stage2_best.pt   --stage1-bundle /scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json   --output-dir ${BASE}/eval_n${N}   --families val --m-eval 32 --k-eval 50 --seed 0   --cache-dir ${BASE}/feature_cache_tr_k50_eval   --variance-maps 0
