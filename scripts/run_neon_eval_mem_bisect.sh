#!/bin/bash
set -uo pipefail
REPO=/home/jrj6wm/GINO_Model/neuraloperator_clean_mcdropout
CONTAINER=/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif
BASE=/scratch/jrj6wm/GINO_Model/neon_stage2_full_train
cd "${REPO}"
module purge; module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"; slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
COMMON="--config ${BASE}/config/coast_fgn_neon_tr450.yaml --stage2-checkpoint ${BASE}/tr_n450/neon_stage2_best.pt --stage1-bundle /scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json --families val --m-eval 32 --k-eval 50 --seed 0 --cache-dir ${BASE}/feature_cache_tr_k50_eval --max-families 1"
echo '######## RUN 1: metrics only ########'
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" python -m neuralop.flood.cli.eval_neon_stage2 ${COMMON} --output-dir ${BASE}/mem_bisect/r1 --variance-maps 0
echo "run1 exit=$?"
echo '######## RUN 2: + impact metrics ########'
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" python -m neuralop.flood.cli.eval_neon_stage2 ${COMMON} --output-dir ${BASE}/mem_bisect/r2 --variance-maps 0 --impact-metrics
echo "run2 exit=$?"
echo '######## RUN 3: + artifacts + maps ########'
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" python -m neuralop.flood.cli.eval_neon_stage2 ${COMMON} --output-dir ${BASE}/mem_bisect/r3 --variance-maps 1 --impact-metrics --write-artifacts
echo "run3 exit=$?"
echo 'BISECT DONE'
