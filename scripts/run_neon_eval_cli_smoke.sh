#!/bin/bash
set -euo pipefail
REPO=/home/jrj6wm/GINO_Model/neuraloperator_clean_mcdropout
CONTAINER=/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif
cd "${REPO}"
module purge; module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"; slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" python -m neuralop.flood.cli.eval_neon_stage2   --config /scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_fgn_eval60/coast_fgn3x20_eval_currentviz_20260506_152059/config/coast_fgn3x20_eval_currentviz.yaml   --stage2-checkpoint /scratch/jrj6wm/GINO_Model/neon_stage2_full_train/real_ref_full/neon_stage2_best.pt   --stage1-bundle /scratch/jrj6wm/GINO_Model/model_bundles/coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json   --output-dir /scratch/jrj6wm/GINO_Model/neon_stage2_full_train/eval_cli_smoke   --families all --max-families 2 --rollout-length 12 --m-eval 8 --k-eval 8   --impact-metrics --write-artifacts 2>&1
