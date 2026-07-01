#!/bin/bash
set -euo pipefail
REPO=/home/jrj6wm/GINO_Model/neuraloperator_clean_mcdropout
CONTAINER=/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif
cd "${REPO}"
module purge; module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"; slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
export APPTAINERENV_MPLBACKEND=Agg
apptainer exec ${APPTAINER_BIND_ARGS} --nv "${CONTAINER}" python "${REPO}/scripts/neon_stage2_smoke.py" 2>&1
