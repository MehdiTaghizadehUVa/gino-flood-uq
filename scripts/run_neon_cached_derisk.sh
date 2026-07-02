#!/bin/bash
set -euo pipefail
REPO=/home/jrj6wm/GINO_Model/neuraloperator_clean_mcdropout
CONTAINER=/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif
cd "${REPO}"
module purge; module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"; slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
apptainer exec --nv ${APPTAINER_BIND_ARGS} "${CONTAINER}" python scripts/neon_stage2_cached_derisk.py 2>&1
