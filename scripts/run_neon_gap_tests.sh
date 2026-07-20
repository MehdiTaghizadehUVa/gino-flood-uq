#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${NEON_REPO:-$(git -C "${SCRIPT_DIR}/.." rev-parse --show-toplevel)}
CONTAINER=${NEON_CONTAINER:-/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif}
cd "${REPO}"
module purge; module load apptainer
source "${REPO}/scripts/slurm/lib/common.sh"; slurm_configure_host_ca
export APPTAINERENV_PYTHONPATH="${REPO}"
if [[ "$#" -eq 0 ]]; then
    set -- tests/test_neon_*.py
fi
apptainer exec ${APPTAINER_BIND_ARGS} "${CONTAINER}" python -m pytest -q --no-header --tb=short "$@" 2>&1
