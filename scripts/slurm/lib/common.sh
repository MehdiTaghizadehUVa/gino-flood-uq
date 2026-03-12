#!/usr/bin/env bash

    slurm_load_apptainer() {
      module purge
      module load apptainer
    }

    slurm_resolve_scripts_root() {
      local canonical_dir="$1"
      if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        printf '%s
' "${SLURM_SUBMIT_DIR}"
        return 0
      fi
      case "$(basename "$canonical_dir")" in
        train|eval)
          (cd "$canonical_dir/../.." && pwd)
          ;;
        *)
          (cd "$canonical_dir" && pwd)
          ;;
      esac
    }

    slurm_configure_host_ca() {
      HOST_CA_BUNDLE=""
      for cand in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt; do
        if [[ -f "$cand" ]]; then
          HOST_CA_BUNDLE="$cand"
          break
        fi
      done
      APPTAINER_BIND_ARGS="--nv"
      if [[ -n "${HOST_CA_BUNDLE}" ]]; then
        APPTAINER_BIND_ARGS="--nv --bind ${HOST_CA_BUNDLE}:/host_ca_bundle.crt:ro"
        export APPTAINERENV_SSL_CERT_FILE=/host_ca_bundle.crt
        export APPTAINERENV_REQUESTS_CA_BUNDLE=/host_ca_bundle.crt
      fi
    }

    slurm_assert_container_gpus() {
      local container_path="$1"
      local min_gpus="${2:-1}"
      apptainer exec ${APPTAINER_BIND_ARGS} "${container_path}" python - <<PY
import sys
import torch

available = torch.cuda.is_available()
count = torch.cuda.device_count() if available else 0
required = int(${min_gpus})
if (not available) or count < required:
    raise SystemExit(
        f"Container GPU preflight failed: cuda_available={available} "
        f"device_count={count} required>={required}. "
        "Check the Slurm GPU allocation and apptainer --nv wiring."
    )
print(f"Container GPU preflight OK: cuda_available={available} device_count={count}")
PY
    }
