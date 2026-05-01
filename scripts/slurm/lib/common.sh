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


    slurm_prepare_geo_venv() {
      local container_path="$1"
      if [[ "${ENABLE_GEO_VENV:-1}" != "1" ]]; then
        echo "Geo visualization venv disabled (ENABLE_GEO_VENV=${ENABLE_GEO_VENV:-0})."
        return 0
      fi

      local geo_venv_root="${GEO_VENV_ROOT:-/scratch/$USER/GINO_Model/venvs/neuralop_geo}"
      local geo_target="${GEO_SITEPACKAGES_ROOT:-${geo_venv_root}/target}"
      local required="${GEO_VENV_REQUIRED:-0}"
      local install_cmd="${GEO_VENV_INSTALL_CMD:-contextily pyproj rasterio xyzservices mercantile geopy geographiclib affine cligj click-plugins}"
      local pip_flags="${GEO_VENV_PIP_FLAGS:---no-cache-dir --upgrade --only-binary=:all: --no-deps}"
      local import_check='import contextily, pyproj, rasterio, xyzservices'

      if apptainer exec ${APPTAINER_BIND_ARGS} "${container_path}" python - <<GEO_IMPORT_CHECK >/dev/null 2>&1
${import_check}
GEO_IMPORT_CHECK
      then
        echo "Geo visualization dependencies are already available in the container."
        return 0
      fi

      if [[ -d "${geo_target}" ]] && \
        apptainer exec ${APPTAINER_BIND_ARGS} "${container_path}" env PYTHONPATH="${geo_target}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}" python - <<GEO_TARGET_IMPORT_CHECK >/dev/null 2>&1
${import_check}
GEO_TARGET_IMPORT_CHECK
      then
        export APPTAINERENV_PYTHONPATH="${geo_target}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
        echo "Geo visualization packages active: ${geo_target}"
        return 0
      fi

      if [[ -x "${geo_venv_root}/bin/python" ]] && \
        apptainer exec ${APPTAINER_BIND_ARGS} "${container_path}" "${geo_venv_root}/bin/python" - <<GEO_VENV_IMPORT_CHECK >/dev/null 2>&1
${import_check}
GEO_VENV_IMPORT_CHECK
      then
        local geo_site
        geo_site="$(apptainer exec ${APPTAINER_BIND_ARGS} "${container_path}" "${geo_venv_root}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
        export APPTAINERENV_PATH="${geo_venv_root}/bin:${APPTAINERENV_PATH:-/usr/local/bin:/usr/bin:/bin}"
        export APPTAINERENV_PYTHONPATH="${geo_site}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
        echo "Geo visualization venv active: ${geo_venv_root}"
        return 0
      fi

      echo "Preparing geo visualization packages at ${geo_target}"
      mkdir -p "${geo_target}"
      if ! apptainer exec ${APPTAINER_BIND_ARGS} "${container_path}" python -m pip install ${pip_flags} --target "${geo_target}" ${install_cmd}; then
        if [[ "${required}" == "1" ]]; then
          echo "ERROR: failed to install required geo visualization packages." >&2
          return 2
        fi
        echo "WARNING: failed to install geo visualization packages; renderer will use configured fallback if possible." >&2
        return 0
      fi

      if ! apptainer exec ${APPTAINER_BIND_ARGS} "${container_path}" env PYTHONPATH="${geo_target}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}" python - <<GEO_TARGET_FINAL_CHECK >/dev/null 2>&1
${import_check}
GEO_TARGET_FINAL_CHECK
      then
        if [[ "${required}" == "1" ]]; then
          echo "ERROR: geo visualization packages installed but import validation failed." >&2
          return 2
        fi
        echo "WARNING: geo visualization packages installed but import validation failed; renderer will use configured fallback if possible." >&2
        return 0
      fi

      export APPTAINERENV_PYTHONPATH="${geo_target}${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
      echo "Geo visualization packages active: ${geo_target}"
    }
