#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 36:00:00
#SBATCH -J eval_seed_q_m40
#SBATCH -o logs/out/eval_seed_q_m40-%A.out
#SBATCH -e logs/err/eval_seed_q_m40-%A.err
# Submit from scripts/: cd scripts && sbatch eval_seed_quality_testM40.sh

set -euo pipefail

module purge
module load apptainer

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PROJECT_DIR="/scratch/$USER/Data_Generation_UQ/GINO_Model/neuraloperator_no_physics"
TRAIN_CONFIG="${TRAIN_PROJECT_DIR}/config/gino_pluvial_flood_config_WV_depth_only.yaml"
CONTAINER_PATH="/share/resources/containers/apptainer/archive/pytorch-2.0.1.sif"
CHECKPOINT_ROOT="/scratch/$USER/Data_Generation_UQ/GINO_Model/neuraloperator_no_physics/scripts/checkpoints_WV_depth_only_300ep"
ROLLOUT_ROOT="/scratch/$USER/Data_Generation_UQ/Results_Test/M40"
TRAIN_NORMALIZER="/scratch/$USER/Data_Generation_UQ/Results/M40/normalizers_depth_only.pt"
ENSEMBLE_PER_SEED=25

RUN_TAG="eval_seed_quality_testM40_$(date +%Y%m%d_%H%M%S)"
BASE_OUT_DIR="${SCRIPT_DIR}/eval_outputs/${RUN_TAG}"
mkdir -p "${BASE_OUT_DIR}" logs/out logs/err

TEST_TXT="${ROLLOUT_ROOT}/test.txt"
find "${ROLLOUT_ROOT}" -maxdepth 1 -type f -name "*.hdf" -printf "%f\n" | sed 's/\.hdf$//' | sort > "${TEST_TXT}"
if [[ ! -s "${TEST_TXT}" ]]; then
  echo "ERROR: ${TEST_TXT} is empty."
  exit 1
fi
TRAIN_STUB_TXT="${ROLLOUT_ROOT}/train_eval_stub.txt"
head -n 1 "${TEST_TXT}" > "${TRAIN_STUB_TXT}"

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

if [[ ! -f "${TRAIN_CONFIG}" ]]; then
  echo "ERROR: Training config not found: ${TRAIN_CONFIG}"
  exit 1
fi

echo "Seed-quality evaluation"
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Rollout root:    ${ROLLOUT_ROOT}"
echo "Training config: ${TRAIN_CONFIG}"
echo "Ensemble/seed:   ${ENSEMBLE_PER_SEED}"
echo "Output root:     ${BASE_OUT_DIR}"

seed_count=0
for ckpt_dir in "${CHECKPOINT_ROOT}"/*; do
  [[ -d "${ckpt_dir}" ]] || continue
  if [[ ! -f "${ckpt_dir}/best_model_state_dict.pt" && ! -f "${ckpt_dir}/model_state_dict.pt" ]]; then
    continue
  fi
  seed_name="$(basename "${ckpt_dir}")"
  out_dir="${BASE_OUT_DIR}/${seed_name}"
  mkdir -p "${out_dir}"
  seed_count=$((seed_count + 1))
  echo
  echo "=== Running seed ${seed_name} (${seed_count}) ==="
  apptainer run ${APPTAINER_BIND_ARGS} "${CONTAINER_PATH}" "${PROJECT_DIR}/scripts/evaluate_post_training_flood_WV.py" \
    --config_path "${TRAIN_CONFIG}" \
    --checkpoint.save_dir "${ckpt_dir}" \
    --data.root "${ROLLOUT_ROOT}" \
    --data.train_txt "train_eval_stub.txt" \
    --data.write_train_txt false \
    --data.normalizer_path "${TRAIN_NORMALIZER}" \
    --rollout_data.root "${ROLLOUT_ROOT}" \
    --rollout_data.test_txt "test.txt" \
    --rollout.out_dir "${out_dir}" \
    --rollout.n_ensemble_samples "${ENSEMBLE_PER_SEED}" \
    --wandb.log false \
    --run_rollout \
    --skip_single_step
done

if [[ "${seed_count}" -lt 1 ]]; then
  echo "ERROR: no checkpoint seed runs found under ${CHECKPOINT_ROOT}"
  exit 1
fi

python3 - <<PY
import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

base = Path("${BASE_OUT_DIR}")
rows = []
for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    j = d / "uq_overall_metrics.json"
    if not j.exists():
        continue
    with open(j, "r", encoding="utf-8") as f:
        m = json.load(f)
    rows.append(
        {
            "seed": d.name,
            "rmse_wd_overall_mean": float(m.get("rmse_wd_overall_mean", np.nan)),
            "crps_wd_overall_mean": float(m.get("crps_wd_overall_mean", np.nan)),
            "brier_wd_exceed_overall_mean": float(m.get("brier_wd_exceed_overall_mean", np.nan)),
            "wd_exceed_reliability_ece": float(m.get("wd_exceed_reliability_ece", np.nan)),
            "pit_l1_distance": float(m.get("pit_l1_distance", np.nan)),
            "rank_hist_l1_distance": float(m.get("rank_hist_l1_distance", np.nan)),
            "spread_skill_corr": float(m.get("spread_skill_corr", np.nan)),
        }
    )

if not rows:
    raise SystemExit("No per-seed uq_overall_metrics.json found to summarize.")

csv_path = base / "seed_quality_summary.csv"
fieldnames = list(rows[0].keys())
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

if plt is not None:
    seed_names = [r["seed"] for r in rows]
    metrics = [
        "rmse_wd_overall_mean",
        "crps_wd_overall_mean",
        "brier_wd_exceed_overall_mean",
        "wd_exceed_reliability_ece",
        "pit_l1_distance",
    ]
    fig, axs = plt.subplots(
        1,
        len(metrics),
        figsize=(4.2 * len(metrics), 4.8),
        dpi=260,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axs = [axs]
    for ax, key in zip(axs, metrics):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        x = np.arange(len(seed_names))
        ax.bar(x, vals, color="#4c78a8", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(seed_names, rotation=35, ha="right", fontsize=7)
        ax.set_title(key)
        ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(base / "seed_quality_overall_metrics.png", bbox_inches="tight")
    plt.close(fig)
else:
    print("matplotlib unavailable on host python; skipped seed_quality_overall_metrics.png")

best = min(rows, key=lambda r: r["crps_wd_overall_mean"])
with open(base / "seed_quality_best_by_crps.txt", "w", encoding="utf-8") as f:
    f.write(f"{best['seed']}\\n")
print(f"Wrote seed quality summary to {csv_path}")
print(f"Best seed by CRPS: {best['seed']}")
PY

echo
echo "Seed-quality evaluation finished. Output root: ${BASE_OUT_DIR}"
