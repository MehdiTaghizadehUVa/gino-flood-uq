#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -J gino_gauss_e4
#SBATCH --array=0-3
#SBATCH -o logs/out/gino_gauss_e4-%A_%a.out
#SBATCH -e logs/err/gino_gauss_e4-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/slurm/train/pytorch_gpu_job_train_gaussian_wv_ensemble4.sh" "$@"
