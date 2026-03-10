#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -J ddofs_wv_e4
#SBATCH --array=0-3
#SBATCH -o runtime/logs/out/ddofs_wv_e4-%A_%a.out
#SBATCH -e runtime/logs/err/ddofs_wv_e4-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/slurm/train/pytorch_gpu_job_train_diffusion_wv.sh" "$@"
