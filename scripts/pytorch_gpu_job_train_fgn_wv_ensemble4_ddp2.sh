#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 72:00:00
#SBATCH -J gino_fgn_e4_ddp2
#SBATCH --array=0-3
#SBATCH -o runtime/logs/out/gino_fgn_e4_ddp2-%A_%a.out
#SBATCH -e runtime/logs/err/gino_fgn_e4_ddp2-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/slurm/train/pytorch_gpu_job_train_fgn_wv_ensemble4_ddp2.sh" "$@"
