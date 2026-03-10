#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J ddofs_wv_ev
#SBATCH -o logs/out/ddofs_wv_ev-%j.out
#SBATCH -e logs/err/ddofs_wv_ev-%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/slurm/eval/pytorch_gpu_job_eval_diffusion_wv.sh" "$@"
