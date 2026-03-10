#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J fgn_wv_ev
#SBATCH -o runtime/logs/out/fgn_wv_ev-%j.out
#SBATCH -e runtime/logs/err/fgn_wv_ev-%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/slurm/eval/pytorch_gpu_job_eval_fgn_wv.sh" "$@"
