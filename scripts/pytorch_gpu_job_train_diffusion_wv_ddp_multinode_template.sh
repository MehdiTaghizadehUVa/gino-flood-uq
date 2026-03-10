#!/bin/bash
#SBATCH -A uqgroup
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 72:00:00
#SBATCH -J ddofs_wv_e4_ddpMN
#SBATCH --array=0-3
#SBATCH -o runtime/logs/out/ddofs_wv_e4_ddpMN-%A_%a.out
#SBATCH -e runtime/logs/err/ddofs_wv_e4_ddpMN-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/slurm/train/pytorch_gpu_job_train_diffusion_wv_ddp_multinode_template.sh" "$@"
