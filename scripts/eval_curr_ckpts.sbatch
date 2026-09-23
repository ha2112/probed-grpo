#!/bin/bash
#SBATCH --job-name=vla-ev
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --nodelist=worker-0
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0-06:00:00
#SBATCH --output=/home/cuongvd6/sc/probed-grpo/results/logs/vla-ev-%j.out
#SBATCH --error=/home/cuongvd6/sc/probed-grpo/results/logs/vla-ev-%j.err

set -euo pipefail
cd /home/cuongvd6/sc/probed-grpo
mkdir -p results/logs
source /home/cuongvd6/miniconda3/etc/profile.d/conda.sh
conda activate probed-probe
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset CUDA_VISIBLE_DEVICES || true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export JUDGE_WORKERS=4
export VENUS_JUDGE=local
export N_GPUS=2

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds) mode=eval-only-2gpu"
nvidia-smi -L
python -c "import torch; n=torch.cuda.device_count(); print('gpus', n); assert n >= 2, n"
chmod +x scripts/eval_curr_ckpts.sh
bash scripts/eval_curr_ckpts.sh
echo "done=$(date --iso-8601=seconds)"
