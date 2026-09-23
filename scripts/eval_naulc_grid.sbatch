#!/bin/bash
#SBATCH --job-name=vla-naulc
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --nodelist=worker-2
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-18:00:00
#SBATCH --output=/home/cuongvd6/sc/probed-grpo/results/logs/vla-naulc-%j.out
#SBATCH --error=/home/cuongvd6/sc/probed-grpo/results/logs/vla-naulc-%j.err

# Prefer attaching to an existing interactive allocation instead:
#   bash scripts/start_naulc_on_worker2.sh
# Or inside the worker-2 srun pty:
#   N_GPUS=1 bash scripts/eval_naulc_grid.sh all
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
export N_GPUS=1

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi -L
python -c "import torch; n=torch.cuda.device_count(); print('gpus', n); assert n >= 1, n"
chmod +x scripts/eval_naulc_grid.sh
SCOPE="${1:-all}"
bash scripts/eval_naulc_grid.sh "${SCOPE}"
echo "done=$(date --iso-8601=seconds)"
