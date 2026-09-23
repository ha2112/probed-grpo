#!/bin/bash
#SBATCH --job-name=vla
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --nodelist=worker-0
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=5
#SBATCH --cpus-per-task=40
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/cuongvd6/sc/probed-grpo/results/logs/vla-official8-%j.out
#SBATCH --error=/home/cuongvd6/sc/probed-grpo/results/logs/vla-official8-%j.err

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
export N_GPUS=5
export STEPS=1000

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi -L
python -c "import torch; n=torch.cuda.device_count(); print('gpus', n); assert n >= 5, n"
python -m unittest -v grpo.tests.test_venus_solve
chmod +x scripts/run_venus_official8.sh
bash scripts/run_venus_official8.sh
echo "done=$(date --iso-8601=seconds)"
