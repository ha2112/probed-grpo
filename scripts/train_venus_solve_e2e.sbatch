#!/bin/bash
#SBATCH --job-name=vla
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/cuongvd6/sc/probed-grpo/results/logs/vla-%j.out
#SBATCH --error=/home/cuongvd6/sc/probed-grpo/results/logs/vla-%j.err

set -euo pipefail
cd /home/cuongvd6/sc/probed-grpo
mkdir -p results/logs results/venus_solve

source /home/cuongvd6/miniconda3/etc/profile.d/conda.sh
conda activate probed-probe
export PATH="${CONDA_PREFIX}/bin:${PATH}"
# IMPORTANT: do NOT override CUDA_VISIBLE_DEVICES — Slurm binds the 2 allocated GPUs.
unset CUDA_VISIBLE_DEVICES || true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
export VENUS_JUDGE="${VENUS_JUDGE:-local}"
# Train probe from scratch for this e2e run (set 0 to reuse model-cache/vla_wm/best).
export FORCE_PROBE="${FORCE_PROBE:-1}"
export PROBE_EPOCHS="${PROBE_EPOCHS:-500}"
export PROBE_SAVE_EVERY_EPOCHS="${PROBE_SAVE_EVERY_EPOCHS:-100}"
# Optimizer-step budget off by default (0 = use epochs only).
export PROBE_MAX_STEPS="${PROBE_MAX_STEPS:-0}"
export PROBE_SAVE_EVERY_STEPS="${PROBE_SAVE_EVERY_STEPS:-0}"
# GRPO: base and each curriculum stage use 600 steps by default.
export TOTAL_STEPS="${TOTAL_STEPS:-600}"
export STAGE_STEPS="${STAGE_STEPS:-600}"

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L
python -c "import peft, torch; print('peft', peft.__version__, 'torch', torch.__version__, 'gpus', torch.cuda.device_count())"
if ! python -c "import torch; assert torch.cuda.device_count() >= 2, torch.cuda.device_count()"; then
  echo "ERROR: need 2 visible GPUs inside the job" >&2
  nvidia-smi
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -m unittest -v grpo.tests.test_venus_solve
python -m grpo.venus_solve_reward --check

chmod +x scripts/run_venus_solve_e2e.sh
bash scripts/run_venus_solve_e2e.sh

echo "done=$(date --iso-8601=seconds)"
