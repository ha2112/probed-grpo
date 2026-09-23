#!/bin/bash
# Eval base + curriculum easy/medium/hard @ 600 steps each, 1 GPU.
#SBATCH --job-name=vla
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --nodelist=worker-2
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/cuongvd6/sc/probed-grpo/results/logs/vla-curr-%j.out
#SBATCH --error=/home/cuongvd6/sc/probed-grpo/results/logs/vla-curr-%j.err

set -euo pipefail
cd /home/cuongvd6/sc/probed-grpo
source /home/cuongvd6/miniconda3/etc/profile.d/conda.sh
conda activate probed-probe
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset CUDA_VISIBLE_DEVICES || true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
export VENUS_JUDGE="${VENUS_JUDGE:-local}"
export FORCE_PROBE=0
export SKIP_PROBE=1
export N_GPUS=1
export TOTAL_STEPS=600
export STAGE_STEPS=600
export GRPO_MAX_STEPS_OVERRIDE=600

MARK=results/venus_solve/e2e/markers
RUN=results/venus_solve/e2e
mkdir -p "$MARK" results/logs "$RUN/curriculum"

if [[ ! -f "$MARK/train_base.done" ]]; then
  echo "No train_base.done — exit"
  exit 0
fi
if [[ ! -d "$RUN/base/final/adapter" ]]; then
  echo "Missing $RUN/base/final/adapter — exit"
  exit 1
fi

# Skip probe/corpus/base-train; (re)run eval_base + curriculum.
rm -f "$MARK"/eval_base.done "$MARK"/train_easy.done "$MARK"/train_medium.done \
  "$MARK"/train_hard.done "$MARK"/eval_curriculum.done
rm -rf "$RUN/base/eval" "$RUN/curriculum/easy" "$RUN/curriculum/medium" \
  "$RUN/curriculum/hard" "$RUN/curriculum/eval"
touch "$MARK/probe.done" "$MARK/corpus.done" "$MARK/train_base.done"

echo "job=${SLURM_JOB_ID} host=$(hostname) N_GPUS=${N_GPUS} STAGE_STEPS=${STAGE_STEPS}"
nvidia-smi -L
python -c "import torch; print('cuda', torch.cuda.device_count())"
bash scripts/run_venus_solve_e2e.sh
echo "done=$(date --iso-8601=seconds)"
