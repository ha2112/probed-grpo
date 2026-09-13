#!/bin/bash
#SBATCH --job-name=group-lora
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cuongvd6/sc/probed-grpo/results/logs/group-lora-%j.out
#SBATCH --error=/home/cuongvd6/sc/probed-grpo/results/logs/group-lora-%j.err

set -euo pipefail
cd /home/cuongvd6/sc/probed-grpo
mkdir -p results/logs model-cache/group-probe-lora

source /home/cuongvd6/miniconda3/etc/profile.d/conda.sh
conda activate probed-probe
export PATH="/home/cuongvd6/miniconda3/envs/probed-probe/bin:$PATH"

echo "host=$(hostname) gpus=${SLURM_GPUS:-$CUDA_VISIBLE_DEVICES} job=$SLURM_JOB_ID"
echo "python=$(which python)"
echo "torchrun=$(which torchrun || true)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import peft,torch; print('peft', peft.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.device_count())"
python -m unittest -v test_group_probe_lora.py

# 2 GPU DDP. --batch-size is PER GPU.
# batch 8 x 2 GPUs = effective 16 (safe on H100 for long prompts).
# For effective 32, set --batch-size 16 (may OOM on very long statements).
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run --standalone --nproc_per_node=2 group_probe_lora.py \
  --epochs 12 \
  --patience 4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --last-n-layers 12 \
  --lr 8e-5 \
  --batch-size 8 \
  --eval-batch-size 4 \
  --grad-accum 1 \
  --log-every 20 \
  --output-dir model-cache/group-probe-lora

echo "DONE"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
