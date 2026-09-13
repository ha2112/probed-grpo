#!/bin/bash
#SBATCH --job-name=group-top
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cuongvd6/sc/probed-grpo/results/logs/group-top-%j.out
#SBATCH --error=/home/cuongvd6/sc/probed-grpo/results/logs/group-top-%j.err

set -euo pipefail
cd /home/cuongvd6/sc/probed-grpo
mkdir -p results/logs model-cache/group-probe-top

source /home/cuongvd6/miniconda3/etc/profile.d/conda.sh
conda activate probed-probe
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
echo "python=$(which python)"
python -c "import peft, torch; print('peft', peft.__version__, 'torch', torch.__version__, 'gpus', torch.cuda.device_count())"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -m unittest -v test_group_probe_top.py

# Per-GPU batch 8 x 2 GPUs x accumulation 2 = effective batch 32.
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  group_probe_top.py \
  --epochs 10 \
  --patience 3 \
  --batch-size 8 \
  --eval-batch-size 4 \
  --grad-accum 2 \
  --lr 3e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --last-n-layers 16 \
  --output-dir model-cache/group-probe-top

echo "done=$(date --iso-8601=seconds)"
