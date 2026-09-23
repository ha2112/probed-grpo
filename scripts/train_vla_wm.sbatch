#!/bin/bash
#SBATCH --job-name=vla_wm
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/cuongvd6/sc/probed-grpo/results/logs/vla_wm-%j.out
#SBATCH --error=/home/cuongvd6/sc/probed-grpo/results/logs/vla_wm-%j.err

set -euo pipefail
cd /home/cuongvd6/sc/probed-grpo
mkdir -p results/logs results/vla_wm model-cache/vla_wm

source /home/cuongvd6/miniconda3/etc/profile.d/conda.sh
conda activate probed-probe
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

echo "job=${SLURM_JOB_ID} name=vla_wm host=$(hostname) start=$(date --iso-8601=seconds)"
echo "python=$(which python)"
python -c "import peft, torch, matplotlib; print('peft', peft.__version__, 'torch', torch.__version__, 'gpus', torch.cuda.device_count(), 'mpl', matplotlib.__version__)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -m unittest -v test_group_probe_top.py

# 3 groups: easy / medium / hard (train tertiles)
# 500 epochs, checkpoint every 100 epochs, live plots in results/vla_wm
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  group_probe_top.py \
  --num-groups 3 \
  --epochs 500 \
  --patience 0 \
  --save-every 100 \
  --batch-size 16 \
  --eval-batch-size 8 \
  --grad-accum 2 \
  --lr 3e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --last-n-layers 16 \
  --output-dir model-cache/vla_wm \
  --plot-dir results/vla_wm

echo "done=$(date --iso-8601=seconds)"
