#!/usr/bin/env bash
# Official-label Venus GRPO on 8 GPUs. Does not use the probe.
# 1) eval base model on the official test split
# 2) GRPO on all 984 train rows, 1000 steps
# 3) official Easy -> Medium -> Hard, 1000 steps each, resume LoRA
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source /home/cuongvd6/miniconda3/etc/profile.d/conda.sh
conda activate probed-probe
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset CUDA_VISIBLE_DEVICES || true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
export VENUS_JUDGE=local

N_GPUS="${N_GPUS:-8}"
STEPS="${STEPS:-1000}"
DATA="${ROOT}/grpo/data/venus_solve"
RUN="${RUN_DIR:-${ROOT}/results/venus_solve/official8}"
TEST="${DATA}/shared/test.parquet"
mkdir -p "${RUN}/logs" results/logs

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "${RUN}/logs/orchestrator.log"; }

python -c "import torch; n=torch.cuda.device_count(); print(f'torch.cuda.device_count={n}'); assert n >= ${N_GPUS}, n"
python - <<'PY'
from pathlib import Path
src = Path("grpo/solve_grpo_train.py").read_text()
if "dist.broadcast(payload" in src:
    raise SystemExit("GRPO still broadcasts rank0 rewards; refuse to train")
if "Each rank must score" not in src:
    raise SystemExit("missing per-rank reward comment/path")
print("grpo_reward_path=per_rank")
PY

log "Building official Easy/Medium/Hard train splits (dataset labels, not probe)"
python scripts/build_official_curriculum.py

torchrun_train() {
  python -m torch.distributed.run --standalone --nproc_per_node="${N_GPUS}" \
    grpo/solve_grpo_train.py "$@"
}

eval_one() {
  local adapter="$1"
  local output="$2"
  local args=(
    --eval-file "${TEST}"
    --output "${output}"
    --max-new-tokens 1536
  )
  if [[ -n "${adapter}" ]]; then
    args+=(--adapter "${adapter}")
  fi
  rm -rf "${output}"
  mkdir -p "${output}"
  log "Eval sharded across ${N_GPUS} GPUs -> ${output}"
  python -m torch.distributed.run --standalone --nproc_per_node="${N_GPUS}" \
    grpo/solve_grpo_eval.py "${args[@]}"
}

log "1/3 Eval BASE model (no LoRA) on official test n=300"
eval_one "" "${RUN}/base_model/eval"

log "2/3 Train FULL train (984) for ${STEPS} steps on ${N_GPUS} GPUs"
torchrun_train \
  --train-file "${DATA}/base/train.parquet" \
  --save-dir "${RUN}/full" \
  --max-steps "${STEPS}" \
  --prompts-per-step 1 \
  --rollouts 4 \
  --save-every 200
log "Eval FULL"
eval_one "${RUN}/full/final" "${RUN}/full/eval"

CURR=""
for stage in easy medium hard; do
  log "3/3 Curriculum official stage=${stage} steps=${STEPS}"
  if [[ -n "${CURR}" ]]; then
    torchrun_train \
      --train-file "${DATA}/official/${stage}/train.parquet" \
      --save-dir "${RUN}/official/${stage}" \
      --max-steps "${STEPS}" \
      --prompts-per-step 1 \
      --rollouts 4 \
      --save-every 200 \
      --resume "${CURR}"
  else
    torchrun_train \
      --train-file "${DATA}/official/${stage}/train.parquet" \
      --save-dir "${RUN}/official/${stage}" \
      --max-steps "${STEPS}" \
      --prompts-per-step 1 \
      --rollouts 4 \
      --save-every 200
  fi
  CURR="${RUN}/official/${stage}/final/adapter"
done

log "Eval official curriculum (hard/final)"
eval_one "${RUN}/official/hard/final" "${RUN}/official/eval"

python - <<PY
import json
from pathlib import Path
run = Path("${RUN}")
base = json.loads((run / "base_model/eval/metrics.json").read_text())
full = json.loads((run / "full/eval/metrics.json").read_text())
curr = json.loads((run / "official/eval/metrics.json").read_text())
report = {
    "test_n": 300,
    "steps_per_run": int("${STEPS}"),
    "gpus": int("${N_GPUS}"),
    "curriculum_label": "official Easy/Medium/Hard, not probe",
    "base_model_pass_rate": base["pass_rate"],
    "full_grpo_pass_rate": full["pass_rate"],
    "official_curriculum_pass_rate": curr["pass_rate"],
    "base_model": base,
    "full_grpo": full,
    "official_curriculum": curr,
}
(run / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
log "done"
