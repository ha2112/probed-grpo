#!/usr/bin/env bash
# Baseline arm: eval base model, then GRPO on full Venus train (984), then eval.
# 2-GPU DDP: each rank scores its own rollouts; KV-cache on during generate.
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

N_GPUS="${N_GPUS:-2}"
STEPS="${STEPS:-1000}"
DATA="${ROOT}/grpo/data/venus_solve"
RUN="${RUN_DIR:-${ROOT}/results/venus_solve/official2/baseline}"
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
if "use_cache=True" not in src or "generate_rollouts" not in src:
    raise SystemExit("generate path must enable KV cache (generate_rollouts)")
print("grpo_reward_path=per_rank generate=kv_cache")
PY

torchrun() {
  python -m torch.distributed.run --standalone --nproc_per_node="${N_GPUS}" "$@"
}

eval_one() {
  local adapter="$1"
  local output="$2"
  local args=(--eval-file "${TEST}" --output "${output}" --max-new-tokens 1536)
  if [[ -n "${adapter}" ]]; then
    args+=(--adapter "${adapter}")
  fi
  rm -rf "${output}"
  mkdir -p "${output}"
  log "Eval sharded across ${N_GPUS} GPUs -> ${output}"
  torchrun grpo/solve_grpo_eval.py "${args[@]}"
}

log "BASELINE 1/2 Eval BASE model (no LoRA) on official test n=300"
eval_one "" "${RUN}/base_model/eval"

log "BASELINE 2/2 Train FULL train (984) for ${STEPS} steps on ${N_GPUS} GPUs"
rm -rf "${RUN}/full"
torchrun grpo/solve_grpo_train.py \
  --train-file "${DATA}/base/train.parquet" \
  --save-dir "${RUN}/full" \
  --max-steps "${STEPS}" \
  --prompts-per-step 1 \
  --rollouts 4 \
  --save-every 200

log "Eval FULL"
eval_one "${RUN}/full/final" "${RUN}/full/eval"

python - <<PY
import json
from pathlib import Path
run = Path("${RUN}")
base = json.loads((run / "base_model/eval/metrics.json").read_text())
full = json.loads((run / "full/eval/metrics.json").read_text())
report = {
    "arm": "baseline_full",
    "test_n": 300,
    "steps": int("${STEPS}"),
    "gpus": int("${N_GPUS}"),
    "base_model_pass_rate": base["pass_rate"],
    "full_grpo_pass_rate": full["pass_rate"],
    "base_model": base,
    "full_grpo": full,
}
(run / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
log "baseline done"
