#!/usr/bin/env bash
# Curriculum arm: official Easy -> Medium -> Hard, 1000 steps each, resume LoRA.
# Independent of the full-train baseline (fresh LoRA on Easy).
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
RUN="${RUN_DIR:-${ROOT}/results/venus_solve/official2/curriculum}"
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

log "Building official Easy/Medium/Hard train splits (dataset labels, not probe)"
python scripts/build_official_curriculum.py

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

rm -rf "${RUN}/official"
CURR=""
for stage in easy medium hard; do
  log "CURRICULUM stage=${stage} steps=${STEPS} gpus=${N_GPUS}"
  if [[ -n "${CURR}" ]]; then
    torchrun grpo/solve_grpo_train.py \
      --train-file "${DATA}/official/${stage}/train.parquet" \
      --save-dir "${RUN}/official/${stage}" \
      --max-steps "${STEPS}" \
      --prompts-per-step 1 \
      --rollouts 4 \
      --save-every 200 \
      --resume "${CURR}"
  else
    torchrun grpo/solve_grpo_train.py \
      --train-file "${DATA}/official/${stage}/train.parquet" \
      --save-dir "${RUN}/official/${stage}" \
      --max-steps "${STEPS}" \
      --prompts-per-step 1 \
      --rollouts 4 \
      --save-every 200
  fi
  CURR="${RUN}/official/${stage}/final/adapter"
done

log "Eval curriculum (hard/final)"
eval_one "${RUN}/official/hard/final" "${RUN}/official/eval"

python - <<PY
import json
from pathlib import Path
run = Path("${RUN}")
curr = json.loads((run / "official/eval/metrics.json").read_text())
report = {
    "arm": "official_curriculum",
    "test_n": 300,
    "steps_per_stage": int("${STEPS}"),
    "gpus": int("${N_GPUS}"),
    "curriculum_label": "official Easy/Medium/Hard, not probe",
    "official_curriculum_pass_rate": curr["pass_rate"],
    "official_curriculum": curr,
}
(run / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
log "curriculum done"
