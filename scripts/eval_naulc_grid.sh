#!/usr/bin/env bash
# Held-out Venus test evals for nAULC learning curves.
# Usage:
#   N_GPUS=1 bash scripts/eval_naulc_grid.sh              # all missing
#   N_GPUS=1 bash scripts/eval_naulc_grid.sh random       # only random route
#   N_GPUS=1 bash scripts/eval_naulc_grid.sh curriculum   # only official E→M→H
# Skip dirs that already have metrics.json unless FORCE=1.
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

N_GPUS="${N_GPUS:-1}"
FORCE="${FORCE:-0}"
SCOPE="${1:-all}"
TEST="${ROOT}/grpo/data/venus_solve/shared/test.parquet"
BASE="${ROOT}/results/venus_solve/official2/baseline"
CURR="${ROOT}/results/venus_solve/official2/curriculum/official"
OUT="${ROOT}/results/venus_solve/official2/naulc_evals"
mkdir -p "${OUT}/logs" results/logs

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "${OUT}/logs/orchestrator.log"; }

python -c "import torch; n=torch.cuda.device_count(); print(f'torch.cuda.device_count={n}'); assert n >= ${N_GPUS}, n"

eval_one() {
  local adapter="$1"
  local output="$2"
  local name="$3"
  if [[ "${FORCE}" != "1" && -f "${output}/metrics.json" ]]; then
    log "skip ${name} (exists)"
    return 0
  fi
  if [[ "${adapter}" != "base" ]]; then
    if [[ ! -f "${adapter}/adapter_config.json" && ! -f "${adapter}/adapter/adapter_config.json" ]]; then
      echo "missing adapter under: ${adapter}" >&2
      exit 1
    fi
  fi
  rm -rf "${output}"
  mkdir -p "${output}"
  log "Eval ${name} on ${N_GPUS} GPU(s) -> ${output}"
  local adapter_args=()
  if [[ "${adapter}" != "base" ]]; then
    adapter_args=(--adapter "${adapter}")
  fi
  if [[ "${N_GPUS}" -gt 1 ]]; then
    python -m torch.distributed.run --standalone --nproc_per_node="${N_GPUS}" \
      grpo/solve_grpo_eval.py \
        --eval-file "${TEST}" "${adapter_args[@]}" --output "${output}" --max-new-tokens 1536
  else
    python grpo/solve_grpo_eval.py \
      --eval-file "${TEST}" "${adapter_args[@]}" --output "${output}" \
      --device cuda:0 --max-new-tokens 1536
  fi
}

# Cold start + Random full-GRPO intermediates (global step = train step).
if [[ "${SCOPE}" == "all" || "${SCOPE}" == "random" ]]; then
  eval_one "base" "${OUT}/cold_start/eval" "cold_start"
  for step in 200 400 600 800 1000; do
    printf -v tag "%06d" "${step}"
    eval_one "${BASE}/full/step-${tag}" "${OUT}/random_step${step}/eval" "random/step-${step}"
  done
  eval_one "${BASE}/full/final" "${OUT}/random_final/eval" "random/final"
fi

# Official curriculum points used for Official (→H400) and Probed (→H200) curves.
# global_step = easy(0..1000) + medium(1000..2000) + hard(2000..3000)
if [[ "${SCOPE}" == "all" || "${SCOPE}" == "curriculum" ]]; then
  for step in 200 400 600 800 1000; do
    printf -v tag "%06d" "${step}"
    eval_one "${CURR}/easy/step-${tag}" "${OUT}/easy_step${step}/eval" "easy/step-${step}"
  done
  for step in 200 400 600 800 1000; do
    printf -v tag "%06d" "${step}"
    eval_one "${CURR}/medium/step-${tag}" "${OUT}/medium_step${step}/eval" "medium/step-${step}"
  done
  for step in 200 400 600 800 1000; do
    printf -v tag "%06d" "${step}"
    eval_one "${CURR}/hard/step-${tag}" "${OUT}/hard_step${step}/eval" "hard/step-${step}"
  done
fi

python scripts/recompute_eval_rewards.py --glob 'venus_solve/official2/naulc_evals/**/eval'
python scripts/compute_naulc.py
log "naulc grid done"
