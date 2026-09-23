#!/usr/bin/env bash
# End-to-end: Codeforces 3-class probe -> Venus solve GRPO (base vs curriculum).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source /home/cuongvd6/miniconda3/etc/profile.d/conda.sh
conda activate probed-probe
export PATH="${CONDA_PREFIX}/bin:${PATH}"
# Let Slurm bind GPUs. Only set CUDA_VISIBLE_DEVICES if the caller exported it.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Using caller CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  unset CUDA_VISIBLE_DEVICES || true
fi
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
export VENUS_JUDGE="${VENUS_JUDGE:-local}"

RUN_DIR="${RUN_DIR:-${ROOT}/results/venus_solve/e2e}"
PROBE_DIR="${PROBE_DIR:-${ROOT}/model-cache/vla_wm}"
DATA_DIR="${DATA_DIR:-${ROOT}/grpo/data/venus_solve}"
N_GPUS="${N_GPUS:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-600}"
# Each curriculum stage matches base length by default (not TOTAL/3).
STAGE_STEPS="${STAGE_STEPS:-${TOTAL_STEPS}}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-1}"
ROLLOUTS="${ROLLOUTS:-4}"
MAX_TRAIN="${MAX_TRAIN:--1}"
MAX_VALIDATION="${MAX_VALIDATION:--1}"
SMOKE="${SMOKE:-0}"
SKIP_PROBE="${SKIP_PROBE:-0}"
FORCE_PROBE="${FORCE_PROBE:-0}"

if [[ "${SMOKE}" == "1" ]]; then
  TOTAL_STEPS=6
  STAGE_STEPS=2
  ROLLOUTS=2
  MAX_TRAIN=8
  MAX_VALIDATION=8
  RUN_DIR="${RUN_DIR}-smoke"
  DATA_DIR="${DATA_DIR}-smoke"
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/markers" "${RUN_DIR}/base" "${RUN_DIR}/curriculum" "${DATA_DIR}"
MARKER="${RUN_DIR}/markers"

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "${RUN_DIR}/logs/orchestrator.log"; }

mark_done() { touch "${MARKER}/$1.done"; }
is_done() { [[ -f "${MARKER}/$1.done" ]]; }

torchrun() {
  python -c "import torch; n=torch.cuda.device_count(); print(f'torch.cuda.device_count={n}'); assert n >= ${N_GPUS}, n"
  python -m torch.distributed.run --standalone --nproc_per_node="${N_GPUS}" "$@"
}

########################################
# 1) Probe (3-class Codeforces)
########################################
if [[ "${SKIP_PROBE}" == "1" ]]; then
  log "SKIP_PROBE=1 — not training probe"
elif [[ "${FORCE_PROBE}" == "1" ]] || ! is_done probe || [[ ! -f "${PROBE_DIR}/best/heads.pt" ]]; then
  if [[ "${FORCE_PROBE}" == "1" && -d "${PROBE_DIR}/best" ]]; then
    stamp="$(date +%Y%m%d-%H%M%S)"
    log "FORCE_PROBE=1 — moving old probe to ${PROBE_DIR}/best.bak-${stamp}"
    mv "${PROBE_DIR}/best" "${PROBE_DIR}/best.bak-${stamp}"
    rm -f "${MARKER}/probe.done"
  fi
  log "Training group_probe_top (3 groups) -> ${PROBE_DIR}"
  PROBE_ARGS=(
    --num-groups 3
    --epochs "${PROBE_EPOCHS:-500}"
    --patience 0
    --save-every "${PROBE_SAVE_EVERY_EPOCHS:-100}"
    --batch-size 8
    --eval-batch-size 4
    --grad-accum 2
    --lr 3e-5
    --lora-r 32
    --lora-alpha 64
    --last-n-layers 16
    --output-dir "${PROBE_DIR}"
    --plot-dir "${RUN_DIR}/probe_plots"
  )
  if [[ "${PROBE_MAX_STEPS:-0}" != "0" ]]; then
    PROBE_ARGS+=(--max-steps "${PROBE_MAX_STEPS}")
    log "Probe step budget: max_steps=${PROBE_MAX_STEPS} save_every_steps=${PROBE_SAVE_EVERY_STEPS:-0}"
  else
    log "Probe epoch budget: epochs=${PROBE_EPOCHS:-500} save_every=${PROBE_SAVE_EVERY_EPOCHS:-100}"
  fi
  if [[ "${PROBE_SAVE_EVERY_STEPS:-0}" != "0" ]]; then
    PROBE_ARGS+=(--save-every-steps "${PROBE_SAVE_EVERY_STEPS}")
  fi
  torchrun group_probe_top.py "${PROBE_ARGS[@]}"
  [[ -f "${PROBE_DIR}/best/heads.pt" ]] || { log "ERROR: missing ${PROBE_DIR}/best/heads.pt"; exit 1; }
  mark_done probe
  # Corpus depends on probe scores — force rebuild after a fresh probe.
  rm -f "${MARKER}/corpus.done"
else
  log "Reusing existing probe checkpoint at ${PROBE_DIR}/best (set FORCE_PROBE=1 to retrain)"
  mark_done probe
fi

########################################
# 2) Venus solve corpora
########################################
if ! is_done corpus; then
  log "Building Venus solve corpora -> ${DATA_DIR}"
  EXTRA=()
  if [[ "${MAX_TRAIN}" != "-1" ]]; then EXTRA+=(--max-train "${MAX_TRAIN}"); fi
  if [[ "${MAX_VALIDATION}" != "-1" ]]; then EXTRA+=(--max-validation "${MAX_VALIDATION}"); fi
  python grpo/venus_solve_corpus.py \
    --probe-dir "${PROBE_DIR}/best" \
    --output-dir "${DATA_DIR}" \
    --device cuda:0 \
    "${EXTRA[@]}"
  [[ -f "${DATA_DIR}/manifest.json" ]] || exit 1
  mark_done corpus
else
  log "Skip corpus stage"
fi

########################################
# 3) Base GRPO on full train
########################################
if ! is_done train_base; then
  log "Training BASE strategy for ${TOTAL_STEPS} steps"
  torchrun grpo/solve_grpo_train.py \
    --train-file "${DATA_DIR}/base/train.parquet" \
    --save-dir "${RUN_DIR}/base" \
    --max-steps "${TOTAL_STEPS}" \
    --prompts-per-step "${PROMPTS_PER_STEP}" \
    --rollouts "${ROLLOUTS}" \
    --save-every 100
  mark_done train_base
else
  log "Skip base train"
fi

if ! is_done eval_base; then
  log "Evaluating BASE"
  python grpo/solve_grpo_eval.py \
    --eval-file "${DATA_DIR}/shared/test.parquet" \
    --adapter "${RUN_DIR}/base/final" \
    --output "${RUN_DIR}/base/eval" \
    --device cuda:0
  mark_done eval_base
fi

########################################
# 4) Curriculum easy -> medium -> hard
########################################
CURR_INIT=""
for stage in easy medium hard; do
  stage_dir="${RUN_DIR}/curriculum/${stage}"
  marker="train_${stage}"
  train_file="${DATA_DIR}/curriculum/${stage}/train.parquet"
  if [[ ! -f "${train_file}" ]]; then
    log "ERROR: missing ${train_file}"
    exit 1
  fi
  # Skip empty bucket parquet (0 rows) but keep chain resume path.
  row_count="$(python - <<PY
import pandas as pd
print(len(pd.read_parquet("${train_file}")))
PY
)"
  if [[ "${row_count}" == "0" ]]; then
    log "WARNING: bucket ${stage} is empty; skipping train, keeping previous adapter"
    mkdir -p "${stage_dir}/final"
    if [[ -n "${CURR_INIT}" && -d "${CURR_INIT}" ]]; then
      cp -a "${CURR_INIT}/." "${stage_dir}/final/adapter/"
    fi
    mark_done "${marker}"
    CURR_INIT="${stage_dir}/final/adapter"
    continue
  fi
  if ! is_done "${marker}"; then
    log "Training CURRICULUM stage=${stage} steps=${STAGE_STEPS} rows=${row_count}"
    RESUME_ARGS=()
    if [[ -n "${CURR_INIT}" ]]; then
      RESUME_ARGS=(--resume "${CURR_INIT}")
    fi
    torchrun grpo/solve_grpo_train.py \
      --train-file "${train_file}" \
      --save-dir "${stage_dir}" \
      --max-steps "${STAGE_STEPS}" \
      --prompts-per-step "${PROMPTS_PER_STEP}" \
      --rollouts "${ROLLOUTS}" \
      --save-every 50 \
      "${RESUME_ARGS[@]}"
    mark_done "${marker}"
  else
    log "Skip curriculum stage ${stage}"
  fi
  CURR_INIT="${stage_dir}/final/adapter"
done

if ! is_done eval_curriculum; then
  log "Evaluating CURRICULUM (final=hard)"
  python grpo/solve_grpo_eval.py \
    --eval-file "${DATA_DIR}/shared/test.parquet" \
    --adapter "${RUN_DIR}/curriculum/hard/final" \
    --output "${RUN_DIR}/curriculum/eval" \
    --device cuda:0
  mark_done eval_curriculum
fi

########################################
# 5) Comparison
########################################
python - <<PY
import json
from pathlib import Path
run = Path("${RUN_DIR}")
base = json.loads((run / "base/eval/metrics.json").read_text())
curr = json.loads((run / "curriculum/eval/metrics.json").read_text())
out = {
    "base_pass_rate": base["pass_rate"],
    "curriculum_pass_rate": curr["pass_rate"],
    "delta_pass_rate": curr["pass_rate"] - base["pass_rate"],
    "base_format_rate": base["format_rate"],
    "curriculum_format_rate": curr["format_rate"],
    "total_steps": int("${TOTAL_STEPS}"),
    "stage_steps": int("${STAGE_STEPS}"),
}
path = run / "comparison.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY

log "E2E complete. Results under ${RUN_DIR}"
