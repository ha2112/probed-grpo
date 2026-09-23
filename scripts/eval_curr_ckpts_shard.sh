#!/usr/bin/env bash
# Eval-only a subset of curriculum checkpoints. Args: name=adapter_relpath ...
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
CURR="${ROOT}/results/venus_solve/official2/curriculum/official"
OUT="${ROOT}/results/venus_solve/official2/curriculum/ckpt_evals"
TEST="${ROOT}/grpo/data/venus_solve/shared/test.parquet"
mkdir -p "${OUT}/logs" results/logs

log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "${OUT}/logs/orchestrator.log"; }

python -c "import torch; n=torch.cuda.device_count(); print(f'torch.cuda.device_count={n}'); assert n >= ${N_GPUS}, n"

eval_one() {
  local adapter="$1"
  local output="$2"
  local name="$3"
  if [[ ! -f "${adapter}/adapter_config.json" && ! -f "${adapter}/adapter/adapter_config.json" ]]; then
    echo "missing adapter under: ${adapter}" >&2
    exit 1
  fi
  rm -rf "${output}"
  mkdir -p "${output}"
  log "Eval-only ${name} on ${N_GPUS} GPU(s) -> ${output}"
  if [[ "${N_GPUS}" -gt 1 ]]; then
    python -m torch.distributed.run --standalone --nproc_per_node="${N_GPUS}" \
      grpo/solve_grpo_eval.py \
        --eval-file "${TEST}" --adapter "${adapter}" --output "${output}" --max-new-tokens 1536
  else
    python grpo/solve_grpo_eval.py \
      --eval-file "${TEST}" --adapter "${adapter}" --output "${output}" \
      --device cuda:0 --max-new-tokens 1536
  fi
}

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 out_name=rel_adapter_path ..." >&2
  exit 1
fi

for spec in "$@"; do
  name="${spec%%=*}"
  rel="${spec#*=}"
  eval_one "${CURR}/${rel}" "${OUT}/${name}/eval" "${rel}"
done

python - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
wanted = ["hard_step200", "hard_step400", "medium_step800", "medium_final"]
report = {}
ready = True
for name in wanted:
    path = out / name / "eval" / "metrics.json"
    if not path.is_file():
        ready = False
        continue
    metrics = json.loads(path.read_text())
    report[f"{name}_pass_rate"] = metrics["pass_rate"]
    report[name] = metrics
if ready:
    base_path = Path("results/venus_solve/official2/baseline/summary.json")
    curr_path = Path("results/venus_solve/official2/curriculum/summary.json")
    if base_path.is_file():
        base = json.loads(base_path.read_text())
        report["ref_base_model_pass_rate"] = base["base_model_pass_rate"]
        report["ref_full_grpo_pass_rate"] = base["full_grpo_pass_rate"]
    if curr_path.is_file():
        curr = json.loads(curr_path.read_text())
        report["ref_hard_final_pass_rate"] = curr["official_curriculum_pass_rate"]
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k.endswith("_pass_rate") or k.startswith("ref_")}, indent=2))
else:
    print("partial done; waiting for sibling job before comparison.json")
PY
log "shard evals done"
