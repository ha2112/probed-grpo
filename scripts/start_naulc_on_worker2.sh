#!/usr/bin/env bash
# Launch nAULC held-out eval grid on the existing worker-2 interactive allocation.
# Usage (from login, attaches to job 4294):
#   bash scripts/start_naulc_on_worker2.sh
# Or inside the srun pty on worker-2:
#   N_GPUS=1 nohup bash scripts/eval_naulc_grid.sh all > results/venus_solve/official2/naulc_evals/logs/run.out 2>&1 &
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mkdir -p results/venus_solve/official2/naulc_evals/logs results/logs

JOBID="${NAULC_JOBID:-4294}"
LOG="${ROOT}/results/venus_solve/official2/naulc_evals/logs/naulc_worker2.out"

echo "attaching to Slurm job ${JOBID} on worker-2; log=${LOG}"
nohup srun --jobid="${JOBID}" --overlap --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=64G \
  -w worker-2 \
  bash -lc "cd '${ROOT}' && export N_GPUS=1 FORCE=0 && bash scripts/eval_naulc_grid.sh all" \
  >"${LOG}" 2>&1 &
echo "launcher pid=$!"
echo "tail -f ${LOG}"
