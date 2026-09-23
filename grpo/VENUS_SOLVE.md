# Venus solve-from-scratch GRPO (correctness only, no Docker)

This experiment checks whether a **Codeforces 3-class probe curriculum**
improves pass@tests GRPO on Venus versus training on the full train set at once.

## Setup

| Stage | Data / model |
|-------|----------------|
| Probe | `group_probe_top --num-groups 3` on Codeforces |
| GRPO | `Elfsong/Venus_Python`, **solve from statement only** |
| Reward | pass all tests + format (no time/memory) |
| Hardware | 2 GPUs, conda `probed-probe` (no Docker/verl) |

Strategies (default **600 steps per arm**; curriculum is 3× longer wall-clock):

1. **base** — GRPO on all Venus train rows (600 steps).
2. **curriculum** — easy → medium → hard (**600+600+600**), resume LoRA each stage.

## Launch

```bash
mkdir -p results/logs
sbatch scripts/train_venus_solve_e2e.sbatch
tail -f results/logs/vla-<JOB>.out
```

Smoke (tiny subset) inside an interactive 2-GPU shell:

```bash
conda activate probed-probe
export PATH="$CONDA_PREFIX/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1
SMOKE=1 bash scripts/run_venus_solve_e2e.sh
```

Reuse an existing probe checkpoint:

```bash
SKIP_PROBE=1 PROBE_DIR=model-cache/vla_wm bash scripts/run_venus_solve_e2e.sh
```

## Outputs

```text
results/venus_solve/e2e/
  base/final/          # LoRA policy
  base/eval/metrics.json
  curriculum/hard/final/
  curriculum/eval/metrics.json
  comparison.json      # delta pass_rate
grpo/data/venus_solve/
  base/train.parquet
  curriculum/{easy,medium,hard}/train.parquet
  shared/test.parquet
  manifest.json
```

## Notes

- Needs a working judge. Default is **local** Python harness
  (`VENUS_JUDGE=local`) because cluster nodes often cannot resolve
  `monolith.cool`. Set `VENUS_JUDGE=monolith` only if Monolith is reachable.
- This trainer is **not** the Docker/verl Afterburner stack; base vs curriculum
  comparisons are valid within this codebase.
- `#SBATCH --time=7-00:00:00` is only the **Slurm wall-clock limit** (max job
  lifetime before kill). It is **not** the expected training duration.
- Default e2e uses `FORCE_PROBE=1` so Codeforces 3-class probe is trained
  (not reused). Set `FORCE_PROBE=0` to reuse `model-cache/vla_wm/best`.
- Do **not** export `CUDA_VISIBLE_DEVICES=0,1` inside Slurm; let Slurm bind
  both GPUs (`--gpus-per-task=2`). The job aborts if `torch.cuda.device_count() < 2`.
