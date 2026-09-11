# Afterburner GRPO on three CodeContests orderings

For a new machine, follow the
[complete mentor workflow](../README.md#complete-mentor-workflow). It includes the
pinned Docker setup, local judge, preflight, full probe, corpus build, required GPU
smoke step, full runs and checkpoint export.

This experiment holds the model, selected rows, prompt construction, reward,
validation set, seed, and `verl` settings constant. Only the physical order of
the training parquet changes:

- `random`: seeded shuffle of canonical problem IDs.
- `official`: ascending Codeforces `cf_rating`, then problem ID.
- `probed`: ascending learned difficulty, then problem ID.

Every retained problem must be a rated Codeforces row with a Python reference
solution and at least one test. `manifest.json` records every row hash and the
three orderings so corpus equality is auditable. Prompt filtering and trimming to
whole training batches occur on the shared row set before ordering; training does
not drop a different partial batch from each route.

## Build the corpora

First train the full probe described in `../LINEAR_PROBE.md`. Then run:

```bash
python3 grpo/codeforces_corpus.py
python3 grpo/preflight.py --stage corpus
```

For an offline smoke build from the cached dataset:

```bash
HF_HUB_OFFLINE=1 python3 grpo/codeforces_corpus.py \
  --max-train 10 --max-validation 10 \
  --probe model-cache/probe/smoke/probe.pt \
  --allow-smoke-probe --train-batch-size 1 \
  --output-dir /tmp/codecontests-grpo-smoke
```

The smoke probe proves execution only. It is not suitable for the actual
probed-ranking experiment.

## Train

Training uses the pinned `verl` 0.5.0 image and a working judge. The default GPU
count is eight; source `grpo/env.sh` and set `N_GPUS`/`TENSOR_PARALLEL_SIZE` for the
actual machine. Console logging is the default; no W&B login is required.
Each launcher performs preflight before starting `verl`:

```bash
source grpo/env.sh
bash grpo/run_three_routes.sh random
bash grpo/run_three_routes.sh official
bash grpo/run_three_routes.sh probed
```

Use `all` to run the three jobs sequentially. Additional arguments are passed
unchanged to `verl`.

The bundled local judge setup is documented in the run guide. The public
`monolith.cool` endpoint is an optional external dependency, not an availability
guarantee. Judge requests are limited to eight test cases each and default to
eight concurrent requests. Infrastructure failures abort the run.

The CodeContests schema has accepted solutions and tests but no Venus baseline
runtime, memory, or integral measurements. The shared reward therefore retains
Afterburner's correctness-transition and format terms with weights `0.5` and
`0.2`; it does not invent an efficiency delta. A candidate must pass every
available test to receive the passing transition reward.

## Verify

```bash
VERL_SOURCE_DIR=/opt/verl VERL_INTEGRATION=1 python3 -m unittest discover -s grpo/tests -v
```
