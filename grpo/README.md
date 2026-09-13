# Afterburner GRPO on three Venus orderings

For a new machine, follow the
[complete mentor workflow](../README.md#complete-mentor-workflow). It includes the
pinned Docker setup, profiling judge, preflight, full probe, corpus build, required GPU
smoke step, full runs and checkpoint export.

This experiment holds the model, selected rows, prompt construction, reward,
validation set, seed, and `verl` settings constant. Only the physical order of
the training parquet changes:

- `random`: seeded shuffle of canonical Venus example IDs.
- `official`: Venus Easy → Medium → Hard, then problem/example ID.
- `probed`: ascending Codeforces-trained probe score, then problem/example ID.

Codeforces is used only to train the probe. GRPO uses the pinned
[`Elfsong/Venus_Python`](https://huggingface.co/datasets/Elfsong/Venus_Python)
dataset (984 train / 300 test problems), with native solution baselines,
profiling measurements, test runners and evaluators. Each problem expands into
time, memory and integral optimization examples. `manifest.json` records every row hash and the
three orderings so corpus equality is auditable. Prompt filtering and trimming to
whole training batches occur on the shared row set before ordering; training does
not drop a different partial batch from each route.

## Build the corpora

Venus is downloaded automatically if absent. It can be cached before training
the full Codeforces probe described in `../LINEAR_PROBE.md`:

```bash
python3 grpo/venus_corpus.py --download-only
```

Then build using the trained probe:

```bash
python3 grpo/venus_corpus.py
python3 grpo/preflight.py --stage corpus
```

For an offline smoke build from the cached dataset:

```bash
HF_HUB_OFFLINE=1 python3 grpo/venus_corpus.py \
  --max-train 10 --max-validation 10 \
  --probe model-cache/probe/smoke/probe.pt \
  --allow-smoke-probe --train-batch-size 1 \
  --output-dir /tmp/venus-grpo-smoke
```

The smoke probe proves execution only. It is not suitable for the actual
probed-ranking experiment. Default corpora go to `grpo/data/venus/`; use
`VENUS_GRPO_DATA_DIR` to train from another output directory. Existing Codeforces
GRPO corpora must be rebuilt. Checkpoints default to `grpo/checkpoints/venus/` to
avoid resuming legacy Codeforces training. Changed pipeline sources require a new
`--run-dir`.

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

Use a profiling [Monolith service](https://github.com/Elfsong/Monolith) and set
`MONOLITH_URL` to its `/execute` endpoint. The old bundled `judge_server.py` cannot
provide efficiency measurements. The default public endpoint may be unavailable;
`python3 grpo/venus_reward.py --check` verifies both correctness and profiling.
Requests default to eight concurrent workers, each running all supplied test cases
64 times, matching Afterburner. Infrastructure failures abort the run.

The shared reward follows `Afterburner/grpo/afterburner_reward_function.py`:
`0.5 * (correctness_transition + 0.5 * efficiency_delta_if_passed) + 0.2 * format`.
The efficiency delta is the original clipped, tanh-scaled relative improvement for
the requested objective; the original clipping constants and measurement values
are preserved. Length reward remains disabled. A candidate must pass every test.

## Verify

```bash
VERL_SOURCE_DIR=/opt/verl VERL_INTEGRATION=1 python3 -m unittest discover -s grpo/tests -v
```
