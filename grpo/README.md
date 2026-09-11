# Afterburner GRPO on three CodeContests orderings

This experiment holds the model, selected rows, prompt construction, reward,
validation set, seed, and `verl` settings constant. Only the physical order of
the training parquet changes:

- `random`: seeded shuffle of canonical problem IDs.
- `official`: ascending Codeforces `cf_rating`, then problem ID.
- `probed`: ascending learned difficulty, then problem ID.

Every retained problem must be a rated Codeforces row with a Python reference
solution and at least one test. `manifest.json` records every row hash and the
three orderings so corpus equality is auditable.

## Build the corpora

First train the full probe described in `../LINEAR_PROBE.md`. Then run:

```bash
python3 grpo/codeforces_corpus.py
```

For an offline smoke build from the cached dataset:

```bash
HF_HUB_OFFLINE=1 python3 grpo/codeforces_corpus.py \
  --max-train 10 --max-validation 10 \
  --probe results/linear-probe-afterburner-smoke/probe.pt \
  --output-dir /tmp/codecontests-grpo-smoke
```

The smoke probe proves execution only. It is not suitable for the actual
probed-ranking experiment.

## Train

Training needs the same Linux/CUDA `verl` environment as Afterburner and a
reachable Monolith-compatible execution endpoint:

```bash
MONOLITH_URL=https://monolith.cool/execute grpo/run_three_routes.sh random
MONOLITH_URL=https://monolith.cool/execute grpo/run_three_routes.sh official
MONOLITH_URL=https://monolith.cool/execute grpo/run_three_routes.sh probed
```

Use `all` to run the three jobs sequentially. Additional arguments are passed
unchanged to `verl`.

The CodeContests schema has accepted solutions and tests but no Venus baseline
runtime, memory, or integral measurements. The shared reward therefore retains
Afterburner's correctness-transition and format terms with weights `0.5` and
`0.2`; it does not invent an efficiency delta. A candidate must pass every
available test to receive the passing transition reward.

## Verify

```bash
python3 -m unittest discover -s grpo/tests -v
```
