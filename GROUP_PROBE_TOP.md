# Top-tier five-group difficulty model

This is the recommended research model when the goal is to predict one of five
ordered Codeforces-difficulty groups as accurately as possible.

The implementation is in `group_probe_top.py`. It keeps the earlier frozen MLP
and simple LoRA models unchanged as reproducible baselines.

Important scientific wording: the implementation is a new combination for this
project. Do not claim that each component is novel in the literature without a
formal related-work search. The contribution to test is whether their
combination improves group accuracy and curriculum ordering on this fixed split.

## Motivation

The previous models reached:

- frozen embedding MLP: about 42.2% exact test accuracy and 0.690 QWK;
- simple LoRA model: about 44.3% exact test accuracy and 0.704 QWK.

The simple LoRA model was optimized mainly for smooth ordinal behavior:
neighbor-soft labels, CORAL, and a QWK-heavy checkpoint metric. That is useful
for ordering, but it does not directly prioritize exact five-group accuracy.

The top model addresses four failure modes:

1. A generic final chat token is a weak summary of a full programming problem.
2. Hard group labels lose the continuous relationship between nearby ratings.
3. Flat classification does not penalize large ordering errors sufficiently.
4. Raw `argmax` predictions are not calibrated to the validation distribution.

## Input

Source dataset:

```text
deepmind/code_contests
revision 802411c3010cb00d1b05bad57ca77365a3c699d6
split: original train split
```

Eligible examples:

- `source == 2` (Codeforces);
- finite `cf_rating > 0`;
- unique problem descriptions;
- prompt fits within 4,096 tokens.

The model input contains only:

```text
system prompt: competitive-programming solve instruction
user prompt: problem description
assistant generation marker
```

The input does **not** contain `cf_rating`, tests, reference solutions, problem
answers, or reward results. This prevents direct target leakage.

## Labels and splits

The deterministic split uses seed 42:

- 64% train;
- 16% validation;
- 20% held-out test.

The four group edges are fitted from **train ratings only** at the 20th, 40th,
60th, and 80th percentiles. In the current data they are:

```text
group 0: rating < 1200                 easiest
group 1: 1200 <= rating < 1600
group 2: 1600 <= rating < 2000
group 3: 2000 <= rating < 2500
group 4: rating >= 2500                hardest
```

Validation and test labels use the frozen train-derived edges. Test labels are
never used for model selection or calibration.

## Model architecture

```text
problem text
  |
  v
Qwen2.5-Coder-3B Afterburner backbone
  + LoRA on attention and MLP projections in the last 16 layers
  |
  +--> learned attention pooling over every non-padding token
  |
  +--> final non-padding token
  |
  v
concatenate(attention_pool, final_token)       [4096 dimensions]
  |
LayerNorm
Linear(4096 -> 2048), GELU, Dropout
Linear(2048 -> 1024), GELU, Dropout
  |
  +--> class head:      5 logits
  +--> monotonic ordinal head: 4 ordered-threshold logits
  +--> regression head: 1 normalized rating score
```

Why learned whole-sequence pooling:

- it can attend to constraints, algorithms, input structure, and mathematical
  content anywhere in the statement;
- it is not forced to represent difficulty using only a shared generation
  marker near the end;
- concatenating the final token preserves the pretrained chat representation.

Only LoRA adapters and the new heads are trainable. The base 3B parameters stay
frozen.

## Multi-task objective

The default total loss is:

```text
L =
    1.00 * hard five-class cross entropy
  + 0.30 * Earth Mover ordinal loss
  + 0.20 * monotonic threshold BCE
  + 0.30 * Huber normalized-rating regression
  + 0.30 * RankNet pairwise ranking
```

### Hard cross entropy

Directly optimizes exact group prediction. Label smoothing is only 0.02, unlike
the old model's larger neighbor-soft target.

### Earth Mover loss

Compares cumulative class distributions. Predicting group 1 for a group-0
example is penalized less than predicting group 4.

### Monotonic ordinal threshold head

Learns four questions:

```text
is difficulty > group 0?
is difficulty > group 1?
is difficulty > group 2?
is difficulty > group 3?
```

One latent difficulty score and strictly ordered learned cutpoints guarantee
that these probabilities cannot cross. This regularizes the class head to
respect ordering.

### Huber regression

Predicts the continuous rating after train-only z-score normalization. Huber is
less sensitive to extreme ratings than MSE and preserves information lost by
binning.

### RankNet

Across the global two-GPU microbatch, problem pairs separated by at least 0.30
train-rating standard deviations are trained in the correct order.
Differentiable all-gather includes cross-GPU pairs. Ambiguous near-ties are
ignored.

## Leakage-safe exact calibration

Each validation pass produces three scalar scores:

1. expected group from the five-class probabilities;
2. expected group from monotonic threshold probabilities;
3. continuous rating-head score.

The script searches non-negative blends of those scores. For each blend, dynamic
programming finds the globally exact-accuracy-optimal four ordered thresholds on
the **validation set only**. Equal score values are never split across groups.

The validation partition is split deterministically and class-stratified into:

- a calibration half that fits the blend and thresholds;
- a disjoint selection half that chooses the best epoch.

The selected blend and thresholds are frozen before the held-out test is
evaluated. This avoids fitting calibration and selecting epochs on the same
examples, and the test set remains untouched.

Checkpoint selection emphasizes the requested metric:

```text
validation selection =
    exact accuracy
  + 0.10 * QWK
  + 0.025 * adjacent accuracy
```

## Training environment

Activate the existing Python 3.11 environment:

```bash
cd ~/sc/probed-grpo
conda activate probed-probe
pip install -r requirements-probe.txt
```

Run CPU mathematical tests:

```bash
python -m unittest -v test_group_probe_top.py
```

## Train on two GPUs

If already inside an interactive two-GPU `srun` allocation:

```bash
cd ~/sc/probed-grpo
conda activate probed-probe
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  group_probe_top.py \
  --epochs 10 \
  --patience 3 \
  --batch-size 8 \
  --eval-batch-size 4 \
  --grad-accum 2 \
  --lr 3e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --last-n-layers 16 \
  --output-dir model-cache/group-probe-top
```

`--batch-size 8` is per GPU. The effective optimization batch is:

```text
8 per GPU * 2 GPUs * 2 gradient accumulation = 32
```

Submit as a detached Slurm job instead:

```bash
mkdir -p results/logs
sbatch scripts/train_group_probe_top.sbatch
squeue -u "$USER"
```

Follow its log:

```bash
tail -f results/logs/group-top-<JOB_ID>.out
```

## Output

The default output directory is `model-cache/group-probe-top/`.

It contains:

- `adapter/`: best LoRA adapter weights;
- `heads.pt`: attention pool, shared trunk, all three heads, train-only label
  statistics, and validation calibration;
- `metrics.json`: validation/test metrics, loss configuration, calibration, and
  comparison with the simple LoRA baseline;
- `history.csv`: per-epoch train loss and validation metrics;
- `curriculum_order.csv`: all examples sorted by predicted group, then calibrated
  continuous score;
- `best/`: internal best checkpoint selected on validation.

Model output for each problem:

```text
pred_group: integer in {0, 1, 2, 3, 4}
pred_score: calibrated continuous ordering score
```

`pred_group` is for exact bucket assignment. `pred_score` breaks ties inside a
bucket for curriculum ordering.

Predict one UTF-8 problem statement after training:

```bash
python group_probe_top.py \
  --device cuda:0 \
  --output-dir model-cache/group-probe-top \
  --predict-file problem.txt
```

The command prints `pred_group`, calibrated `pred_score`, five class
probabilities, estimated continuous rating, and the train-derived group edges.

## Evaluation criteria

Primary metric:

```text
exact accuracy = mean(pred_group == true_group)
```

Secondary metrics:

- QWK: penalizes distant group errors more heavily;
- adjacent accuracy: fraction with absolute group error <= 1;
- Spearman: monotonic ordering quality;
- confusion matrix: identifies systematic boundary or extreme-group errors.

Inspect results:

```bash
python - <<'PY'
import json

m = json.load(open("model-cache/group-probe-top/metrics.json"))
for split in ("validation", "test"):
    print(split, {
        key: round(m[split][key], 4)
        for key in ("accuracy", "qwk", "adjacent_accuracy", "mae_groups")
    })
print("comparison", m.get("comparison"))
print("calibration", m["calibration"])
PY
```

The method is successful only if held-out test metrics improve over the simple
LoRA baseline. Do not report validation improvements as final results.

Current baseline to beat:

```text
test exact accuracy: 0.4427
test QWK:            0.7039
test adjacent:       0.8322
```

A strong target is exact accuracy above 0.50 with positive QWK improvement.
Values such as 0.60 are aspirational, not guaranteed: Codeforces rating contains
information (contest population, problem position, historical outcomes) that may
not be fully recoverable from statement text alone.

## Required ablation study

For a credible top-tier project, run these with the same split and seed:

1. frozen linear rating probe;
2. frozen CORAL MLP (`group_probe.py`);
3. simple LoRA (`group_probe_lora.py`);
4. top model without RankNet (`--ranking-weight 0`);
5. top model without regression (`--regression-weight 0`);
6. top model without calibration (report raw class argmax separately);
7. complete top model.

For final claims, repeat the complete model with at least three seeds and report
mean plus standard deviation. One run is evidence of a result, not evidence of
robustness.

Keep the split fixed while changing optimization seeds:

```bash
for seed in 42 43 44; do
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
    group_probe_top.py \
    --seed "$seed" \
    --output-dir "model-cache/group-probe-top-seed-${seed}"
done
```
