# Top-tier five-group difficulty model

This is the recommended research model when the goal is to predict one of five
ordered Codeforces-difficulty groups as accurately as possible.

The implementation is in `group_probe_top.py`. It keeps the earlier frozen MLP
and simple LoRA models unchanged as reproducible baselines.

The sections **Problem statement**, **Motivation**, **Model architecture**,
**Multi-task training objective**, and **Decoding** below are written so they
can be adapted into a paper Methods section (with equations and explicit
motivations). Keep the scientific caution in the next paragraph when claiming
novelty.

Important scientific wording: the implementation is a new combination for this
project. Do not claim that each component is novel in the literature without a
formal related-work search. The contribution to test is whether their
combination improves group accuracy and curriculum ordering on this fixed split.

## Problem statement (paper-ready)

**Task.** Given only the natural-language statement of a competitive-programming
problem, predict a discrete difficulty bucket \(y \in \{0,1,2,3,4\}\), where
larger \(y\) means harder. Buckets are ordered, so mistaking \(0\) for \(1\) is
less severe than mistaking \(0\) for \(4\), but the primary research goal of
this model is **exact bucket accuracy** \(\mathbb{1}[\hat y = y]\).

**Why this matters.** Exact groups support curriculum construction (serve easier
problems first) and analysis of whether a coding LLM's internal representation
encodes contest difficulty. Ordering-only metrics (QWK, Spearman) can look good
while exact buckets remain wrong; therefore architecture, loss, and model
selection are all aligned toward exact accuracy, with ordinal quality as a
secondary objective.

**What must not leak.** The model never sees Codeforces rating, tests, editorial
solutions, or reward signals at inference time. Labels are derived offline from
`cf_rating` for supervision only.

---

## Motivation

### Empirical starting point

On the fixed Codeforces subset and split used in this repo:

| Model | Exact test acc. | QWK |
|-------|-----------------|-----|
| Frozen embedding + CORAL MLP | ~42.2% | ~0.690 |
| Simple LoRA (soft-ordinal / QWK-heavy) | ~44.3% | ~0.704 |

The simple LoRA baseline already adapts the backbone, but it was tuned mainly
for smooth ordinal behavior (neighbor-soft labels, CORAL, QWK-oriented
checkpointing). That helps ranking and adjacent correctness more than **exact**
five-way classification.

### Failure modes we target

1. **Weak problem summary.** Using only the final chat token compresses a long
   statement into a shared generation marker. Difficulty cues (constraints,
   required algorithms, edge cases, mathematical structure) appear anywhere in
   the text.
2. **Information loss from hard bins.** Mapping continuous rating to five labels
   discards within-bin structure; nearby ratings should remain close in
   representation space.
3. **Symmetric classification cost.** Plain cross-entropy treats all wrong
   classes equally, so a large ordinal jump is not specially discouraged.
4. **Uncalibrated decoding.** Training logits + `argmax` need not match the
   validation label margins; a small post-hoc score blend and ordered thresholds
   can improve exact accuracy without changing the network.

### Design principle

Use a **shared encoder representation**, train it with **complementary
supervisory signals** (exact class + ordinal geometry + continuous rating +
pairwise order), then decode with a **validation-only, leakage-safe calibrator**
optimized for exact accuracy. Individual building blocks are standard; the claim
to evaluate is whether this combination improves held-out exact accuracy over
the LoRA baseline on the fixed split (and across seeds).

---

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

---

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

For calibration vs checkpoint selection, the validation set is further split
50/50 (stratified by group, seed 42) into **calibration** and **selection**
halves (see Decoding).

---

## Model architecture

Implementation: `TopGroupModel` in `group_probe_top.py`. Notation: batch size
\(B\), sequence length \(T \le 4096\), backbone width \(H = 2048\).

### High-level diagram

```text
tokenized problem (chat template)
              |
              v
 ┌────────────────────────────────────────────┐
 │ Encoder: Qwen2.5-Coder-3B Afterburner      │
 │ AutoModel last_hidden_state ∈ R^{B×T×H}    │
 │ + LoRA (r=32, α=64) on last 16 Transformer │
 │   layers: attn {q,k,v,o} and MLP           │
 │   {gate,up,down}. Base weights frozen.     │
 └────────────────────────────────────────────┘
              |
              |  hidden h_{1:T}
              |
      ┌───────┴────────┐
      v                v
 attention pool     last non-pad token
 α_t · h_t          h_{t*}
      |                |
      +-------cat------+
              |
              v
        u ∈ R^{B×2H}   (4096-d)
              |
              v
        shared trunk MLP
        LN → Linear(2H→H) → GELU → Dropout
           → Linear(H→H/2) → GELU → Dropout
              |
              v
        z ∈ R^{B×1024}     shared difficulty feature
              |
     ┌────────┼────────┐
     v        v        v
  class     ordinal   regress
  5 logits  4 logits  1 scalar
```

### Stage A — Parameter-efficient encoder

**Motivation.** A 3B coding LM already encodes algorithmic language; full
fine-tuning is expensive and easy to overfit on ~4k–7k eligible problems. LoRA
on the **last 16 layers** adapts high-level semantic features while freezing
early layers that mostly do generic token mixing.

Formally, if \(f_\theta\) is the frozen backbone and \(\Delta\theta\) are LoRA
adapters, the contextual states are

\[
h_{1:T} = f_{\theta+\Delta\theta}(x_{1:T}) \in \mathbb{R}^{T \times H}.
\]

Only \(\Delta\theta\), the pooling module, trunk, and heads are trained
(~0.85% of parameters in the default config).

### Stage B — Dual sequence summary (attention pool + last token)

**Motivation.** Difficulty is a **global** property of the statement, not a
property of the final assistant marker alone. We therefore build two summaries:

1. **Learned soft attention over all non-padding tokens** (additive / Bahdanau-
   style pooling — *not* an extra Transformer self-attention block):

\[
e_t = w^\top \tanh(W\,\mathrm{LN}(h_t)), \quad
\alpha_t = \frac{\exp(e_t)}{\sum_{t':m_{t'}=1}\exp(e_{t'})}, \quad
a = \sum_{t=1}^{T} \alpha_t h_t.
\]

   Padding positions are masked (\(e_t \leftarrow -\infty\)) before the
   softmax. The weights \(\alpha\) are free to focus on constraints, formulas,
   I/O specs, or algorithm hints wherever they appear.

2. **Final content token** \(h_{t^*}\) with \(t^* = \sum_t m_t - 1\), which
   preserves the pretrained chat representation near the generation boundary.

**Why concatenate both.** Pooling alone can ignore useful end-of-prompt
geometry; last-token alone underuses the long statement. The concat
\(u = [a; h_{t^*}] \in \mathbb{R}^{2H}\) lets the trunk keep both signals.

Code: `attention_pool` and last-index gather in `TopGroupModel.forward`.

### Stage C — Shared trunk

\[
z = \mathrm{MLP}(u) \in \mathbb{R}^{H/2}.
\]

**Motivation.** One difficulty embedding should serve all tasks so gradients
from classification, ordinal structure, and regression reinforce the same
geometry rather than diverging into disconnected heads.

### Stage D — Three prediction heads (still one model)

All heads read the **same** \(z\):

| Head | Output | Role |
|------|--------|------|
| Classifier | \(\ell^{\mathrm{cls}} \in \mathbb{R}^{5}\) | Direct 5-way group prediction |
| Monotonic ordinal | \(\ell^{\mathrm{ord}} \in \mathbb{R}^{4}\) | \(P(y > k)\) for \(k=0..3\) with ordered cutpoints |
| Regressor | \(\hat r_z \in \mathbb{R}\) | Continuous train-normalized rating |

**Ordinal head details.** A scalar score \(s = w_s^\top z\) and strictly
increasing cutpoints \(c_0 < c_1 < c_2 < c_3\) (first cut free; positive
softplus increments) yield

\[
\ell^{\mathrm{ord}}_k = s - c_k,
\]

so estimated \(P(y>k)\) cannot cross as \(k\) increases. This encodes the
ordered nature of difficulty without requiring five independent ordinal logits.

**Paper wording tip:** describe this as a *multi-task ordinal difficulty
encoder with dual pooling*, not as three separate models.

---

## Multi-task training objective

Parameters update **every optimizer step** (after `grad_accum` micro-batches),
not once per epoch. Default loss:

\[
\begin{aligned}
\mathcal{L}
&= 1.00\,\mathcal{L}_{\mathrm{CE}}
 + 0.30\,\mathcal{L}_{\mathrm{EMD}}
 + 0.20\,\mathcal{L}_{\mathrm{CORAL}}
 + 0.30\,\mathcal{L}_{\mathrm{Huber}}
 + 0.30\,\mathcal{L}_{\mathrm{Rank}}.
\end{aligned}
\]

### Cross-entropy (primary for exact accuracy)

\[
\mathcal{L}_{\mathrm{CE}} = \mathrm{CE}(\ell^{\mathrm{cls}}, y)
\]

with light label smoothing \(0.02\). This is the direct surrogate for
\(\hat y = \arg\max_k \ell^{\mathrm{cls}}_k\). Unlike the earlier LoRA baseline,
we do **not** heavily soften the target toward neighbors, so the classifier is
not biased away from exact hits.

### Squared Earth-Mover / CDF loss (ordinal geometry on the class head)

Let \(p = \mathrm{softmax}(\ell^{\mathrm{cls}})\) and \(q = \mathrm{onehot}(y)\).
Compare cumulative distributions:

\[
\mathcal{L}_{\mathrm{EMD}}
= \frac{1}{K}\sum_{k=0}^{K-1}
  \Bigl(\sum_{j\le k} p_j - \sum_{j\le k} q_j\Bigr)^2.
\]

**Motivation.** Moving probability mass to an adjacent wrong class costs less
than moving it to a distant class, matching the ordered label space.

### Monotonic CORAL / threshold BCE

Targets \(t_k = \mathbf{1}[y > k]\). With logits \(\ell^{\mathrm{ord}}\),

\[
\mathcal{L}_{\mathrm{CORAL}}
= \mathrm{BCEWithLogits}(\ell^{\mathrm{ord}}, t).
\]

**Motivation.** Forces the shared feature \(z\) to answer nested difficulty
questions consistently with rating order.

### Huber regression on normalized rating

Train-only mean/std yield \(r_z = (r - \mu)/\sigma\). Then

\[
\mathcal{L}_{\mathrm{Huber}} = \mathrm{SmoothL1}(\hat r_z, r_z;\,\beta=0.5).
\]

**Motivation.** Recovers continuous difficulty discarded by quintile binning;
Huber limits the influence of extreme ratings versus MSE.

### RankNet pairwise ranking (batch-global, including DDP)

From class probabilities form an expected group score
\(s = \sum_{k=0}^{4} k\,p_k\). For pairs \((i,j)\) in the **all-gathered**
two-GPU micro-batch with \(|r_{z,i}-r_{z,j}| \ge 0.30\),

\[
\mathcal{L}_{\mathrm{Rank}}
= \mathrm{mean}\,\mathrm{softplus}\bigl(
  -\mathrm{sign}(r_{z,i}-r_{z,j})\,(s_i-s_j)
\bigr).
\]

Near ties are ignored. **Motivation.** Directly trains comparative curriculum
order (“A harder than B”) even when absolute bins are noisy at boundaries.

---

## Decoding: leakage-safe exact calibration

Training produces three continuous scores per example:

1. **Class expectation** \(s^{\mathrm{cls}} = \sum_k k\,p_k\);
2. **Ordinal expectation** \(s^{\mathrm{ord}} = \sum_k \sigma(\ell^{\mathrm{ord}}_k)\);
3. **Regression score** \(s^{\mathrm{reg}} = \hat r_z\).

**Motivation for blending.** The heads make correlated but non-identical
errors; a non-negative convex combination can outperform any single head for
exact cuts.

### Procedure (each validation epoch, rank 0)

1. Fit blend weights and four **ordered** thresholds on the **calibration**
   half only: standardize the three scores, grid-search simplex weights, and
   for each blend run dynamic programming to maximize exact accuracy under
   ordered thresholds (equal scores stay in one group).
2. Apply the frozen blend + thresholds to the disjoint **selection** half.
3. Checkpoint by

\[
\mathrm{selection}
= \mathrm{exact}
+ 0.10\,\mathrm{QWK}
+ 0.025\,\mathrm{adjacent}.
\]

4. After training, reload the best checkpoint and evaluate **full validation**
   and **held-out test** with the stored calibration. Test never participates
   in fitting thresholds or choosing the epoch.

This avoids the optimistic bias of fitting thresholds and picking epochs on the
same validation examples.

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
