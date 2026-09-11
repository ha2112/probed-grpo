# Afterburner coding-difficulty probe

`linear_probe.py` trains a new probe for the cached
`model-cache/afterburner/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start`
checkpoint. This is the cold-start model, not the final GRPO checkpoint. Its
configuration specifies a hidden size of 2,048; the script derives the probe
width from its extracted features.

## Run

From this workspace, use the existing environment:

```bash
conda activate probing-difficulty-linear
python linear_probe.py
```

For a new environment, install `requirements-probe.txt`. The script selects CUDA,
then MPS, then CPU. Extraction defaults to float16 on accelerators and float32 on
CPU. Only the small linear probe trains, on CPU. To select a GPU explicitly:

```bash
python linear_probe.py --device cuda:0
```

The dataset cache defaults to `data-cache/huggingface/datasets`. To use the
existing cache without contacting Hugging Face, set `HF_HUB_OFFLINE=1`.
An end-to-end smoke run uses a separate output directory:

```bash
HF_HUB_OFFLINE=1 python linear_probe.py --max-samples 10 --epochs 2 \
  --output-dir results/linear-probe-afterburner-smoke
```

The smoke run verifies execution only; ten examples and two epochs are
insufficient to assess the research hypothesis. The full defaults are all rated
Codeforces training problems and 80 probe epochs.

The locally cached dataset yields 6,673 distinct rated problems after filtering:
4,270 training, 1,068 validation, and 1,335 test rows, with ratings from 800 to 3,500.

Score a new UTF-8 problem statement with the trained checkpoint:

```bash
python linear_probe.py --predict-file problem.txt
```

Prediction loads the model identity, precision, and token limit from `probe.pt`
and shares the training prompt. Select a different trained probe with
`--output-dir`. Scores are continuous estimates on the Codeforces rating scale,
not probabilities or measured correctness/efficiency improvements.

## Method and source evidence

The reference implementation is the sibling
`Probing-Difficulty-Perception-of-LLMs` repository:

| Component | Implementation here | Source |
| --- | --- | --- |
| Representation | Frozen model, final normalized layer, last prompt token, assistant generation marker appended; no generation | `README.md`, “Data Preparation of Training Probe”, step 2 |
| Labels | `cf_rating > 0` and `source == 2`, from CodeContests' original training split | `train_codecontests_probe.py`, dataset filtering |
| Partition | Seed 42, 20% held-out test; remaining rows divided 80/20 into train/validation (approximately 64/16/20 overall) | Reference README step 3 and CodeContests trainer |
| Regressor | One `Linear(hidden_size, 1)` layer with bias | Reference `RegressionNN` |
| Optimization | MSE, Adam, learning rate 0.0005, weight decay 0.0002, batch size 32, 80 epochs | Reference README step 3 |
| Coding-scale adaptation | Standardize both features and ratings; select the lowest validation-loss epoch; fold both scalers into saved weights | `train_codecontests_probe.py` |
| Response format | Python solution with `<thinking>` and `<solution>` tags | Adapted from `Afterburner/grpo/afterburner_dataset.py` |

The original README fits feature scaling before splitting. The sibling coding
trainer fits scaling on train plus validation. This script intentionally fits
both scalers on **training rows only** to keep held-out statistics out of training.
Identical problem statements are deduplicated before splitting. These are
documented departures from literal replication.

`AutoModel.last_hidden_state[0, -1]` avoids computing vocabulary logits and
retaining every layer. The feature-equivalence test compares it against the
reference `AutoModelForCausalLM(..., output_hidden_states=True).hidden_states[-1]`
on a small Qwen2 model with the actual cached tokenizer.

The prompt is:

```text
You are an expert competitive programmer. Solve the following programming problem in Python, respecting its input/output format and constraints. Enclose your reasoning in <thinking> </thinking> and your complete solution in <solution> </solution>. Put the solution code in one markdown code block with the python language identifier.
```

The user turn contains only the problem description. Ratings, solutions, and
private tests are excluded. This probes problem-solving difficulty; adding
Afterburner's original-solution/performance context would change the task to
revision difficulty and would require corresponding labels.

The cached tokenizer supplies its own `System:`, `Human:`, and `Assistant:` chat
template. The script uses that template, including its final `Assistant:` marker,
instead of hardcoding the base Qwen model's ChatML role markers.

## Outputs and resuming

The default output directory is `results/linear-probe-afterburner`:

- `embeddings.npz`: float32 raw features and cache metadata; saves every 25 rows
  and on completion. A rerun resumes the saved prefix.
- `probe.pt`: state dictionary, feature dimension, and extraction metadata;
  loadable with `torch.load(..., weights_only=True)`.
- `predictions.csv`: problem names, partition membership, actual ratings, predictions.
- `metrics.json`: validation-selected epoch, settings, provenance, and per-partition
  Pearson/Spearman correlations, RMSE, and MAE in rating units.
- `losses.csv`: training and validation MSE on standardized ratings per epoch.

The cache checks the selected records, prompt, model identity/local file stats,
precision, and token limit. Use a separate `--output-dir` when changing these.
Local file stats detect ordinary model replacement but are not cryptographic
weight hashes. For reproducible runs with `--model`, use a fixed local snapshot.

Prompts longer than `--max-length` (default 4,096 tokens) stop with the problem
name and actual token count. Increase the limit, within the model's context
window, and use a new output directory. No question or assistant marker is
silently truncated, and failed extractions do not silently remove examples.

Run focused checks with:

```bash
HF_HUB_OFFLINE=1 python -m unittest -v test_linear_probe.py
```

These checks exercise actual Qwen2 feature equivalence, the coding chat template,
overflow rejection, scaler folding, split isolation, partial-cache recovery,
cache mismatch rejection, and safe checkpoint reload. They require the cached
tokenizer, but not the full checkpoint weights or a GPU.

Verified locally on 2026-09-11: all five tests passed, and a CPU/float32 run on
the real cold-start checkpoint completed ten Codeforces problems and two probe
epochs. Its artifacts are in `results/linear-probe-afterburner-smoke`. The full
6,673-problem, 80-epoch experiment has not been run. The saved probe also reloaded
successfully through `--predict-file` and scored a new problem with the real model.
