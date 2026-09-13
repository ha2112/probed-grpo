# Strong 5-group probe (LoRA) — runbook

Frozen MLP on static embeddings plateaued (~42% exact, ~0.69 QWK).  
This path adapts the **Afterburner backbone with LoRA** so group prediction can actually get strong.

For the newer exact-accuracy-focused multi-task model with whole-sequence
attention pooling and validation-only calibration, use
[`GROUP_PROBE_TOP.md`](GROUP_PROBE_TOP.md). This file documents the simpler LoRA
baseline retained for ablation.

| File | Role |
| --- | --- |
| `group_probe_lora.py` | Train / predict |
| `test_group_probe_lora.py` | CPU unit tests |
| `model-cache/group-probe-lora/` | Checkpoints + metrics |
| `GROUP_PROBE.md` | Weaker frozen-embedding baseline |

---

## Architecture

```text
Problem statement
  → chat template (same as linear_probe)
  → Qwen2.5-Coder-3B Afterburner (LoRA on last 8 layers)
  → features = concat(last_token, mean(last 8 tokens))
  → MLP head → 5 logits   (soft-ordinal CE)
  → CORAL aux head → 4 logits (order regularizer)
  → pred_group = argmax(softmax)
  → pred_score = expected group ∈ [0,4]
```

**Labels:** train-only quintiles → groups `0..4` (same edges rule as `group_probe.py`).  
**Loss:** soft CE (mass on true ± neighbor) + `0.25 * CORAL BCE`.

---

## Setup (once)

```bash
cd ~/sc/probed-grpo
conda activate probed-probe
pip install -r requirements-probe.txt   # includes peft
python -m unittest -v test_group_probe_lora.py
```

Need **GPU** (`nvidia-smi`). 1 GPU is enough; 2 GPUs optional later.

---

## Train — copy into your `srun` terminal

```bash
cd ~/sc/probed-grpo
conda activate probed-probe

python group_probe_lora.py --device cuda:0
```

Defaults: 8 epochs, early stop patience 3, LoRA r=16 on last 8 layers, grad accum 16.

Stronger (recommended if you have time):

```bash
python group_probe_lora.py --device cuda:0 \
  --epochs 12 --patience 4 \
  --lora-r 32 --lora-alpha 64 \
  --last-n-layers 12 \
  --lr 8e-5 --grad-accum 16
```

Smoke (minutes, not research quality):

```bash
python group_probe_lora.py --device cuda:0 \
  --max-samples 64 --epochs 2 --patience 1 \
  --output-dir model-cache/group-probe-lora/smoke
```

Expect **hours** for a full run (each epoch ≈ full pass over ~4.2k train problems through a 3B model). Use `tmux`/`screen` if SSH may drop.

---

## After train — verify

```bash
python - <<'PY'
import json
m=json.load(open("model-cache/group-probe-lora/metrics.json"))
print("best_epoch", m["best_epoch"])
print("val ", {k: m["validation"][k] for k in ("qwk","accuracy","adjacent_accuracy")})
print("test", {k: m["test"][k] for k in ("qwk","accuracy","adjacent_accuracy")})
print("vs MLP", m.get("compare_to_frozen_mlp"))
PY
```

Target (full data):

- test **accuracy ≥ 0.55** (stretch ≥ 0.60)  
- test **adjacent ≥ 0.90**  
- test **QWK ≥ 0.78**  
- `delta_acc` / `delta_qwk` vs frozen MLP **> 0**

Outputs: `adapter/`, `head.pt`, `metrics.json`, `curriculum_order.csv`.

Predict one problem:

```bash
python group_probe_lora.py --device cuda:0 --predict-file problem.txt
```

---

## Why this should beat the MLP

| Frozen MLP (`group_probe.py`) | LoRA (`group_probe_lora.py`) |
| --- | --- |
| Embedding fixed forever | Last layers adapt to difficulty |
| Only linear/MLP on 2k-d vector | Representation shifts per group |
| Exact ~42% plateau | Room to move exact + QWK up |

If LoRA still undershoots targets: raise `--last-n-layers` / `--lora-r`, or lower LR and train longer — do **not** go back to 10k-epoch linear heads.
