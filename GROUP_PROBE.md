# Five-group ordinal difficulty probe — runbook

Goal: predict **one of 5 ordered difficulty groups** (easy → hard), not exact CF rating.

> **Want a strong model?** The frozen-embedding MLP below plateaus (~42% exact).
> Use the LoRA trainer instead: **[GROUP_PROBE_LORA.md](GROUP_PROBE_LORA.md)**
> (`python group_probe_lora.py --device cuda:0`).

| File | Role |
| --- | --- |
| `group_probe.py` | Train / predict (frozen embeddings — fast baseline) |
| `group_probe_lora.py` | **Strong** LoRA backbone trainer |
| `test_group_probe.py` | Unit tests |
| `linear_probe.py` | Builds `embeddings.npz` (already done for you) |
| `model-cache/group-probe/` | Outputs after train |

---

## Train command (you run this)

In your `srun` shell / conda env (`probed-probe`):

```bash
cd ~/sc/probed-grpo
conda activate probed-probe

# 1) sanity checks
python -m unittest -v test_group_probe.py

# 2) train (CPU is enough; reuses embeddings — no GPU required)
python group_probe.py
```

Done when you see: `Saved group probe to .../model-cache/group-probe`

Optional stronger / longer:

```bash
python group_probe.py --hidden-dim 1536 --ensemble-size 7 --epochs 300 --patience 40
```

Smoke (fast):

```bash
python group_probe.py --ensemble-size 2 --epochs 40 --patience 8 \
  --output-dir model-cache/group-probe/smoke
```

Prerequisite (only if `model-cache/probe/embeddings.npz` missing):

```bash
python linear_probe.py --device cuda:0 --dtype float16
```

---

## Strategy (verified)

### Labels — 5 equal-count groups (train-only)

| Group | Approx band (your data) |
| --- | --- |
| 0 | ~800–1200 easiest |
| 1 | ~1200–1600 |
| 2 | ~1600–2000 |
| 3 | ~2000–2500 |
| 4 | ~2500–3500 hardest |

- Edges = train quantiles `[0.2, 0.4, 0.6, 0.8]` only → no leakage  
- Equal count ≈ balanced classes (unlike raw CF tiers)

### Model — strongest practical head on frozen embeddings

1. **StandardScaler** on embeddings (train-only)  
2. **LayerNorm + MLP** (1024 → 512, GELU, dropout)  
3. **CORAL** ordinal logits (`P(y>k)` for k=0..3) — respects order  
4. **5-seed ensemble** (average logits)  
5. **Balanced sampling**  
6. Early stop on **`qwk + 0.15 * adjacent_accuracy`**  
7. Defaults tuned against overfit: `lr=3e-4`, `dropout=0.35`, `weight_decay=5e-2`

### Metrics (what “good” means)

| Metric | Why |
| --- | --- |
| **QWK** | Ordinal agreement (primary) |
| **Adjacent accuracy** | `\|pred−true\|≤1` — safe for curriculum |
| Exact accuracy | Secondary |
| **delta vs baseline** | Same edges applied to old linear rating probe |

After train, `metrics.json` includes `baseline_linear_rating_quintile` and per-split `delta_qwk`.

### Curriculum use

Sort easy→hard by:

1. `pred_group` ascending  
2. `pred_score` ascending (soft expected group in `[0,4]`)

File: `model-cache/group-probe/curriculum_order.csv`

---

## After training — verify

```bash
python - <<'PY'
import json
m=json.load(open("model-cache/group-probe/metrics.json"))
for split in ("validation","test"):
    s=m[split]
    b=(m.get("baseline_linear_rating_quintile") or {}).get(split) or {}
    print(split,
          "qwk", round(s["qwk"],4),
          "acc", round(s["accuracy"],4),
          "adj", round(s["adjacent_accuracy"],4),
          "baseline_qwk", round(b.get("qwk", float("nan")),4),
          "delta_qwk", round(s["qwk"]-b["qwk"],4) if b else None)
PY
```

Pass checklist:
- [ ] `test.qwk` ≥ baseline QWK (prefer clear `delta_qwk > 0`)  
- [ ] `test.adjacent_accuracy` ≳ 0.85  
- [ ] train QWK not wildly above val/test (if train≫test by >0.25, re-run with higher `--dropout`)  
- [ ] `curriculum_order.csv` exists and groups increase along the file  

Score one problem (needs GPU for backbone):

```bash
python group_probe.py --device cuda:0 --predict-file problem.txt
```

---

## Outputs

| File | Content |
| --- | --- |
| `group_probe.pt` | Ensemble + scaler + edges |
| `metrics.json` | Val/test + baseline deltas |
| `predictions.csv` | Per-split true/pred group |
| `curriculum_order.csv` | Full easy→hard order |
| `group_schema.json` | Edge legend |
| `losses.csv` | Curves per seed |

---

## What this is / isn’t

- **Is:** best fast method on existing `embeddings.npz` for 5-group curriculum  
- **Isn’t:** LoRA / backbone finetune (larger GPU job; next upgrade if QWK plateaus)  
- **Isn’t:** GRPO training (needs container later)

Pointer from rating probe docs: [LINEAR_PROBE.md](LINEAR_PROBE.md).
