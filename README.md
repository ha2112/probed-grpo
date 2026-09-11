# Probed GRPO

This repository studies whether a learned estimate of Codeforces problem
difficulty can improve curriculum ordering for GRPO training. It contains two
connected workflows:

1. Train a linear difficulty probe on frozen Afterburner representations.
2. Build equivalent CodeContests corpora in random, official-rating, and
   probe-predicted order, then train one GRPO route at a time.

**Running on a new GPU machine? Start with [RUN_PIPELINE.md](RUN_PIPELINE.md).**
It covers the pinned Docker environment, a bundled execution judge, the full probe,
corpus construction, a required one-step GPU smoke test, training, resumption and export.

## Repository layout

| Path | Purpose |
| --- | --- |
| `linear_probe.py` | Train or apply the coding-difficulty probe. |
| `artifact_cache.py` | Pinned model/dataset downloads and repository-local caches. |
| `RUN_PIPELINE.md` | Step-by-step mentor handoff for the complete pipeline. |
| `grpo/Dockerfile`, `grpo/requirements-overlay.lock` | Pinned GPU runtime and Python dependencies. |
| `grpo/preflight.py`, `grpo/judge_server.py` | Runtime/corpus checks and optional Docker-isolated judge. |
| `test_linear_probe.py` | Focused probe tests. |
| `grpo/` | Corpus construction, reward logic, tests, and training launchers. |
| `LINEAR_PROBE.md` | Probe setup, methodology, outputs, and reproduction notes. |
| `grpo/README.md` | Three-route GRPO workflow. |
| `GRPO_METHODOLOGY_REPORT.md` | Full experimental methodology. |
| `GRPO_METHODOLOGY_VERIFICATION.md` | Verification record and evidence. |
| `data-cache/`, `model-cache/` | Local datasets, model weights, and probe artifacts; ignored by Git. |
| `results/`, `grpo/checkpoints/` | Generated experiment artifacts; ignored by Git. |

## Probe-only quick start

Create an environment for the linear-probe workflow (Python 3.10–3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-probe.txt
```

Run the checks:

```bash
python artifact_cache.py
HF_HUB_OFFLINE=1 python -m unittest -v test_artifact_cache.py test_linear_probe.py
python -m unittest discover -s grpo/tests -v
```

Train the probe. If the model checkpoint is absent, it is downloaded once into
`model-cache/afterburner`:

```bash
python linear_probe.py
```

For GRPO use the pinned Linux/CUDA image in [RUN_PIPELINE.md](RUN_PIPELINE.md).
The probe-only requirements do not install `verl`, vLLM or a CUDA environment.

## Local artifacts

Missing Hugging Face models are downloaded into `model-cache`; dataset downloads
use `data-cache/huggingface/datasets`; and the trained linear probe is stored in
`model-cache/probe`. These caches, checkpoints, logs, and generated results stay
local by default. The tracked `results/.gitkeep` preserves the results directory
without committing run artifacts. If a small artifact must become part of the
research record, add it intentionally with `git add -f` and document its provenance.
