# Probed GRPO

This repository studies whether a learned estimate of Codeforces problem
difficulty can improve curriculum ordering for GRPO training. It contains two
connected workflows:

1. Train a linear difficulty probe on frozen Afterburner representations.
2. Build equivalent CodeContests corpora in random, official-rating, and
   probe-predicted order, then train one GRPO route at a time.

## Repository layout

| Path | Purpose |
| --- | --- |
| `linear_probe.py` | Train or apply the coding-difficulty probe. |
| `test_linear_probe.py` | Focused probe tests. |
| `grpo/` | Corpus construction, reward logic, tests, and training launchers. |
| `LINEAR_PROBE.md` | Probe setup, methodology, outputs, and reproduction notes. |
| `grpo/README.md` | Three-route GRPO workflow. |
| `GRPO_METHODOLOGY_REPORT.md` | Full experimental methodology. |
| `GRPO_METHODOLOGY_VERIFICATION.md` | Verification record and evidence. |
| `data-cache/`, `model-cache/` | Local datasets and model weights; ignored by Git. |
| `results/`, `grpo/checkpoints/` | Generated experiment artifacts; ignored by Git. |

## Quick start

Create an environment for the linear-probe workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-probe.txt
```

Run the checks:

```bash
HF_HUB_OFFLINE=1 python -m unittest -v test_linear_probe.py
python -m unittest discover -s grpo/tests -v
```

Train the probe after placing the model checkpoint at the default path described
in `LINEAR_PROBE.md`:

```bash
python linear_probe.py
```

The GRPO stage additionally requires the Linux/CUDA `verl` environment used by
Afterburner and a compatible code-execution endpoint. See `grpo/README.md` for
corpus-building and training commands.

## Local artifacts

Model weights, downloaded datasets, caches, checkpoints, logs, and generated
results stay local by default. The tracked `results/.gitkeep` preserves the
expected output directory without committing run artifacts. If a small artifact
must become part of the research record, add it intentionally with `git add -f`
and document its provenance.
