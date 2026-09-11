# Probed GRPO

This repository studies whether a learned estimate of Codeforces problem
difficulty can improve curriculum ordering for GRPO training. It contains two
connected workflows:

1. Train a linear difficulty probe on frozen Afterburner representations.
2. Build equivalent CodeContests corpora in random, official-rating, and
   probe-predicted order, then train one GRPO route at a time.

**Mentor handoff:** the complete fresh-machine workflow is included directly in
this README under [Complete mentor workflow](#complete-mentor-workflow).

## Repository layout

| Path | Purpose |
| --- | --- |
| `linear_probe.py` | Train or apply the coding-difficulty probe. |
| `artifact_cache.py` | Pinned model/dataset downloads and repository-local caches. |
| `RUN_PIPELINE.md` | Compatibility pointer to the workflow in this README. |
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

## Complete mentor workflow

Run the stages in this order: environment → judge → full probe → three corpora →
one-step GRPO smoke test → full GRPO runs. The trained probe is a prerequisite
for the probed-order corpus; a fresh clone does not contain its weights. All
commands below start from the repository root.

### 1. Host prerequisites

Use a Linux x86_64 host with NVIDIA GPUs, a working NVIDIA driver, Docker Engine,
and the NVIDIA Container Toolkit. The pinned image targets Ampere/Ada/Hopper
GPUs. Blackwell, ARM servers, macOS and Windows are outside this environment's
supported target. See [NVIDIA's container installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

The reference experiment uses **8 GPUs**, tensor parallel size **2**, 32 prompts
per batch, 32 responses per prompt, and 200 epochs. A100/H100-class machines with
ample GPU memory are the intended target. Memory fit and throughput must be
confirmed by the smoke step; the repository does not claim a measured minimum
GPU-memory requirement. Smaller GPU counts are configurable below.

Allow substantial disk space for Docker's image/build cache, the source dataset,
weights and checkpoints. The pinned base image alone is about 27.4 GB compressed;
reserve roughly 200 GB of free disk as an initial planning allowance, then monitor
usage during training. The launcher retains two actor checkpoints per route.
Internet access is needed for the initial image, package, model and dataset downloads.

Verify on the host:

```bash
nvidia-smi
docker version
docker info
python3 --version
```

Docker must be usable by the account running these commands. Use Python 3.10+
on the host for the optional bundled judge. The training container provides its
own Python 3.10 and compiled CUDA stack; do not install host Python packages into it.

Obtain the repository using its actual Git URL or the supplied source archive,
then `cd` to its root. The caches are deliberately ignored by Git and are
downloaded or created on the mentor's machine.

### 2. Build the pinned training image

On the host:

```bash
bash grpo/container.sh build
```

This pins the upstream image by digest, installs the locked Python overlay, and
installs `verl` at commit `8fdc4d3f202f41461f4de9f42a637228e342668b` (v0.5.0).
It preserves the upstream PyTorch 2.6.0, vLLM 0.8.5.post1 and FlashAttention binary
stack. It checks key imports during the build. Do not run `pip install -U`,
`pip install verl`, or reinstall PyTorch afterward.

The resulting image is `probed-grpo:verl0.5.0`. To record the exact built image:

```bash
docker image inspect probed-grpo:verl0.5.0 --format '{{.Id}}'
```

### 3. Start a code-execution judge

Choose one option. The service must remain available throughout GRPO training,
including validation.

#### Option A: bundled local Docker judge

In a separate **host** terminal, from this repository root:

```bash
python3 grpo/judge_server.py --pull-image
python3 grpo/judge_server.py --port 8000 --workers 8
```

Wait for `Judge ready at http://127.0.0.1:8000/execute`. Startup tests that Docker
can actually execute Python in the pinned judge image. Leave the terminal running
(use a persistent SSH/tmux session for long jobs).

Each request runs in an ephemeral container with networking disabled, no host
mounts or Docker socket, a read-only root filesystem, an unprivileged user,
and CPU/memory/process/time limits. The server itself runs on the host and needs
Docker access. It listens only on loopback. The training container uses host
networking so it can reach it. The judge runs Python 3.11 standard-library code;
reference solutions in CodeContests may be Python 2, but generated candidates
are explicitly requested in Python 3. Judge containers are limited to 1 GiB and
2 CPUs each; this is a correctness benchmark, not an official Codeforces judge.

In the host terminal that will start the training container:

```bash
export MONOLITH_URL=http://127.0.0.1:8000/execute
export JUDGE_WORKERS=8
```

#### Option B: existing Monolith-compatible service

Use the service's full `/execute` URL instead:

```bash
export MONOLITH_URL=https://YOUR-JUDGE-HOST/execute
export JUDGE_WORKERS=8
```

Replace `YOUR-JUDGE-HOST`; it is a placeholder. The adapter posts Python code,
tests, `language`, `libraries`, `timeout`, and `run_profiling` fields and requires
stdout containing the runner's case counts. A successful health check is required
below. Authentication headers are not implemented by this adapter.

The default public URL is `https://monolith.cool/execute`; its availability is not
guaranteed. Use Option A if no maintained service is available. Do not change judge
implementations or limits between curriculum routes in the same experiment.

### 4. Enter the container and check the environment

On the host, after setting the judge URL:

```bash
bash grpo/container.sh shell
```

All remaining commands run **inside this container**, at
`/workspace/probed-grpo`. That fixed mount keeps model paths in probe metadata
consistent across container restarts. Files created there persist in the host
repository. The wrapper uses the image's default container user, so newly created
host files may be root-owned.

```bash
source grpo/env.sh
set -o pipefail
mkdir -p results/logs
python grpo/preflight.py --stage runtime
python grpo/codeforces_reward.py --check
```

Both must succeed. Runtime preflight imports the pinned packages, checks visible
GPU count/architecture and executes a small CUDA operation on each requested GPU.
Judge preflight verifies both acceptance of a correct answer and rejection of
an incorrect one. Each training launch repeats the runtime, corpus and judge checks.

For a different GPU count, set it **before the runtime check**. For example, with
two visible compatible GPUs:

```bash
export N_GPUS=2
export TENSOR_PARALLEL_SIZE=2
python grpo/preflight.py --stage runtime
```

For one GPU use `N_GPUS=1` and `TENSOR_PARALLEL_SIZE=1`. These settings do not
guarantee memory fit. Keep the same configuration for all routes. Set these again
when entering a new container; shell exports do not survive container removal.

The complete installed package list is recorded in the image. Save a copy:

```bash
cp /opt/probed-grpo/environment.freeze.txt results/environment.freeze.txt
```

### 5. Download and verify the probe inputs

```bash
python artifact_cache.py
python grpo/preflight.py --stage probe
HF_HUB_OFFLINE=1 python -m unittest -v test_artifact_cache.py test_linear_probe.py
VERL_SOURCE_DIR=/opt/verl VERL_INTEGRATION=1 \
  python -m unittest discover -s grpo/tests -v
```

The model and dataset revisions are pinned in `artifact_cache.py`. Downloads use:

| Artifact | Repository location |
| --- | --- |
| Frozen Afterburner checkpoint | `model-cache/afterburner/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start/` |
| Dataset Arrow cache | `data-cache/huggingface/datasets/` |
| Hub downloads and Xet cache | `data-cache/huggingface/hub/`, `data-cache/huggingface/xet/` |
| Full linear probe and its embeddings/results | `model-cache/probe/` |
| GRPO corpora | `grpo/data/` |
| GRPO actor/optimizer checkpoints | `grpo/checkpoints/` |

Complete local model snapshots are reused without a Hub call. Missing default
weights are downloaded from the pinned revision; nested optimizer/trainer snapshots
are excluded. Hugging Face reuses its dataset/download cache. Set `HF_HUB_OFFLINE=1`
for later fully offline reuse after downloads finish; it cannot populate an empty
cache. Do not copy caches generated under arbitrary newer package versions into this
environment. Explicit `HF_*` variables can override cache placement.

Probe-input preflight tokenizes every rated training statement before the expensive
feature extraction. On the audited dataset it finds **6,673** distinct rated problems,
with longest prompt **2,158** tokens, below the 4,096-token limit. A tokenizer-dependent
unit test may skip if run before model download; it must run after this step.

### 6. Train the full linear probe

```bash
python linear_probe.py --device cuda:0 --dtype float16 \
  2>&1 | tee results/logs/probe-full.log
```

Wait for `Saved probe and results to .../model-cache/probe`. This extracts features
from all eligible statements and trains the CPU linear layer for 80 epochs.
The backbone remains frozen. Expected outputs include `probe.pt`, `embeddings.npz`,
`metrics.json`, `predictions.csv`, and `losses.csv` in `model-cache/probe`.

If interrupted, rerun the exact command. Embeddings resume from the last saved
25-row prefix; the inexpensive probe optimization restarts. Do not change model,
precision, record selection or token limit while reusing an embeddings cache.
A mismatch stops with an explanation. Keep the repository/container mount stable.

Do not substitute the 10-problem smoke probe for this stage. Corpus construction
requires the full-run metadata (`max_samples=-1`, at least 80 epochs) by default.

### 7. Build all three corpora

```bash
python grpo/codeforces_corpus.py --device cuda:0 \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  2>&1 | tee results/logs/corpus-full.log
python grpo/preflight.py --stage corpus
```

The builder uses the original CodeContests `train` and `valid` splits. It selects
rated Codeforces problems with reference Python code and tests, filters the shared
set to the GRPO prompt limit, and trims that common set to a whole number of training
batches **before** applying random, official-rating and probe-score ordering. Thus
`verl`'s `drop_last=True` cannot discard a different tail for each route. The shared
validation set is filtered to the same prompt limit and is not batch-trimmed.

It writes:

```text
grpo/data/
  random/train.parquet
  official/train.parquet
  probed/train.parquet
  shared/validation.parquet
  manifest.json
  probe_scores.jsonl
```

`manifest.json` records selection counts, excluded overlong prompts, common tail
trimming, model path, probe hash, dataset revision, row orders and parquet hashes.
Preflight verifies the files and that all routes contain identical records. The
GRPO row count is smaller than the probe's 6,673 because its selection/prompt differs.
With the pinned tokenizer, a 2,048-token prompt limit and batch size 32, the audited
data yields **5,024 training rows and 72 validation rows**: 5,228 initial training
rows minus 198 overlong prompts and a common 6-row tail; 76 initial validation rows
minus 4 overlong prompts. Reference code can make some GRPO prompts much longer
than the statement-only probe prompt, which is why this shared filtering matters.

Scoring resumes from `probe_scores.jsonl` after interruption. If the probe, statements
or preparation settings change, use a new corpus output directory. All routes must
be rebuilt together after changing batch size or prompt length. To use a new corpus
directory, pass `--output-dir` to the builder and export `CODECONTESTS_GRPO_DATA_DIR`
to the same absolute path before training.

### 8. Run one actual GRPO optimizer step

First inspect the final composed configuration without launching training:

```bash
bash grpo/train.sh --config-only random
```

Then run a one-step smoke job **using the full corpus and full training settings**:

```bash
AFTERBURNER_CHECKPOINT_DIR="$PWD/grpo/checkpoints/smoke" \
  bash grpo/train.sh random \
    trainer.total_training_steps=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.experiment_name=codecontests-smoke \
  2>&1 | tee results/logs/grpo-smoke.log
```

Success requires exit status 0, a completed training step (not just model loading),
and `grpo/checkpoints/smoke/random-seed-42/global_step_1/actor/` containing checkpoint
files. This checks the real GPU allocation, vLLM rollout, judge round trip, GRPO
loss/backward step and checkpoint saving. `verl` may also perform end-of-run
validation; allow the judge to remain running until the command exits.

If it fails, use the troubleshooting table below and repeat it. Do not start the
200-epoch experiment until this step succeeds on the mentor's actual machine.

### 9. Train GRPO for each curriculum

After the smoke step succeeds, run the three full jobs sequentially:

```bash
bash grpo/run_three_routes.sh all \
  2>&1 | tee results/logs/grpo-all.log
```

Or run them separately:

```bash
bash grpo/run_three_routes.sh random 2>&1 | tee results/logs/grpo-random.log
bash grpo/run_three_routes.sh official 2>&1 | tee results/logs/grpo-official.log
bash grpo/run_three_routes.sh probed 2>&1 | tee results/logs/grpo-probed.log
```

Use one of these alternatives, not both. Each route starts from the same cold-start
checkpoint and saves to a separate directory (`random-seed-42`, `official-seed-42`,
`probed-seed-42`). The default is 200 epochs, saving/validation every 10 steps.
Console logging needs no W&B account. To opt into W&B, authenticate inside the
container and append `trainer.logger='[console,wandb]'` consistently to every route.

The launcher's effective config is saved in `results/launch-ROUTE.yaml`; keep it,
the logs, environment freeze, probe metrics and corpus manifest with the experiment.
`verl` saves model, optimizer and data-loader state for resumption. Its default
`trainer.resume_mode=auto` resumes the most recent checkpoint in each route directory.
After an interruption, rerun only the interrupted route's exact command. Do not
reuse that directory for changed hyperparameters; set a new `AFTERBURNER_CHECKPOINT_DIR`.
Running `all` again also revisits earlier routes, so use individual commands when
some routes have already completed.

### 10. Export a trained checkpoint

GRPO checkpoints are FSDP shards, not immediately usable Hugging Face directories.
Choose an existing `global_step_N` for the route and replace `N` below:

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir grpo/checkpoints/random-seed-42/global_step_N/actor \
  --target_dir model-cache/grpo-random
```

Repeat with different output directories for `official` and `probed`. This exports
locally; it does not upload weights. Retain the original shards if resuming training.

### Troubleshooting

| Failure | Action |
| --- | --- |
| CUDA unavailable / wrong GPU count | Check host `nvidia-smi`, NVIDIA Container Toolkit, Docker GPU access, `N_GPUS` and visible devices. |
| Version/import check fails | Rebuild the pinned image and enter it; do not repair it with unpinned package upgrades. |
| Judge connection refused | Keep the host judge terminal running; use `http://127.0.0.1:8000/execute` with this wrapper's host networking. |
| Judge busy / timeout | Keep `JUDGE_WORKERS` at or below server workers; check host CPU/RAM capacity and the judge log. Infrastructure failures abort training rather than become incorrect-answer rewards. |
| Out of GPU memory | The defaults already use one example per actor/log-prob micro-batch. Use higher-memory/more GPUs or deliberate offloading overrides. Changing response length, rollout count or batch size changes experiment settings: use the same values for all routes and rerun the smoke test. Rebuild corpora if batch size/prompt limit changes. |
| Missing probe / smoke probe rejected | Complete step 6 and use `model-cache/probe/probe.pt` with its full-run metrics. |
| Embedding/score-cache mismatch | Use a new output directory for changed settings; keep previous artifacts for provenance. |
| Corpus hash/config mismatch | Rebuild all routes together and verify that trainer batch/prompt settings match preparation. |
| Disk fills | Inspect checkpoint retention and Docker cache usage; archive completed outputs intentionally. Do not blindly remove active checkpoints/caches. |

### Verification record and limits

The setup was audited on 2026-09-11. Local checks use pinned PyTorch 2.6.0,
Transformers 4.51.3 and datasets 4.0.0; the real `verl` 0.5.0 dataset loader,
collation, DataProto and batch reward manager are exercised. All three launchers
are composed against that release's real Hydra schema. Download defaults,
cache reuse, parquet checksums, shared selection, judge protocol and cleanup
behavior have regression tests. Full probe input tokenization was checked offline.
The real cold-start backbone also loaded on CPU and produced a finite 2,048-value
embedding under the pinned PyTorch/Transformers versions. The local suite passed
11 cache/probe tests and 19 GRPO/handoff tests, including actual upstream integration.

The development machine has no Linux CUDA runtime and no running Docker daemon.
The Docker image build, actual judge container execution and actual GRPO optimizer
step therefore remain **host acceptance checks**, not completed GPU validation.
Steps 2, 4 and 8 make those checks explicit. A successful unit suite alone is not
evidence that an arbitrary GPU machine can complete the full experiment.

Upstream evidence: [verl v0.5 installation](https://verl.readthedocs.io/en/v0.5.x/start/install.html),
[pinned batch reward interface](https://github.com/volcengine/verl/blob/8fdc4d3f202f41461f4de9f42a637228e342668b/verl/workers/reward_manager/batch.py),
[pinned trainer and drop-last behavior](https://github.com/volcengine/verl/blob/8fdc4d3f202f41461f4de9f42a637228e342668b/verl/trainer/ppo/ray_trainer.py),
[checkpoint export](https://github.com/volcengine/verl/blob/8fdc4d3f202f41461f4de9f42a637228e342668b/docs/advance/checkpoint.rst).

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

For GRPO use the pinned Linux/CUDA workflow under
[Complete mentor workflow](#complete-mentor-workflow). The probe-only requirements
do not install `verl`, vLLM or a CUDA environment.

## Local artifacts

Missing Hugging Face models are downloaded into `model-cache`; dataset downloads
use `data-cache/huggingface/datasets`; and the trained linear probe is stored in
`model-cache/probe`. These caches, checkpoints, logs, and generated results stay
local by default. The tracked `results/.gitkeep` preserves the results directory
without committing run artifacts. If a small artifact must become part of the
research record, add it intentionally with `git add -f` and document its provenance.
