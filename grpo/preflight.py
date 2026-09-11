"""Check the pinned runtime, corpora and effective verl configuration before training."""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artifact_cache import model_snapshot_available  # noqa: E402
from grpo.codeforces_corpus import ROUTES, probe_fingerprint  # noqa: E402

VERL_COMMIT = "8fdc4d3f202f41461f4de9f42a637228e342668b"
RUNTIME_VERSIONS = {"torch": "2.6.0", "vllm": "0.8.5.post1", "verl": "0.5.0"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_runtime(gpu_count):
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "The pinned GRPO image requires Linux x86_64 with NVIDIA GPUs.")
    require(sys.version_info[:2] == (3, 10), "Use Python 3.10 from grpo/Dockerfile.")
    expected = dict(RUNTIME_VERSIONS)
    for line in (ROOT / "requirements-grpo.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==")
            expected[name.split("[")[0]] = version
    versions = {name: importlib.metadata.version(name) for name in expected}
    for name, version in expected.items():
        require(versions[name].split("+")[0] == version,
                f"{name}: expected {version}, found {versions[name]}; rebuild the pinned image.")
    for module in ("torch", "vllm", "flash_attn", "verl.trainer.main_ppo",
                   "verl.workers.reward_manager.batch", "pyarrow", "sklearn"):
        importlib.import_module(module)
    import torch
    import verl

    source_dir = Path(verl.__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
    require(revision == VERL_COMMIT, "Installed verl source differs from the pinned commit")

    require(gpu_count > 0, "N_GPUS must be positive")
    require(torch.cuda.is_available(), "CUDA is unavailable inside this container.")
    require(torch.cuda.device_count() >= gpu_count,
            f"Requested {gpu_count} GPUs, but only {torch.cuda.device_count()} are visible.")
    devices = []
    for index in range(gpu_count):
        props = torch.cuda.get_device_properties(index)
        require(8 <= props.major < 10,
                f"GPU {index} ({props.name}) is outside this image's Ampere/Ada/Hopper target.")
        # Exercise the actual CUDA installation before a long preparation run.
        tensor = torch.ones((16, 16), device=f"cuda:{index}")
        require(float((tensor @ tensor)[0, 0]) == 16.0, "CUDA matrix multiplication failed")
        devices.append({"name": props.name, "memory_gib": round(props.total_memory / 2**30, 1)})
    print(json.dumps({"versions": versions, "verl_commit": revision, "gpus": devices}, indent=2))


def check_probe_inputs():
    import linear_probe
    from artifact_cache import ensure_model
    from transformers import AutoConfig, AutoTokenizer

    args = linear_probe.parse_args([])
    args.model = str(ensure_model(args.model))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    records = linear_probe.load_records(args)
    limit = min(args.max_length, model_config.max_position_embeddings)
    longest = 0
    for row in records:
        tokens = tokenizer.apply_chat_template(linear_probe.build_messages(row["description"]),
                                               tokenize=True, add_generation_prompt=True)
        longest = max(longest, len(tokens))
        require(len(tokens) <= limit, f"Probe statement {row['name']} exceeds {limit} tokens")
    print(f"Probe inputs verified: {len(records)} distinct rated problems; longest prompt {longest}/{limit} tokens")


def compose_config(overrides, config_dir=None):
    from hydra import compose, initialize_config_dir

    if config_dir is None:
        distribution = importlib.metadata.distribution("verl")
        # Editable installs are used in the pinned image.
        import verl
        config_dir = Path(verl.__file__).parent / "trainer/config"
        require(distribution.version == "0.5.0", "Use the pinned verl 0.5.0 environment")
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)


def validate_config(config):
    c = config
    gpus = c.trainer.n_gpus_per_node
    tp = c.actor_rollout_ref.rollout.tensor_model_parallel_size
    batch = c.data.train_batch_size
    mini = c.actor_rollout_ref.actor.ppo_mini_batch_size
    micro = c.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
    n = c.actor_rollout_ref.rollout.n
    require(c.trainer.nnodes == 1, "This launcher supports a single GPU node")
    require(all(isinstance(x, int) and x > 0 for x in (gpus, tp, batch, mini, micro, n)),
            "GPU, tensor parallel and batch sizes must be positive integers")
    require(n >= 2, "GRPO needs at least two sampled responses per prompt")
    require(gpus % tp == 0, "N_GPUS must be divisible by TENSOR_PARALLEL_SIZE")
    require(batch >= mini and batch % mini == 0, "TRAIN_BATCH_SIZE must be a multiple of PPO_MINI_BATCH_SIZE")
    require((mini * n) % (gpus * micro) == 0,
            "PPO mini-batch × rollout count must divide evenly across GPUs and micro-batches")
    require(not c.data.shuffle and not c.data.validation_shuffle,
            "Curriculum comparison requires shuffle=False for training and validation")
    require(not c.actor_rollout_ref.actor.shuffle and not c.trainer.balance_batch,
            "Actor shuffle and trainer.balance_batch must be False to preserve curriculum order")
    require(not c.data.filter_overlong_prompts and c.data.truncation == "error",
            "Build the shared prompt-filtered corpus first; runtime filtering/truncation changes row sets")
    require(c.algorithm.adv_estimator == "grpo" and c.reward_model.reward_manager == "batch",
            "This pipeline requires GRPO and the batch reward manager")
    require(c.custom_reward_function.name == "codeforces_reward_fn_batch",
            "The CodeContests batch reward function is required")
    require(Path(c.custom_reward_function.path).resolve() == ROOT / "grpo/codeforces_reward.py",
            "Unexpected reward implementation path")
    require(c.data.max_prompt_length > 0 and c.data.max_response_length > 0, "Token limits must be positive")
    require(c.trainer.total_epochs > 0, "TOTAL_EPOCHS must be positive")


def verify_corpora(data_dir, train_batch_size=None, max_prompt_length=None):
    import pyarrow.parquet as pq

    manifest_path = data_dir / "manifest.json"
    require(manifest_path.is_file(), f"Missing {manifest_path}; build the corpora first")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema_version") == 2, "Rebuild corpora with the current code (manifest v2 required)")
    preparation = manifest["preparation"]
    if train_batch_size is not None:
        require(preparation.get("train_batch_size") == train_batch_size,
                "Batch size differs from corpus preparation; rebuild all routes with that batch size")
    if max_prompt_length is not None:
        require(preparation.get("max_prompt_length") == max_prompt_length,
                "Prompt limit differs from corpus preparation; rebuild all routes with that limit")
    canonical = None
    for relative in [*(f"{route}/train.parquet" for route in ROUTES), "shared/validation.parquet"]:
        path = data_dir / relative
        require(path.is_file(), f"Missing corpus file: {path}")
        require(probe_fingerprint(path) == manifest["files_sha256"].get(relative),
                f"Corpus checksum mismatch: {path}; rebuild it")
        rows = pq.read_table(path).to_pylist()
        require(len(rows) > 0, f"Empty corpus: {path}")
        ids = [row["extra_info"]["problem_id"] for row in rows]
        require(len(ids) == len(set(ids)), f"Duplicate problem IDs in {path}")
        for row in rows:
            truth = json.loads(row["reward_model"]["ground_truth"])
            require(len(truth["tests"]) > 0, "Every training problem must have tests")
        if relative.startswith("shared/"):
            require(ids == manifest["validation_order"], "Validation order differs from manifest")
            require(len(rows) == manifest["validation_count"], "Validation count differs from manifest")
        else:
            route = relative.split("/")[0]
            require(ids == manifest["route_orders"][route], f"{route}: order differs from manifest")
            require(len(rows) == manifest["train_count"], "Training count differs from manifest")
            size = train_batch_size or preparation.get("train_batch_size", 1)
            require(len(rows) % size == 0, "Training corpus has an incomplete batch")
            contents = sorted(json.dumps(row, sort_keys=True) for row in rows)
            if canonical is not None:
                require(contents == canonical, "Routes contain different records")
            canonical = contents
    print(f"Corpora verified: {manifest['train_count']} train / {manifest['validation_count']} validation; identical route contents")
    return manifest


def check_training(config, check_judge=True):
    validate_config(config)
    check_runtime(config.trainer.n_gpus_per_node)
    train_files = list(config.data.train_files)
    validation_files = list(config.data.val_files)
    require(len(train_files) == len(validation_files) == 1, "Expected one train and one validation parquet")
    train_path = Path(train_files[0]).resolve()
    data_dir = train_path.parents[1]
    require(train_path.parent.name in ROUTES and train_path.name == "train.parquet", "Invalid route path")
    require(Path(validation_files[0]).resolve() == data_dir / "shared/validation.parquet",
            "All routes must use the same validation corpus")
    manifest = verify_corpora(data_dir, config.data.train_batch_size, config.data.max_prompt_length)
    if manifest["preparation"].get("smoke"):
        require(config.trainer.total_training_steps == 1,
                "Smoke-probe corpora are allowed only for a one-step training smoke test")
    model_path = Path(config.actor_rollout_ref.model.path).resolve()
    require(model_snapshot_available(model_path), f"Incomplete model: {model_path}")
    require(Path(manifest["preparation"]["model"]).resolve() == model_path,
            "Model path differs from corpus tokenizer; build and train in the same container mount")
    model_config = json.loads((model_path / "config.json").read_text())
    require(config.data.max_prompt_length + config.data.max_response_length <= model_config["max_position_embeddings"],
            "Prompt plus response exceeds the model context window")
    require(model_config["num_attention_heads"] % config.actor_rollout_ref.rollout.tensor_model_parallel_size == 0,
            "Tensor parallel size must divide the model's attention heads")
    # Exercise the actual verl parquet/tokenizer boundary on both splits.
    from transformers import AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from grpo.codeforces_reward import codeforces_reward_fn_batch

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    for paths in (train_files, validation_files):
        dataset = RLHFDataset(paths, tokenizer, config.data)
        batch = collate_fn([dataset[0], dataset[min(1, len(dataset) - 1)]])
        # Empty responses exercise collation without making network requests.
        scores = codeforces_reward_fn_batch(
            batch["data_source"], ["", ""],
            [item["ground_truth"] for item in batch["reward_model"]], batch["extra_info"],
        )
        require(len(scores) == 2, "The verl reward interface check failed")
    if check_judge:
        from grpo.codeforces_reward import check_judge as judge
        judge()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("runtime", "probe", "corpus", "training", "config"), default="runtime")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "grpo/data")
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("N_GPUS", "8")))
    parser.add_argument("--record", type=Path, help="Save the effective config after successful preflight")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.stage == "runtime":
            check_runtime(args.gpus)
        elif args.stage == "probe":
            check_probe_inputs()
        elif args.stage == "corpus":
            verify_corpora(args.data_dir)
        else:
            overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
            config = compose_config(overrides)
            validate_config(config)
            if args.stage == "training":
                check_training(config)
            from omegaconf import OmegaConf
            if args.record:
                args.record.parent.mkdir(parents=True, exist_ok=True)
                args.record.write_text(OmegaConf.to_yaml(config, resolve=True))
            if args.stage == "config":
                print(OmegaConf.to_yaml(config, resolve=True))
        print("Preflight passed")
    except Exception as error:
        raise SystemExit(f"Preflight failed: {error}") from error


if __name__ == "__main__":
    main()
