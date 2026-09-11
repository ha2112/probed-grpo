"""Resolve model artifacts into this repository's local cache."""

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Set these before importing datasets/transformers/huggingface_hub.
os.environ.setdefault("HF_HOME", str(ROOT / "data-cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "data-cache/huggingface/hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "data-cache/huggingface/datasets"))
os.environ.setdefault("HF_XET_CACHE", str(ROOT / "data-cache/huggingface/xet"))
MODEL_CACHE = ROOT / "model-cache"
DEFAULT_MODEL_ID = "Elfsong/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start"
DEFAULT_MODEL_REVISION = "de85a42c256f2d9deb6eb45ce940a006734dfaa6"
DEFAULT_DATASET_ID = "deepmind/code_contests"
DEFAULT_DATASET_REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"
DEFAULT_MODEL_RELATIVE = Path("afterburner/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start")
DEFAULT_MODEL_DIR = MODEL_CACHE / DEFAULT_MODEL_RELATIVE


def _weight_files_from_index(index_path):
    try:
        contents = json.loads(index_path.read_text(encoding="utf-8"))
        return set(contents["weight_map"].values())
    except (KeyError, OSError, TypeError, ValueError):
        return set()


def model_snapshot_available(path):
    """Return whether a local Transformers snapshot has its required files."""
    path = Path(path)
    if not path.is_dir() or not all(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in ("config.json", "tokenizer_config.json")
    ):
        return False
    if not any((path / name).is_file() for name in (
        "tokenizer.json", "tokenizer.model", "vocab.json",
    )):
        return False
    if any((path / name).is_file() and (path / name).stat().st_size > 0
           for name in ("model.safetensors", "pytorch_model.bin")):
        return True
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if index_path.is_file():
            weight_files = _weight_files_from_index(index_path)
            return bool(weight_files) and all(
                (path / name).is_file() and (path / name).stat().st_size > 0 for name in weight_files
            )
    return False


def _is_hugging_face_id(value):
    value = str(value)
    parts = value.split("/")
    return (
        not value.startswith(("/", ".", "~"))
        and len(parts) == 2
        and all(part not in {"", ".", ".."} for part in parts)
    )


def ensure_model(model=DEFAULT_MODEL_DIR, model_cache=MODEL_CACHE):
    """Reuse a complete local model, or download a Hub model into model-cache."""
    value = str(model)
    local_path = Path(value).expanduser()
    if model_snapshot_available(local_path):
        return local_path.resolve()

    if local_path.resolve() == DEFAULT_MODEL_DIR.resolve():
        repo_id = DEFAULT_MODEL_ID
        destination = local_path
    elif value == DEFAULT_MODEL_ID:
        repo_id = DEFAULT_MODEL_ID
        destination = Path(model_cache).expanduser() / DEFAULT_MODEL_RELATIVE
    elif local_path.exists():
        raise FileNotFoundError(f"Local model path is incomplete: {local_path}")
    elif _is_hugging_face_id(value):
        repo_id = value
        destination = Path(model_cache).expanduser() / "huggingface" / value
    else:
        state = "incomplete" if local_path.exists() else "missing"
        raise FileNotFoundError(f"Local model path is {state}: {local_path}")

    if model_snapshot_available(destination):
        return destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    print(
        f"Model not available locally; downloading {repo_id} to {destination}",
        file=sys.stderr,
        flush=True,
    )
    from huggingface_hub import snapshot_download

    options = {}
    if repo_id == DEFAULT_MODEL_ID:
        options["revision"] = DEFAULT_MODEL_REVISION
        # Exclude the nested optimizer/trainer checkpoint and training logs.
        options["allow_patterns"] = [
            "config.json", "generation_config.json", "tokenizer*.json", "*.model",
            "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
            "model*.safetensors", "model.safetensors.index.json",
        ]
    snapshot_download(repo_id=repo_id, local_dir=str(destination), **options)
    if not model_snapshot_available(destination):
        raise RuntimeError(f"Downloaded model snapshot is incomplete: {destination}")
    if repo_id == DEFAULT_MODEL_ID:
        (destination / "SOURCE_REVISION").write_text(DEFAULT_MODEL_REVISION + "\n")
    return destination.resolve()


def load_cached_dataset(dataset, split, cache_dir):
    from datasets import load_dataset

    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    revision = DEFAULT_DATASET_REVISION if dataset == DEFAULT_DATASET_ID else None
    return load_dataset(dataset, split=split, cache_dir=str(cache_dir), revision=revision)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", default=str(DEFAULT_MODEL_DIR),
        help="Complete local snapshot or Hugging Face owner/model ID",
    )
    args = parser.parse_args(argv)
    print(ensure_model(args.model))


if __name__ == "__main__":
    main()
