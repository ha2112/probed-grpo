"""Train a coding-difficulty linear probe on the frozen Afterburner model."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model-cache/afterburner/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start"
SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Solve the following programming "
    "problem in Python, respecting its input/output format and constraints. "
    "Enclose your reasoning in <thinking> </thinking> and your complete solution "
    "in <solution> </solution>. Put the solution code in one markdown code block "
    "with the python language identifier."
)
SEED = 42


def build_messages(description):
    # Ratings, reference solutions, and test answers never enter the prompt.
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": description},
    ]


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_backbone(model_path, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # AutoModel returns the same final normalized hidden state without LM logits.
    model = AutoModel.from_pretrained(model_path, torch_dtype=getattr(torch, dtype))
    model.to(device).eval().requires_grad_(False)
    return tokenizer, model


def embed_problem(description, tokenizer, model, device, max_length):
    inputs = tokenizer.apply_chat_template(
        build_messages(description), tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(device)
    # Do not truncate away the assistant marker or silently change the problem.
    limit = min(max_length, model.config.max_position_embeddings)
    if inputs["input_ids"].shape[1] > limit:
        raise ValueError(f"Prompt has {inputs['input_ids'].shape[1]} tokens, exceeding limit {limit}")
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
    embedding = outputs.last_hidden_state[0, -1].float().cpu().numpy().copy()
    if not np.isfinite(embedding).all():
        raise ValueError("Model produced a non-finite embedding")
    return embedding


def load_records(args):
    # The random 64/16/20 partition uses only CodeContests' original train split.
    dataset = load_dataset(args.data, split="train", cache_dir=str(args.data_cache))
    dataset = dataset.select_columns(["name", "description", "source", "cf_rating"])
    records = [
        {"name": row["name"], "description": row["description"],
         "rating": float(row["cf_rating"])}
        for row in dataset
        if row["source"] == 2 and row["cf_rating"] is not None
        and np.isfinite(row["cf_rating"]) and row["cf_rating"] > 0
    ]
    frame = pd.DataFrame(records, columns=["name", "description", "rating"])
    frame = frame.sort_values("rating", kind="stable").reset_index(drop=True)
    # Identical statements must not occur on both sides of the probe split.
    frame = frame.drop_duplicates("description").reset_index(drop=True)
    if args.max_samples > 0 and args.max_samples < len(frame):
        frame = frame.sample(args.max_samples, random_state=SEED)
        frame = frame.sort_values("rating", kind="stable").reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("At least 10 rated, distinct Codeforces problems are required")
    return frame.to_dict("records")


def cache_metadata(args, records, dtype):
    path = Path(args.model).expanduser()
    model_id = str(path.resolve()) if path.exists() else args.model
    # Detect replacement of a local checkpoint without hashing GB of weights.
    files = []
    if path.is_dir():
        for item in sorted(path.iterdir()):
            if item.is_file() and item.suffix in {".json", ".safetensors", ".bin", ".txt"}:
                stat = item.stat()
                files.append([item.name, stat.st_size, stat.st_mtime_ns])
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"version": 1, "model": model_id, "model_files": files,
            "prompt": SYSTEM_PROMPT, "feature": "final_layer_last_prompt_token",
            "dtype": dtype, "max_length": args.max_length, "records_sha256": digest}


def extract_embeddings(args, records, device, dtype):
    metadata = cache_metadata(args, records, dtype)
    cache = args.output_dir / "embeddings.npz"
    embeddings = []
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(saved["metadata"].item()) != metadata:
                raise ValueError("Embedding cache does not match this run; use a new --output-dir")
            embeddings = list(saved["embeddings"])
        if len(embeddings) > len(records) or not np.isfinite(embeddings).all():
            raise ValueError("Invalid embedding cache; use a new --output-dir")
        print(f"Resuming embeddings: {len(embeddings)}/{len(records)}", flush=True)
    if len(embeddings) < len(records):
        tokenizer, model = load_backbone(args.model, device, dtype)
        for index in range(len(embeddings), len(records)):
            try:
                embedding = embed_problem(records[index]["description"], tokenizer,
                                          model, device, args.max_length)
            except ValueError as error:
                raise ValueError(f"{records[index]['name']}: {error}") from error
            embeddings.append(embedding)
            print(f"Embedded {len(embeddings)}/{len(records)}: {records[index]['name']}", flush=True)
            # Save a contiguous prefix atomically; interrupted runs resume safely.
            if len(embeddings) % 25 == 0 or len(embeddings) == len(records):
                temporary = cache.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, embeddings=np.stack(embeddings),
                             metadata=json.dumps(metadata, sort_keys=True))
                temporary.replace(cache)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    return np.stack(embeddings).astype(np.float32), metadata


def split_indices(count):
    train_full, test = train_test_split(np.arange(count), test_size=0.2, random_state=SEED)
    boundary = int(0.8 * len(train_full))
    return train_full[:boundary], train_full[boundary:], test


def fold_scalers(probe, feature_scaler, target_scaler):
    # y = sy * (W * ((x - mx) / sx) + b) + my.
    folded = copy.deepcopy(probe).cpu().eval()
    with torch.no_grad():
        folded.weight.div_(torch.tensor(feature_scaler.scale_, dtype=torch.float32))
        folded.bias.sub_((folded.weight * torch.tensor(
            feature_scaler.mean_, dtype=torch.float32)).sum(-1))
        folded.weight.mul_(float(target_scaler.scale_[0]))
        folded.bias.mul_(float(target_scaler.scale_[0]))
        folded.bias.add_(float(target_scaler.mean_[0]))
    return folded


def metrics(prediction, target):
    pred, real = pd.Series(prediction.ravel()), pd.Series(target.ravel())
    correlated = len(pred) > 2 and pred.nunique() > 1 and real.nunique() > 1
    return {
        "n": len(pred), "rmse": float(np.sqrt(np.mean((pred - real) ** 2))),
        "mae": float(np.mean(np.abs(pred - real))),
        "pearson": float(pred.corr(real)) if correlated else None,
        "spearman": float(pred.rank().corr(real.rank())) if correlated else None,
    }


def fit_probe(features, targets, epochs):
    torch.manual_seed(SEED)
    train, val, test = split_indices(len(features))
    # Fit on training rows only; keep validation and test statistics held out.
    x_scaler = StandardScaler().fit(features[train])
    y_scaler = StandardScaler().fit(targets[train])
    x = torch.from_numpy(x_scaler.transform(features).astype(np.float32))
    y = torch.from_numpy(y_scaler.transform(targets).astype(np.float32))
    loader = DataLoader(TensorDataset(x[train], y[train]), batch_size=32, shuffle=True)
    # Only this small linear layer trains; CPU avoids retaining GPU model memory.
    probe = nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=5e-4, weight_decay=2e-4)
    criterion = nn.MSELoss()
    best_loss, best_state, best_epoch = float("inf"), None, None
    history = []
    for epoch in range(1, epochs + 1):
        probe.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        probe.eval()
        with torch.no_grad():
            val_loss = criterion(probe(x[val]), y[val]).item()
        if not np.isfinite(val_loss):
            raise ValueError("Probe validation loss is non-finite")
        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, copy.deepcopy(probe.state_dict()), epoch
        history.append({"epoch": epoch, "train_mse": total / len(train), "val_mse": val_loss})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}: train={total / len(train):.5f} val={val_loss:.5f}", flush=True)
    probe.load_state_dict(best_state)
    return fold_scalers(probe, x_scaler, y_scaler), (train, val, test), history, best_epoch


def train(args, device, dtype):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args)
    features, metadata = extract_embeddings(args, records, device, dtype)
    targets = np.array([row["rating"] for row in records], dtype=np.float32).reshape(-1, 1)
    probe, splits, history, best_epoch = fit_probe(features, targets, args.epochs)
    with torch.inference_mode():
        prediction = probe(torch.from_numpy(features)).numpy()
    report = {"best_epoch": best_epoch, "epochs": args.epochs, "seed": SEED,
              "batch_size": 32, "lr": 5e-4, "weight_decay": 2e-4,
              "normalization_fit": "train_only", "metadata": metadata}
    rows = []
    for name, indices in zip(("train", "validation", "test"), splits):
        report[name] = metrics(prediction[indices], targets[indices])
        for index in indices:
            rows.append({"name": records[index]["name"], "split": name,
                         "real_difficulty": float(targets[index, 0]),
                         "pred_difficulty": float(prediction[index, 0])})
    torch.save({"state_dict": probe.state_dict(), "hidden_size": features.shape[1],
                "metadata": metadata}, args.output_dir / "probe.pt")
    pd.DataFrame(rows).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "losses.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("best_epoch", "validation", "test")}, indent=2))
    print(f"Saved probe and results to {args.output_dir}")


def predict(args, device):
    checkpoint = torch.load(args.output_dir / "probe.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    # Prediction uses exactly the checkpoint's model, prompt, dtype, and token limit.
    if metadata["prompt"] != SYSTEM_PROMPT:
        raise ValueError("Checkpoint prompt differs from this script")
    args.model, args.max_length = metadata["model"], metadata["max_length"]
    current = cache_metadata(args, [], metadata["dtype"])
    if current["model_files"] != metadata["model_files"]:
        raise ValueError("Local model files changed since this probe was trained")
    tokenizer, model = load_backbone(args.model, device, metadata["dtype"])
    embedding = embed_problem(args.predict_file.read_text(), tokenizer, model, device, args.max_length)
    probe = nn.Linear(checkpoint["hidden_size"], 1)
    probe.load_state_dict(checkpoint["state_dict"])
    with torch.inference_mode():
        score = probe(torch.from_numpy(embedding)).item()
    print(json.dumps({"pred_difficulty": score, "scale": "Codeforces rating"}))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Local checkpoint or Hugging Face model ID")
    parser.add_argument("--data", default="deepmind/code_contests")
    parser.add_argument("--data-cache", type=Path, default=ROOT / "data-cache/huggingface/datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/linear-probe-afterburner")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--max-length", type=int, default=4096, help="Fail on longer prompts; never silently truncate")
    parser.add_argument("--max-samples", type=int, default=-1, help="Seeded sample cap; -1 uses all rated problems")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--predict-file", type=Path, help="Score a UTF-8 problem statement with a saved probe")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_length < 1 or (args.max_samples != -1 and args.max_samples < 10):
        parser.error("epochs/max-length must be positive; max-samples must be -1 or at least 10")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    selected_device = resolve_device(arguments.device)
    selected_dtype = arguments.dtype or ("float32" if selected_device == "cpu" else "float16")
    if arguments.predict_file:
        predict(arguments, selected_device)
    else:
        train(arguments, selected_device, selected_dtype)
