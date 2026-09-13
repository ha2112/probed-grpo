"""Research-grade 5-group difficulty predictor.

Method: LoRA-adapted Afterburner encoder, learned whole-sequence attention
pooling, and multi-task classification/regression/ordinal/ranking supervision.
The final 5 groups are decoded by thresholds fitted on validation only.

This is intentionally separate from group_probe.py and group_probe_lora.py so
all baselines remain reproducible.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

import group_probe as gp
import linear_probe as lp


NUM_GROUPS = 5
SEED = 42
DEFAULT_OUTPUT_DIR = lp.ROOT / "model-cache/group-probe-top"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(args):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
        return True, dist.get_rank(), dist.get_world_size(), device
    device = torch.device(lp.resolve_device(args.device))
    if device.type != "cuda":
        raise SystemExit("This model requires CUDA; use --device cuda:0 or torchrun.")
    return False, 0, 1, device


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


class ProblemDataset(Dataset):
    def __init__(self, records, ratings, groups, indices, rating_mean, rating_std):
        self.records = [records[int(i)] for i in indices]
        self.ratings = np.asarray(ratings, dtype=np.float32)[indices]
        self.groups = np.asarray(groups, dtype=np.int64)[indices]
        self.rating_z = (self.ratings - rating_mean) / rating_std
        self.coral = gp.coral_targets(self.groups, NUM_GROUPS)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {
            "description": self.records[index]["description"],
            "name": self.records[index]["name"],
            "group": int(self.groups[index]),
            "rating": float(self.ratings[index]),
            "rating_z": float(self.rating_z[index]),
            "coral": self.coral[index],
        }


def identity_collate(batch):
    return batch


def tokenize_batch(tokenizer, descriptions, device, max_length):
    encoded = [
        tokenizer.apply_chat_template(
            lp.build_messages(text),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        for text in descriptions
    ]
    lengths = [int(row["input_ids"].shape[1]) for row in encoded]
    if max(lengths) > max_length:
        index = int(np.argmax(lengths))
        raise ValueError(
            f"Prompt {index} has {lengths[index]} tokens, exceeding {max_length}"
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    width = max(lengths)
    ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for index, row in enumerate(encoded):
        sequence = row["input_ids"][0]
        ids[index, : len(sequence)] = sequence
        mask[index, : len(sequence)] = 1
    return ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)


def lora_targets(num_layers, last_n_layers):
    start = max(0, int(num_layers) - int(last_n_layers))
    names = []
    for layer in range(start, int(num_layers)):
        names.extend(
            f"layers.{layer}.self_attn.{name}"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        names.extend(
            f"layers.{layer}.mlp.{name}"
            for name in ("gate_proj", "up_proj", "down_proj")
        )
    return names, list(range(start, int(num_layers)))


class TopGroupModel(nn.Module):
    """Whole-sequence attentive pooling with four complementary objectives."""

    def __init__(self, backbone, hidden_size, dropout=0.15):
        super().__init__()
        self.backbone = backbone
        backbone_dtype = next(backbone.parameters()).dtype
        self.attention_pool = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1, bias=False),
        ).to(dtype=backbone_dtype)
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        width = hidden_size // 2
        self.classifier = nn.Linear(width, NUM_GROUPS)
        self.ordinal = MonotonicOrdinalHead(width, NUM_GROUPS)
        self.regressor = nn.Linear(width, 1)

    def forward(self, input_ids, attention_mask):
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = output.last_hidden_state
        attention_logits = self.attention_pool(hidden).squeeze(-1)
        attention_logits = attention_logits.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(attention_logits.float(), dim=-1).to(hidden.dtype)
        attentive = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        last_indices = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        last = hidden[batch_indices, last_indices]
        # Only pooled O(B*H) features become FP32, not the O(B*T*H) sequence.
        feature = self.trunk(torch.cat([attentive, last], dim=-1).float())
        return {
            "class_logits": self.classifier(feature),
            "coral_logits": self.ordinal(feature),
            "rating_z": self.regressor(feature).squeeze(-1),
            "attention": weights,
        }


class MonotonicOrdinalHead(nn.Module):
    """One latent score with strictly ordered learned cutpoints."""

    def __init__(self, input_size, num_groups):
        super().__init__()
        self.score = nn.Linear(input_size, 1)
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.0))
        self.raw_increments = nn.Parameter(torch.zeros(num_groups - 2))

    def cutpoints(self):
        increments = F.softplus(self.raw_increments) + 1e-4
        return torch.cat(
            [self.first_cutpoint.unsqueeze(0), self.first_cutpoint + increments.cumsum(0)]
        )

    def forward(self, features):
        # P(y > k) decreases monotonically as threshold k increases.
        return self.score(features) - self.cutpoints().unsqueeze(0)


def build_model(args, device, dtype):
    model_path = lp.ensure_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, dtype),
        local_files_only=True,
    )
    targets, layers = lora_targets(backbone.config.num_hidden_layers, args.last_n_layers)
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=targets,
    )
    backbone = get_peft_model(backbone, config)
    backbone.enable_input_require_grads()
    backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = TopGroupModel(backbone, backbone.config.hidden_size, args.dropout)
    return tokenizer, model.to(device), layers


def earth_mover_loss(logits, groups):
    """Squared distance between predicted and target cumulative distributions."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    predicted_cdf = probabilities.cumsum(dim=-1)
    target = F.one_hot(groups, NUM_GROUPS).float()
    target_cdf = target.cumsum(dim=-1)
    return torch.mean((predicted_cdf - target_cdf) ** 2)


def pairwise_rank_loss(scores, ratings_z, minimum_gap):
    """RankNet loss over informative pairs from the global DDP microbatch."""
    if dist.is_available() and dist.is_initialized():
        scores = torch.cat(tuple(dist_nn.all_gather(scores)), dim=0)
        # Labels need no gradient; ordinary all_gather avoids needless autograd.
        gathered_ratings = [torch.empty_like(ratings_z) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_ratings, ratings_z)
        ratings_z = torch.cat(gathered_ratings, dim=0)
    difference = ratings_z[:, None] - ratings_z[None, :]
    upper = torch.triu(torch.ones_like(difference, dtype=torch.bool), diagonal=1)
    mask = upper & (difference.abs() >= minimum_gap)
    if not bool(mask.any()):
        return scores.sum() * 0.0
    direction = difference[mask].sign()
    score_difference = scores[:, None] - scores[None, :]
    return F.softplus(-direction * score_difference[mask]).mean()


def multi_task_loss(outputs, groups, coral_targets, rating_z, args):
    ce = F.cross_entropy(
        outputs["class_logits"].float(),
        groups,
        label_smoothing=args.label_smoothing,
    )
    emd = earth_mover_loss(outputs["class_logits"], groups)
    coral = F.binary_cross_entropy_with_logits(
        outputs["coral_logits"].float(), coral_targets
    )
    regression = F.smooth_l1_loss(outputs["rating_z"].float(), rating_z, beta=0.5)
    class_score = (
        torch.softmax(outputs["class_logits"].float(), dim=-1)
        * torch.arange(NUM_GROUPS, device=groups.device).float()
    ).sum(dim=-1)
    ranking = pairwise_rank_loss(class_score, rating_z, args.pair_min_gap)
    total = (
        args.ce_weight * ce
        + args.emd_weight * emd
        + args.coral_weight * coral
        + args.regression_weight * regression
        + args.ranking_weight * ranking
    )
    parts = {
        "ce": ce.detach(),
        "emd": emd.detach(),
        "coral": coral.detach(),
        "regression": regression.detach(),
        "ranking": ranking.detach(),
    }
    return total, parts


def groups_from_thresholds(scores, thresholds):
    return np.searchsorted(
        np.asarray(thresholds, dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
        side="right",
    ).astype(np.int64)


def fit_exact_ordered_thresholds(scores, groups, num_groups=NUM_GROUPS):
    """Globally optimal ordered thresholds for validation exact accuracy.

    Scores with equal values are kept together. Dynamic programming partitions
    sorted unique scores into K contiguous predicted groups and maximizes the
    number of exact labels. No test labels are used.
    """
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    unique, inverse = np.unique(scores, return_inverse=True)
    if len(unique) < num_groups:
        raise ValueError("Not enough unique scores for ordered calibration")
    counts = np.zeros((len(unique), num_groups), dtype=np.int64)
    np.add.at(counts, (inverse, groups), 1)
    prefix = np.vstack([np.zeros((1, num_groups), dtype=np.int64), counts.cumsum(0)])
    size = len(unique)
    negative = -10**12
    dp = np.full((num_groups + 1, size + 1), negative, dtype=np.int64)
    back = np.full((num_groups + 1, size + 1), -1, dtype=np.int64)
    dp[0, 0] = 0
    for group_count in range(1, num_groups + 1):
        label = group_count - 1
        for end in range(group_count, size + 1):
            starts = np.arange(group_count - 1, end)
            values = dp[group_count - 1, starts] + (
                prefix[end, label] - prefix[starts, label]
            )
            best_local = int(np.argmax(values))
            dp[group_count, end] = values[best_local]
            back[group_count, end] = starts[best_local]
    cuts = []
    end = size
    for group_count in range(num_groups, 1, -1):
        start = int(back[group_count, end])
        cuts.append(start)
        end = start
    cuts.reverse()
    return np.asarray(
        [(unique[cut - 1] + unique[cut]) / 2.0 for cut in cuts],
        dtype=np.float64,
    )


def calibration_selection(report):
    """Exact accuracy is primary; ordinal quality breaks close ties."""
    return (
        float(report["accuracy"])
        + 0.10 * float(report["qwk"])
        + 0.025 * float(report["adjacent_accuracy"])
    )


def split_calibration_selection(validation_indices, validation_groups):
    """Split validation labels for calibration and unbiased epoch selection."""
    local = np.arange(len(validation_indices))
    counts = np.bincount(validation_groups, minlength=NUM_GROUPS)
    stratify = validation_groups if np.all(counts[counts > 0] >= 2) else None
    calibration_local, selection_local = train_test_split(
        local,
        test_size=0.5,
        random_state=SEED,
        stratify=stratify,
    )
    return (
        np.asarray(validation_indices)[calibration_local],
        np.asarray(validation_indices)[selection_local],
    )


def calibrate_decoder(class_scores, coral_scores, rating_scores, labels):
    """Select a validation-only score blend and exact-optimal thresholds."""
    components = np.stack([class_scores, coral_scores, rating_scores], axis=1)
    components = (components - components.mean(axis=0)) / np.maximum(
        components.std(axis=0), 1e-6
    )
    candidates = []
    for class_weight in np.linspace(0, 1, 5):
        for coral_weight in np.linspace(0, 1 - class_weight, 5):
            rating_weight = 1 - class_weight - coral_weight
            weights = np.asarray([class_weight, coral_weight, rating_weight])
            score = components @ weights
            thresholds = fit_exact_ordered_thresholds(score, labels)
            prediction = groups_from_thresholds(score, thresholds)
            report = gp.classification_report(labels, prediction, score)
            candidates.append(
                (calibration_selection(report), weights, thresholds, components.mean(0), components.std(0), report)
            )
    _, weights, thresholds, _, _, report = max(candidates, key=lambda row: row[0])
    # Store raw-component normalization, not the already standardized matrix.
    raw = np.stack([class_scores, coral_scores, rating_scores], axis=1)
    return {
        "weights": weights.tolist(),
        "thresholds": thresholds.tolist(),
        "component_mean": raw.mean(axis=0).tolist(),
        "component_std": np.maximum(raw.std(axis=0), 1e-6).tolist(),
        "validation_report": report,
    }


def decode_with_calibration(class_scores, coral_scores, rating_scores, calibration):
    raw = np.stack([class_scores, coral_scores, rating_scores], axis=1)
    normalized = (
        raw - np.asarray(calibration["component_mean"])
    ) / np.asarray(calibration["component_std"])
    score = normalized @ np.asarray(calibration["weights"])
    groups = groups_from_thresholds(score, calibration["thresholds"])
    return groups, score


@torch.no_grad()
def predict_loader(model, tokenizer, loader, device, max_length):
    model.eval()
    result = {
        "names": [],
        "labels": [],
        "ratings": [],
        "class_scores": [],
        "coral_scores": [],
        "rating_scores": [],
    }
    rank_values = torch.arange(NUM_GROUPS, device=device).float()
    for batch in loader:
        ids, mask = tokenize_batch(
            tokenizer, [row["description"] for row in batch], device, max_length
        )
        output = model(ids, mask)
        probability = torch.softmax(output["class_logits"].float(), dim=-1)
        result["names"].extend(row["name"] for row in batch)
        result["labels"].extend(row["group"] for row in batch)
        result["ratings"].extend(row["rating"] for row in batch)
        result["class_scores"].extend((probability * rank_values).sum(-1).cpu().tolist())
        result["coral_scores"].extend(
            torch.sigmoid(output["coral_logits"].float()).sum(-1).cpu().tolist()
        )
        result["rating_scores"].extend(output["rating_z"].float().cpu().tolist())
    return {key: np.asarray(value) for key, value in result.items()}


def make_loader(dataset, batch_size, sampler=None, shuffle=False, workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        collate_fn=identity_collate,
        num_workers=workers,
        pin_memory=True,
    )


def write_history_and_plots(history, plot_dir: Path) -> None:
    """Persist train/eval curves after every epoch for live monitoring."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history)
    frame.to_csv(plot_dir / "history.csv", index=False)
    (plot_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    if frame.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = frame["epoch"].to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    train_axes = axes[0]
    train_axes.plot(epochs, frame["train_loss"], label="loss", linewidth=2)
    for key, style in (
        ("train_ce", "-"),
        ("train_emd", "--"),
        ("train_coral", "-."),
        ("train_regression", ":"),
        ("train_ranking", "--"),
    ):
        if key in frame.columns:
            train_axes.plot(epochs, frame[key], style, label=key.replace("train_", ""), alpha=0.85)
    train_axes.set_title("Train losses")
    train_axes.set_xlabel("epoch")
    train_axes.set_ylabel("loss")
    train_axes.grid(True, alpha=0.3)
    train_axes.legend(fontsize=8)

    eval_axes = axes[1]
    eval_axes.plot(epochs, frame["val_accuracy"], label="exact acc", linewidth=2)
    eval_axes.plot(epochs, frame["val_qwk"], label="QWK")
    eval_axes.plot(epochs, frame["val_adjacent_accuracy"], label="adjacent")
    if "selection" in frame.columns:
        eval_axes.plot(epochs, frame["selection"], label="selection", linestyle="--")
    eval_axes.set_title("Eval metrics (selection half)")
    eval_axes.set_xlabel("epoch")
    eval_axes.set_ylabel("score")
    eval_axes.set_ylim(0.0, 1.05)
    eval_axes.grid(True, alpha=0.3)
    eval_axes.legend(fontsize=8)

    temporary = plot_dir / "_metrics_live_write.png"
    final = plot_dir / "metrics_live.png"
    figure.savefig(temporary, dpi=140, format="png")
    plt.close(figure)
    temporary.replace(final)


def save_checkpoint(args, model, edges, rating_mean, rating_std, layers, directory, calibration=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(directory / "adapter")
    torch.save(
        {
            "attention_pool": model.attention_pool.state_dict(),
            "trunk": model.trunk.state_dict(),
            "classifier": model.classifier.state_dict(),
            "ordinal": model.ordinal.state_dict(),
            "regressor": model.regressor.state_dict(),
            "edges": np.asarray(edges),
            "rating_mean": float(rating_mean),
            "rating_std": float(rating_std),
            "layers": layers,
            "dtype": args.dtype,
            "dropout": args.dropout,
            "base_model": str(Path(lp.ensure_model(args.model)).resolve()),
            "hidden_size": int(model.backbone.config.hidden_size),
            "num_hidden_layers": int(model.backbone.config.num_hidden_layers),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "last_n_layers": int(args.last_n_layers),
            "max_length": int(args.max_length),
            "calibration": calibration,
        },
        directory / "heads.pt",
    )


def load_checkpoint(args, directory, device):
    directory = Path(directory)
    bundle = torch.load(directory / "heads.pt", map_location="cpu", weights_only=False)
    model_path = lp.ensure_model(args.model)
    resolved_model = str(Path(model_path).resolve())
    if resolved_model != bundle["base_model"]:
        raise ValueError(
            f"Checkpoint base model is {bundle['base_model']}, not {resolved_model}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, bundle["dtype"]),
        local_files_only=True,
    )
    backbone = PeftModel.from_pretrained(backbone, directory / "adapter")
    model = TopGroupModel(backbone, backbone.config.hidden_size, bundle["dropout"])
    if int(backbone.config.hidden_size) != bundle["hidden_size"]:
        raise ValueError("Checkpoint and base-model hidden sizes differ")
    if int(backbone.config.num_hidden_layers) != bundle["num_hidden_layers"]:
        raise ValueError("Checkpoint and base-model layer counts differ")
    for name in ("attention_pool", "trunk", "classifier", "ordinal", "regressor"):
        getattr(model, name).load_state_dict(bundle[name])
    return tokenizer, model.to(device).eval(), bundle


def train(args):
    distributed, rank, world_size, device = setup_distributed(args)
    set_seed(args.seed + rank)
    dtype = args.dtype
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        dtype = "float16"
        args.dtype = dtype

    records = lp.load_records(
        argparse.Namespace(
            data=args.data,
            data_cache=args.data_cache,
            max_samples=args.max_samples,
            model=args.model,
            max_length=args.max_length,
        )
    )
    ratings = np.asarray([row["rating"] for row in records], dtype=np.float32)
    # Keep the benchmark partition fixed; --seed changes optimization only.
    train_idx, val_idx, test_idx = gp.split_indices(len(records), SEED)
    edges = gp.fit_quintile_edges(ratings, train_idx, NUM_GROUPS)
    groups = gp.ratings_to_groups(ratings, edges)
    rating_mean = float(ratings[train_idx].mean())
    rating_std = float(ratings[train_idx].std())
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using {world_size} GPU(s), per-GPU batch={args.batch_size}", flush=True)
        print("Train-only edges:", edges.tolist(), flush=True)

    tokenizer, raw_model, layers = build_model(args, device, dtype)
    # Fail deterministically before DDP collectives if any prompt is too long.
    if rank == 0:
        for start in range(0, len(records), 64):
            tokenize_batch(
                tokenizer,
                [row["description"] for row in records[start : start + 64]],
                torch.device("cpu"),
                args.max_length,
            )
        print(f"Validated token lengths for {len(records)} inputs", flush=True)
    if distributed:
        dist.barrier()
    if rank == 0:
        raw_model.backbone.print_trainable_parameters()
    model = (
        DDP(raw_model, device_ids=[device.index], find_unused_parameters=False)
        if distributed
        else raw_model
    )

    datasets = {
        "train": ProblemDataset(records, ratings, groups, train_idx, rating_mean, rating_std),
        "validation": ProblemDataset(records, ratings, groups, val_idx, rating_mean, rating_std),
        "test": ProblemDataset(records, ratings, groups, test_idx, rating_mean, rating_std),
        "all": ProblemDataset(records, ratings, groups, np.arange(len(records)), rating_mean, rating_std),
    }
    calibration_idx, selection_idx = split_calibration_selection(
        val_idx, groups[val_idx]
    )
    datasets["calibration"] = ProblemDataset(
        records, ratings, groups, calibration_idx, rating_mean, rating_std
    )
    datasets["selection"] = ProblemDataset(
        records, ratings, groups, selection_idx, rating_mean, rating_std
    )
    sampler = (
        DistributedSampler(datasets["train"], shuffle=True, seed=args.seed)
        if distributed
        else None
    )
    train_loader = make_loader(
        datasets["train"], args.batch_size, sampler, shuffle=True, workers=0
    )
    calibration_loader = make_loader(datasets["calibration"], args.eval_batch_size)
    selection_loader = make_loader(datasets["selection"], args.eval_batch_size)
    val_loader = make_loader(datasets["validation"], args.eval_batch_size)
    test_loader = make_loader(datasets["test"], args.eval_batch_size)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup = max(1, int(total_updates * args.warmup_ratio))
    global_step = 0
    best = {"score": -math.inf, "epoch": 0}
    patience_left = args.patience
    history = []
    plot_dir = Path(args.plot_dir)
    if rank == 0:
        plot_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "ce": 0.0, "emd": 0.0, "coral": 0.0, "regression": 0.0, "ranking": 0.0}
        seen = 0
        accumulation_divisor = args.grad_accum
        for step, batch in enumerate(train_loader, 1):
            if (step - 1) % args.grad_accum == 0:
                accumulation_divisor = min(
                    args.grad_accum, len(train_loader) - step + 1
                )
            ids, mask = tokenize_batch(
                tokenizer, [row["description"] for row in batch], device, args.max_length
            )
            group = torch.tensor([row["group"] for row in batch], device=device)
            coral = torch.tensor(
                np.stack([row["coral"] for row in batch]), device=device
            )
            rating_z = torch.tensor(
                [row["rating_z"] for row in batch], device=device, dtype=torch.float32
            )
            should_update = step % args.grad_accum == 0 or step == len(train_loader)
            # Avoid an unnecessary DDP all-reduce on accumulation-only steps.
            sync_context = (
                nullcontext()
                if should_update or not isinstance(model, DDP)
                else model.no_sync()
            )
            with sync_context:
                output = model(ids, mask)
                loss, parts = multi_task_loss(output, group, coral, rating_z, args)
                (loss / accumulation_divisor).backward()
            batch_size = len(batch)
            totals["loss"] += float(loss.detach()) * batch_size
            for name, value in parts.items():
                totals[name] += float(value) * batch_size
            seen += batch_size
            if should_update:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if global_step < warmup:
                    scale = (global_step + 1) / warmup
                else:
                    progress = (global_step - warmup) / max(1, total_updates - warmup)
                    scale = 0.5 * (1 + math.cos(math.pi * progress))
                for group_config in optimizer.param_groups:
                    group_config["lr"] = args.lr * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if rank == 0 and (step % args.log_every == 0 or step == len(train_loader)):
                summary = " ".join(f"{key}={value / seen:.4f}" for key, value in totals.items())
                print(f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} {summary}", flush=True)

        if distributed:
            dist.barrier()
        stop = False
        if rank == 0:
            calibration_raw = predict_loader(
                unwrap(model), tokenizer, calibration_loader, device, args.max_length
            )
            calibration = calibrate_decoder(
                calibration_raw["class_scores"],
                calibration_raw["coral_scores"],
                calibration_raw["rating_scores"],
                calibration_raw["labels"],
            )
            selection_raw = predict_loader(
                unwrap(model), tokenizer, selection_loader, device, args.max_length
            )
            prediction, score = decode_with_calibration(
                selection_raw["class_scores"],
                selection_raw["coral_scores"],
                selection_raw["rating_scores"],
                calibration,
            )
            report = gp.classification_report(
                selection_raw["labels"], prediction, score
            )
            selection = calibration_selection(report)
            row = {
                "epoch": epoch,
                "train_loss": totals["loss"] / seen,
                "train_ce": totals["ce"] / seen,
                "train_emd": totals["emd"] / seen,
                "train_coral": totals["coral"] / seen,
                "train_regression": totals["regression"] / seen,
                "train_ranking": totals["ranking"] / seen,
                "val_accuracy": report["accuracy"],
                "val_qwk": report["qwk"],
                "val_adjacent_accuracy": report["adjacent_accuracy"],
                "selection": selection,
            }
            history.append(row)
            write_history_and_plots(history, plot_dir)
            print(json.dumps({"validation": row}, indent=2), flush=True)
            print(f"Updated live plots: {plot_dir / 'metrics_live.png'}", flush=True)
            if selection > best["score"] + 1e-6:
                best = {"score": selection, "epoch": epoch}
                if args.patience > 0:
                    patience_left = args.patience
                save_checkpoint(
                    args,
                    unwrap(model),
                    edges,
                    rating_mean,
                    rating_std,
                    layers,
                    args.output_dir / "best",
                    calibration,
                )
                print(f"New best exact-focused checkpoint: epoch {epoch}", flush=True)
            elif args.patience > 0:
                patience_left -= 1
                stop = patience_left <= 0
                if stop:
                    print(f"Early stop at {epoch}; best={best['epoch']}", flush=True)
            if args.save_every > 0 and epoch % args.save_every == 0:
                periodic = args.output_dir / "checkpoints" / f"epoch-{epoch:04d}"
                save_checkpoint(
                    args,
                    unwrap(model),
                    edges,
                    rating_mean,
                    rating_std,
                    layers,
                    periodic,
                    calibration,
                )
                print(f"Saved periodic checkpoint: {periodic}", flush=True)
        if distributed:
            flag = torch.tensor([int(stop)], device=device)
            dist.broadcast(flag, 0)
            stop = bool(flag.item())
            dist.barrier()
        if stop:
            break

    if distributed:
        dist.barrier()
    if rank == 0:
        # Release the training copy before loading the best checkpoint.
        del optimizer
        del model
        del raw_model
        torch.cuda.empty_cache()
        tokenizer, best_model, bundle = load_checkpoint(
            args, args.output_dir / "best", device
        )
        split_predictions = {}
        reports = {}
        for split, loader in (("validation", val_loader), ("test", test_loader)):
            raw = predict_loader(best_model, tokenizer, loader, device, args.max_length)
            predicted, score = decode_with_calibration(
                raw["class_scores"],
                raw["coral_scores"],
                raw["rating_scores"],
                bundle["calibration"],
            )
            reports[split] = gp.classification_report(raw["labels"], predicted, score)
            split_predictions[split] = (raw, predicted, score)

        metrics = {
            "method": "attention_lora_multitask_calibrated",
            "seed": args.seed,
            "best_epoch": best["epoch"],
            "edges": edges.tolist(),
            "rating_mean_train": rating_mean,
            "rating_std_train": rating_std,
            "lora_layers": layers,
            "world_size": world_size,
            "effective_batch_size": args.batch_size * world_size * args.grad_accum,
            "loss_weights": {
                key: getattr(args, f"{key}_weight")
                for key in ("ce", "emd", "coral", "regression", "ranking")
            },
            "calibration": bundle["calibration"],
            "validation_protocol": {
                "calibration_rows": int(len(calibration_idx)),
                "selection_rows": int(len(selection_idx)),
                "calibration_role": "fit score blend and thresholds",
                "selection_role": "select epoch without reusing calibration labels",
            },
            **reports,
        }
        baseline_path = Path("model-cache/group-probe-lora/metrics.json")
        if baseline_path.is_file():
            baseline = json.loads(baseline_path.read_text())
            metrics["comparison"] = {
                "baseline": "group_probe_lora.py",
                "baseline_test_accuracy": baseline["test"]["accuracy"],
                "baseline_test_qwk": baseline["test"]["qwk"],
                "delta_accuracy": reports["test"]["accuracy"] - baseline["test"]["accuracy"],
                "delta_qwk": reports["test"]["qwk"] - baseline["test"]["qwk"],
            }

        all_loader = make_loader(datasets["all"], args.eval_batch_size)
        raw = predict_loader(best_model, tokenizer, all_loader, device, args.max_length)
        predicted, score = decode_with_calibration(
            raw["class_scores"],
            raw["coral_scores"],
            raw["rating_scores"],
            bundle["calibration"],
        )
        order = np.lexsort((score, predicted))
        curriculum = pd.DataFrame(
            {
                "name": raw["names"][order],
                "real_rating": raw["ratings"][order],
                "true_group": raw["labels"][order],
                "pred_group": predicted[order],
                "pred_score": score[order],
                "curriculum_rank": np.arange(len(order)),
            }
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        curriculum.to_csv(args.output_dir / "curriculum_order.csv", index=False)
        save_checkpoint(
            args,
            best_model,
            edges,
            rating_mean,
            rating_std,
            layers,
            args.output_dir,
            bundle["calibration"],
        )
        print(json.dumps({"validation": reports["validation"], "test": reports["test"], "comparison": metrics.get("comparison")}, indent=2), flush=True)
        print(f"Saved top-tier group model to {args.output_dir}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def predict_file(args):
    device = torch.device(lp.resolve_device(args.device))
    if device.type != "cuda":
        raise SystemExit("Prediction requires CUDA for the 3B backbone.")
    tokenizer, model, bundle = load_checkpoint(args, args.output_dir, device)
    description = args.predict_file.read_text(encoding="utf-8")
    ids, mask = tokenize_batch(
        tokenizer, [description], device, args.max_length
    )
    with torch.no_grad():
        output = model(ids, mask)
        probability = torch.softmax(output["class_logits"].float(), dim=-1)[0]
        class_score = float(
            (probability * torch.arange(NUM_GROUPS, device=device).float()).sum()
        )
        coral_score = float(torch.sigmoid(output["coral_logits"].float()).sum())
        rating_score = float(output["rating_z"].float()[0])
    predicted, score = decode_with_calibration(
        np.asarray([class_score]),
        np.asarray([coral_score]),
        np.asarray([rating_score]),
        bundle["calibration"],
    )
    estimated_rating = rating_score * bundle["rating_std"] + bundle["rating_mean"]
    print(
        json.dumps(
            {
                "pred_group": int(predicted[0]),
                "pred_score": float(score[0]),
                "class_probabilities": probability.cpu().tolist(),
                "estimated_rating": float(estimated_rating),
                "group_edges": bundle["edges"].tolist(),
            },
            indent=2,
        )
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="deepmind/code_contests")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-cache", type=Path, default=lp.DEFAULT_DATA_CACHE)
    parser.add_argument("--model", default=str(lp.DEFAULT_MODEL))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early-stop patience on selection metric; 0 disables early stopping",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Per GPU")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--last-n-layers", type=int, default=16)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--pair-min-gap", type=float, default=0.30)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--emd-weight", type=float, default=0.30)
    parser.add_argument("--coral-weight", type=float, default=0.20)
    parser.add_argument("--regression-weight", type=float, default=0.30)
    parser.add_argument("--ranking-weight", type=float, default=0.30)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save a periodic checkpoint every N epochs (0 disables)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=lp.ROOT / "results/group-probe-top",
        help="Directory for live history.csv and metrics_live.png",
    )
    parser.add_argument("--predict-file", type=Path)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.grad_accum) < 1:
        parser.error("epochs/batch-size/grad-accum must be positive")
    if args.patience < 0:
        parser.error("patience must be >= 0 (0 disables early stopping)")
    if args.save_every < 0:
        parser.error("save-every must be >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.predict_file is not None:
        predict_file(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
