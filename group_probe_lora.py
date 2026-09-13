"""Strong 5-group difficulty probe: LoRA backbone + soft ordinal classification.

Frozen-embedding MLP heads plateau (~42% exact / ~0.69 QWK). This script adapts
the last transformer layers with LoRA and trains a classification head with
neighbor-soft labels so group predictions are accurate and order-aware.

Requires GPU. Reuses the same records/splits/quintile edges discipline as
`group_probe.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoTokenizer

import group_probe as gp
import linear_probe as lp


NUM_GROUPS = gp.NUM_GROUPS
SEED = gp.SEED
DEFAULT_OUTPUT_DIR = lp.ROOT / "model-cache/group-probe-lora"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_ordinal_targets(groups, num_groups=NUM_GROUPS, neighbor=0.15):
    """Put most mass on true group, residual on immediate neighbors."""
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    targets = np.zeros((len(groups), num_groups), dtype=np.float32)
    for i, group in enumerate(groups):
        left = group > 0
        right = group < num_groups - 1
        center = 1.0 - neighbor * (int(left) + int(right))
        targets[i, group] = center
        if left:
            targets[i, group - 1] = neighbor
        if right:
            targets[i, group + 1] = neighbor
    return targets


class ProblemDataset(Dataset):
    def __init__(self, records, groups, indices):
        self.records = [records[i] for i in indices]
        self.groups = np.asarray(groups, dtype=np.int64)[indices]
        self.soft = soft_ordinal_targets(self.groups)
        self.coral = gp.coral_targets(self.groups, NUM_GROUPS)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {
            "description": self.records[index]["description"],
            "name": self.records[index]["name"],
            "group": int(self.groups[index]),
            "soft": self.soft[index],
            "coral": self.coral[index],
        }


def collate_identity(batch):
    return batch


class LoRAGroupModel(nn.Module):
    """LoRA-adapted encoder + pooled features + 5-way head (+ CORAL aux)."""

    def __init__(self, backbone, hidden_size, pool_last_n=8, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.pool_last_n = pool_last_n
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, NUM_GROUPS),
        )
        self.coral = nn.Linear(hidden_size * 2, NUM_GROUPS - 1)

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        hidden = outputs.last_hidden_state  # [B, T, H]
        # Last non-pad token per row.
        lengths = attention_mask.long().sum(dim=1) - 1
        batch = torch.arange(hidden.size(0), device=hidden.device)
        last = hidden[batch, lengths]
        pooled = []
        for i in range(hidden.size(0)):
            end = int(lengths[i].item()) + 1
            start = max(0, end - self.pool_last_n)
            pooled.append(hidden[i, start:end].mean(dim=0))
        mean_last = torch.stack(pooled, dim=0)
        # Head stays float32 for stable LayerNorm / CE under bf16/fp16 backbone.
        return torch.cat([last, mean_last], dim=-1).float()

    def forward(self, input_ids, attention_mask):
        features = self.encode(input_ids, attention_mask)
        return self.head(features), self.coral(features), features


def tokenize_batch(tokenizer, descriptions, device, max_length):
    messages = [lp.build_messages(text) for text in descriptions]
    # Tokenize one-by-one then pad: chat templates differ in length.
    encoded = [
        tokenizer.apply_chat_template(
            msg,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        for msg in messages
    ]
    lengths = [int(item["input_ids"].shape[1]) for item in encoded]
    if max(lengths) > max_length:
        bad = lengths.index(max(lengths))
        raise ValueError(
            f"Prompt has {lengths[bad]} tokens, exceeding limit {max_length}"
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    width = max(lengths)
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for i, item in enumerate(encoded):
        seq = item["input_ids"][0]
        input_ids[i, : seq.shape[0]] = seq
        attention_mask[i, : seq.shape[0]] = 1
    return input_ids.to(device), attention_mask.to(device)


def lora_target_module_names(num_layers, last_n_layers):
    """Full module paths — peft layers_to_transform breaks on bare Qwen2Model."""
    start = max(0, int(num_layers) - int(last_n_layers))
    targets = []
    for index in range(start, int(num_layers)):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            targets.append(f"layers.{index}.self_attn.{name}")
        for name in ("gate_proj", "up_proj", "down_proj"):
            targets.append(f"layers.{index}.mlp.{name}")
    return targets, list(range(start, int(num_layers)))


def build_model(model_path, device, dtype, lora_r, lora_alpha, lora_dropout, last_n_layers):
    model_path = lp.ensure_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, dtype),
        local_files_only=True,
    )
    num_layers = int(backbone.config.num_hidden_layers)
    target_modules, layers = lora_target_module_names(num_layers, last_n_layers)
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    backbone = get_peft_model(backbone, config)
    backbone.enable_input_require_grads()
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = LoRAGroupModel(backbone, hidden_size=backbone.config.hidden_size)
    model.to(device)
    return tokenizer, model, layers


def trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def batch_loss(class_logits, coral_logits, soft, coral, coral_weight):
    log_probs = F.log_softmax(class_logits.float(), dim=-1)
    ce = -(soft * log_probs).sum(dim=-1).mean()
    bce = F.binary_cross_entropy_with_logits(coral_logits.float(), coral)
    return ce + coral_weight * bce, ce.detach(), bce.detach()


@torch.no_grad()
def evaluate(model, tokenizer, loader, device, max_length):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    for batch in loader:
        descriptions = [row["description"] for row in batch]
        input_ids, attention_mask = tokenize_batch(
            tokenizer, descriptions, device, max_length
        )
        class_logits, _, _ = model(input_ids, attention_mask)
        probs = torch.softmax(class_logits.float(), dim=-1)
        pred = probs.argmax(dim=-1)
        score = (probs * torch.arange(NUM_GROUPS, device=probs.device).float()).sum(dim=-1)
        y_true.extend(row["group"] for row in batch)
        y_pred.extend(pred.cpu().tolist())
        y_score.extend(score.cpu().tolist())
    return gp.classification_report(y_true, y_pred, y_score)


def setup_distributed(args):
    """Return (is_distributed, rank, world_size, device)."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, dist.get_rank(), dist.get_world_size(), torch.device(f"cuda:{local_rank}")
    device = lp.resolve_device(args.device)
    if not str(device).startswith("cuda"):
        raise SystemExit("LoRA group probe requires CUDA (use torchrun or --device cuda:0).")
    return False, 0, 1, torch.device(device)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def is_main(rank):
    return rank == 0


def train(args):
    distributed, rank, world_size, device = setup_distributed(args)
    set_seed(SEED + rank)
    if is_main(rank):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = args.dtype or "bfloat16"
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        dtype = "float16"

    record_args = argparse.Namespace(
        data=args.data,
        data_cache=args.data_cache,
        max_samples=args.max_samples,
        model=args.model,
        max_length=args.max_length,
    )
    records = lp.load_records(record_args)
    ratings = np.array([row["rating"] for row in records], dtype=np.float64)
    train_idx, val_idx, test_idx = gp.split_indices(len(records), seed=SEED)
    edges = gp.fit_quintile_edges(ratings, train_idx, NUM_GROUPS)
    groups = gp.ratings_to_groups(ratings, edges)
    if is_main(rank):
        print("Train-only quintile edges:", edges.tolist(), flush=True)
        print(f"Using {world_size} GPU(s); per-GPU batch_size={args.batch_size}", flush=True)
        for name, idx in ("train", train_idx), ("validation", val_idx), ("test", test_idx):
            print(name, "counts", np.bincount(groups[idx], minlength=NUM_GROUPS).tolist(), flush=True)

    tokenizer, model, lora_layers = build_model(
        args.model, device, dtype, args.lora_r, args.lora_alpha, args.lora_dropout, args.last_n_layers
    )
    if is_main(rank):
        model.backbone.print_trainable_parameters()
    if distributed:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )

    train_dataset = ProblemDataset(records, groups, train_idx)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=SEED) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_identity,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ProblemDataset(records, groups, val_idx),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_identity,
    )
    test_loader = DataLoader(
        ProblemDataset(records, groups, test_idx),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_identity,
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters(unwrap_model(model)), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(0.05 * total_steps))

    def lr_at(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    best = {"selection": -1.0, "epoch": 0, "path": args.output_dir / "best"}
    history = []
    global_step = 0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        running_ce = 0.0
        running_bce = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            descriptions = [row["description"] for row in batch]
            soft = torch.tensor(
                np.stack([row["soft"] for row in batch]), device=device, dtype=torch.float32
            )
            coral = torch.tensor(
                np.stack([row["coral"] for row in batch]), device=device, dtype=torch.float32
            )
            input_ids, attention_mask = tokenize_batch(
                tokenizer, descriptions, device, args.max_length
            )
            class_logits, coral_logits, _ = model(input_ids, attention_mask)
            loss, ce, bce = batch_loss(
                class_logits, coral_logits, soft, coral, args.coral_weight
            )
            (loss / args.grad_accum).backward()
            running += float(loss.item()) * len(batch)
            running_ce += float(ce.item()) * len(batch)
            running_bce += float(bce.item()) * len(batch)
            seen += len(batch)
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_parameters(unwrap_model(model)), 1.0)
                scale = lr_at(global_step)
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if is_main(rank) and (step % args.log_every == 0 or step == len(train_loader)):
                print(
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={running / max(seen, 1):.4f} ce={running_ce / max(seen, 1):.4f} "
                    f"coral={running_bce / max(seen, 1):.4f}",
                    flush=True,
                )

        if distributed:
            dist.barrier()
        val_report = None
        if is_main(rank):
            val_report = evaluate(
                unwrap_model(model), tokenizer, val_loader, device, args.max_length
            )
            row = {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "val_qwk": val_report["qwk"],
                "val_accuracy": val_report["accuracy"],
                "val_adjacent_accuracy": val_report["adjacent_accuracy"],
                "val_selection": val_report["selection_score"],
            }
            history.append(row)
            print(json.dumps({"validation": row}, indent=2), flush=True)
            improved = val_report["selection_score"] > best["selection"] + 1e-6
            stop = False
            if improved:
                best.update({
                    "selection": val_report["selection_score"],
                    "epoch": epoch,
                    "validation": val_report,
                })
                save_checkpoint(
                    args, unwrap_model(model), tokenizer, edges, lora_layers, dtype, best["path"]
                )
                patience_left = args.patience
                print(f"New best checkpoint at epoch {epoch}", flush=True)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stop at epoch {epoch} (best {best['epoch']})", flush=True)
                    stop = True
        else:
            stop = False
            improved = False

        if distributed:
            # Broadcast early-stop decision from rank 0.
            flag = torch.tensor([1 if stop else 0], device=device, dtype=torch.int)
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
            dist.barrier()
        if stop:
            break

    if distributed:
        dist.barrier()

    # Load best on all ranks for final eval artifacts (rank 0 only writes).
    if is_main(rank):
        tokenizer, raw_model = load_checkpoint(best["path"], args.model, device, dtype)
        metrics = {
            "method": "lora_soft_ordinal",
            "edges": edges.tolist(),
            "lora_layers": lora_layers,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "last_n_layers": args.last_n_layers,
            "pool_last_n": 8,
            "coral_weight": args.coral_weight,
            "best_epoch": best["epoch"],
            "dtype": dtype,
            "seed": SEED,
            "world_size": world_size,
            "per_gpu_batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * world_size * args.grad_accum,
            "validation": evaluate(raw_model, tokenizer, val_loader, device, args.max_length),
            "test": evaluate(raw_model, tokenizer, test_loader, device, args.max_length),
            "train_subset_note": "full train metrics omitted (expensive); see losses.csv",
        }
        mlp_metrics = Path("model-cache/group-probe/metrics.json")
        if mlp_metrics.is_file():
            prev = json.loads(mlp_metrics.read_text())
            metrics["compare_to_frozen_mlp"] = {
                "test_qwk_mlp": prev.get("test", {}).get("qwk"),
                "test_acc_mlp": prev.get("test", {}).get("accuracy"),
                "test_qwk_lora": metrics["test"]["qwk"],
                "test_acc_lora": metrics["test"]["accuracy"],
                "delta_qwk": metrics["test"]["qwk"] - float(prev.get("test", {}).get("qwk", 0.0)),
                "delta_acc": metrics["test"]["accuracy"] - float(prev.get("test", {}).get("accuracy", 0.0)),
            }
        print(json.dumps({
            "validation": {
                "qwk": metrics["validation"]["qwk"],
                "accuracy": metrics["validation"]["accuracy"],
                "adjacent_accuracy": metrics["validation"]["adjacent_accuracy"],
            },
            "test": {
                "qwk": metrics["test"]["qwk"],
                "accuracy": metrics["test"]["accuracy"],
                "adjacent_accuracy": metrics["test"]["adjacent_accuracy"],
            },
            "compare_to_frozen_mlp": metrics.get("compare_to_frozen_mlp"),
            "effective_batch_size": metrics["effective_batch_size"],
        }, indent=2), flush=True)

        all_loader = DataLoader(
            ProblemDataset(records, groups, np.arange(len(records))),
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_identity,
        )
        raw_model.eval()
        names, true_g, pred_g, scores, rate = [], [], [], [], []
        with torch.no_grad():
            offset = 0
            for batch in all_loader:
                input_ids, attention_mask = tokenize_batch(
                    tokenizer, [row["description"] for row in batch], device, args.max_length
                )
                class_logits, _, _ = raw_model(input_ids, attention_mask)
                probs = torch.softmax(class_logits.float(), dim=-1)
                pred = probs.argmax(dim=-1).cpu().numpy()
                score = (probs * torch.arange(NUM_GROUPS, device=probs.device).float()).sum(dim=-1)
                score = score.cpu().numpy()
                for i, row in enumerate(batch):
                    names.append(row["name"])
                    true_g.append(row["group"])
                    pred_g.append(int(pred[i]))
                    scores.append(float(score[i]))
                    rate.append(float(ratings[offset + i]))
                offset += len(batch)

        order = np.lexsort((np.asarray(scores), np.asarray(pred_g)))
        curriculum = [{
            "curriculum_rank": rank_i,
            "name": names[index],
            "real_rating": rate[index],
            "true_group": true_g[index],
            "pred_group": pred_g[index],
            "pred_score": scores[index],
        } for rank_i, index in enumerate(order)]

        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pd.DataFrame(history).to_csv(args.output_dir / "losses.csv", index=False)
        pd.DataFrame(curriculum).to_csv(args.output_dir / "curriculum_order.csv", index=False)
        (args.output_dir / "group_schema.json").write_text(
            json.dumps({
                "edges": edges.tolist(),
                "groups": {
                    "0": "easiest quintile",
                    "1": "easy-mid",
                    "2": "mid",
                    "3": "mid-hard",
                    "4": "hardest quintile",
                },
                "sort_keys": ["pred_group ascending", "pred_score ascending"],
                "method": "lora_soft_ordinal",
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        save_checkpoint(args, raw_model, tokenizer, edges, lora_layers, dtype, args.output_dir)
        print(f"Saved LoRA group probe to {args.output_dir}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def save_checkpoint(args, model, tokenizer, edges, lora_layers, dtype, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    backbone = model.backbone
    backbone.save_pretrained(directory / "adapter")
    torch.save({
        "head": model.head.state_dict(),
        "coral": model.coral.state_dict(),
        "edges": np.asarray(edges, dtype=np.float64),
        "num_groups": NUM_GROUPS,
        "pool_last_n": model.pool_last_n,
        "lora_layers": lora_layers,
        "dtype": dtype,
        "model": str(args.model),
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "last_n_layers": args.last_n_layers,
    }, directory / "head.pt")
    (directory / "tokenizer_name.txt").write_text(str(lp.ensure_model(args.model)), encoding="utf-8")


def load_checkpoint(directory, model_name, device, dtype):
    directory = Path(directory)
    bundle = torch.load(directory / "head.pt", map_location="cpu", weights_only=False)
    model_path = lp.ensure_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, bundle.get("dtype", dtype)),
        local_files_only=True,
    )
    backbone = PeftModel.from_pretrained(backbone, directory / "adapter")
    model = LoRAGroupModel(
        backbone,
        hidden_size=backbone.config.hidden_size,
        pool_last_n=bundle.get("pool_last_n", 8),
    )
    model.head.load_state_dict(bundle["head"])
    model.coral.load_state_dict(bundle["coral"])
    model.to(device).eval()
    return tokenizer, model


def predict_file(args):
    device = lp.resolve_device(args.device)
    dtype = args.dtype or "bfloat16"
    tokenizer, model = load_checkpoint(args.output_dir, args.model, device, dtype)
    text = args.predict_file.read_text(encoding="utf-8")
    input_ids, attention_mask = tokenize_batch(
        tokenizer, [text], device, args.max_length
    )
    with torch.no_grad():
        class_logits, _, _ = model(input_ids, attention_mask)
        probs = torch.softmax(class_logits.float(), dim=-1)[0]
        group = int(probs.argmax().item())
        score = float((probs * torch.arange(NUM_GROUPS, device=probs.device).float()).sum().item())
    bundle = torch.load(args.output_dir / "head.pt", map_location="cpu", weights_only=False)
    print(json.dumps({
        "pred_group": group,
        "pred_score": score,
        "probs": probs.cpu().tolist(),
        "edges": bundle["edges"].tolist(),
    }, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="deepmind/code_contests")
    parser.add_argument("--data-cache", type=Path, default=lp.DEFAULT_DATA_CACHE)
    parser.add_argument("--model", default=str(lp.DEFAULT_MODEL))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--dtype", default=None, help="bfloat16|float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--last-n-layers", type=int, default=8)
    parser.add_argument("--coral-weight", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--predict-file", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1:
        parser.error("epochs/batch-size/grad-accum must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.predict_file is not None:
        predict_file(args)
        return
    train(args)


if __name__ == "__main__":
    main()
