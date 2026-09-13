"""Five-group ordinal difficulty probe on frozen Afterburner embeddings.

Strongest practical head (without LoRA):
  - train-only equal-count quintile labels (balanced ordinal groups)
  - LayerNorm + MLP trunk
  - CORAL ordinal logits (respects 0 < 1 < 2 < 3 < 4)
  - multi-seed ensemble with soft averaging
  - early stop on validation quadratic weighted kappa (QWK)

Reuses `model-cache/probe/embeddings.npz` from `linear_probe.py` by default.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import linear_probe as lp


NUM_GROUPS = 5
DEFAULT_EMBED_DIR = lp.DEFAULT_PROBE_DIR
DEFAULT_OUTPUT_DIR = lp.ROOT / "model-cache/group-probe"
SEED = lp.SEED


class CoralMLP(nn.Module):
    """Shared MLP trunk + CORAL thresholds (K-1 logits)."""

    def __init__(self, input_dim, hidden_dim=1024, dropout=0.2, num_groups=NUM_GROUPS):
        super().__init__()
        if num_groups < 2:
            raise ValueError("num_groups must be at least 2")
        self.num_groups = num_groups
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.coral = nn.Linear(hidden_dim // 2, num_groups - 1)

    def forward(self, x):
        return self.coral(self.trunk(x))


def split_indices(count, seed=SEED):
    train_full, test = train_test_split(np.arange(count), test_size=0.2, random_state=seed)
    boundary = int(0.8 * len(train_full))
    return train_full[:boundary], train_full[boundary:], test


def fit_quintile_edges(ratings, train_idx, num_groups=NUM_GROUPS):
    """Equal-count edges from train ratings only. Returns interior thresholds."""
    values = np.asarray(ratings, dtype=np.float64)[train_idx]
    quantiles = np.linspace(0.0, 1.0, num_groups + 1)[1:-1]
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) != num_groups - 1:
        # Rare duplicate quantile collapse: fall back to linspace over train range.
        lo, hi = float(values.min()), float(values.max())
        edges = np.linspace(lo, hi, num_groups + 1)[1:-1]
    return edges.astype(np.float64)


def ratings_to_groups(ratings, edges):
    """Map continuous ratings to ordinal groups 0..K-1 using interior edges."""
    return np.digitize(np.asarray(ratings, dtype=np.float64), edges, right=False).astype(np.int64)


def coral_targets(groups, num_groups=NUM_GROUPS):
    """CORAL multi-hot: target[k] = 1{y > k} for k = 0..K-2."""
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    thresholds = np.arange(num_groups - 1, dtype=np.int64)[None, :]
    return (groups[:, None] > thresholds).astype(np.float32)


def coral_probs(logits):
    return torch.sigmoid(logits)


def coral_expected_rank(logits):
    """Soft expected group in [0, K-1]."""
    return coral_probs(logits).sum(dim=-1)


def coral_predict_group(logits):
    # expected rank ∈ [0, K-1]; K-1 == logits.shape[-1]
    max_group = logits.shape[-1]
    return torch.round(coral_expected_rank(logits)).clamp(0, max_group).long()


def ensemble_logits(models, x):
    stacked = torch.stack([model(x) for model in models], dim=0)
    return stacked.mean(dim=0)


def selection_score(report):
    """Primary: QWK; secondary: adjacent accuracy (curriculum-safe)."""
    return float(report["qwk"]) + 0.15 * float(report["adjacent_accuracy"])


def classification_report(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    report = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "adjacent_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mae_groups": float(np.mean(np.abs(y_true - y_pred))),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1
        else None,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(NUM_GROUPS))
        ).tolist(),
    }
    if y_score is not None:
        report["score_spearman"] = float(
            pd.Series(y_true).corr(pd.Series(np.asarray(y_score)), method="spearman")
        )
    report["selection_score"] = selection_score(report)
    return report


def linear_rating_quintile_baseline(embed_dir, names, groups, edges, split_indices_map):
    """Fair baseline: digitize linear_probe continuous ratings with the same edges."""
    path = Path(embed_dir) / "predictions.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if not {"name", "pred_difficulty"}.issubset(frame.columns):
        return None
    pred_by_name = dict(zip(frame["name"], frame["pred_difficulty"]))
    if any(name not in pred_by_name for name in names):
        return None
    pred_rating = np.asarray([pred_by_name[name] for name in names], dtype=np.float64)
    pred_group = ratings_to_groups(pred_rating, edges)
    baseline = {}
    for split_name, idx in split_indices_map.items():
        baseline[split_name] = classification_report(
            groups[idx], pred_group[idx], pred_rating[idx]
        )
    return baseline


def load_embeddings_and_records(args):
    records = lp.load_records(argparse.Namespace(
        data=args.data,
        data_cache=args.data_cache,
        max_samples=args.max_samples,
        model=args.model,
        max_length=args.max_length,
    ))
    dtype = args.dtype or "float16"
    metadata = lp.cache_metadata(
        argparse.Namespace(model=args.model, max_length=args.max_length),
        records,
        dtype,
    )
    cache = Path(args.embed_dir) / "embeddings.npz"
    if not cache.is_file():
        raise SystemExit(
            f"Missing {cache}. Run: python linear_probe.py --device cuda:0 "
            f"--output-dir {args.embed_dir}"
        )
    with np.load(cache, allow_pickle=False) as saved:
        saved_meta = json.loads(saved["metadata"].item())
        embeddings = np.asarray(saved["embeddings"], dtype=np.float32)
    if saved_meta.get("records_sha256") != metadata.get("records_sha256"):
        raise SystemExit(
            "Embedding cache does not match current Codeforces record filter; "
            "re-run linear_probe.py with the same data settings."
        )
    if len(embeddings) != len(records):
        raise SystemExit(
            f"Embedding rows ({len(embeddings)}) != records ({len(records)}); "
            "finish linear_probe embedding extraction first."
        )
    if not np.isfinite(embeddings).all():
        raise SystemExit("Non-finite values in embeddings.npz")
    return records, embeddings, metadata


def make_loader(x, y_groups, indices, batch_size, shuffle, balance):
    subset_x = x[indices]
    subset_y = y_groups[indices]
    targets = torch.from_numpy(coral_targets(subset_y))
    data = TensorDataset(torch.from_numpy(subset_x), targets, torch.from_numpy(subset_y))
    if balance and shuffle:
        counts = np.bincount(subset_y, minlength=NUM_GROUPS).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / counts[subset_y]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights.astype(np.float64)),
            num_samples=len(subset_y),
            replacement=True,
        )
        return DataLoader(data, batch_size=batch_size, sampler=sampler)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def train_one_model(x_train, y_train, x_val, y_val, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CoralMLP(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_groups=NUM_GROUPS,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    train_loader = make_loader(
        x_train, y_train, np.arange(len(y_train)), args.batch_size, True, args.balance
    )
    best = {"selection": -1.0, "state": None, "epoch": 0, "report": None}
    patience_left = args.patience
    history = []

    x_val_t = torch.from_numpy(x_val)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb, _ in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            logits = model(x_val_t)
            pred = coral_predict_group(logits).numpy()
            score = coral_expected_rank(logits).numpy()
            report = classification_report(y_val, pred, score)
            val_loss = criterion(logits, torch.from_numpy(coral_targets(y_val))).item()
        row = {
            "epoch": epoch,
            "train_bce": total / max(n, 1),
            "val_bce": val_loss,
            "val_qwk": report["qwk"],
            "val_accuracy": report["accuracy"],
            "val_adjacent_accuracy": report["adjacent_accuracy"],
            "val_selection": report["selection_score"],
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"seed={seed} epoch={epoch}/{args.epochs} "
                f"train_bce={row['train_bce']:.4f} val_qwk={row['val_qwk']:.4f} "
                f"acc={row['val_accuracy']:.4f} adj={row['val_adjacent_accuracy']:.4f}",
                flush=True,
            )
        if report["selection_score"] > best["selection"] + 1e-6:
            best = {
                "selection": report["selection_score"],
                "state": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "report": report,
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(
                    f"seed={seed} early stop at epoch {epoch} (best {best['epoch']})",
                    flush=True,
                )
                break

    model.load_state_dict(best["state"])
    model.eval()
    return model, best, history


def evaluate_models(models, x, y):
    with torch.no_grad():
        logits = ensemble_logits(models, torch.from_numpy(x))
        pred = coral_predict_group(logits).numpy()
        score = coral_expected_rank(logits).numpy()
    return classification_report(y, pred, score), pred, score


def train(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, embeddings, embed_meta = load_embeddings_and_records(args)
    ratings = np.array([row["rating"] for row in records], dtype=np.float64)
    train_idx, val_idx, test_idx = split_indices(len(records), seed=SEED)
    edges = fit_quintile_edges(ratings, train_idx, NUM_GROUPS)
    groups = ratings_to_groups(ratings, edges)

    # Feature scaling on train only (same discipline as linear_probe).
    scaler = StandardScaler().fit(embeddings[train_idx])
    features = scaler.transform(embeddings).astype(np.float32)

    print("Train-only quintile edges (interior):", edges.tolist(), flush=True)
    for split_name, idx in ("train", train_idx), ("validation", val_idx), ("test", test_idx):
        counts = np.bincount(groups[idx], minlength=NUM_GROUPS).tolist()
        print(f"{split_name} group counts: {counts}", flush=True)

    seeds = list(range(SEED, SEED + args.ensemble_size))
    models = []
    member_reports = []
    all_history = []
    for seed in seeds:
        model, best, history = train_one_model(
            features[train_idx],
            groups[train_idx],
            features[val_idx],
            groups[val_idx],
            args,
            seed,
        )
        models.append(model)
        member_reports.append(
            {"seed": seed, "best_epoch": best["epoch"], "validation": best["report"]}
        )
        all_history.extend({"seed": seed, **row} for row in history)

    split_map = {"train": train_idx, "validation": val_idx, "test": test_idx}
    names = [row["name"] for row in records]
    baseline = linear_rating_quintile_baseline(
        args.embed_dir, names, groups, edges, split_map
    )
    metrics = {
        "num_groups": NUM_GROUPS,
        "method": "coral_mlp_ensemble",
        "edges": edges.tolist(),
        "edge_rule": "train_only_equal_count_quantiles",
        "ensemble_seeds": seeds,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "balance": args.balance,
        "embed_dir": str(Path(args.embed_dir).resolve()),
        "embedding_metadata": embed_meta,
        "members": member_reports,
        "normalization_fit": "train_only_standard_scaler",
        "selection_rule": "qwk + 0.15 * adjacent_accuracy",
        "seed": SEED,
        "baseline_linear_rating_quintile": baseline,
    }
    rows = []
    for split_name, idx in split_map.items():
        report, pred, score = evaluate_models(models, features[idx], groups[idx])
        metrics[split_name] = report
        payload = {
            split_name: {
                "accuracy": report["accuracy"],
                "adjacent_accuracy": report["adjacent_accuracy"],
                "qwk": report["qwk"],
                "mae_groups": report["mae_groups"],
                "score_spearman": report.get("score_spearman"),
            }
        }
        if baseline is not None:
            payload[split_name]["baseline_qwk"] = baseline[split_name]["qwk"]
            payload[split_name]["baseline_accuracy"] = baseline[split_name]["accuracy"]
            payload[split_name]["baseline_adjacent_accuracy"] = baseline[split_name][
                "adjacent_accuracy"
            ]
            payload[split_name]["delta_qwk"] = (
                report["qwk"] - baseline[split_name]["qwk"]
            )
        print(json.dumps(payload, indent=2), flush=True)
        for local_i, global_i in enumerate(idx):
            rows.append({
                "name": records[global_i]["name"],
                "split": split_name,
                "real_rating": float(ratings[global_i]),
                "true_group": int(groups[global_i]),
                "pred_group": int(pred[local_i]),
                "pred_score": float(score[local_i]),
            })

    # Full-corpus scores for curriculum sorting (group asc, then score asc).
    full_report, full_pred, full_score = evaluate_models(models, features, groups)
    metrics["full"] = full_report
    curriculum = []
    order = np.lexsort((full_score, full_pred))  # stable: score then group
    for rank, index in enumerate(order):
        curriculum.append({
            "curriculum_rank": rank,
            "name": records[index]["name"],
            "real_rating": float(ratings[index]),
            "true_group": int(groups[index]),
            "pred_group": int(full_pred[index]),
            "pred_score": float(full_score[index]),
        })

    bundle = {
        "state_dicts": [model.state_dict() for model in models],
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "edges": edges.astype(np.float64),
        "num_groups": NUM_GROUPS,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "input_dim": int(features.shape[1]),
        "seeds": seeds,
        "embed_dir": str(Path(args.embed_dir).resolve()),
    }
    torch.save(bundle, args.output_dir / "group_probe.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(curriculum).to_csv(args.output_dir / "curriculum_order.csv", index=False)
    pd.DataFrame(all_history).to_csv(args.output_dir / "losses.csv", index=False)
    group_map = {
        "edges": edges.tolist(),
        "groups": {
            "0": "easiest quintile",
            "1": "easy-mid",
            "2": "mid",
            "3": "mid-hard",
            "4": "hardest quintile",
        },
        "sort_keys": ["pred_group ascending", "pred_score ascending"],
    }
    (args.output_dir / "group_schema.json").write_text(
        json.dumps(group_map, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved group probe to {args.output_dir}", flush=True)


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    models = []
    for state in bundle["state_dicts"]:
        model = CoralMLP(
            input_dim=bundle["input_dim"],
            hidden_dim=bundle["hidden_dim"],
            dropout=bundle["dropout"],
            num_groups=bundle["num_groups"],
        )
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return bundle, models


def predict_file(args):
    """Score one statement: needs backbone + trained group head."""
    bundle_path = args.output_dir / "group_probe.pt"
    if not bundle_path.is_file():
        raise SystemExit(f"Missing {bundle_path}; train first")
    bundle, models = load_bundle(bundle_path)
    device = lp.resolve_device(args.device)
    dtype = args.dtype or ("float32" if device == "cpu" else "float16")
    text = args.predict_file.read_text(encoding="utf-8")
    tokenizer, backbone = lp.load_backbone(args.model, device, dtype)
    embedding = lp.embed_problem(text, tokenizer, backbone, device, args.max_length)
    x = (embedding - bundle["scaler_mean"]) / bundle["scaler_scale"]
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        logits = ensemble_logits(models, x)
        group = int(coral_predict_group(logits).item())
        score = float(coral_expected_rank(logits).item())
    print(json.dumps({
        "pred_group": group,
        "pred_score": score,
        "edges": bundle["edges"].tolist(),
        "num_groups": bundle["num_groups"],
    }, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="deepmind/code_contests")
    parser.add_argument("--data-cache", type=Path, default=lp.DEFAULT_DATA_CACHE)
    parser.add_argument("--model", default=str(lp.DEFAULT_MODEL))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embed-dir", type=Path, default=DEFAULT_EMBED_DIR,
                        help="Directory with embeddings.npz from linear_probe.py")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    # Stronger regularization: early runs overfit by epoch ~3 without this.
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--predict-file", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or args.ensemble_size < 1:
        parser.error("epochs/patience/ensemble-size must be positive")
    if args.hidden_dim < 32 or not (0.0 <= args.dropout < 1.0):
        parser.error("invalid hidden-dim/dropout")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.predict_file is not None:
        predict_file(args)
        return
    train(args)


if __name__ == "__main__":
    main()
