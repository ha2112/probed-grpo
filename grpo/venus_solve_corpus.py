"""Build Venus solve-from-scratch corpora bucketed by group_probe_top (3-class)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_cache import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    VENUS_DATASET_ID,
    VENUS_DATASET_REVISION,
    ensure_model,
    load_cached_dataset,
)
from grpo.codeforces_corpus import probe_fingerprint, _record_hash  # noqa: E402
from grpo.venus_corpus import select_venus  # noqa: E402

BUCKET_NAMES = {0: "easy", 1: "medium", 2: "hard"}
# Code-first: long <thinking> was exhausting max_new_tokens (~1024) before any code.
SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Solve the problem in Python 3. "
    "Output ONLY code. No essays, no <thinking>, no explanation before or after. "
    "Finish the function, then immediately close the code block and stop. "
    "Do not repeat tokens."
)
USER_TEMPLATE = """## Problem Description
{description}

## Output Format (mandatory)
- Only code, already starting inside ```python.
- One complete Python solution. No comments that restate the problem.
- When the code is complete, emit ``` then </solution> and stop.
- Never keep emitting the same token or list item after the code is done.
- No <thinking> tags.
"""
# Forced generation prefix so rollouts start inside the code fence.
SOLVE_RESPONSE_PREFIX = "<solution>\n```python\n"


def make_solve_prompt(description: str):
    """Chat messages for Venus solve-from-scratch (code-first)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(description=description)},
    ]


def make_solve_record(problem, split):
    instance = {
        k: problem[k]
        for k in ("test_case_runners", "test_case_evaluator", "test_cases")
    }
    record_id = f"venus-solve:{problem['problem_id']}"
    return {
        "schema_version": 1,
        "data_source": VENUS_DATASET_ID,
        "ability": "code",
        "prompt": make_solve_prompt(problem["question_content"]),
        "reward_model": {"style": "rule", "ground_truth": "{}"},
        "extra_info": {
            "split": split,
            "problem_id": int(problem["problem_id"]),
            "record_id": record_id,
            "name": problem["title"],
            "difficulty": problem["difficulty"],
            "description": problem["question_content"],
            "probe_group": -1,
            "probe_score": 0.0,
            "bucket": "",
            "case_multiply": 1,
            "instance": json.dumps(instance, ensure_ascii=False),
        },
    }


def filter_prompt_length(records, tokenizer, max_prompt_length):
    retained = []
    for row in records:
        length = len(
            tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True
            )
        )
        if length <= max_prompt_length:
            retained.append(row)
    return retained, len(records) - len(retained)


class GroupProbeTopScorer:
    """Score statements with a frozen group_probe_top checkpoint."""

    def __init__(self, probe_dir, device="cuda:0", batch_size=4):
        import argparse

        import group_probe_top as top
        import torch

        self.torch = torch
        self.top = top
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise SystemExit("group_probe_top scoring requires CUDA")
        args = argparse.Namespace(
            model=str(DEFAULT_MODEL_DIR),
            output_dir=Path(probe_dir),
            max_length=4096,
            device=str(self.device),
        )
        self.tokenizer, self.model, self.bundle = top.load_checkpoint(
            args, Path(probe_dir), self.device
        )
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = int(self.bundle.get("max_length", 4096))
        self.num_groups = int(self.bundle.get("num_groups", self.model.num_groups))
        if self.num_groups != 3:
            print(
                f"Warning: probe has num_groups={self.num_groups}; expected 3",
                flush=True,
            )

    def score_batch(self, descriptions):
        torch = self.torch
        top = self.top
        groups = []
        scores = []
        for start in range(0, len(descriptions), self.batch_size):
            chunk = descriptions[start : start + self.batch_size]
            ids, mask = top.tokenize_batch(
                self.tokenizer, chunk, self.device, self.max_length
            )
            with torch.no_grad():
                output = self.model(ids, mask)
                probability = torch.softmax(output["class_logits"].float(), dim=-1)
                rank = torch.arange(self.num_groups, device=self.device).float()
                class_scores = (probability * rank).sum(-1).cpu().numpy()
                coral_scores = (
                    torch.sigmoid(output["coral_logits"].float()).sum(-1).cpu().numpy()
                )
                rating_scores = output["rating_z"].float().cpu().numpy()
            predicted, score = top.decode_with_calibration(
                class_scores,
                coral_scores,
                rating_scores,
                self.bundle["calibration"],
            )
            groups.extend(int(x) for x in predicted)
            scores.extend(float(x) for x in score)
        return groups, scores


def assign_buckets(records, groups, scores):
    for row, group, score in zip(records, groups, scores):
        group = int(np.clip(group, 0, 2))
        row["extra_info"]["probe_group"] = group
        row["extra_info"]["probe_score"] = float(score)
        row["extra_info"]["bucket"] = BUCKET_NAMES[group]


def write_parquet(rows, path: Path):
    from datasets import Dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))
    print(f"wrote {len(rows)} rows -> {path}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="test")
    parser.add_argument(
        "--data-cache", type=Path, default=ROOT / "data-cache/huggingface/datasets"
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=ROOT / "model-cache/vla_wm/best",
        help="group_probe_top checkpoint directory (heads.pt + adapter/)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get("VENUS_SOLVE_DATA_DIR", ROOT / "grpo/data/venus_solve")
        ),
    )
    parser.add_argument("--max-train", type=int, default=-1)
    parser.add_argument("--max-validation", type=int, default=-1)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--fake-probe-groups",
        action="store_true",
        help="CPU/dev only: map Venus Easy/Med/Hard -> 0/1/2 without loading probe",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_problems = select_venus(
        load_cached_dataset(VENUS_DATASET_ID, args.train_split, args.data_cache),
        args.max_train,
    )
    test_problems = select_venus(
        load_cached_dataset(VENUS_DATASET_ID, args.validation_split, args.data_cache),
        args.max_validation,
    )
    if {p["problem_id"] for p in train_problems} & {
        p["problem_id"] for p in test_problems
    }:
        raise ValueError("Venus train/test problem IDs overlap")
    print(
        f"Venus solve corpus: {len(train_problems)} train / {len(test_problems)} test",
        flush=True,
    )
    if args.download_only:
        return

    from transformers import AutoTokenizer

    model_path = ensure_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    train = [make_solve_record(p, "train") for p in train_problems]
    test = [make_solve_record(p, "test") for p in test_problems]
    train, train_overlong = filter_prompt_length(train, tokenizer, args.max_prompt_length)
    test, test_overlong = filter_prompt_length(test, tokenizer, args.max_prompt_length)
    if not train or not test:
        raise SystemExit("No rows remain after prompt-length filtering")

    if args.fake_probe_groups:
        official = {"Easy": 0, "Medium": 1, "Hard": 2}
        groups = [official[r["extra_info"]["difficulty"]] for r in train]
        scores = [float(g) for g in groups]
        probe_meta = {"mode": "fake_official_difficulty"}
    else:
        probe_dir = args.probe_dir
        if not (probe_dir / "heads.pt").is_file():
            alt = probe_dir.parent / "best"
            if (alt / "heads.pt").is_file():
                probe_dir = alt
        if not (probe_dir / "heads.pt").is_file():
            raise SystemExit(f"Missing group_probe_top checkpoint at {probe_dir}")
        scorer = GroupProbeTopScorer(probe_dir, device=args.device)
        descriptions = [r["extra_info"]["description"] for r in train]
        groups, scores = scorer.score_batch(descriptions)
        probe_meta = {
            "mode": "group_probe_top",
            "probe_dir": str(probe_dir.resolve()),
            "probe_sha256": probe_fingerprint(probe_dir / "heads.pt"),
            "num_groups": scorer.num_groups,
        }

    assign_buckets(train, groups, scores)
    # Stable order within each bucket by probe score then problem id.
    train.sort(
        key=lambda r: (
            r["extra_info"]["probe_group"],
            r["extra_info"]["probe_score"],
            r["extra_info"]["problem_id"],
        )
    )
    test.sort(key=lambda r: r["extra_info"]["problem_id"])

    output_dir = args.output_dir
    write_parquet(train, output_dir / "base" / "train.parquet")
    bucket_counts = {}
    for name in BUCKET_NAMES.values():
        subset = [r for r in train if r["extra_info"]["bucket"] == name]
        bucket_counts[name] = len(subset)
        write_parquet(subset, output_dir / "curriculum" / name / "train.parquet")
    write_parquet(test, output_dir / "shared" / "test.parquet")

    manifest = {
        "schema_version": 1,
        "experiment": "venus_solve_correctness",
        "seed": args.seed,
        "dataset": VENUS_DATASET_ID,
        "dataset_revision": VENUS_DATASET_REVISION,
        "train_count": len(train),
        "test_count": len(test),
        "bucket_counts": bucket_counts,
        "train_overlong": train_overlong,
        "test_overlong": test_overlong,
        "max_prompt_length": args.max_prompt_length,
        "probe": probe_meta,
        "train_record_ids": [r["extra_info"]["record_id"] for r in train],
        "test_record_ids": [r["extra_info"]["record_id"] for r in test],
        "files_sha256": {
            str(path.relative_to(output_dir)): probe_fingerprint(path)
            for path in sorted(output_dir.rglob("*.parquet"))
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {"train": [_record_hash(r) for r in train], "probe": probe_meta},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest["content_fingerprint"] = fingerprint
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"bucket_counts": bucket_counts, "fingerprint": fingerprint}, indent=2))


if __name__ == "__main__":
    main()
