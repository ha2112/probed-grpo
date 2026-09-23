"""Evaluate a Venus solve GRPO policy on the held-out test parquet."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_cache import DEFAULT_MODEL_DIR, ensure_model  # noqa: E402
from grpo.solve_grpo_train import (  # noqa: E402
    ParquetPromptDataset,
    build_prompt_ids,
    generation_controls,
)
from grpo.venus_solve_corpus import SOLVE_RESPONSE_PREFIX  # noqa: E402
from grpo.venus_solve_reward import (  # noqa: E402
    combine_solve_reward,
    format_score,
    venus_solve_score_batch,
)


def load_policy(args, device):
    model_path = ensure_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, args.dtype),
        local_files_only=True,
    )
    adapter = None if not args.adapter else Path(args.adapter)
    if adapter is None:
        print("eval policy: base model, no LoRA", flush=True)
    elif (adapter / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, adapter)
    elif (adapter / "adapter" / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, adapter / "adapter")
    else:
        raise SystemExit(f"No LoRA adapter under {adapter}")
    return tokenizer, model.to(device).eval()


def _mean(values, mask=None):
    chosen = [value for value, keep in zip(values, mask or [True] * len(values)) if keep]
    return float(np.mean(chosen)) if chosen else 0.0


def _bool_series(series: pd.Series) -> list[bool]:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool).tolist()
    return series.astype(str).str.strip().str.lower().isin(("true", "1")).tolist()


SHARD_COLUMNS = [
    "index",
    "name",
    "difficulty",
    "bucket",
    "passed",
    "syntax_ok",
    "tests_passed",
    "tests_total",
    "test_fraction",
    "response",
]


def _rewards_from_frame(frame: pd.DataFrame) -> list[float]:
    passed = _bool_series(frame["passed"])
    fractions = frame["test_fraction"].astype(float).tolist()
    responses = frame["response"].astype(str).tolist()
    return [
        combine_solve_reward(ok, format_score(text), frac)
        for ok, text, frac in zip(passed, responses, fractions)
    ]


def summarize(frame: pd.DataFrame, adapter, eval_file) -> dict:
    passed = _bool_series(frame["passed"])
    syntax_ok = _bool_series(frame["syntax_ok"])
    fractions = frame["test_fraction"].astype(float).tolist()
    difficulties = frame["difficulty"].astype(str).tolist()
    rewards = _rewards_from_frame(frame)
    syntax_and_not_pass = [ok and not ok_all for ok, ok_all in zip(syntax_ok, passed)]
    report = {
        "n": int(len(frame)),
        "pass_rate": _mean(passed),
        "mean_reward": _mean(rewards),
        "syntax_ok_rate": _mean(syntax_ok),
        "mean_test_fraction": _mean(fractions),
        "mean_test_fraction_if_syntax_ok": _mean(fractions, syntax_ok),
        "mean_test_fraction_if_syntax_ok_and_not_all_pass": _mean(
            fractions, syntax_and_not_pass
        ),
        "n_syntax_ok": int(sum(syntax_ok)),
        "n_not_all_pass": int(sum(not item for item in passed)),
        "adapter": adapter,
        "eval_file": eval_file,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    by_difficulty = {}
    for difficulty in sorted(set(difficulties)):
        mask = [item == difficulty for item in difficulties]
        by_difficulty[difficulty] = {
            "n": int(sum(mask)),
            "pass_rate": _mean(passed, mask),
            "mean_reward": _mean(rewards, mask),
            "syntax_ok_rate": _mean(syntax_ok, mask),
            "mean_test_fraction": _mean(fractions, mask),
        }
    report["by_official_difficulty"] = by_difficulty
    return report


@torch.no_grad()
def evaluate(args):
    distributed = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if world_size > visible or local_rank >= max(visible, 1):
        raise SystemExit(
            f"eval world_size={world_size} local_rank={local_rank} but only "
            f"{visible} visible CUDA device(s); refusing to pile every rank onto one GPU"
        )
    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("gloo")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("Evaluation requires CUDA")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        args.dtype = "float16"
    print(
        f"eval rank={rank}/{world_size} device={device} "
        f"visible={torch.cuda.device_count()} "
        f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')!r}",
        flush=True,
    )
    tokenizer, model = load_policy(args, device)
    dataset = ParquetPromptDataset(args.eval_file)
    rows = dataset.rows
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    shard = [(index, row) for index, row in enumerate(rows) if index % world_size == rank]
    records = []
    for seen, (index, row) in enumerate(shard, start=1):
        ids, mask = build_prompt_ids(
            tokenizer, row["prompt"], device, args.max_prompt_length
        )
        generated = model.generate(
            input_ids=ids,
            attention_mask=mask,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **generation_controls(tokenizer, ids.shape[1]),
        )
        text = SOLVE_RESPONSE_PREFIX + tokenizer.decode(
            generated[0, ids.shape[1] :], skip_special_tokens=True
        )
        detail = venus_solve_score_batch([text], [row["extra_info"]])[0]
        info = row["extra_info"]
        records.append(
            {
                "index": index,
                "name": info.get("name", ""),
                "difficulty": info.get("difficulty", ""),
                "bucket": info.get("bucket", ""),
                "passed": bool(detail["passed"]),
                "syntax_ok": bool(detail["syntax_ok"]),
                "tests_passed": int(detail["tests_passed"]),
                "tests_total": int(detail["tests_total"]),
                "test_fraction": float(detail["test_fraction"]),
                "response": text,
            }
        )
        if seen % 5 == 0 or seen == len(shard):
            print(
                f"rank={rank} generated {seen}/{len(shard)} of shard "
                f"(global {len(rows)}, world={world_size})",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    shard_path = args.output / f"shard-{rank:02d}.csv"
    pd.DataFrame(records, columns=SHARD_COLUMNS).to_csv(shard_path, index=False)
    if distributed:
        dist.barrier()
    if rank != 0:
        if distributed:
            dist.destroy_process_group()
        return None

    frame = pd.concat(
        [pd.read_csv(args.output / f"shard-{idx:02d}.csv") for idx in range(world_size)],
        ignore_index=True,
    )
    frame = frame.sort_values("index").reset_index(drop=True)
    if len(frame) != len(rows):
        raise SystemExit(f"eval merge got {len(frame)} rows, expected {len(rows)}")
    adapter = "base" if not args.adapter else str(Path(args.adapter).resolve())
    report = summarize(frame, adapter, str(Path(args.eval_file).resolve()))
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    frame.to_csv(args.output / "predictions.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)
    if distributed:
        dist.destroy_process_group()
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv=None):
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
