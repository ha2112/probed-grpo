"""Conda-native 2-GPU GRPO trainer for Venus solve-from-scratch (pass/fail only)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_cache import DEFAULT_MODEL_DIR, ensure_model  # noqa: E402
from grpo.venus_solve_corpus import (  # noqa: E402
    SOLVE_RESPONSE_PREFIX,
    make_solve_prompt,
)
from grpo.venus_solve_reward import (  # noqa: E402
    extract_solve_code,
    format_score,
    gradient_continuation,
    venus_solve_reward_batch,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ.get("WORLD_SIZE", "1"))
        visible = torch.cuda.device_count()
        if visible < world or local_rank >= visible:
            raise SystemExit(
                f"DDP world_size={world} local_rank={local_rank} but only "
                f"{visible} visible CUDA device(s); "
                "check Slurm --gpus-per-task and do not override CUDA_VISIBLE_DEVICES"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device)
        print(
            f"rank={dist.get_rank()} local_rank={local_rank} "
            f"device={device} visible={torch.cuda.device_count()} "
            f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')!r}",
            flush=True,
        )
        return True, dist.get_rank(), dist.get_world_size(), device
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for solve_grpo_train")
    return False, 0, 1, torch.device("cuda:0")


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


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
    return names


class ParquetPromptDataset(Dataset):
    def __init__(self, path: Path, *, rewrite_code_first: bool = True):
        frame = pd.read_parquet(path)
        self.rows = frame.to_dict(orient="records")
        for row in self.rows:
            prompt = row["prompt"]
            if isinstance(prompt, str):
                prompt = json.loads(prompt)
            elif hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            row["prompt"] = list(prompt)
            info = row["extra_info"]
            if isinstance(info, str):
                info = json.loads(info)
            elif hasattr(info, "item"):
                info = info.item()
            row["extra_info"] = dict(info)
            # Rebuild prompts so older parquet (thinking-first) still trains code-first.
            if rewrite_code_first:
                description = row["extra_info"].get("description")
                if not description and row["prompt"]:
                    description = row["prompt"][-1].get("content", "")
                if description:
                    row["prompt"] = make_solve_prompt(description)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_identity(batch):
    return batch


def build_prompt_ids(
    tokenizer,
    prompt,
    device,
    max_prompt_length,
    *,
    response_prefix: str = SOLVE_RESPONSE_PREFIX,
):
    """Tokenize chat prompt and append a forced code-fence prefix."""
    encoded = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    ids = encoded["input_ids"]
    mask = encoded["attention_mask"]
    if response_prefix:
        prefix_ids = tokenizer.encode(
            response_prefix, add_special_tokens=False, return_tensors="pt"
        )
        ones = torch.ones_like(prefix_ids)
        ids = torch.cat([ids, prefix_ids], dim=1)
        mask = torch.cat([mask, ones], dim=1)
    if ids.shape[1] > max_prompt_length:
        raise ValueError(f"Prompt length {ids.shape[1]} exceeds {max_prompt_length}")
    return ids.to(device), mask.to(device)


def build_policy(args, device, dtype):
    model_path = ensure_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, dtype),
        local_files_only=True,
    )
    if args.resume and (Path(args.resume) / "adapter_config.json").is_file():
        backbone = PeftModel.from_pretrained(backbone, args.resume, is_trainable=True)
    else:
        targets = lora_targets(backbone.config.num_hidden_layers, args.last_n_layers)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
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
    backbone.config.use_cache = False
    return tokenizer, backbone.to(device)


def sequence_logprob(model, input_ids, attention_mask, prompt_lengths, response_token_counts=None):
    """Mean log-prob of the useful response tokens only.

    A sum lets long token loops dominate GRPO. Tokens after the closing fence or
    after a trailing cycle are excluded so they cannot be reinforced.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1].float()
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logprob = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(token_logprob)
    for index, prompt_length in enumerate(prompt_lengths):
        # labels align to tokens[1:]; response starts at prompt_length.
        start = max(int(prompt_length) - 1, 0)
        usable = attention_mask[index, 1:][start:]
        if response_token_counts is not None:
            usable = usable.clone()
            keep = max(int(response_token_counts[index]), 0)
            if keep < usable.numel():
                usable[keep:] = 0
        mask[index, start:] = usable
    lengths = mask.sum(dim=-1).clamp_min(1)
    return (token_logprob * mask).sum(dim=-1) / lengths


def kept_new_token_count(tokenizer, new_ids) -> int:
    """How many generated tokens belong to the code, not the loop after it."""
    total = int(new_ids.numel())
    if total == 0:
        return 0
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    kept = gradient_continuation(text)
    if not kept:
        return 0
    if kept == text:
        return total
    lo, hi = 1, total
    best = total
    while lo <= hi:
        mid = (lo + hi) // 2
        decoded = tokenizer.decode(new_ids[:mid], skip_special_tokens=True)
        if decoded.startswith(kept) or kept.startswith(decoded) and len(decoded) >= len(kept):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def _encode_stop_ids(tokenizer, texts):
    """Token id sequences used by the fast fence stopper (no per-step decode)."""
    stops = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            stops.append(ids)
    return stops


def _endswith_any(haystack, needles):
    for needle in needles:
        n = len(needle)
        if n and len(haystack) >= n and haystack[-n:] == needle:
            return True
    return False


class CloseFenceStop(StoppingCriteria):
    """Stop when a closing fence / </solution> token sequence appears. No decode."""

    def __init__(self, stop_ids, prompt_len: int):
        self.stop_ids = list(stop_ids)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids, scores, **kwargs):
        flags = []
        for seq in input_ids:
            new = seq[self.prompt_len :].tolist()
            flags.append(_endswith_any(new, self.stop_ids))
        return torch.tensor(flags, device=input_ids.device, dtype=torch.bool)


class BanTokenCycle(LogitsProcessor):
    """Block the next token of a short cycle so '1, 1, 1' cannot run to max tokens."""

    def __init__(self, max_cycle: int = 6, repeats: int = 4):
        self.max_cycle = max_cycle
        self.repeats = repeats
        self._need = max_cycle * repeats

    def __call__(self, input_ids, scores):
        for index, seq in enumerate(input_ids):
            if seq.numel() < self._need:
                continue
            tokens = seq[-self._need :].tolist()
            for cycle in range(1, self.max_cycle + 1):
                span = cycle * self.repeats
                if len(tokens) < span:
                    continue
                tail = tokens[-span:]
                unit = tail[:cycle]
                if tail == unit * self.repeats:
                    scores[index, unit[0]] = torch.finfo(scores.dtype).min
                    break
        return scores


def generation_controls(tokenizer, prompt_len: int):
    stop_ids = _encode_stop_ids(
        tokenizer,
        ("```", "\n```", "</solution>", "\n</solution>"),
    )
    return {
        "stopping_criteria": StoppingCriteriaList(
            [CloseFenceStop(stop_ids, prompt_len)]
        ),
        "logits_processor": LogitsProcessorList([BanTokenCycle()]),
        "repetition_penalty": 1.05,
    }


def generate_rollouts(model, tokenizer, batch_ids, batch_mask, args, pad_id, max_len):
    """Generate with KV-cache on. Checkpointing stays off only for this call.

    Leaving use_cache=False during generate made 8×H100 as slow as 1 GPU.
    """
    policy = unwrap(model)
    had_checkpoint = bool(getattr(policy, "is_gradient_checkpointing", False))
    if had_checkpoint:
        policy.gradient_checkpointing_disable()
    policy.config.use_cache = True
    policy.eval()
    try:
        with torch.no_grad():
            return policy.generate(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.rollouts,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                **generation_controls(tokenizer, max_len),
            )
    finally:
        policy.config.use_cache = False
        if had_checkpoint:
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )


def grpo_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """rewards: [batch * group_size] -> within-group standardized advantages."""
    reshaped = rewards.view(-1, group_size)
    mean = reshaped.mean(dim=-1, keepdim=True)
    std = reshaped.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return ((reshaped - mean) / std).reshape(-1)


def save_policy(model, tokenizer, directory: Path, meta: dict):
    directory.mkdir(parents=True, exist_ok=True)
    unwrap(model).save_pretrained(directory / "adapter")
    tokenizer.save_pretrained(directory / "tokenizer")
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def train(args):
    distributed, rank, world_size, device = setup_distributed()
    set_seed(args.seed + rank)
    dtype = args.dtype
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        dtype = "float16"

    dataset = ParquetPromptDataset(args.train_file)
    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=args.seed)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.prompts_per_step,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collate_identity,
        drop_last=True,
    )
    if len(loader) == 0:
        raise SystemExit("Train loader is empty; reduce prompts-per-step or grow corpus")

    tokenizer, policy = build_policy(args, device, dtype)
    if rank == 0:
        policy.print_trainable_parameters()
    model = (
        DDP(policy, device_ids=[device.index], find_unused_parameters=False)
        if distributed
        else policy
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    epoch = 0
    history = []
    if rank == 0:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"GRPO solve train: rows={len(dataset)} prompts/step={args.prompts_per_step} "
            f"rollouts={args.rollouts} max_steps={args.max_steps} world={world_size}",
            flush=True,
        )

    while step < args.max_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= args.max_steps:
                break
            prompt_tensors = []
            prompt_masks = []
            prompt_lengths = []
            extras = []
            for row in batch:
                ids, mask = build_prompt_ids(
                    tokenizer, row["prompt"], device, args.max_prompt_length
                )
                prompt_tensors.append(ids[0])
                prompt_masks.append(mask[0])
                prompt_lengths.append(int(ids.shape[1]))
                extras.append(row["extra_info"])

            # Left-pad for decoder-only generate.
            max_len = max(prompt_lengths)
            pad_id = tokenizer.pad_token_id
            batch_ids = torch.full(
                (len(batch), max_len), pad_id, dtype=torch.long, device=device
            )
            batch_mask = torch.zeros(
                (len(batch), max_len), dtype=torch.long, device=device
            )
            for index, (ids, mask) in enumerate(zip(prompt_tensors, prompt_masks)):
                batch_ids[index, max_len - ids.numel() :] = ids
                batch_mask[index, max_len - mask.numel() :] = mask

            generated = generate_rollouts(
                model, tokenizer, batch_ids, batch_mask, args, pad_id, max_len
            )
            response_texts = []
            expanded_extras = []
            expanded_prompt_lengths = []
            response_token_counts = []
            for prompt_index, prompt_length in enumerate(prompt_lengths):
                for rollout in range(args.rollouts):
                    flat = prompt_index * args.rollouts + rollout
                    # Left-padded prompt occupies the last prompt_length tokens
                    # of the max_len prefix; new tokens start after max_len.
                    new_tokens = generated[flat, max_len:]
                    # Prefix was part of the prompt (forced); prepend for reward/extract.
                    continuation = tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    )
                    response_texts.append(SOLVE_RESPONSE_PREFIX + continuation)
                    expanded_extras.append(extras[prompt_index])
                    # Left-padded generate window ends at max_len; response tokens
                    # start there (not at the unpadded prompt length).
                    expanded_prompt_lengths.append(max_len)
                    response_token_counts.append(
                        kept_new_token_count(tokenizer, new_tokens)
                    )

            # Each rank must score *its own* rollouts. Broadcasting rank0 rewards
            # onto rank1 sequences (different prompts/samples) corrupts GRPO grads
            # under DDP — that bug affected the 2-GPU base run (world=2).
            rewards_list, passed_list = venus_solve_reward_batch(
                response_texts,
                expanded_extras,
                raise_on_error=False,
            )
            if rank == 0 and (
                step == 0
                or (step % args.log_every == 0 and float(sum(rewards_list)) == 0.0)
            ):
                preview = (response_texts[0][:240] if response_texts else "").replace(
                    "\n", "\\n"
                )
                print(
                    f"judge_mode={os.environ.get('VENUS_JUDGE', 'auto')} "
                    f"sample_response_chars={len(response_texts[0]) if response_texts else 0} "
                    f"nonempty_code={sum(1 for t in response_texts if extract_solve_code(t))} "
                    f"format_gt0={sum(1 for t in response_texts if format_score(t) > 0)} "
                    f"preview={preview!r}",
                    flush=True,
                )

            rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
            advantages = grpo_advantages(rewards, args.rollouts)

            # generated is left-padded to max_len; response tokens start after that.
            attn = (generated != pad_id).long()
            unwrap(model).train()
            logp = sequence_logprob(
                model,
                generated,
                attn,
                expanded_prompt_lengths,
                response_token_counts,
            )
            loss = -(advantages.detach() * logp).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            # Sync scalar metrics for logging (optional; ranks may see different prompts).
            if distributed:
                stats = torch.tensor(
                    [
                        float(rewards.mean()),
                        float(np.mean(passed_list)),
                        float(advantages.std()),
                        float(loss.detach()),
                    ],
                    device=device,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.AVG)

            step += 1
            if rank == 0 and (step % args.log_every == 0 or step == args.max_steps):
                if distributed:
                    reward_mean, pass_rate, adv_std, loss_v = stats.tolist()
                else:
                    reward_mean = float(rewards.mean())
                    pass_rate = float(np.mean(passed_list))
                    adv_std = float(advantages.std())
                    loss_v = float(loss.detach())
                row = {
                    "step": step,
                    "loss": loss_v,
                    "reward_mean": reward_mean,
                    "pass_rate": pass_rate,
                    "advantage_std": adv_std,
                }
                history.append(row)
                print(json.dumps(row), flush=True)
                pd.DataFrame(history).to_csv(args.save_dir / "history.csv", index=False)
            if rank == 0 and args.save_every > 0 and step % args.save_every == 0:
                save_policy(
                    model,
                    tokenizer,
                    args.save_dir / f"step-{step:06d}",
                    {"step": step, "train_file": str(args.train_file)},
                )

    if distributed:
        dist.barrier()
    if rank == 0:
        save_policy(
            model,
            tokenizer,
            args.save_dir / "final",
            {
                "step": step,
                "train_file": str(args.train_file),
                "max_steps": args.max_steps,
                "rollouts": args.rollouts,
            },
        )
        print(f"Saved final policy to {args.save_dir / 'final'}", flush=True)
    if distributed:
        dist.destroy_process_group()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--resume", type=Path, default=None, help="LoRA adapter dir")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--prompts-per-step", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1536,
        help="Generation budget after the forced <solution>```python prefix",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--last-n-layers", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args(argv)
    # Job 4074 (and older e2e launches) baked STAGE_STEPS=TOTAL/3=200 into the
    # parent bash. Curriculum torchrun still re-reads this file, so bump here.
    override = os.environ.get("GRPO_MAX_STEPS_OVERRIDE")
    if override:
        args.max_steps = int(override)
        print(f"GRPO_MAX_STEPS_OVERRIDE -> max_steps={args.max_steps}", flush=True)
    elif "curriculum" in Path(args.save_dir).as_posix() and args.max_steps == 200:
        print(
            f"Curriculum step budget override: max_steps 200 -> 600 "
            f"(save_dir={args.save_dir})",
            flush=True,
        )
        args.max_steps = 600
    if not args.train_file.is_file():
        parser.error(f"missing train file: {args.train_file}")
    if min(args.max_steps, args.prompts_per_step, args.rollouts) < 1:
        parser.error("max-steps/prompts-per-step/rollouts must be positive")
    return args


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
