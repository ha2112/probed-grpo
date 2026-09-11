"""Build three order-controlled CodeContests corpora for Afterburner GRPO."""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCE = "deepmind/code_contests"
CODEFORCES_SOURCE = 2
ROUTES = ("random", "official", "probed")

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Solve the following programming "
    "problem in Python, respecting its input/output format and constraints. "
    "Enclose your reasoning in <thinking> </thinking> and your complete solution "
    "in <solution> </solution>. Put the solution code in one markdown code block "
    "with the python language identifier."
)

USER_TEMPLATE = """## Problem Description
{description}

## Original Solution
{baseline_solution}

## Output Format
- First reason inside <thinking> </thinking>.
- Then provide the complete Python 3 solution inside <solution> </solution>.
- Put the solution in one markdown code block with the python language identifier.
- Use the original solution as a reference, preserve correctness, and improve it when possible.
"""


def _rating(problem):
    return int(problem.get("cf_rating", 0) or 0)


def problem_id(problem):
    return f"cf:{int(problem.get('cf_contest_id', 0) or 0)}:{problem.get('cf_index', '')}"


def problem_key(split, problem):
    return f"{split}:{problem_id(problem)}"


def _is_python(language):
    return str(language).lower() in {"1", "python", "python2", "python3"}


def baseline_code(problem):
    """Return the first Python solution, independent of route ordering."""
    solutions = problem.get("solutions") or {}
    if isinstance(solutions, dict):
        codes = solutions.get("solution") or solutions.get("code") or []
        languages = solutions.get("language") or []
        if isinstance(codes, str):
            codes = [codes]
        if not isinstance(languages, list):
            languages = [languages]
        candidates = zip(codes, languages or [None] * len(codes))
    elif isinstance(solutions, list):
        candidates = (
            (
                item.get("code") or item.get("solution") or "",
                item.get("language"),
            )
            if isinstance(item, dict)
            else (item, None)
            for item in solutions
        )
    else:
        candidates = ()

    unlabelled = ""
    for code, language in candidates:
        if not isinstance(code, str) or not code.strip():
            continue
        if language is None and not unlabelled:
            unlabelled = code.strip()
        if _is_python(language):
            return code.strip()
    return unlabelled


def collect_tests(problem):
    tests = []
    for group_name in ("public_tests", "private_tests", "generated_tests"):
        group = problem.get(group_name) or {}
        inputs = group.get("input") or []
        outputs = group.get("output") or []
        tests.extend(
            {"group": group_name, "input": input_text, "output": output_text}
            for input_text, output_text in zip(inputs, outputs)
        )
    return tests


def select_codeforces(problems, limit=-1):
    """Select one canonical row set before applying any route ordering."""
    selected = [
        problem
        for problem in problems
        if problem.get("source") == CODEFORCES_SOURCE
        and _rating(problem) > 0
        and baseline_code(problem)
        and collect_tests(problem)
    ]
    selected.sort(key=lambda problem: problem_id(problem))
    if limit > 0:
        selected = selected[:limit]
    return selected


def order_problems(problems, split, route, scores, seed):
    if route not in ROUTES:
        raise ValueError(f"Unknown route {route!r}; expected one of {ROUTES}")
    ordered = list(problems)
    if route == "random":
        # Start from canonical IDs so source traversal order cannot affect the shuffle.
        ordered.sort(key=problem_id)
        random.Random(f"{seed}:{split}").shuffle(ordered)
    elif route == "official":
        ordered.sort(key=lambda problem: (_rating(problem), problem_id(problem)))
    else:
        missing = [problem_key(split, problem) for problem in ordered if problem_key(split, problem) not in scores]
        if missing:
            raise KeyError(f"Missing probe scores for {len(missing)} problems: {', '.join(missing[:3])}")
        ordered.sort(key=lambda problem: (float(scores[problem_key(split, problem)]), problem_id(problem)))
    return ordered


def make_record(problem, split, probe_score):
    """Create route-independent content; only physical row order may differ."""
    baseline = baseline_code(problem)
    tests = collect_tests(problem)
    return {
        "schema_version": 1,
        "data_source": "codecontests_afterburner",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    description=problem.get("description", ""),
                    baseline_solution=baseline,
                ),
            },
        ],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "baseline_passed": True,
                "baseline_solution": baseline,
                "tests": tests,
            },
        },
        "extra_info": {
            "split": split,
            "problem_id": problem_id(problem),
            "name": problem.get("name", ""),
            "cf_rating": _rating(problem),
            "probe_difficulty": float(probe_score),
            "time_limit_seconds": float((problem.get("time_limit") or {}).get("seconds", 0) or 0),
            "memory_limit_bytes": int(problem.get("memory_limit_bytes", 0) or 0),
        },
    }


def build_records(problems, split, route, scores, seed=42, limit=-1):
    selected = select_codeforces(problems, limit)
    missing = [problem_key(split, problem) for problem in selected if problem_key(split, problem) not in scores]
    if missing:
        raise KeyError(f"Missing probe scores for {len(missing)} problems: {', '.join(missing[:3])}")
    records_by_id = {
        problem_id(problem): make_record(problem, split, scores[problem_key(split, problem)])
        for problem in selected
    }
    ordered = order_problems(selected, split, route, scores, seed)
    return [records_by_id[problem_id(problem)] for problem in ordered]


def probe_fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_score_cache(path, fingerprint):
    if not path.exists():
        return {}
    scores = {}
    with path.open(encoding="utf-8") as handle:
        metadata = json.loads(handle.readline())
        if metadata != {"type": "metadata", "probe_sha256": fingerprint}:
            raise ValueError(f"Probe score cache does not match {path}")
        for line in handle:
            if line.strip():
                item = json.loads(line)
                scores[item["key"]] = float(item["probe_difficulty"])
    return scores


def initialize_score_cache(path, fingerprint):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "metadata", "probe_sha256": fingerprint}) + "\n",
        encoding="utf-8",
    )


def append_score(path, key, score):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "probe_difficulty": float(score)}) + "\n")


class DifficultyScorer:
    """Score statements with the probe produced by this repository."""

    def __init__(self, checkpoint_path, device):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch
        from torch import nn
        import linear_probe

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metadata = checkpoint["metadata"]
        if metadata["prompt"] != linear_probe.SYSTEM_PROMPT:
            raise ValueError("Probe checkpoint prompt differs from linear_probe.SYSTEM_PROMPT")
        self.device = linear_probe.resolve_device(device)
        self.tokenizer, self.model = linear_probe.load_backbone(
            metadata["model"], self.device, metadata["dtype"]
        )
        self.max_length = metadata["max_length"]
        self.embed_problem = linear_probe.embed_problem
        self.np = np
        self.torch = torch
        self.probe = nn.Linear(checkpoint["hidden_size"], 1)
        self.probe.load_state_dict(checkpoint["state_dict"])
        self.probe.eval()

    def score(self, description):
        embedding = self.embed_problem(
            description, self.tokenizer, self.model, self.device, self.max_length
        )
        with self.torch.inference_mode():
            return float(self.probe(self.torch.from_numpy(self.np.asarray(embedding))).item())


def ensure_scores(problems, split, scores, cache_path, scorer_factory, scorer=None):
    for position, problem in enumerate(problems, start=1):
        key = problem_key(split, problem)
        if key in scores:
            continue
        if scorer is None:
            scorer = scorer_factory()
        score = scorer.score(problem.get("description", ""))
        scores[key] = score
        append_score(cache_path, key, score)
        print(f"{split}: probed {position}/{len(problems)} {key} -> {score:.3f}", flush=True)
    return scorer


def _record_hash(record):
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_corpora(train_problems, validation_problems, scores, output_dir, seed):
    from datasets import Dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    route_orders = {}
    content_hashes = None
    for route in ROUTES:
        records = build_records(train_problems, "train", route, scores, seed)
        route_dir = output_dir / route
        route_dir.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(records).to_parquet(str(route_dir / "train.parquet"))
        current_hashes = sorted(_record_hash(record) for record in records)
        if content_hashes is not None and current_hashes != content_hashes:
            raise RuntimeError("Route corpora do not contain identical records")
        content_hashes = current_hashes
        route_orders[route] = [record["extra_info"]["problem_id"] for record in records]
        print(f"{route}: wrote {len(records)} training rows", flush=True)

    validation = build_records(validation_problems, "validation", "official", scores, seed)
    shared_dir = output_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(validation).to_parquet(str(shared_dir / "validation.parquet"))
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "train_count": len(content_hashes or []),
        "validation_count": len(validation),
        "train_record_hashes": content_hashes or [],
        "route_orders": route_orders,
        "validation_order": [record["extra_info"]["problem_id"] for record in validation],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATA_SOURCE)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="valid")
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=ROOT / "data-cache/huggingface/datasets",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=ROOT / "results/linear-probe-afterburner/probe.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("CODECONTESTS_GRPO_DATA_DIR", ROOT / "grpo/data")),
    )
    parser.add_argument("--max-train", type=int, default=-1)
    parser.add_argument("--max-validation", type=int, default=-1)
    args = parser.parse_args()
    if args.max_train == 0 or args.max_validation == 0:
        parser.error("sample limits must be -1 or positive")
    return args


def main():
    from datasets import load_dataset

    args = parse_args()
    if not args.probe.is_file():
        raise SystemExit(f"Probe checkpoint not found: {args.probe}")
    train = select_codeforces(
        load_dataset(args.dataset, split=args.train_split, cache_dir=str(args.data_cache)),
        args.max_train,
    )
    validation = select_codeforces(
        load_dataset(
            args.dataset,
            split=args.validation_split,
            cache_dir=str(args.data_cache),
        ),
        args.max_validation,
    )
    fingerprint = probe_fingerprint(args.probe)
    cache_path = args.output_dir / "probe_scores.jsonl"
    initialize_score_cache(cache_path, fingerprint)
    scores = load_score_cache(cache_path, fingerprint)
    scorer_factory = lambda: DifficultyScorer(args.probe, args.device)
    scorer = ensure_scores(train, "train", scores, cache_path, scorer_factory)
    ensure_scores(
        validation,
        "validation",
        scores,
        cache_path,
        scorer_factory,
        scorer,
    )
    write_corpora(train, validation, scores, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
