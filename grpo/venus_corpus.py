"""Build three order-controlled Venus corpora for Afterburner GRPO."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from artifact_cache import (  # noqa: E402
    DEFAULT_MODEL_DIR, VENUS_DATASET_ID, VENUS_DATASET_REVISION,
    ensure_model, load_cached_dataset,
)
from grpo.codeforces_corpus import (  # noqa: E402
    DifficultyScorer, ROUTES, append_score, initialize_score_cache,
    load_score_cache, probe_fingerprint, _record_hash,
)

DATA_SOURCE = VENUS_DATASET_ID
DIFFICULTIES = {"Easy": 0, "Medium": 1, "Hard": 2}
EFFICIENCY_INSTRUCTIONS = {
    "time": "time efficient", "memory": "memory efficient",
    "integral": "both time and memory efficient",
}
# Prompt and objectives follow Afterburner/grpo/afterburner_dataset.py.
SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question and provides an original solution, then the Assistant improve it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the improved solution.
The reasoning process and solution are enclosed within <thinking> </thinking> and <solution> </solution> tags, respectively.
For example, "<thinking>reasoning_process</thinking><solution>improved_solution</solution>".
"""
USER_TEMPLATE = """## Instructions
You are an expert competitive programmer who excels at solving algorithm problems in multiple programming languages.
Your task is to implement a solution to the following problem in python.
## Problem Description
{description}
## Original Solution
{code}
## Original Performance
Passed: {passed} / Time: {time} / Memory: {memory} / Integral: {integral}
## Output Format
- Provide the complete solution code in **one markdown code block** with appropriate language identifier.
- Fix the original solution if it was not passed. Optimize the {efficiency} performance if the original solution was passed.
- EXCLUDE ALL explanations, code comments, import/package/library statements, additional classes or functions outside of the starter code scope, or starting code like `if __name__ == "__main__":` or `func main()` or `package main` or `using namespace std;`.
"""


def select_venus(problems, limit=-1):
    """Validate the native schema; never require Codeforces ratings or sources."""
    selected = sorted(problems, key=lambda p: int(p["problem_id"]))
    ids = [int(p["problem_id"]) for p in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Venus problem IDs")
    if limit > 0:
        selected = selected[:limit]
    for p in selected:
        if p["difficulty"] not in DIFFICULTIES or not p["question_content"].strip():
            raise ValueError(f"Invalid Venus statement/difficulty: {p['problem_id']}")
        tests = json.loads(p["test_cases"])
        if not tests or any(not isinstance(t.get(k), str) for t in tests for k in ("input", "output")):
            raise ValueError(f"Invalid Venus tests: {p['problem_id']}")
        if "==Code Submission==" not in p["test_case_runners"] or not p["test_case_evaluator"].strip():
            raise ValueError(f"Missing Venus test harness: {p['problem_id']}")
        if not p["solutions"]:
            raise ValueError(f"Missing Venus baseline solutions: {p['problem_id']}")
        for baseline in p["solutions"]:
            if not isinstance(baseline["code"], str) or not isinstance(baseline["passed"], bool):
                raise ValueError(f"Invalid Venus baseline: {p['problem_id']}")
            # Failed Venus attempts can have empty code and +inf timeout metrics.
            if any(math.isnan(baseline[k]) or baseline[k] < 0 or
                   (baseline["passed"] and not math.isfinite(baseline[k])) for k in EFFICIENCY_INSTRUCTIONS):
                raise ValueError(f"Invalid Venus baseline measurements: {p['problem_id']}")
    if not selected:
        raise ValueError("No Venus problems selected")
    return selected


def make_records(problems, split, seed):
    records = []
    for p in select_venus(problems):
        for objective, instruction in EFFICIENCY_INSTRUCTIONS.items():
            record_id = f"venus:{p['problem_id']}:{objective}"
            # Choose once per example, independent of route and source traversal.
            baseline = random.Random(f"{seed}:{split}:{record_id}").choice(p["solutions"])
            instance = {k: p[k] for k in ("test_case_runners", "test_case_evaluator", "test_cases")}
            records.append({
                "schema_version": 1, "data_source": DATA_SOURCE, "ability": "code",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        description=p["question_content"], efficiency=instruction, **baseline)},
                ],
                "reward_model": {"style": "rule", "ground_truth": json.dumps(baseline, ensure_ascii=False)},
                "extra_info": {
                    "split": split, "problem_id": int(p["problem_id"]), "record_id": record_id,
                    "name": p["title"], "difficulty": p["difficulty"],
                    "description": p["question_content"], "probe_difficulty": 0.0,
                    "efficiency_instruction": objective, "case_multiply": 64,
                    # Strings avoid nested NumPy object arrays at the verl boundary.
                    "instance": json.dumps(instance, ensure_ascii=False),
                },
            })
    return records


def prepare_records(records, tokenizer, max_prompt_length, batch_size=1):
    retained = [r for r in records if len(tokenizer.apply_chat_template(
        r["prompt"], tokenize=True, add_generation_prompt=True)) <= max_prompt_length]
    filtered = len(records) - len(retained)
    tail = len(retained) % batch_size
    if tail:
        retained = retained[:-tail]
    if not retained:
        raise ValueError("No complete batch remains after Venus prompt filtering; check sample/batch limits")
    return retained, {"selected": len(records), "overlong": filtered, "batch_tail": tail}


def ensure_scores(records, scores, cache_path, scorer_factory, scorer=None):
    for row in records:
        info = row["extra_info"]
        key = f"{info['split']}:venus:{info['problem_id']}"
        if key not in scores:
            if scorer is None:
                scorer = scorer_factory()
            score = scorer.score(info["description"])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite probe score: {key}")
            scores[key] = score
            append_score(cache_path, key, score)
            print(f"{key} -> {score:.3f}", flush=True)
        info["probe_difficulty"] = float(scores[key])
    return scorer


def order_records(records, split, route, seed):
    ordered = sorted(records, key=lambda r: (r["extra_info"]["problem_id"], r["extra_info"]["record_id"]))
    if route == "random":
        random.Random(f"{seed}:{split}").shuffle(ordered)
    elif route == "official":
        ordered.sort(key=lambda r: DIFFICULTIES[r["extra_info"]["difficulty"]])
    elif route == "probed":
        ordered.sort(key=lambda r: r["extra_info"]["probe_difficulty"])
    else:
        raise ValueError(f"Unknown route {route!r}")
    return ordered


def write_corpora(train, validation, output_dir, seed, preparation):
    from datasets import Dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    route_orders = {}
    for route in ROUTES:
        rows = order_records(train, "train", route, seed)
        destination = output_dir / route / "train.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(rows).to_parquet(str(destination))
        route_orders[route] = [r["extra_info"]["record_id"] for r in rows]
        print(f"{route}: wrote {len(rows)} Venus training examples", flush=True)
    validation = order_records(validation, "test", "official", seed)
    destination = output_dir / "shared/validation.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(validation).to_parquet(str(destination))
    manifest = {
        "schema_version": 3, "seed": seed, "train_count": len(train),
        "validation_count": len(validation), "preparation": preparation,
        "train_record_hashes": sorted(_record_hash(r) for r in train),
        "route_orders": route_orders,
        "validation_order": [r["extra_info"]["record_id"] for r in validation],
        "files_sha256": {str(p.relative_to(output_dir)): probe_fingerprint(p) for p in
                         [*(output_dir / route / "train.parquet" for route in ROUTES), destination]},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="test")
    parser.add_argument("--data-cache", type=Path, default=ROOT / "data-cache/huggingface/datasets")
    parser.add_argument("--download-only", action="store_true", help="Cache and validate Venus without a probe or GPU")
    parser.add_argument("--probe", type=Path, default=ROOT / "model-cache/probe/probe.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-smoke-probe", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.environ.get("VENUS_GRPO_DATA_DIR", ROOT / "grpo/data/venus")))
    parser.add_argument("--max-train", type=int, default=-1, help="Limit problems before expanding objectives")
    parser.add_argument("--max-validation", type=int, default=-1)
    args = parser.parse_args(argv)
    if any(n != -1 and n < 1 for n in (args.max_train, args.max_validation)):
        parser.error("sample limits must be -1 or positive")
    if min(args.max_prompt_length, args.train_batch_size) < 1:
        parser.error("prompt length and batch size must be positive")
    return args


def main():
    args = parse_args()
    train = select_venus(load_cached_dataset(DATA_SOURCE, args.train_split, args.data_cache), args.max_train)
    validation = select_venus(load_cached_dataset(DATA_SOURCE, args.validation_split, args.data_cache), args.max_validation)
    if {p["problem_id"] for p in train} & {p["problem_id"] for p in validation}:
        raise ValueError("Venus training and validation problem IDs overlap")
    print(f"Venus cached and validated: {len(train)} train / {len(validation)} validation problems", flush=True)
    if args.download_only:
        return
    if not args.probe.is_file():
        raise SystemExit(f"Codeforces-trained probe checkpoint not found: {args.probe}")
    if not args.allow_smoke_probe:
        report_path = args.probe.parent / "metrics.json"
        if not report_path.is_file():
            raise SystemExit("Full probe metrics.json is required beside probe.pt")
        report = json.loads(report_path.read_text())
        if report.get("max_samples") != -1 or report.get("epochs", 0) < 80:
            raise SystemExit("Train the full 80-epoch probe first; --allow-smoke-probe is for smoke tests only")
    from transformers import AutoTokenizer

    model_path = ensure_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    train, train_filter = prepare_records(make_records(train, args.train_split, args.seed), tokenizer,
                                         args.max_prompt_length, args.train_batch_size)
    validation, validation_filter = prepare_records(make_records(validation, args.validation_split, args.seed),
                                                   tokenizer, args.max_prompt_length)
    preparation = {
        "dataset": DATA_SOURCE, "dataset_revision": VENUS_DATASET_REVISION,
        "train_split": args.train_split, "validation_split": args.validation_split,
        "seed": args.seed, "model": str(model_path), "max_prompt_length": args.max_prompt_length,
        "train_batch_size": args.train_batch_size, "max_train": args.max_train,
        "max_validation": args.max_validation, "train_filter": train_filter,
        "validation_filter": validation_filter, "probe_sha256": probe_fingerprint(args.probe),
        "smoke": args.allow_smoke_probe,
    }
    print(json.dumps(preparation, indent=2), flush=True)
    fingerprint = hashlib.sha256(json.dumps({
        "preparation": preparation, "records": [_record_hash(r) for r in train + validation],
    }, sort_keys=True).encode()).hexdigest()
    cache_path = args.output_dir / "probe_scores.jsonl"
    initialize_score_cache(cache_path, fingerprint)
    scores = load_score_cache(cache_path, fingerprint)
    factory = lambda: DifficultyScorer(args.probe, args.device)
    scorer = ensure_scores(train, scores, cache_path, factory)
    ensure_scores(validation, scores, cache_path, factory, scorer)
    write_corpora(train, validation, args.output_dir, args.seed, preparation)


if __name__ == "__main__":
    main()
