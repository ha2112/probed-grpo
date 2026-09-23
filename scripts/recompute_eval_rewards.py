#!/usr/bin/env python3
"""Attach mean_reward to existing solve_grpo_eval metrics.json files (offline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpo.solve_grpo_eval import summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "eval_dirs",
        nargs="*",
        type=Path,
        help="Directories that contain predictions.csv (+ metrics.json)",
    )
    parser.add_argument(
        "--glob",
        default="",
        help="Optional glob under results/, e.g. 'venus_solve/official2/**/eval'",
    )
    args = parser.parse_args()
    dirs = list(args.eval_dirs)
    if args.glob:
        dirs.extend(sorted((ROOT / "results").glob(args.glob)))
    if not dirs:
        raise SystemExit("pass eval dirs or --glob")

    updated = []
    for directory in dirs:
        pred = directory / "predictions.csv"
        if not pred.is_file():
            continue
        frame = pd.read_csv(pred)
        meta = {}
        metrics_path = directory / "metrics.json"
        if metrics_path.is_file():
            meta = json.loads(metrics_path.read_text())
        report = summarize(
            frame,
            meta.get("adapter", "unknown"),
            meta.get("eval_file", ""),
        )
        if "world_size" in meta:
            report["world_size"] = meta["world_size"]
        metrics_path.write_text(json.dumps(report, indent=2) + "\n")
        updated.append((str(directory), report["pass_rate"], report["mean_reward"]))
        print(
            f"{directory}: pass={report['pass_rate']:.4f} reward={report['mean_reward']:.4f}"
        )

    out = ROOT / "results/venus_solve/official2/figures/endpoint_rewards.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(updated, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
