"""Split Venus train rows by the dataset's official Easy/Medium/Hard label.

This does not use the Codeforces probe. The shared test file is unchanged.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "grpo/data/venus_solve/base/train.parquet"
DST = ROOT / "grpo/data/venus_solve/official"
NAMES = {"Easy": "easy", "Medium": "medium", "Hard": "hard"}


def _info(value):
    if isinstance(value, str):
        return json.loads(value)
    if hasattr(value, "item"):
        return value.item()
    return dict(value)


def main():
    frame = pd.read_parquet(SRC)
    rows = frame.to_dict(orient="records")
    counts = Counter()
    DST.mkdir(parents=True, exist_ok=True)
    for name in NAMES.values():
        (DST / name).mkdir(parents=True, exist_ok=True)
    grouped = {name: [] for name in NAMES.values()}
    for row in rows:
        info = _info(row["extra_info"])
        label = info.get("difficulty")
        if label not in NAMES:
            raise SystemExit(f"Unexpected official difficulty: {label!r}")
        bucket = NAMES[label]
        info["bucket"] = bucket
        info["probe_group"] = -1
        row["extra_info"] = info
        grouped[bucket].append(row)
        counts[bucket] += 1
    for bucket, subset in grouped.items():
        path = DST / bucket / "train.parquet"
        pd.DataFrame(subset).to_parquet(path)
        print(f"wrote {len(subset)} -> {path}", flush=True)
    expected = {"easy": 238, "medium": 516, "hard": 230}
    if dict(counts) != expected:
        raise SystemExit(f"official counts {dict(counts)} != {expected}")
    print("official train counts", dict(counts), flush=True)


if __name__ == "__main__":
    main()
