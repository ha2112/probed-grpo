#!/usr/bin/env python3
"""Compute nAULC from held-out naulc_evals metrics (mean_reward vs global step)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/venus_solve/official2/naulc_evals"


def load_reward(name: str) -> float | None:
    path = OUT / name / "eval" / "metrics.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if "mean_reward" in data:
        return float(data["mean_reward"])
    return float(data["pass_rate"])  # fallback only if rewards not recomputed


def naulc(points: list[tuple[int, float]]) -> float:
    points = sorted(points)
    if len(points) < 2:
        raise ValueError(f"need >=2 points, got {points}")
    s0, sK = points[0][0], points[-1][0]
    area = 0.0
    for (s0_, r0), (s1, r1) in zip(points[:-1], points[1:]):
        area += 0.5 * (r0 + r1) * (s1 - s0_)
    return area / (sK - s0)


def collect(prefix: str, steps: list[int], offset: int = 0) -> list[tuple[int, float]]:
    pts = []
    for step in steps:
        reward = load_reward(f"{prefix}_step{step}")
        if reward is None:
            continue
        pts.append((offset + step, reward))
    return pts


def main():
    cold = load_reward("cold_start")
    report: dict = {"missing": [], "curves": {}}

    # Random: cold + steps on full GRPO (horizon 1000)
    random_pts: list[tuple[int, float]] = []
    if cold is not None:
        random_pts.append((0, cold))
    else:
        report["missing"].append("cold_start")
    for step in (200, 400, 600, 800, 1000):
        r = load_reward(f"random_step{step}")
        if r is None and step == 1000:
            r = load_reward("random_final")
        if r is None:
            report["missing"].append(f"random_step{step}")
            continue
        random_pts.append((step, r))
    if len(random_pts) >= 2:
        report["curves"]["random"] = {
            "points": random_pts,
            "nAULC": naulc(random_pts),
            "horizon": random_pts[-1][0],
        }

    # Official report endpoint = hard step 400 => global 2400
    # Probed report endpoint = hard step 200 => global 2200
    # Shared early curriculum trajectory when both are checkpoint selections.
    shared: list[tuple[int, float]] = []
    if cold is not None:
        shared.append((0, cold))
    for stage, offset in (("easy", 0), ("medium", 1000), ("hard", 2000)):
        for step in (200, 400, 600, 800, 1000):
            r = load_reward(f"{stage}_step{step}")
            if r is None:
                report["missing"].append(f"{stage}_step{step}")
                continue
            shared.append((offset + step, r))

    def truncate(pts: list[tuple[int, float]], horizon: int) -> list[tuple[int, float]]:
        kept = [p for p in pts if p[0] <= horizon]
        if kept and kept[-1][0] != horizon:
            # allow exact endpoint missing only if we have it named
            pass
        return kept

    for name, horizon in (("official", 2400), ("probed", 2200)):
        pts = truncate(shared, horizon)
        # ensure endpoint present
        end_name = "hard_step400" if name == "official" else "hard_step200"
        end_r = load_reward(end_name)
        if end_r is not None and (not pts or pts[-1][0] != horizon):
            pts = [p for p in pts if p[0] < horizon] + [(horizon, end_r)]
        if len(pts) >= 2:
            report["curves"][name] = {
                "points": pts,
                "nAULC": naulc(pts),
                "horizon": horizon,
                "note": "same official E→M→H trajectory; endpoints differ",
            }

    if "random" in report["curves"] and "probed" in report["curves"]:
        # Compare on common horizon of random (1000) using curriculum early pts
        # vs full random — only if we also want matched-horizon; primary table
        # uses each route's own training horizon as in the paper text.
        report["deltas"] = {}
        for other in ("official", "probed"):
            if other in report["curves"]:
                report["deltas"][f"{other}_minus_random"] = (
                    report["curves"][other]["nAULC"] - report["curves"]["random"]["nAULC"]
                )

    out = OUT / "naulc_summary.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
