#!/usr/bin/env python3
"""Learning curves on matched [0,1000] with E/M/H thirds.

Probed and Official share one curriculum run and are drawn as two truncated
lines (different stop points)—not duplicate full curves and not endpoint markers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/venus_solve/official2/figures"
NAULC = ROOT / "results/venus_solve/official2/naulc_evals"
OUT.mkdir(parents=True, exist_ok=True)

STAGE = 1000.0 / 3.0


def load(name: str) -> dict:
    return json.loads((NAULC / name / "eval" / "metrics.json").read_text())


def lerp(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def to_display(stage: str, local: float) -> float:
    local = max(0.0, min(1000.0, float(local)))
    return {"easy": 0.0, "medium": STAGE, "hard": 2 * STAGE}[stage] + local / 1000.0 * STAGE


def smooth(y, win=25):
    y = np.asarray(y, float)
    if len(y) < win:
        return y
    return np.convolve(y, np.ones(win) / win, mode="same")


def compress(df: pd.DataFrame, stage_idx: int) -> pd.DataFrame:
    return df.assign(x=stage_idx * STAGE + df["step"] / 1000.0 * STAGE)


def shade(ax):
    ax.axvspan(0, STAGE, color="C2", alpha=0.07, lw=0)
    ax.axvspan(STAGE, 2 * STAGE, color="C1", alpha=0.07, lw=0)
    ax.axvspan(2 * STAGE, 1000, color="C3", alpha=0.07, lw=0)
    ax.set_xlim(0, 1000)
    ax.set_xlabel("Training progress (matched budget)")


def main():
    random = pd.read_csv(ROOT / "results/venus_solve/official2/baseline/full/history.csv")
    easy = pd.read_csv(ROOT / "results/venus_solve/official2/curriculum/official/easy/history.csv")
    medium = pd.read_csv(
        ROOT / "results/venus_solve/official2/curriculum/official/medium/history.csv"
    )
    hard = pd.read_csv(ROOT / "results/venus_solve/official2/curriculum/official/hard/history.csv")

    # Train-batch curriculum on display axis
    curr_tb = pd.concat(
        [compress(easy, 0), compress(medium, 1), compress(hard, 2)], ignore_index=True
    )
    curr_tb = curr_tb.assign(reward_s=smooth(curr_tb["reward_mean"].to_numpy()))
    x_probed = to_display("hard", 200)
    x_official = to_display("hard", 400)
    probed_tb = curr_tb[curr_tb["x"] <= x_probed + 1e-9]
    official_tb = curr_tb[curr_tb["x"] <= x_official + 1e-9]

    # Held-out curriculum points
    curr_ho = []
    for local, name in (
        (0, "cold_start"),
        (200, "easy_step200"),
        (400, "easy_step400"),
        (600, "easy_step600"),
        (800, "easy_step800"),
        (1000, "easy_step1000"),
    ):
        m = load(name)
        curr_ho.append((to_display("easy", local), m["mean_reward"], m["pass_rate"] * 100, True))
    m200, m800 = load("medium_step200"), load("medium_step800")
    for local, name, measured in (
        (200, "medium_step200", True),
        (400, None, False),
        (600, None, False),
        (800, "medium_step800", True),
        (1000, "medium_step1000", True),
    ):
        if measured:
            m = load(name)
            curr_ho.append(
                (to_display("medium", local), m["mean_reward"], m["pass_rate"] * 100, True)
            )
        else:
            curr_ho.append(
                (
                    to_display("medium", local),
                    lerp(200, m200["mean_reward"], 800, m800["mean_reward"], local),
                    lerp(200, m200["pass_rate"], 800, m800["pass_rate"], local) * 100,
                    False,
                )
            )
    for local, name in ((200, "hard_step200"), (400, "hard_step400")):
        m = load(name)
        curr_ho.append((to_display("hard", local), m["mean_reward"], m["pass_rate"] * 100, True))
    curr_ho.sort(key=lambda row: row[0])

    def truncate_ho(x_end):
        return [row for row in curr_ho if row[0] <= x_end + 1e-9]

    probed_ho = truncate_ho(x_probed)
    official_ho = truncate_ho(x_official)

    rand_ho = []
    for step in (0, 200, 400, 600, 800, 1000):
        m = load("cold_start" if step == 0 else f"random_step{step}")
        rand_ho.append((step, m["mean_reward"], m["pass_rate"] * 100))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.35))
    for ax in axes:
        shade(ax)

    # (a) train-batch: Random + Probed + Official (truncated lines)
    ax = axes[0]
    ax.plot(random["step"], smooth(random["reward_mean"]), color="C0", lw=1.7, label="Random")
    ax.plot(
        official_tb["x"],
        official_tb["reward_s"],
        color="C1",
        lw=1.8,
        label="Official",
    )
    ax.plot(
        probed_tb["x"],
        probed_tb["reward_s"],
        color="C2",
        ls="--",
        lw=1.8,
        label="Probed",
    )
    ax.set_ylabel("Train-batch reward")
    ax.set_title("(a) Training dynamics")
    ax.set_ylim(0.05, 0.85)
    for x, lab in ((STAGE / 2, "Easy"), (STAGE * 1.5, "Medium"), (STAGE * 2.5, "Hard")):
        ax.text(x, 0.82, lab, ha="center", va="top", fontsize=8, color="0.35")
    ax.legend(fontsize=8, loc="upper right")

    # (b) held-out Pass@1: same truncation logic
    ax = axes[1]
    ax.plot(
        [p[0] for p in rand_ho],
        [p[2] for p in rand_ho],
        color="C0",
        lw=1.7,
        marker="o",
        ms=3.2,
        label="Random",
    )
    ax.plot(
        [p[0] for p in official_ho],
        [p[2] for p in official_ho],
        color="C1",
        lw=1.8,
        marker="o",
        ms=3.2,
        label="Official",
    )
    ax.plot(
        [p[0] for p in probed_ho],
        [p[2] for p in probed_ho],
        color="C2",
        ls="--",
        lw=1.8,
        marker="o",
        ms=3.2,
        label="Probed",
    )
    # open squares only on estimated midpoints of official (shared fills)
    est = [(p[0], p[2]) for p in official_ho if not p[3]]
    if est:
        ax.plot(
            [p[0] for p in est],
            [p[1] for p in est],
            "s",
            color="0.4",
            ms=3.8,
            mfc="white",
            mew=1.0,
            label="Mid-stage fill",
        )
    ax.set_ylabel("Held-out Pass@1 (%)")
    ax.set_title("(b) Held-out evaluation")
    ymin, ymax = ax.get_ylim()
    for x, lab in ((STAGE / 2, "Easy"), (STAGE * 1.5, "Medium"), (STAGE * 2.5, "Hard")):
        ax.text(x, ymax - 0.02 * (ymax - ymin), lab, ha="center", va="top", fontsize=8, color="0.35")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT / "fig_learning_curves.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_learning_curves.pdf")
    print(f"stop x: probed={x_probed:.1f}, official={x_official:.1f}")


if __name__ == "__main__":
    main()
