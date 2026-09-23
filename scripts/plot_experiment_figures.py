#!/usr/bin/env python3
"""Generate experiment figures from existing Venus/probe artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/venus_solve/official2/figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name: str):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_probe_analysis():
    pred = pd.read_csv(ROOT / "model-cache/probe/predictions.csv")
    test = pred[pred["split"] == "test"]
    # Venus train probe scores
    frame = pd.read_parquet(ROOT / "grpo/data/venus_solve/base/train.parquet")
    rows = []
    for item in frame["extra_info"].tolist():
        if isinstance(item, str):
            item = json.loads(item)
        elif hasattr(item, "item"):
            item = item.item()
        rows.append(item)
    venus = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    hb = ax.hexbin(
        test["real_difficulty"],
        test["pred_difficulty"],
        gridsize=35,
        cmap="viridis",
        mincnt=1,
    )
    lims = [
        min(test["real_difficulty"].min(), test["pred_difficulty"].min()),
        max(test["real_difficulty"].max(), test["pred_difficulty"].max()),
    ]
    ax.plot(lims, lims, color="white", lw=1.0, alpha=0.8)
    ax.set_xlabel("True Codeforces rating")
    ax.set_ylabel("Predicted rating")
    ax.set_title("(a) Held-out linear probe")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="count")

    ax = axes[1]
    order = ["Easy", "Medium", "Hard"]
    data = [venus.loc[venus["difficulty"] == d, "probe_score"].to_numpy() for d in order]
    parts = ax.violinplot(data, positions=range(len(order)), showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.7)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("Probe score")
    ax.set_title("(b) Venus score by official label")
    rho, _ = spearmanr(
        venus["probe_score"],
        venus["difficulty"].map({"Easy": 0, "Medium": 1, "Hard": 2}),
    )
    ax.text(
        0.02,
        0.98,
        f"Spearman vs label = {rho:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_probe_analysis.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_probe_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_probe_analysis.pdf")
    print("wrote", OUT / "fig_probe_analysis.png")
    return rho


def _smooth(y, win=25):
    if len(y) < win:
        return y
    kernel = np.ones(win) / win
    return np.convolve(y, kernel, mode="same")


def fig_learning_curves():
    """Held-out reward/Pass@1 on a matched [0,1000] axis (E/M/H thirds)."""
    # Prefer the dedicated redraw if naulc evals exist; else train-batch fallback.
    naulc = ROOT / "results/venus_solve/official2/naulc_evals"
    if (naulc / "cold_start/eval/metrics.json").is_file():
        # Delegate to compact held-out plot (no internal ckpt aliases in legend).
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/plot_heldout_learning_curves.py")],
            check=True,
        )
        return
    random = pd.read_csv(ROOT / "results/venus_solve/official2/baseline/full/history.csv")
    easy = pd.read_csv(ROOT / "results/venus_solve/official2/curriculum/official/easy/history.csv")
    medium = pd.read_csv(
        ROOT / "results/venus_solve/official2/curriculum/official/medium/history.csv"
    )
    hard = pd.read_csv(ROOT / "results/venus_solve/official2/curriculum/official/hard/history.csv")
    # Compress E/M/H into equal thirds of [0,1000]
    stage = 1000.0 / 3.0

    def compress(df, stage_idx):
        return df.assign(x=stage_idx * stage + df["step"] / 1000.0 * stage)

    official = pd.concat(
        [compress(easy, 0), compress(medium, 1), compress(hard, 2)],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    for ax in axes:
        ax.axvspan(0, stage, color="C2", alpha=0.06, lw=0)
        ax.axvspan(stage, 2 * stage, color="C1", alpha=0.06, lw=0)
        ax.axvspan(2 * stage, 1000, color="C3", alpha=0.06, lw=0)
        ax.set_xlim(0, 1000)
        ax.set_xlabel("Training progress (matched budget)")
    ax = axes[0]
    ax.plot(random["step"], _smooth(random["reward_mean"].to_numpy()), label="Random", lw=1.6)
    ax.plot(
        official["x"],
        _smooth(official["reward_mean"].to_numpy()),
        label="Official",
        lw=1.6,
    )
    ax.set_ylabel("Train batch reward")
    ax.set_title("(a) Reward")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    ax.plot(random["step"], _smooth(random["pass_rate"].to_numpy()), label="Random", lw=1.6)
    ax.plot(
        official["x"],
        _smooth(official["pass_rate"].to_numpy()),
        label="Official",
        lw=1.6,
    )
    ax.set_ylabel("Train batch pass rate")
    ax.set_title("(b) Pass rate")
    fig.tight_layout()
    fig.savefig(OUT / "fig_learning_curves.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_learning_curves.pdf")


def fig_curriculum_diagnostics():
    frame = pd.read_parquet(ROOT / "grpo/data/venus_solve/base/train.parquet")
    rows = []
    for item in frame["extra_info"].tolist():
        if isinstance(item, str):
            item = json.loads(item)
        elif hasattr(item, "item"):
            item = item.item()
        rows.append(item)
    venus = pd.DataFrame(rows)
    # probed order = sort by probe_score ascending (easy first)
    probed = venus.sort_values("probe_score").reset_index(drop=True)
    probed["pos"] = np.arange(len(probed))
    # official order = Easy, Medium, Hard blocks
    official = pd.concat(
        [
            venus[venus["difficulty"] == d].sample(frac=1.0, random_state=0)
            for d in ("Easy", "Medium", "Hard")
        ],
        ignore_index=True,
    )
    official["pos"] = np.arange(len(official))
    rng = np.random.default_rng(0)
    random = venus.sample(frac=1.0, random_state=0).reset_index(drop=True)
    random["pos"] = np.arange(len(random))

    def rolling_score(df, win=40):
        return df["probe_score"].rolling(win, min_periods=1).mean()

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    ax = axes[0]
    ax.plot(random["pos"], rolling_score(random), label="Random", lw=1.5)
    ax.plot(official["pos"], rolling_score(official), label="Official", lw=1.5)
    ax.plot(probed["pos"], rolling_score(probed), label="Probed", lw=1.5)
    ax.set_xlabel("Training-row position")
    ax.set_ylabel("Rolling mean probe score")
    ax.set_title("(a) Difficulty order")
    ax.legend(fontsize=8)

    ax = axes[1]
    # composition in 10 bins
    bins = 10

    def composition(df):
        edges = np.linspace(0, len(df), bins + 1).astype(int)
        mat = []
        for i in range(bins):
            chunk = df.iloc[edges[i] : edges[i + 1]]
            mat.append(
                [
                    (chunk["difficulty"] == d).mean()
                    for d in ("Easy", "Medium", "Hard")
                ]
            )
        return np.asarray(mat)

    mat = composition(probed)
    ax.stackplot(
        np.arange(bins),
        mat[:, 0],
        mat[:, 1],
        mat[:, 2],
        labels=["Easy", "Medium", "Hard"],
        alpha=0.85,
    )
    ax.set_xlabel("Curriculum stage bin")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.set_title("(b) Probed composition")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    # with prompts_per_step=1 and 2 GPUs -> 2 distinct IDs/step
    routes = ["Random", "Official", "Probed"]
    values = [2, 2, 2]
    ax.bar(routes, values, color=["C0", "C1", "C2"])
    ax.set_ylabel("Distinct problem IDs / step")
    ax.set_title("(c) Matched batch width")
    ax.set_ylim(0, 4)
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_curriculum_diagnostics.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_curriculum_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig_curriculum_diagnostics.pdf")


def main():
    fig_probe_analysis()
    fig_learning_curves()
    fig_curriculum_diagnostics()
    pred = pd.read_csv(ROOT / "model-cache/probe/predictions.csv")
    tr = pred[pred["split"] == "train"]
    te = pred[pred["split"] == "test"]
    mean = float(tr["real_difficulty"].mean())
    err = te["real_difficulty"].to_numpy() - mean
    linear = json.loads((ROOT / "model-cache/probe/metrics.json").read_text())["test"]
    frame = pd.read_parquet(ROOT / "grpo/data/venus_solve/base/train.parquet")
    rows = [
        json.loads(x) if isinstance(x, str) else (x.item() if hasattr(x, "item") else x)
        for x in frame["extra_info"]
    ]
    venus = pd.DataFrame(rows)
    meta = {
        "linear_probe_test": linear,
        "mean_predictor_test": {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
        },
        "venus_probe_label_spearman": float(
            spearmanr(
                venus["probe_score"],
                venus["difficulty"].map({"Easy": 0, "Medium": 1, "Hard": 2}),
            ).correlation
        ),
        "batch_distinct_ids_per_step": {"random": 2, "official": 2, "probed": 2},
    }
    (OUT / "figure_metrics.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", OUT / "figure_metrics.json")


if __name__ == "__main__":
    main()
