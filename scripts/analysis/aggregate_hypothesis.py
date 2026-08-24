"""Aggregate a hypothesis run's fold_results.csv into stats, paired tests, and a plot.

Works for H1, H2, and H4 (any run produced by scripts/training/run_hypothesis.py).
For H4, pass --configs-dir to relabel arms by their *actual* encoder backbone
(the "small"/"large" folder names only reflect CLI arguments, not the backbone).

Examples:
    python scripts/analysis/aggregate_hypothesis.py \\
        --fold-results ../results_from_apl19/h1_effb0_32fold_8gpu/fold_results.csv \\
        --pairs baseline_single_phase:multiphase_train_single_infer \\
        --output-dir results/h1 --title "H1: single-phase vs multiphase training"

    python scripts/analysis/aggregate_hypothesis.py \\
        --fold-results ../results_from_apl19/h4_effb0_reuseSmall_res34_trainLarge_32fold_8gpu/fold_results.csv \\
        --configs-dir ../results_from_apl19/h4_effb0_reuseSmall_res34_trainLarge_32fold_8gpu/configs \\
        --pairs small_single_phase:small_multiphase --pairs large_single_phase:large_multiphase \\
        --output-dir results/h4 --title "H4: backbone size vs multiphase benefit"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sstats

from common import ROOT, load_yaml, resolve_path

CAUTION_NOTE = (
    "Caution: folds are k-fold CV splits, not independent repeats (training sets overlap heavily "
    "across folds), so these p-values/CIs are optimistic relative to true independent replication."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold-results", type=str, required=True)
    p.add_argument("--configs-dir", type=str, default=None, help="Relabel arms using model.encoder_backbone from saved configs.")
    p.add_argument("--pairs", type=str, action="append", default=[], help="arm_a:arm_b, repeatable. Paired stats computed as (b - a).")
    p.add_argument("--dice-column", type=str, default="best_val_dice")
    p.add_argument("--order", type=str, nargs="+", default=None, help="Arm display order for the plot.")
    p.add_argument("--title", type=str, default="Hypothesis results")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def _arm_label(arm: str, configs_dir: Optional[Path]) -> str:
    if configs_dir is None:
        return arm
    cfg_path = configs_dir / f"{arm}.yaml"
    if not cfg_path.exists():
        return arm
    cfg = load_yaml(cfg_path)
    backbone = str(cfg.get("model", {}).get("encoder_backbone", "?"))
    is_multiphase = cfg.get("data", {}).get("force_phase_input") is None
    regime = "multiphase" if is_multiphase else "single_phase"
    return f"{backbone}_{regime}"


def _paired_stats(df: pd.DataFrame, arm_a: str, arm_b: str, dice_col: str) -> Dict[str, object]:
    a = df.loc[df["arm"] == arm_a].set_index("fold")[dice_col]
    b = df.loc[df["arm"] == arm_b].set_index("fold")[dice_col]
    joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    diff = joined["b"] - joined["a"]
    n = len(diff)
    if n < 2:
        return {"arm_a": arm_a, "arm_b": arm_b, "n": n, "note": "not enough paired folds"}

    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    se_diff = std_diff / np.sqrt(n)
    t_crit = sstats.t.ppf(0.975, n - 1)
    t_stat, p_t = sstats.ttest_rel(joined["b"], joined["a"])
    try:
        _, p_w = sstats.wilcoxon(joined["b"], joined["a"])
    except ValueError:
        p_w = float("nan")

    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_folds": n,
        "mean_a": float(joined["a"].mean()),
        "mean_b": float(joined["b"].mean()),
        "mean_diff_b_minus_a": mean_diff,
        "std_diff": std_diff,
        "ci95_low": mean_diff - t_crit * se_diff,
        "ci95_high": mean_diff + t_crit * se_diff,
        "wins_b": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses_b": int((diff < 0).sum()),
        "paired_ttest_p": float(p_t),
        "wilcoxon_p": float(p_w),
    }


def _plot_boxplot(df: pd.DataFrame, dice_col: str, order: List[str], title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    values_per_arm = [df.loc[df["label"] == label, dice_col].to_numpy(dtype=np.float64) for label in order]

    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(order)), 6))
    positions = np.arange(1, len(order) + 1)
    ax.boxplot(
        values_per_arm,
        positions=positions,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.8},
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
        boxprops={"facecolor": "#b8d8ea", "alpha": 0.9},
    )
    for idx, ys in enumerate(values_per_arm, start=1):
        xs = np.full(len(ys), idx, dtype=np.float64) + rng.uniform(-0.09, 0.09, size=len(ys))
        ax.scatter(xs, ys, s=32, alpha=0.85, color="#1f4e79", edgecolor="white", linewidth=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("Dice")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(resolve_path(args.fold_results))
    configs_dir = resolve_path(args.configs_dir) if args.configs_dir else None
    df["label"] = df["arm"].apply(lambda a: _arm_label(a, configs_dir))

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_stats = (
        df.groupby(["arm", "label"])[args.dice_column]
        .agg(n="count", mean="mean", std="std", median="median", min="min", max="max")
        .reset_index()
    )
    group_stats.to_csv(output_dir / "summary_stats.csv", index=False)

    pair_rows = []
    for pair in args.pairs:
        arm_a, arm_b = pair.split(":")
        pair_rows.append(_paired_stats(df, arm_a.strip(), arm_b.strip(), args.dice_column))
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(output_dir / "paired_tests.csv", index=False)

    order = args.order or sorted(df["label"].unique().tolist())
    _plot_boxplot(df, args.dice_column, order, args.title, output_dir / "dice_boxplot.png")

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"{args.title}\n\n")
        f.write(group_stats.to_string(index=False))
        f.write("\n\n")
        if pair_rows:
            f.write(pd.DataFrame(pair_rows).to_string(index=False))
            f.write("\n\n")
        f.write(CAUTION_NOTE + "\n")

    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
