"""Stratify single-phase vs multiphase training results by lesion size (Hypothesis 3).

Requires:
  - per-patient metrics from scripts/analysis/run_inference.py (needs both the
    single-phase and multiphase arms in one CSV, joined by patient_uid + fold).
  - lesion size categories from scripts/analysis/lesion_sizes.py.

Example:
    python scripts/analysis/stratify_h3.py \\
        --per-patient-metrics results/h1_effb0_32fold_8gpu/per_patient_metrics.csv \\
        --lesion-sizes results/lesion_sizes/lesion_sizes.csv \\
        --output-dir results/h3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sstats

from common import resolve_path

CAUTION_NOTE = (
    "Caution: folds are k-fold CV splits, not independent repeats, and lesion-size subgroups have "
    "small n. Treat p-values/CIs here as indicative, not confirmatory."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-patient-metrics", type=str, required=True)
    p.add_argument("--lesion-sizes", type=str, required=True)
    p.add_argument("--baseline-arm", type=str, default="baseline_single_phase")
    p.add_argument("--multiphase-arm", type=str, default="multiphase_train_single_infer")
    p.add_argument("--dice-column", type=str, default="dice_resized")
    p.add_argument("--category-column", type=str, default="size_category_fixed", choices=["size_category_fixed", "size_category_tertile"])
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def _paired_stats_for_group(sub: pd.DataFrame, dice_col: str) -> Dict[str, object]:
    diff = sub["multiphase"] - sub["baseline"]
    n = len(diff)
    if n < 2:
        return {"n_patients": n, "note": "not enough paired patients"}

    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    se_diff = std_diff / np.sqrt(n)
    t_crit = sstats.t.ppf(0.975, n - 1)
    _, p_t = sstats.ttest_rel(sub["multiphase"], sub["baseline"])
    try:
        _, p_w = sstats.wilcoxon(sub["multiphase"], sub["baseline"])
    except ValueError:
        p_w = float("nan")

    return {
        "n_patients": n,
        "mean_baseline_dice": float(sub["baseline"].mean()),
        "mean_multiphase_dice": float(sub["multiphase"].mean()),
        "mean_diff_multiphase_minus_baseline": mean_diff,
        "ci95_low": mean_diff - t_crit * se_diff,
        "ci95_high": mean_diff + t_crit * se_diff,
        "wins_multiphase": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses_multiphase": int((diff < 0).sum()),
        "paired_ttest_p": float(p_t),
        "wilcoxon_p": float(p_w),
    }


def _plot_by_category(df: pd.DataFrame, categories: List[str], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(max(7, 2.6 * len(categories)), 6))

    positions = []
    labels = []
    values = []
    pos = 1
    for cat in categories:
        sub = df[df["size_category"] == cat]
        for arm_key, arm_label in (("baseline", "single-phase"), ("multiphase", "multiphase")):
            values.append(sub[arm_key].to_numpy(dtype=np.float64))
            positions.append(pos)
            labels.append(f"{cat}\n{arm_label}")
            pos += 1
        pos += 0.6  # gap between category groups

    ax.boxplot(
        values,
        positions=positions,
        widths=0.7,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.6},
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
        boxprops={"facecolor": "#b8d8ea", "alpha": 0.9},
    )
    for x, ys in zip(positions, values):
        xs = np.full(len(ys), x, dtype=np.float64) + rng.uniform(-0.09, 0.09, size=len(ys))
        ax.scatter(xs, ys, s=28, alpha=0.85, color="#1f4e79", edgecolor="white", linewidth=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dice (native resolution)" if "native" in output_path.stem else "Dice")
    ax.set_title("H3: single-phase vs multiphase Dice by lesion size")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(resolve_path(args.per_patient_metrics))
    lesion_sizes = pd.read_csv(resolve_path(args.lesion_sizes))

    dice_col = args.dice_column
    if dice_col not in metrics.columns:
        raise SystemExit(
            f"Column '{dice_col}' not found in per-patient metrics. "
            f"Available: {list(metrics.columns)}. Use --dice-column dice_native if masks were saved."
        )

    baseline = metrics.loc[metrics["arm"] == args.baseline_arm, ["patient_uid", "fold", dice_col]].rename(columns={dice_col: "baseline"})
    multiphase = metrics.loc[metrics["arm"] == args.multiphase_arm, ["patient_uid", "fold", dice_col]].rename(columns={dice_col: "multiphase"})
    joined = baseline.merge(multiphase, on=["patient_uid", "fold"], how="inner")
    joined = joined.merge(lesion_sizes[["patient_uid", args.category_column, "largest_component_equiv_diameter_mm", "total_lesion_volume_ml"]], on="patient_uid", how="left")
    joined = joined.rename(columns={args.category_column: "size_category"})
    joined["size_category"] = joined["size_category"].fillna("unknown")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_dir / "h3_per_patient_joined.csv", index=False)

    categories = [c for c in ("small", "medium", "large") if c in joined["size_category"].unique()]
    rows = []
    for cat in categories:
        sub = joined[joined["size_category"] == cat]
        row = {"size_category": cat}
        row.update(_paired_stats_for_group(sub, dice_col))
        rows.append(row)

    # Overall (all lesion sizes combined) for reference.
    overall_row = {"size_category": "all"}
    overall_row.update(_paired_stats_for_group(joined, dice_col))
    rows.append(overall_row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_dir / "h3_by_lesion_size.csv", index=False)

    if categories:
        _plot_by_category(joined, categories, output_dir / f"h3_dice_by_lesion_size_{dice_col}.png")

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("H3: multiphase-training benefit by lesion size\n\n")
        f.write(result_df.to_string(index=False))
        f.write("\n\n")
        f.write(CAUTION_NOTE + "\n")

    print(f"Saved: {output_dir}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
