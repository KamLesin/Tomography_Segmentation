"""Split Hypothesis 1 (single-phase vs multiphase) Dice results by source dataset.

Produces two separate boxplots (patient-level, not fold-level) comparing the
baseline_single_phase arm against multiphase_train_single_infer:
  - CECT-only patients (multiphase-trained CECT vs single-phase CECT)
  - PG-only patients (multiphase-trained PG vs single-phase PG)

Input is the per-patient inference table produced by run_inference.py
(e.g. results/h1_effb0_32fold_8gpu/per_patient_metrics.csv), which has one row
per (arm, fold, patient) with a `source` column ("cect" or "full"/PG).

Example:
    python scripts/analysis/plot_h1_by_dataset.py \\
        --per-patient-metrics results/h1_effb0_32fold_8gpu/per_patient_metrics.csv \\
        --output-dir results/h1_by_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sstats

from common import resolve_path

BASELINE_ARM = "baseline_single_phase"
MULTIPHASE_ARM = "multiphase_train_single_infer"

ARM_COLORS = {BASELINE_ARM: "#4A90E2", MULTIPHASE_ARM: "#2ECC71"}
ARM_LABELS = {BASELINE_ARM: "Single-Phase Baseline (PV)", MULTIPHASE_ARM: "Multiphase Training (PV Eval)"}

SOURCE_TITLES = {
    "cect": "Hypothesis 1 (CECT only): Single-Phase vs Multiphase Training",
    "full": "Hypothesis 1 (PG only): Single-Phase vs Multiphase Training",
}
SOURCE_FILENAMES = {"cect": "h1_dice_boxplot_cect_only", "full": "h1_dice_boxplot_pg_only"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-patient-metrics", type=str, required=True)
    p.add_argument("--dice-column", type=str, default="dice_native", help="Dice column to plot (default: dice_native).")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _paired_stats(df: pd.DataFrame, dice_col: str) -> Dict[str, float]:
    sub_a = df.loc[df["arm"] == BASELINE_ARM].set_index("patient_uid")[dice_col]
    sub_b = df.loc[df["arm"] == MULTIPHASE_ARM].set_index("patient_uid")[dice_col]
    joined = pd.concat([sub_a, sub_b], axis=1, keys=["a", "b"]).dropna()
    diff = joined["b"] - joined["a"]
    n = len(diff)
    if n < 2:
        return {"n_patients": n}

    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    se_diff = std_diff / np.sqrt(n) if std_diff > 0 else 0.0
    t_crit = sstats.t.ppf(0.975, n - 1)
    _, p_t = sstats.ttest_rel(joined["b"], joined["a"])
    try:
        _, p_w = sstats.wilcoxon(joined["b"], joined["a"])
    except ValueError:
        p_w = float("nan")

    return {
        "n_patients": n,
        "mean_diff": mean_diff,
        "ci95_low": mean_diff - t_crit * se_diff,
        "ci95_high": mean_diff + t_crit * se_diff,
        "paired_ttest_p": float(p_t),
        "wilcoxon_p": float(p_w),
    }


def _plot_boxplot(
    values_per_arm: List[np.ndarray],
    arm_labels: List[str],
    colors: List[str],
    title: str,
    output_png: Path,
    output_pdf: Path,
    paired: Optional[Dict[str, float]],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "axes.edgecolor": "#2C3E50",
            "axes.linewidth": 1.1,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
        }
    )

    n_arms = len(values_per_arm)
    fig, ax = plt.subplots(figsize=(max(7.0, 3.2 * n_arms), 6.2), dpi=dpi)
    rng = np.random.default_rng(42)
    positions = np.arange(1, n_arms + 1)

    box_plot = ax.boxplot(
        values_per_arm,
        positions=positions,
        widths=0.48,
        showmeans=True,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 2.2},
        meanprops={
            "marker": "D",
            "markerfacecolor": "#FFFFFF",
            "markeredgecolor": "#111111",
            "markersize": 7.0,
            "markeredgewidth": 1.5,
        },
        whiskerprops={"color": "#2C3E50", "linewidth": 1.3},
        capprops={"color": "#2C3E50", "linewidth": 1.3},
    )
    for patch, col in zip(box_plot["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.70)
        patch.set_edgecolor("#1A252F")
        patch.set_linewidth(1.4)

    for idx, (ys, col) in enumerate(zip(values_per_arm, colors), start=1):
        xs = np.full(len(ys), idx, dtype=np.float64) + rng.uniform(-0.11, 0.11, size=len(ys))
        ax.scatter(xs, ys, s=26, alpha=0.55, color=col, edgecolors="#1A252F", linewidths=0.4, zorder=4)

    all_y = np.concatenate(values_per_arm)
    y_min, y_max = float(all_y.min()), float(all_y.max())
    y_pad = (y_max - y_min) * 0.12 if y_max > y_min else 0.05
    lower_bound = max(0.0, y_min - y_pad)
    upper_bound = min(1.0, y_max + y_pad * 1.6)

    if paired and paired.get("n_patients", 0) >= 2:
        p_val = paired["paired_ttest_p"]
        p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        signif = " (N.S.)" if p_val >= 0.05 else (" (*)" if p_val >= 0.01 else " (**)")
        stat_label = (
            f"$\\Delta$ Dice = {paired['mean_diff']:+.4f} ({p_text}{signif})\n"
            f"95% CI: [{paired['ci95_low']:+.4f}, {paired['ci95_high']:+.4f}], n={paired['n_patients']}"
        )
        x1, x2 = 1, 2
        bracket_y = y_max + y_pad * 0.4
        bracket_h = y_pad * 0.15
        ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y], lw=1.2, c="#2C3E50")
        ax.text(
            (x1 + x2) * 0.5,
            bracket_y + bracket_h + y_pad * 0.08,
            stat_label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold",
            color="#2C3E50",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9F9", edgecolor="#BDC3C7", alpha=0.95),
        )
        upper_bound = max(upper_bound, bracket_y + bracket_h + y_pad * 1.1)

    for idx, ys in enumerate(values_per_arm, start=1):
        m, s, med = float(np.mean(ys)), float(np.std(ys, ddof=1)), float(np.median(ys))
        txt = f"Mean: {m:.4f} \u00b1 {s:.4f}\nMedian: {med:.4f}\nn = {len(ys)}"
        ax.text(
            idx,
            lower_bound + (y_pad * 0.1),
            txt,
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#2C3E50",
            bbox=dict(boxstyle="square,pad=0.25", facecolor="#FFFFFF", edgecolor="#D5D8DC", alpha=0.9),
            zorder=5,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(arm_labels, fontsize=11.0, fontweight="bold", ha="center")
    ax.set_ylabel("Per-Patient Dice Coefficient", fontsize=11.5, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=14, color="#1A252F")
    ax.set_ylim(lower_bound, upper_bound)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#BDC3C7")
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], color="#111111", lw=2.2, label="Median"),
        Line2D([0], [0], marker="D", color="w", markeredgecolor="#111111", markerfacecolor="#FFFFFF", markersize=7, label="Mean (\u00b1 SEM)"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="#1A252F", markerfacecolor="#4A90E2", markersize=6, label="Patient Observation"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.95, fontsize=9.0, edgecolor="#D5D8DC")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics_path = resolve_path(args.per_patient_metrics)
    df = pd.read_csv(metrics_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in ["cect", "full"]:
        sub = df.loc[(df["source"] == source) & (df["arm"].isin([BASELINE_ARM, MULTIPHASE_ARM]))].copy()
        sub = sub.dropna(subset=[args.dice_column])
        if sub.empty:
            print(f"No data for source={source}, skipping.")
            continue

        values_per_arm = [sub.loc[sub["arm"] == a, args.dice_column].to_numpy(dtype=np.float64) for a in [BASELINE_ARM, MULTIPHASE_ARM]]
        colors = [ARM_COLORS[a] for a in [BASELINE_ARM, MULTIPHASE_ARM]]
        arm_labels = [ARM_LABELS[a] for a in [BASELINE_ARM, MULTIPHASE_ARM]]

        paired = _paired_stats(sub, args.dice_column)
        stem = SOURCE_FILENAMES[source]
        _plot_boxplot(
            values_per_arm,
            arm_labels,
            colors,
            SOURCE_TITLES[source],
            output_dir / f"{stem}.png",
            output_dir / f"{stem}.pdf",
            paired,
            args.dpi,
        )

        stats_rows = []
        for arm, vals in zip([BASELINE_ARM, MULTIPHASE_ARM], values_per_arm):
            stats_rows.append(
                {
                    "arm": arm,
                    "label": ARM_LABELS[arm],
                    "source": source,
                    "n_patients": len(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)),
                    "median": float(np.median(vals)),
                }
            )
        pd.DataFrame(stats_rows).to_csv(output_dir / f"{stem}_summary_stats.csv", index=False)
        if paired.get("n_patients", 0) >= 2:
            pd.DataFrame([paired]).to_csv(output_dir / f"{stem}_paired_test.csv", index=False)

        print(f"[{source}] wrote {stem}.png / .pdf (n_baseline={len(values_per_arm[0])}, n_multiphase={len(values_per_arm[1])})")


if __name__ == "__main__":
    main()
