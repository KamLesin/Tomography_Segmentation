"""Export qualitative visualizations comparing ground truth vs. predicted masks.

Requires per-patient metrics produced with `run_inference.py --save-masks`
(needs ct_path, gt_mask_path, pred_mask_path columns).

For each selected patient this writes a PNG (CT slice windowed, with ground
truth contour in green and prediction contour in red) at the slice with the
largest ground-truth lesion area. The NIfTI files referenced by ct_path /
gt_mask_path / pred_mask_path can be opened together in ITK-SNAP or 3D Slicer
for full 3D inspection.

Example:
    python scripts/analysis/export_visuals.py \\
        --per-patient-metrics results/h1_effb0_32fold_8gpu/per_patient_metrics.csv \\
        --lesion-sizes results/lesion_sizes/lesion_sizes.csv \\
        --arm multiphase_train_single_infer \\
        --output-dir results/h1_effb0_32fold_8gpu/visualizations
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from common import resolve_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-patient-metrics", type=str, required=True)
    p.add_argument("--lesion-sizes", type=str, default=None, help="Optional, used to pick smallest/largest lesion cases.")
    p.add_argument("--arm", type=str, required=True)
    p.add_argument("--dice-column", type=str, default="dice_native")
    p.add_argument("--strategy", type=str, default="extremes", choices=["extremes", "worst", "best", "all"])
    p.add_argument("--num-cases", type=int, default=6)
    p.add_argument("--hu-window", type=float, nargs=2, default=(-200.0, 300.0))
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def _select_patients(df: pd.DataFrame, dice_col: str, strategy: str, num_cases: int) -> pd.DataFrame:
    df = df.dropna(subset=[dice_col]).copy()
    if strategy == "all":
        return df
    if strategy == "worst":
        return df.nsmallest(num_cases, dice_col)
    if strategy == "best":
        return df.nlargest(num_cases, dice_col)

    # "extremes": mix of worst, best, and lesion-size extremes if available.
    picks = []
    half = max(1, num_cases // 2)
    picks.append(df.nsmallest(half, dice_col))
    picks.append(df.nlargest(num_cases - half, dice_col))
    return pd.concat(picks).drop_duplicates(subset=["patient_uid", "fold"])


def _export_case(row: pd.Series, dice_col: str, hu_window, output_dir: Path) -> Optional[Path]:
    ct_path = row.get("ct_path")
    gt_path = row.get("gt_mask_path")
    pred_path = row.get("pred_mask_path")
    if not ct_path or not gt_path or not isinstance(pred_path, str) or not pred_path:
        return None

    ct = np.squeeze(np.asarray(nib.load(ct_path).dataobj))
    gt = np.squeeze(np.asarray(nib.load(gt_path).dataobj)) > 0
    pred = np.squeeze(np.asarray(nib.load(pred_path).dataobj)) > 0

    if gt.sum() > 0:
        z = int(np.argmax(gt.sum(axis=(0, 1))))
    elif pred.sum() > 0:
        z = int(np.argmax(pred.sum(axis=(0, 1))))
    else:
        z = ct.shape[2] // 2

    ct_slice = np.clip(ct[:, :, z], hu_window[0], hu_window[1])
    ct_slice = (ct_slice - hu_window[0]) / max(hu_window[1] - hu_window[0], 1e-6)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ct_slice.T, cmap="gray", origin="lower")
    if gt[:, :, z].any():
        ax.contour(gt[:, :, z].T, levels=[0.5], colors="lime", linewidths=1.5)
    if pred[:, :, z].any():
        ax.contour(pred[:, :, z].T, levels=[0.5], colors="red", linewidths=1.2)
    ax.set_title(
        f"{row.get('patient_id')} | fold={row.get('fold')} | {dice_col}={row.get(dice_col):.3f}\n"
        f"green=ground truth, red=prediction",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{row.get('patient_id')}_fold{int(row.get('fold')):02d}_slice{z:03d}.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    df = pd.read_csv(resolve_path(args.per_patient_metrics))
    df = df[df["arm"] == args.arm]
    if df.empty:
        raise SystemExit(f"No rows found for arm={args.arm} in {args.per_patient_metrics}")

    dice_col = args.dice_column if args.dice_column in df.columns else "dice_resized"

    if args.lesion_sizes:
        lesion_sizes = pd.read_csv(resolve_path(args.lesion_sizes))
        df = df.merge(
            lesion_sizes[["patient_uid", "largest_component_equiv_diameter_mm", "size_category_fixed"]],
            on="patient_uid",
            how="left",
        )

    selected = _select_patients(df, dice_col, args.strategy, args.num_cases)
    output_dir = resolve_path(args.output_dir)

    exported = []
    for _, row in selected.iterrows():
        out_path = _export_case(row, dice_col, tuple(args.hu_window), output_dir)
        if out_path is not None:
            exported.append(str(out_path))
            print(f"[saved] {out_path}")
        else:
            print(f"[skip] {row.get('patient_id')} fold={row.get('fold')} missing mask paths (rerun run_inference.py with --save-masks)")

    print(f"Exported {len(exported)} visualization(s) to {output_dir}")


if __name__ == "__main__":
    main()
