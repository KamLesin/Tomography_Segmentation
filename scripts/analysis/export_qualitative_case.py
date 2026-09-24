#!/usr/bin/env python3
"""Export a 3-view qualitative example for one CT volume with GT and model prediction overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_volume(path: str | Path) -> np.ndarray:
    vol = np.asarray(nib.load(str(path)).dataobj)
    return np.squeeze(vol).astype(np.float32)


def choose_slice(mask: np.ndarray, axis: str) -> int:
    if mask.size == 0:
        return 0
    mask = mask > 0
    if axis == "axial":
        sizes = mask.sum(axis=(0, 1))
    elif axis == "coronal":
        sizes = mask.sum(axis=(0, 2))
    else:
        sizes = mask.sum(axis=(1, 2))
    if np.all(sizes == 0):
        return max(0, mask.shape[axis_index(axis)] // 2)
    return int(np.argmax(sizes))


def axis_index(axis: str) -> int:
    return {"axial": 2, "coronal": 1, "sagittal": 0}[axis]


def window_ct(vol: np.ndarray, lo: float = -200.0, hi: float = 300.0) -> np.ndarray:
    clipped = np.clip(vol, lo, hi)
    return (clipped - lo) / max(hi - lo, 1e-6)


def get_plane_slice(vol: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "axial":
        return vol[:, :, idx]
    if axis == "coronal":
        return vol[:, idx, :]
    return vol[idx, :, :]


def plot_panel(ax, ct, gt, pred, title: str, axis: str, idx: int, show_legend: bool = False):
    ct_slice = get_plane_slice(ct, axis, idx)
    gt_slice = get_plane_slice(gt, axis, idx)
    pred_slice = get_plane_slice(pred, axis, idx)

    ax.imshow(window_ct(ct_slice), cmap="gray", origin="lower")
    if gt_slice.any():
        ax.contour(gt_slice.astype(bool).T, levels=[0.5], colors="lime", linewidths=1.5)
    if pred_slice.any():
        ax.contour(pred_slice.astype(bool).T, levels=[0.5], colors="red", linewidths=1.2)
    ax.set_title(f"{title} ({axis})", fontsize=12)
    ax.axis("off")

    if show_legend:
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="lime", lw=2, label="GT"),
            Line2D([0], [0], color="red", lw=2, label="Prediction"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--patient-id", type=str, default="case")
    args = parser.parse_args()

    ct = load_volume(args.ct)
    gt = (load_volume(args.gt) > 0).astype(np.uint8)
    pred = (load_volume(args.pred) > 0).astype(np.uint8)

    if ct.shape != gt.shape or ct.shape != pred.shape:
        raise ValueError(f"Shape mismatch: CT={ct.shape}, GT={gt.shape}, pred={pred.shape}")

    combined = np.logical_or(gt > 0, pred > 0)
    axial_idx = choose_slice(combined, "axial")
    coronal_idx = choose_slice(combined, "coronal")
    sagittal_idx = choose_slice(combined, "sagittal")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    plot_panel(axes[0], ct, gt, pred, "Axial", "axial", axial_idx, show_legend=False)
    plot_panel(axes[1], ct, gt, pred, "Coronal", "coronal", coronal_idx, show_legend=False)
    plot_panel(axes[2], ct, gt, pred, "Sagittal", "sagittal", sagittal_idx, show_legend=True)
    fig.suptitle(f"Qualitative segmentation example — {args.patient_id}", fontsize=14)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
