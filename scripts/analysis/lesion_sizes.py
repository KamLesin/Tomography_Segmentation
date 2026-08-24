"""Compute per-patient lesion size/volume statistics from ground-truth masks.

Produces a CSV used by stratify_h3.py to test Hypothesis 3 (small vs. large
lesion benefit of multiphase training) without needing to retrain anything.

Example:
    python scripts/analysis/lesion_sizes.py \\
        --cect-root ../CECT_data_aligned \\
        --full-root ../full_data_converted_aligned \\
        --output results/lesion_sizes/lesion_sizes.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from common import ROOT, resolve_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cect-root", type=str, default="../CECT_data_aligned")
    p.add_argument("--full-root", type=str, default="../full_data_converted_aligned")
    p.add_argument("--mode", type=str, default="mixed", choices=["cect", "full", "mixed"])
    p.add_argument("--missing-phase-strategy", type=str, default="keep", choices=["drop", "keep"])
    p.add_argument("--small-max-diameter-mm", type=float, default=20.0, help="Upper bound (exclusive) for 'small'.")
    p.add_argument("--large-min-diameter-mm", type=float, default=50.0, help="Lower bound (inclusive) for 'large'.")
    p.add_argument("--output", type=str, default="results/lesion_sizes/lesion_sizes.csv")
    return p.parse_args()


def _first_existing(paths):
    for p in paths:
        if p is not None and Path(p).exists():
            return p
    return None


def _voxel_volume_mm3(affine: np.ndarray) -> float:
    try:
        return float(abs(np.linalg.det(affine[:3, :3])))
    except Exception:
        return 1.0


def _lesion_stats(mask_path: Optional[Path]) -> dict:
    empty = {
        "total_lesion_voxels": 0,
        "total_lesion_volume_ml": 0.0,
        "num_components": 0,
        "largest_component_voxels": 0,
        "largest_component_volume_ml": 0.0,
        "largest_component_equiv_diameter_mm": 0.0,
    }
    if mask_path is None:
        return empty

    img = nib.load(str(mask_path))
    arr = np.squeeze(np.asarray(img.dataobj))
    if arr.ndim != 3:
        return empty

    mask = arr > 0
    total_voxels = int(mask.sum())
    if total_voxels == 0:
        return empty

    voxel_volume_mm3 = _voxel_volume_mm3(img.affine)

    from scipy.ndimage import label as cc_label

    labeled, num_components = cc_label(mask.astype(np.uint8))
    component_sizes = np.bincount(labeled.ravel())[1:]  # drop background label 0
    largest_component_voxels = int(component_sizes.max()) if component_sizes.size else 0
    largest_component_volume_ml = largest_component_voxels * voxel_volume_mm3 / 1000.0
    equiv_diameter_mm = (6.0 * largest_component_volume_ml * 1000.0 / np.pi) ** (1.0 / 3.0) if largest_component_volume_ml > 0 else 0.0

    return {
        "total_lesion_voxels": total_voxels,
        "total_lesion_volume_ml": total_voxels * voxel_volume_mm3 / 1000.0,
        "num_components": int(num_components),
        "largest_component_voxels": largest_component_voxels,
        "largest_component_volume_ml": largest_component_volume_ml,
        "largest_component_equiv_diameter_mm": float(equiv_diameter_mm),
    }


def _categorize_fixed(diameter_mm: float, small_max: float, large_min: float) -> str:
    if diameter_mm <= 0:
        return "none"
    if diameter_mm < small_max:
        return "small"
    if diameter_mm >= large_min:
        return "large"
    return "medium"


def main() -> None:
    args = parse_args()
    from multiphase_seg.data import build_patient_records

    cect_root = resolve_path(args.cect_root) if args.mode in ("cect", "mixed") else None
    full_root = resolve_path(args.full_root) if args.mode in ("full", "mixed") else None

    records = build_patient_records(
        mode=args.mode,
        cect_root=cect_root,
        full_root=full_root,
        missing_phase_strategy=args.missing_phase_strategy,
    )
    print(f"Discovered {len(records)} patient records (mode={args.mode}).")

    rows = []
    for rec in records:
        mask_path = _first_existing([rec.mask_paths.get("PV"), rec.mask_paths.get("A"), rec.mask_paths.get("D")])
        stats = _lesion_stats(Path(mask_path) if mask_path else None)
        rows.append(
            {
                "patient_uid": rec.patient_uid,
                "patient_id": rec.patient_id,
                "source": rec.source,
                "mask_path": str(mask_path) if mask_path else "",
                **stats,
                "size_category_fixed": _categorize_fixed(
                    stats["largest_component_equiv_diameter_mm"], args.small_max_diameter_mm, args.large_min_diameter_mm
                ),
            }
        )

    df = pd.DataFrame(rows)

    has_lesion = df["total_lesion_volume_ml"] > 0
    df["size_category_tertile"] = "none"
    if has_lesion.sum() >= 3:
        df.loc[has_lesion, "size_category_tertile"] = pd.qcut(
            df.loc[has_lesion, "total_lesion_volume_ml"], q=3, labels=["small", "medium", "large"]
        ).astype(str)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved lesion size stats: {output_path} ({len(df)} patients)")
    print(df["size_category_fixed"].value_counts())


if __name__ == "__main__":
    main()
