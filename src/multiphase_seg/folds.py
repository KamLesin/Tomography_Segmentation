from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from .data import PatientRecord


def create_cv_folds(
    records: Sequence[PatientRecord],
    n_folds: int,
    seed: int = 42,
    balance_by_source: bool = True,
) -> pd.DataFrame:
    """Create patient-level CV fold assignment.

    One patient can only appear in one fold.
    """

    if n_folds <= 1:
        raise ValueError("n_folds must be > 1")

    patient_count = len(records)
    if n_folds > patient_count:
        raise ValueError(f"n_folds={n_folds} is larger than patient count={patient_count}")

    rng = random.Random(seed)

    source_buckets: Dict[str, List[PatientRecord]] = {}
    if balance_by_source:
        for rec in records:
            source_buckets.setdefault(rec.source, []).append(rec)
    else:
        source_buckets = {"all": list(records)}

    rows = []
    # Rotate starting fold per source bucket to avoid leaving higher-index folds empty
    # when some source buckets are smaller than n_folds.
    fold_offset = 0
    for _, bucket in sorted(source_buckets.items()):
        rng.shuffle(bucket)
        for idx, rec in enumerate(bucket):
            rows.append(
                {
                    "patient_uid": rec.patient_uid,
                    "patient_id": rec.patient_id,
                    "source": rec.source,
                    "fold": (fold_offset + idx) % n_folds,
                }
            )
        fold_offset = (fold_offset + len(bucket)) % n_folds

    df = pd.DataFrame(rows).sort_values(["fold", "source", "patient_id"]).reset_index(drop=True)
    return df


def save_folds_csv(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def suggested_fold_counts(n_patients: int) -> List[int]:
    """Return practical fold counts keeping validation fold sizes usable."""
    candidates = [5, 6, 8, 10, 12, 16]
    return [k for k in candidates if k < n_patients and (n_patients // k) >= 8]


def fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["fold", "source"]).size().rename("patients").reset_index().sort_values(["fold", "source"])
    )


def _first_existing_path(paths: Sequence[Path | None]) -> Path | None:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def _safe_volume_shape(path: Path | None) -> tuple[int, int, int] | None:
    if path is None:
        return None
    try:
        arr = np.asarray(nib.load(str(path)).dataobj)
        arr = np.squeeze(arr)
        if arr.ndim != 3:
            return None
        return int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    except Exception:
        return None


def _safe_mask_stats(path: Path | None, compute_components: bool) -> Dict[str, int | float]:
    empty = {
        "lesion_voxels": 0,
        "mask_voxels": 0,
        "lesion_slices": 0,
        "lesion_components_3d": 0,
    }
    if path is None:
        return empty

    try:
        arr = np.asarray(nib.load(str(path)).dataobj)
        arr = np.squeeze(arr)
        if arr.ndim != 3:
            return empty
        mask = arr > 0
        lesion_voxels = int(mask.sum())
        mask_voxels = int(mask.size)
        lesion_slices = int(np.count_nonzero(np.any(mask, axis=(0, 1))))

        lesion_components = 0
        if compute_components and lesion_voxels > 0:
            try:
                from scipy.ndimage import label as cc_label

                _, lesion_components = cc_label(mask.astype(np.uint8))
                lesion_components = int(lesion_components)
            except Exception:
                lesion_components = 0

        return {
            "lesion_voxels": lesion_voxels,
            "mask_voxels": mask_voxels,
            "lesion_slices": lesion_slices,
            "lesion_components_3d": lesion_components,
        }
    except Exception:
        return empty


def build_fold_statistics(
    records: Sequence[PatientRecord],
    folds_df: pd.DataFrame,
    n_folds: int,
    compute_lesion_components: bool = False,
) -> pd.DataFrame:
    """Build per-fold balance statistics.

    Stats are computed at patient level and then aggregated per fold.
    Patient assignment remains intact: one patient_uid -> one fold.
    """

    rec_map: Dict[str, PatientRecord] = {r.patient_uid: r for r in records}

    patient_rows: List[Dict[str, int | float | str]] = []
    for _, row in folds_df.iterrows():
        patient_uid = str(row["patient_uid"])
        fold = int(row["fold"])
        rec = rec_map.get(patient_uid)
        if rec is None:
            continue

        image_path = _first_existing_path([rec.image_paths.get("PV"), rec.image_paths.get("A"), rec.image_paths.get("D")])
        image_shape = _safe_volume_shape(image_path)
        slices = int(image_shape[2]) if image_shape is not None else 0

        mask_path = _first_existing_path([rec.mask_paths.get("PV"), rec.mask_paths.get("A"), rec.mask_paths.get("D")])
        mask_stats = _safe_mask_stats(mask_path, compute_components=compute_lesion_components)

        patient_rows.append(
            {
                "fold": fold,
                "patient_uid": patient_uid,
                "patient_id": rec.patient_id,
                "source": rec.source,
                "slices": slices,
                "has_mask": int(mask_path is not None),
                "lesion_voxels": int(mask_stats["lesion_voxels"]),
                "mask_voxels": int(mask_stats["mask_voxels"]),
                "lesion_slices": int(mask_stats["lesion_slices"]),
                "lesion_components_3d": int(mask_stats["lesion_components_3d"]),
            }
        )

    patient_df = pd.DataFrame(patient_rows)
    if patient_df.empty:
        return pd.DataFrame()

    grouped = (
        patient_df.groupby("fold")
        .agg(
            patients=("patient_uid", "count"),
            cect_patients=("source", lambda s: int((s == "cect").sum())),
            full_patients=("source", lambda s: int((s == "full").sum())),
            total_slices=("slices", "sum"),
            mean_slices_per_patient=("slices", "mean"),
            patients_with_mask=("has_mask", "sum"),
            total_lesion_voxels=("lesion_voxels", "sum"),
            total_mask_voxels=("mask_voxels", "sum"),
            total_lesion_slices=("lesion_slices", "sum"),
            total_lesion_components_3d=("lesion_components_3d", "sum"),
        )
        .reset_index()
    )

    grouped["lesion_voxel_fraction"] = np.where(
        grouped["total_mask_voxels"] > 0,
        grouped["total_lesion_voxels"] / grouped["total_mask_voxels"],
        0.0,
    )
    grouped["mean_lesion_slices_per_patient"] = np.where(
        grouped["patients"] > 0,
        grouped["total_lesion_slices"] / grouped["patients"],
        0.0,
    )
    grouped["mean_lesion_components_per_patient"] = np.where(
        grouped["patients"] > 0,
        grouped["total_lesion_components_3d"] / grouped["patients"],
        0.0,
    )

    missing_folds = [f for f in range(int(n_folds)) if f not in set(grouped["fold"].tolist())]
    if missing_folds:
        zeros = pd.DataFrame(
            {
                "fold": missing_folds,
                "patients": 0,
                "cect_patients": 0,
                "full_patients": 0,
                "total_slices": 0,
                "mean_slices_per_patient": 0.0,
                "patients_with_mask": 0,
                "total_lesion_voxels": 0,
                "total_mask_voxels": 0,
                "total_lesion_slices": 0,
                "total_lesion_components_3d": 0,
                "lesion_voxel_fraction": 0.0,
                "mean_lesion_slices_per_patient": 0.0,
                "mean_lesion_components_per_patient": 0.0,
            }
        )
        grouped = pd.concat([grouped, zeros], ignore_index=True)

    grouped = grouped.sort_values("fold").reset_index(drop=True)
    return grouped


def write_fold_statistics_summary(stats_df: pd.DataFrame, output_txt: Path) -> None:
    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    if stats_df.empty:
        output_txt.write_text("No statistics available.\n", encoding="utf-8")
        return

    lines = []
    lines.append("Fold balance summary")
    lines.append("====================")
    lines.append(f"fold_count: {len(stats_df)}")
    lines.append(f"patients_total: {int(stats_df['patients'].sum())}")
    lines.append("")

    for metric in ["patients", "total_slices", "lesion_voxel_fraction", "total_lesion_components_3d"]:
        vals = stats_df[metric].to_numpy(dtype=np.float64)
        lines.append(f"{metric}:")
        lines.append(f"  min={vals.min():.6f}")
        lines.append(f"  max={vals.max():.6f}")
        lines.append(f"  mean={vals.mean():.6f}")
        lines.append(f"  std={vals.std():.6f}")
        lines.append("")

    output_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
