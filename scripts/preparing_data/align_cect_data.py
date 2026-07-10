"""
Align CECT_data phases to venous (C1) per patient and export in a standardized structure.

Input (default):
- C:\Projekt_badawczy\CECT_data

Output (default):
- C:\Projekt_badawczy\CECT_data_aligned
  - ct_files/
  - mask_files/
  - liver_mask_files/
  - patient_data.csv
  - alignment_report.csv

Phase mapping:
- C1 -> venous (reference)
- P  -> precontrast
- C2 -> arterial
- C3 -> delayed

Notes:
- No flips or rotations are applied.
- Non-reference phases are registered to C1.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    import ants
except ImportError:
    ants = None


PHASE_CODE_TO_NAME = {
    "P": "precontrast",
    "C1": "venous",
    "C2": "arterial",
    "C3": "delayed",
}

PHASE_ORDER = ["C1", "P", "C2", "C3"]


def ensure_3d_volume(data: np.ndarray, path: Path, is_label: bool = False) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 3:
        return arr

    # Drop all singleton dimensions first (e.g. X,Y,Z,1 or X,Y,Z,1,1).
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        return arr.astype(np.float32)

    # Support multi-component encodings such as X,Y,Z,3 or X,Y,Z,1,3.
    if arr.ndim == 4 and 3 in arr.shape:
        comp_axis = next((i for i, s in enumerate(arr.shape) if s == 3), None)
        if comp_axis is not None:
            if is_label:
                # Keep any positive component for labels, then downstream thresholding binarizes it.
                arr = np.max(arr, axis=comp_axis)
            else:
                # If components are equivalent, keep one channel; otherwise average them.
                arr_moved = np.moveaxis(arr, comp_axis, -1)
                if np.allclose(arr_moved[..., 0], arr_moved[..., 1], atol=1e-5) and np.allclose(
                    arr_moved[..., 0], arr_moved[..., 2], atol=1e-5
                ):
                    arr = arr_moved[..., 0]
                else:
                    arr = np.mean(arr_moved, axis=-1)

            arr = np.squeeze(arr)
            if arr.ndim == 3:
                return arr.astype(np.float32)

    raise ValueError(
        f"Expected a 3D-compatible volume, got shape={tuple(data.shape)} -> {tuple(arr.shape)} "
        f"for file: {path}"
    )


def load_nifti(path: Path, is_label: bool = False) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    img = nib.load(str(path))
    data_xyz = ensure_3d_volume(img.get_fdata().astype(np.float32), path, is_label=is_label)
    affine = img.affine.astype(np.float32)
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    return data_xyz, affine, spacing


def save_nifti(path: Path, data_xyz: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data_xyz.astype(np.float32), affine.astype(np.float32)), str(path))


def to_zyx(data_xyz: np.ndarray) -> np.ndarray:
    if data_xyz.ndim != 3:
        raise ValueError(f"to_zyx expects 3D array, got shape={data_xyz.shape}")
    return np.transpose(data_xyz, (2, 1, 0))


def to_xyz(data_zyx: np.ndarray) -> np.ndarray:
    return np.transpose(data_zyx, (2, 1, 0))


def infer_ct_background_value(volume_xyz: np.ndarray) -> float:
    # CECT_data CTs are typically normalized to [0, 255].
    vmin = float(np.min(volume_xyz))
    vmax = float(np.max(volume_xyz))
    if vmin >= -1.0 and vmax <= 260.0:
        return 0.0
    return -1024.0


def center_crop_or_pad(volume_zyx: np.ndarray, ref_shape_zyx: Tuple[int, int, int], fill_value: float) -> np.ndarray:
    if volume_zyx.shape == ref_shape_zyx:
        return volume_zyx

    out = np.full(ref_shape_zyx, fill_value, dtype=volume_zyx.dtype)

    z_min = (ref_shape_zyx[0] - volume_zyx.shape[0]) // 2
    y_min = (ref_shape_zyx[1] - volume_zyx.shape[1]) // 2
    x_min = (ref_shape_zyx[2] - volume_zyx.shape[2]) // 2

    z_dst = max(0, z_min)
    y_dst = max(0, y_min)
    x_dst = max(0, x_min)

    z_src = max(0, -z_min)
    y_src = max(0, -y_min)
    x_src = max(0, -x_min)

    z_len = min(ref_shape_zyx[0] - z_dst, volume_zyx.shape[0] - z_src)
    y_len = min(ref_shape_zyx[1] - y_dst, volume_zyx.shape[1] - y_src)
    x_len = min(ref_shape_zyx[2] - x_dst, volume_zyx.shape[2] - x_src)

    out[z_dst : z_dst + z_len, y_dst : y_dst + y_len, x_dst : x_dst + x_len] = volume_zyx[
        z_src : z_src + z_len,
        y_src : y_src + y_len,
        x_src : x_src + x_len,
    ]
    return out


def prepare_moving_to_reference_grid(
    moving_xyz: np.ndarray,
    reference_shape_xyz: Tuple[int, int, int],
    fill_value: float,
) -> np.ndarray:
    moving_zyx = to_zyx(moving_xyz)
    ref_shape_zyx = (reference_shape_xyz[2], reference_shape_xyz[1], reference_shape_xyz[0])
    prepared_zyx = center_crop_or_pad(moving_zyx, ref_shape_zyx, fill_value)
    return to_xyz(prepared_zyx)


def register_sitk(
    fixed_xyz: np.ndarray,
    moving_xyz: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    default_value: float,
) -> Tuple[np.ndarray, object, str, str]:
    if sitk is None:
        raise RuntimeError("SimpleITK is not available")

    fixed_zyx = to_zyx(fixed_xyz)
    moving_zyx = to_zyx(moving_xyz)

    fixed_img = sitk.GetImageFromArray(fixed_zyx)
    moving_img = sitk.GetImageFromArray(moving_zyx)
    fixed_img.SetSpacing(spacing_xyz)
    moving_img.SetSpacing(spacing_xyz)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-5,
        numberOfIterations=150,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed_img, moving_img)

    resampled = sitk.Resample(
        moving_img,
        fixed_img,
        final_transform,
        sitk.sitkLinear,
        float(default_value),
        moving_img.GetPixelID(),
    )

    transform_type = final_transform.GetName()
    transform_params = ",".join(f"{p:.6f}" for p in final_transform.GetParameters())
    aligned_xyz = to_xyz(sitk.GetArrayFromImage(resampled))
    return aligned_xyz, final_transform, transform_type, transform_params


def apply_sitk_transform(
    fixed_xyz: np.ndarray,
    moving_xyz: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    transform: object,
    is_label: bool,
) -> np.ndarray:
    if sitk is None:
        raise RuntimeError("SimpleITK is not available")

    fixed_zyx = to_zyx(fixed_xyz)
    moving_zyx = to_zyx(moving_xyz)

    fixed_img = sitk.GetImageFromArray(fixed_zyx)
    moving_img = sitk.GetImageFromArray(moving_zyx)
    fixed_img.SetSpacing(spacing_xyz)
    moving_img.SetSpacing(spacing_xyz)

    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    default_value = 0.0 if is_label else -1024.0

    warped = sitk.Resample(
        moving_img,
        fixed_img,
        transform,
        interpolator,
        default_value,
        moving_img.GetPixelID(),
    )
    return to_xyz(sitk.GetArrayFromImage(warped))


def register_ants(
    fixed_xyz: np.ndarray,
    moving_xyz: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> Tuple[np.ndarray, Dict[str, object], str, str]:
    if ants is None:
        raise RuntimeError("ANTsPy is not available")

    fixed_img = ants.from_numpy(fixed_xyz, spacing=spacing_xyz)
    moving_img = ants.from_numpy(moving_xyz, spacing=spacing_xyz)
    # Affine is safer for this dataset than deformable SyNRA.
    reg = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform="AffineFast")

    aligned_xyz = reg["warpedmovout"].numpy().astype(np.float32)
    transform_type = "ants:AffineFast"
    transform_params = ";".join(reg.get("fwdtransforms", []))
    return aligned_xyz, reg, transform_type, transform_params


def apply_ants_transform(
    fixed_xyz: np.ndarray,
    moving_xyz: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    reg: Dict[str, object],
    is_label: bool,
) -> np.ndarray:
    if ants is None:
        raise RuntimeError("ANTsPy is not available")

    fixed_img = ants.from_numpy(fixed_xyz, spacing=spacing_xyz)
    moving_img = ants.from_numpy(moving_xyz, spacing=spacing_xyz)
    interpolator = "nearestNeighbor" if is_label else "linear"
    warped = ants.apply_transforms(
        fixed=fixed_img,
        moving=moving_img,
        transformlist=reg.get("fwdtransforms", []),
        interpolator=interpolator,
    )
    return warped.numpy().astype(np.float32)


def resolve_input_path(input_root: Path, rel_path: str) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return input_root / p


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align CECT_data phases to venous (C1)")
    parser.add_argument("--input-root", type=str, default="C:\\Projekt_badawczy\\CECT_data")
    parser.add_argument("--output-root", type=str, default="C:\\Projekt_badawczy\\CECT_data_aligned")
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--patients", type=str, default=None, help="Comma-separated patient IDs, e.g. P0001,P0002")
    parser.add_argument(
        "--start-from-patient",
        type=str,
        default=None,
        help="Process only patients with ID >= this value in sorted order, e.g. P0094",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ants", "sitk", "none"],
        default="auto",
        help="Registration backend. auto prefers SimpleITK, then ANTs.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else input_root / "patient_data.csv"

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / f"align_cect_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_csv}")

    df = pd.read_csv(metadata_csv, low_memory=False)
    ensure_columns(df, ["patient_id", "phase", "ct_path", "mask_path", "liver_mask_path"])

    if args.patients:
        selected = {p.strip() for p in args.patients.split(",") if p.strip()}
        df = df[df["patient_id"].astype(str).isin(selected)].copy()
    elif args.start_from_patient:
        start_pid = str(args.start_from_patient).strip()
        if start_pid:
            df = df[df["patient_id"].astype(str) >= start_pid].copy()

    df["phase"] = df["phase"].astype(str).str.upper()
    df = df[df["phase"].isin(PHASE_CODE_TO_NAME.keys())].copy()

    if df.empty:
        raise ValueError("No matching rows to process after filters.")

    if args.backend == "auto":
        backend = "sitk" if sitk is not None else "ants" if ants is not None else "none"
    else:
        backend = args.backend

    if backend == "ants" and ants is None:
        raise RuntimeError("Requested ANTs backend but ANTsPy is not available")
    if backend == "sitk" and sitk is None:
        raise RuntimeError("Requested SimpleITK backend but SimpleITK is not available")

    log(f"Backend: {backend}")

    out_ct_dir = output_root / "ct_files"
    out_mask_dir = output_root / "mask_files"
    out_liver_mask_dir = output_root / "liver_mask_files"
    out_ct_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_liver_mask_dir.mkdir(parents=True, exist_ok=True)

    report_rows: List[Dict[str, object]] = []
    patient_rows_out: List[Dict[str, object]] = []

    grouped = df.groupby("patient_id", sort=True)
    total_patients = grouped.ngroups
    processed_patients = 0
    skipped_without_c1 = 0
    skipped_phase_errors = 0

    for patient_idx, (patient_id, rows) in enumerate(grouped, start=1):
        rows = rows.copy()
        rows["phase_rank"] = rows["phase"].map({p: i for i, p in enumerate(PHASE_ORDER)})
        rows = rows.sort_values("phase_rank")

        ref_rows = rows[rows["phase"] == "C1"]
        if ref_rows.empty:
            skipped_without_c1 += 1
            log(f"[{patient_idx}/{total_patients}] {patient_id}: skipped (missing C1 row)")
            continue

        ref_row = ref_rows.iloc[0]
        ref_ct_path = resolve_input_path(input_root, str(ref_row["ct_path"]))
        if not ref_ct_path.exists():
            skipped_without_c1 += 1
            log(f"[{patient_idx}/{total_patients}] {patient_id}: skipped (missing C1 CT file)")
            continue

        try:
            ref_ct_xyz, ref_affine, ref_spacing = load_nifti(ref_ct_path)
        except Exception as e:
            skipped_without_c1 += 1
            log(f"[{patient_idx}/{total_patients}] {patient_id}: skipped (invalid C1 CT: {e})")
            continue

        ref_shape = ref_ct_xyz.shape
        ct_background_value = infer_ct_background_value(ref_ct_xyz)

        processed_patients += 1
        log(f"[{patient_idx}/{total_patients}] {patient_id}: reference=C1 shape={ref_shape} spacing={ref_spacing}")

        for _, row in rows.iterrows():
            phase_code = str(row["phase"]).upper()
            phase_name = PHASE_CODE_TO_NAME.get(phase_code, "unknown")

            ct_in_path = resolve_input_path(input_root, str(row["ct_path"]))
            mask_in_path = resolve_input_path(input_root, str(row["mask_path"]))
            liver_mask_in_path = resolve_input_path(input_root, str(row["liver_mask_path"]))

            if not ct_in_path.exists():
                log(f"  {phase_code}: skipped (missing CT file: {ct_in_path.name})")
                continue

            try:
                start_t = time.time()

                moving_ct_xyz, _, _ = load_nifti(ct_in_path)

                transform_type = "identity"
                transform_params = ""
                reg_ctx: Optional[object] = None

                if phase_code == "C1" or backend == "none":
                    if moving_ct_xyz.shape != ref_shape:
                        moving_ct_xyz = prepare_moving_to_reference_grid(
                            moving_ct_xyz,
                            ref_shape,
                            fill_value=ct_background_value,
                        )
                    aligned_ct_xyz = moving_ct_xyz
                    if phase_code != "C1" and backend == "none":
                        transform_type = "none"
                else:
                    if backend == "ants":
                        aligned_ct_xyz, reg_ctx, transform_type, transform_params = register_ants(
                            ref_ct_xyz,
                            moving_ct_xyz,
                            ref_spacing,
                        )
                    else:
                        aligned_ct_xyz, reg_ctx, transform_type, transform_params = register_sitk(
                            ref_ct_xyz,
                            moving_ct_xyz,
                            ref_spacing,
                            ct_background_value,
                        )

                out_ct_name = f"{patient_id}_ct_{phase_name}.nii.gz"
                out_ct_path = out_ct_dir / out_ct_name
                save_nifti(out_ct_path, aligned_ct_xyz, ref_affine)

                out_mask_path = None
                if mask_in_path.exists():
                    mask_xyz, _, _ = load_nifti(mask_in_path, is_label=True)

                    if phase_code == "C1" or backend == "none" or reg_ctx is None:
                        if mask_xyz.shape != ref_shape:
                            mask_xyz = prepare_moving_to_reference_grid(mask_xyz, ref_shape, fill_value=0.0)
                        aligned_mask_xyz = mask_xyz
                    elif backend == "ants":
                        aligned_mask_xyz = apply_ants_transform(
                            ref_ct_xyz,
                            mask_xyz,
                            ref_spacing,
                            reg_ctx,
                            is_label=True,
                        )
                    else:
                        aligned_mask_xyz = apply_sitk_transform(
                            ref_ct_xyz,
                            mask_xyz,
                            ref_spacing,
                            reg_ctx,
                            is_label=True,
                        )

                    aligned_mask_xyz = (aligned_mask_xyz > 0.5).astype(np.float32)
                    out_mask_name = f"{patient_id}_mask_{phase_name}.nii.gz"
                    out_mask_path = out_mask_dir / out_mask_name
                    save_nifti(out_mask_path, aligned_mask_xyz, ref_affine)

                out_liver_mask_path = None
                if liver_mask_in_path.exists():
                    liver_mask_xyz, _, _ = load_nifti(liver_mask_in_path, is_label=True)

                    if phase_code == "C1" or backend == "none" or reg_ctx is None:
                        if liver_mask_xyz.shape != ref_shape:
                            liver_mask_xyz = prepare_moving_to_reference_grid(liver_mask_xyz, ref_shape, fill_value=0.0)
                        aligned_liver_mask_xyz = liver_mask_xyz
                    elif backend == "ants":
                        aligned_liver_mask_xyz = apply_ants_transform(
                            ref_ct_xyz,
                            liver_mask_xyz,
                            ref_spacing,
                            reg_ctx,
                            is_label=True,
                        )
                    else:
                        aligned_liver_mask_xyz = apply_sitk_transform(
                            ref_ct_xyz,
                            liver_mask_xyz,
                            ref_spacing,
                            reg_ctx,
                            is_label=True,
                        )

                    aligned_liver_mask_xyz = (aligned_liver_mask_xyz > 0.5).astype(np.float32)
                    out_liver_mask_name = f"{patient_id}-livermask_{phase_name}.nii.gz"
                    out_liver_mask_path = out_liver_mask_dir / out_liver_mask_name
                    save_nifti(out_liver_mask_path, aligned_liver_mask_xyz, ref_affine)

                elapsed = time.time() - start_t
                log(f"  {phase_code}->{phase_name}: transform={transform_type} time={elapsed:.1f}s")

                patient_rows_out.append(
                    {
                        "patient_id": patient_id,
                        "age": row.get("age", ""),
                        "gender": row.get("gender", ""),
                        "phase": phase_name,
                        "phase_code": phase_code,
                        "cancer_type": row.get("cancer_type", ""),
                        "ct_path": str(Path("ct_files") / out_ct_name),
                        "mask_path": str(Path("mask_files") / out_mask_path.name) if out_mask_path else "",
                        "liver_mask_path": str(Path("liver_mask_files") / out_liver_mask_path.name)
                        if out_liver_mask_path
                        else "",
                    }
                )

                report_rows.append(
                    {
                        "patient_id": patient_id,
                        "phase_code": phase_code,
                        "phase": phase_name,
                        "reference_phase_code": "C1",
                        "reference_phase": "venous",
                        "input_ct": str(ct_in_path),
                        "input_mask": str(mask_in_path) if mask_in_path.exists() else "",
                        "input_liver_mask": str(liver_mask_in_path) if liver_mask_in_path.exists() else "",
                        "output_ct": str(out_ct_path),
                        "output_mask": str(out_mask_path) if out_mask_path else "",
                        "output_liver_mask": str(out_liver_mask_path) if out_liver_mask_path else "",
                        "spacing_x": ref_spacing[0],
                        "spacing_y": ref_spacing[1],
                        "spacing_z": ref_spacing[2],
                        "shape_x": ref_shape[0],
                        "shape_y": ref_shape[1],
                        "shape_z": ref_shape[2],
                        "transform_type": transform_type,
                        "transform_params": transform_params,
                        "backend": backend,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                )
            except Exception as e:
                skipped_phase_errors += 1
                log(
                    f"  {phase_code}->{phase_name}: skipped (error: {e}; "
                    f"ct={ct_in_path.name})"
                )
                continue

    out_patient_csv = output_root / "patient_data.csv"
    out_report_csv = output_root / "alignment_report.csv"

    patient_df_new = pd.DataFrame(patient_rows_out)
    if out_patient_csv.exists():
        patient_df_old = pd.read_csv(out_patient_csv, low_memory=False)
        patient_df = pd.concat([patient_df_old, patient_df_new], ignore_index=True)
        dedup_cols = [c for c in ["patient_id", "phase_code"] if c in patient_df.columns]
        if dedup_cols:
            patient_df = patient_df.drop_duplicates(subset=dedup_cols, keep="last")
    else:
        patient_df = patient_df_new
    patient_df.to_csv(out_patient_csv, index=False)

    report_df_new = pd.DataFrame(report_rows)
    if out_report_csv.exists():
        report_df_old = pd.read_csv(out_report_csv, low_memory=False)
        report_df = pd.concat([report_df_old, report_df_new], ignore_index=True)
        dedup_cols = [c for c in ["patient_id", "phase_code", "input_ct"] if c in report_df.columns]
        if dedup_cols:
            report_df = report_df.drop_duplicates(subset=dedup_cols, keep="last")
    else:
        report_df = report_df_new
    report_df.to_csv(out_report_csv, index=False)

    log(
        f"Done. Processed patients: {processed_patients}. "
        f"Skipped without C1: {skipped_without_c1}. "
        f"Skipped phase errors: {skipped_phase_errors}. "
        f"Output rows: {len(report_rows)}."
    )
    log(f"Output metadata: {out_patient_csv}")
    log(f"Output report: {out_report_csv}")
    log(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
