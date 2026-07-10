"""
Align SANNA_FULL DICOM series to the labeled reference series per patient.

Sources (default):
- SANNA_FULL/Liver3D_originals + SANNA_FULL/tumors/Liver3D_originals
- SANNA_FULL/Liver3D_labels + SANNA_FULL/tumors/Liver3D_labels

Outputs:
- Aligned images and labels as NIfTI in full_data_converted_aligned/images and labels
- alignment_report.csv with phase, reference series, and transform info
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
from scipy.ndimage import zoom

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    import ants
except ImportError:
    ants = None



FLIP_LABEL_Z = True


PHASE_PATTERNS = {
    "precontrast": [
        r"\bpre\b",
        r"\bnative\b",
        r"\bnon-contrast\b",
        r"\bplain\b",
        r"\bwithout\b",
        r"\bprecontrast\b",
        r"\bpre-contrast\b",
        r"\bbez kontrastu\b",
        r"\bnativna\b",
        r"\bnativ\b",
        r"\bbez cm\b",
        r"pre\s+contrast",
        r"\bpremonitoring\b",
    ],
    "arterial": [
        r"\barterial\b",
        r"\barter\b",
        r"\bart\s+phase\b",
        r"\bart\.\b",
        r"\bearly\b",
        r"\bhap\b",
        r"tetnicz",
        r"tentnicza",
        r"faza tętnicza",
        r"faza arterial",
        r"f\.?\s*tetnic",
        r"faza tetnic",
    ],
    "venous": [
        r"\bvenous\b",
        r"\bven\b",
        r"\bportal\b",
        r"\bpvp\b",
        r"\bportal venous\b",
        r"\bport\b",
        r"\bpv\b",
        r"żyln",
        r"faza żylna",
        r"faza venous",
        r"zyln",
        r"faza portal",
        r"wrotna",
        r"f\.?\s*zyln",
    ],
    "delayed": [
        r"\bdelayed\b",
        r"\blate\b",
        r"\bequilibrium\b",
        r"\beq\b",
        r"\bdelay\b",
        r"późn",
        r"faza późna",
        r"pozn",
        r"równowaga",
        r"faza opóźniona",
        r"faza pozn",
    ],
    "hepatic": [
        r"\bhepatic\b",
        r"\bliver\b",
        r"wątrobowa",
        r"wątroba",
    ],
}


@dataclass
class SeriesEntry:
    patient_id: str
    series_id: int
    series_dir: Path
    phase: str
    spacing_xyz: Tuple[float, float, float]
    volume_zyx: np.ndarray


@dataclass
class LabelEntry:
    patient_id: str
    series_id: int
    label_path: Path
    qualifiers: str


def detect_phase_from_text(text: str) -> str:
    if not text:
        return "unknown"
    text_lower = text.lower()
    for phase, patterns in PHASE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return phase
    return "unknown"


def normalize_label_name(name: str) -> str:
    if name.startswith("Untitled"):
        return name[len("Untitled"):].lstrip("-_ ")
    return name


def parse_series_id_from_label_name(name: str) -> Optional[int]:
    name = normalize_label_name(name)
    # Support both 2-digit and zero-padded 3-digit series IDs (e.g., 11 and 011).
    match = re.search(r"_(\d{2,3})(?:\D|$)", name)
    if not match:
        return None
    return int(match.group(1))


def parse_patient_id_from_label_name(name: str) -> Optional[str]:
    name = normalize_label_name(name)
    match = re.match(r"^(\d+)", name)
    if not match:
        return None
    return match.group(1)


def extract_qualifiers(name: str) -> str:
    if "Vesicle" in name:
        return "Vesicle"
    if re.search(r"\bT\b", name) or re.search(r"tumor", name, re.IGNORECASE):
        return "T"
    if "P+V" in name:
        return "P+V"
    if re.search(r"\bP\b", name) and "V" not in name:
        return "P"
    if re.search(r"\bV\b", name) and "P" not in name:
        return "V"
    return ""


def load_dicom_series(
    series_dir: Path,
) -> Tuple[np.ndarray, Tuple[float, float, float], str]:
    dicom_files = sorted([p for p in series_dir.glob("*") if p.is_file()])
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files in {series_dir}")

    datasets = []
    for path in dicom_files:
        try:
            ds = pydicom.dcmread(path, force=True)
            datasets.append(ds)
        except Exception:
            continue

    if not datasets:
        raise ValueError(f"No valid DICOM files in {series_dir}")

    try:
        datasets.sort(key=lambda ds: int(ds.InstanceNumber))
    except Exception:
        try:
            datasets.sort(key=lambda ds: float(ds.SliceLocation))
        except Exception:
            pass

    slices = []
    shapes = []
    for ds in datasets:
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
        hu = pixel_array * slope + intercept
        slices.append(hu)
        shapes.append(hu.shape)

    # Keep only the most common shape to avoid stack errors
    if shapes:
        shape_counts: Dict[Tuple[int, int], int] = {}
        for shape in shapes:
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        target_shape = sorted(shape_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        filtered = [(ds, s) for ds, s in zip(datasets, slices) if s.shape == target_shape]
        datasets = [ds for ds, _ in filtered]
        slices = [s for _, s in filtered]

    if not slices:
        raise ValueError(f"No consistent slice shapes in {series_dir}")

    volume_zyx = np.stack(slices, axis=0)
    ds0 = datasets[0]
    spacing_y, spacing_x = map(float, ds0.PixelSpacing)
    spacing_z = float(getattr(ds0, "SpacingBetweenSlices", getattr(ds0, "SliceThickness", 1.0)))

    series_desc = str(getattr(ds0, "SeriesDescription", "") or "")
    protocol_name = str(getattr(ds0, "ProtocolName", "") or "")
    study_desc = str(getattr(ds0, "StudyDescription", "") or "")
    phase = detect_phase_from_text(" ".join([series_desc, protocol_name, study_desc]))

    return volume_zyx, (spacing_x, spacing_y, spacing_z), phase


def load_nifti_label(label_path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = nib.load(str(label_path))
    data_xyz = img.get_fdata().astype(np.float32)
    spacing = img.header.get_zooms()[:3]
    volume_zyx = np.transpose(data_xyz, (2, 1, 0))
    return volume_zyx, (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def resample_label_to_ref(
    label_zyx: np.ndarray,
    label_spacing: Tuple[float, float, float],
    ref_shape: Tuple[int, int, int],
    ref_spacing: Tuple[float, float, float],
) -> np.ndarray:
    scale_z = (label_spacing[2] / ref_spacing[2]) * (ref_shape[0] / label_zyx.shape[0])
    scale_y = (label_spacing[1] / ref_spacing[1]) * (ref_shape[1] / label_zyx.shape[1])
    scale_x = (label_spacing[0] / ref_spacing[0]) * (ref_shape[2] / label_zyx.shape[2])
    resampled = zoom(label_zyx, (scale_z, scale_y, scale_x), order=0)

    if resampled.shape != ref_shape:
        result = np.zeros(ref_shape, dtype=resampled.dtype)

        z_min = (ref_shape[0] - resampled.shape[0]) // 2
        y_min = (ref_shape[1] - resampled.shape[1]) // 2
        x_min = (ref_shape[2] - resampled.shape[2]) // 2

        z_start = max(0, z_min)
        y_start = max(0, y_min)
        x_start = max(0, x_min)

        z_src_start = max(0, -z_min)
        y_src_start = max(0, -y_min)
        x_src_start = max(0, -x_min)

        z_len = min(ref_shape[0] - z_start, resampled.shape[0] - z_src_start)
        y_len = min(ref_shape[1] - y_start, resampled.shape[1] - y_src_start)
        x_len = min(ref_shape[2] - x_start, resampled.shape[2] - x_src_start)

        result[z_start:z_start + z_len, y_start:y_start + y_len, x_start:x_start + x_len] = resampled[
            z_src_start:z_src_start + z_len,
            y_src_start:y_src_start + y_len,
            x_src_start:x_src_start + x_len,
        ]
        resampled = result

    return resampled


def center_crop_or_pad(
    volume: np.ndarray,
    ref_shape: Tuple[int, int, int],
    fill_value: float,
) -> np.ndarray:
    if volume.shape == ref_shape:
        return volume

    result = np.full(ref_shape, fill_value, dtype=volume.dtype)

    z_min = (ref_shape[0] - volume.shape[0]) // 2
    y_min = (ref_shape[1] - volume.shape[1]) // 2
    x_min = (ref_shape[2] - volume.shape[2]) // 2

    z_start = max(0, z_min)
    y_start = max(0, y_min)
    x_start = max(0, x_min)

    z_src_start = max(0, -z_min)
    y_src_start = max(0, -y_min)
    x_src_start = max(0, -x_min)

    z_len = min(ref_shape[0] - z_start, volume.shape[0] - z_src_start)
    y_len = min(ref_shape[1] - y_start, volume.shape[1] - y_src_start)
    x_len = min(ref_shape[2] - x_start, volume.shape[2] - x_src_start)

    result[z_start:z_start + z_len, y_start:y_start + y_len, x_start:x_start + x_len] = volume[
        z_src_start:z_src_start + z_len,
        y_src_start:y_src_start + y_len,
        x_src_start:x_src_start + x_len,
    ]
    return result


def resample_volume_to_ref(
    volume_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    ref_shape: Tuple[int, int, int],
    ref_spacing: Tuple[float, float, float],
    fill_value: float,
) -> np.ndarray:
    scale_z = spacing_xyz[2] / ref_spacing[2]
    scale_y = spacing_xyz[1] / ref_spacing[1]
    scale_x = spacing_xyz[0] / ref_spacing[0]
    resampled = zoom(volume_zyx, (scale_z, scale_y, scale_x), order=1)
    return center_crop_or_pad(resampled, ref_shape, fill_value)


def compute_mask_bbox(mask: np.ndarray, margin: int) -> Optional[Tuple[slice, slice, slice]]:
    coords = np.where(mask > 0)
    if coords[0].size == 0:
        return None
    z_min, y_min, x_min = [int(v.min()) for v in coords]
    z_max, y_max, x_max = [int(v.max()) for v in coords]

    z_min = max(0, z_min - margin)
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)

    z_max = min(mask.shape[0] - 1, z_max + margin)
    y_max = min(mask.shape[1] - 1, y_max + margin)
    x_max = min(mask.shape[2] - 1, x_max + margin)

    return (slice(z_min, z_max + 1), slice(y_min, y_max + 1), slice(x_min, x_max + 1))


def to_xyz(volume_zyx: np.ndarray) -> np.ndarray:
    return np.transpose(volume_zyx, (2, 1, 0))


def to_zyx(volume_xyz: np.ndarray) -> np.ndarray:
    return np.transpose(volume_xyz, (2, 1, 0))




def rigid_register_sitk(
    ref_xyz: np.ndarray,
    mov_xyz: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    fixed_mask_xyz: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str, str]:
    if sitk is None:
        raise RuntimeError("SimpleITK is not available")

    ref_img = sitk.GetImageFromArray(ref_xyz)
    mov_img = sitk.GetImageFromArray(mov_xyz)
    ref_img.SetSpacing(spacing_xyz)
    mov_img.SetSpacing(spacing_xyz)

    fixed_mask_img = None
    if fixed_mask_xyz is not None:
        fixed_mask_img = sitk.GetImageFromArray(fixed_mask_xyz.astype(np.uint8))
        fixed_mask_img.SetSpacing(spacing_xyz)

    initial_transform = sitk.CenteredTransformInitializer(
        ref_img,
        mov_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.5)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.5,
        minStep=1e-5,
        numberOfIterations=100,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1.0, 0.0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)
    if fixed_mask_img is not None:
        registration.SetMetricFixedMask(fixed_mask_img)

    final_transform = registration.Execute(ref_img, mov_img)

    resampled = sitk.Resample(
        mov_img,
        ref_img,
        final_transform,
        sitk.sitkLinear,
        -1024.0,
        mov_img.GetPixelID(),
    )

    transform_type = final_transform.GetName()
    transform_params = ",".join([f"{p:.6f}" for p in final_transform.GetParameters()])

    return sitk.GetArrayFromImage(resampled), transform_type, transform_params


def register_antspy(
    ref_xyz: np.ndarray,
    mov_xyz: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    fixed_mask_xyz: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str, str]:
    if ants is None:
        raise RuntimeError("ANTsPy is not available")

    fixed = ants.from_numpy(ref_xyz, spacing=spacing_xyz)
    moving = ants.from_numpy(mov_xyz, spacing=spacing_xyz)

    mask = None
    if fixed_mask_xyz is not None:
        mask = ants.from_numpy((fixed_mask_xyz > 0).astype(np.uint8), spacing=spacing_xyz)

    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyNRA",
        mask=mask,
    )

    warped = registration["warpedmovout"].numpy()
    transform_params = ";".join(registration.get("fwdtransforms", []))
    return warped, "ants:SyNRA", transform_params


def rigid_affine_register_sitk(
    ref_xyz: np.ndarray,
    mov_xyz: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    fixed_mask_xyz: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str, str]:
    if sitk is None:
        raise RuntimeError("SimpleITK is not available")

    ref_img = sitk.GetImageFromArray(ref_xyz)
    mov_img = sitk.GetImageFromArray(mov_xyz)
    ref_img.SetSpacing(spacing_xyz)
    mov_img.SetSpacing(spacing_xyz)

    fixed_mask_img = None
    if fixed_mask_xyz is not None:
        fixed_mask_img = sitk.GetImageFromArray(fixed_mask_xyz.astype(np.uint8))
        fixed_mask_img.SetSpacing(spacing_xyz)

    rigid_init = sitk.CenteredTransformInitializer(
        ref_img,
        mov_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    rigid = sitk.ImageRegistrationMethod()
    rigid.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    rigid.SetMetricSamplingStrategy(rigid.RANDOM)
    rigid.SetMetricSamplingPercentage(0.2)
    rigid.SetInterpolator(sitk.sitkLinear)
    rigid.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8,
    )
    rigid.SetOptimizerScalesFromPhysicalShift()
    rigid.SetShrinkFactorsPerLevel([4, 2, 1])
    rigid.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    rigid.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    rigid.SetInitialTransform(rigid_init, inPlace=False)
    if fixed_mask_img is not None:
        rigid.SetMetricFixedMask(fixed_mask_img)

    rigid_transform = rigid.Execute(ref_img, mov_img)
    rigid_params = rigid_transform
    if hasattr(rigid_transform, "GetNthTransform"):
        rigid_params = rigid_transform.GetNthTransform(0)

    affine = sitk.AffineTransform(3)
    affine.SetCenter(rigid_params.GetCenter())
    affine.SetMatrix(rigid_params.GetMatrix())
    affine.SetTranslation(rigid_params.GetTranslation())

    affine_reg = sitk.ImageRegistrationMethod()
    affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    affine_reg.SetMetricSamplingStrategy(affine_reg.RANDOM)
    affine_reg.SetMetricSamplingPercentage(0.2)
    affine_reg.SetInterpolator(sitk.sitkLinear)
    affine_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-5,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8,
    )
    affine_reg.SetOptimizerScalesFromPhysicalShift()
    affine_reg.SetShrinkFactorsPerLevel([4, 2, 1])
    affine_reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    affine_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    affine_reg.SetInitialTransform(affine, inPlace=False)
    if fixed_mask_img is not None:
        affine_reg.SetMetricFixedMask(fixed_mask_img)

    affine_transform = affine_reg.Execute(ref_img, mov_img)

    resampled = sitk.Resample(
        mov_img,
        ref_img,
        affine_transform,
        sitk.sitkLinear,
        -1024.0,
        mov_img.GetPixelID(),
    )

    transform_type = "rigid+affine"
    transform_params = ";".join(
        [
            ",".join([f"{p:.6f}" for p in rigid_transform.GetParameters()]),
            ",".join([f"{p:.6f}" for p in affine_transform.GetParameters()]),
        ]
    )

    return sitk.GetArrayFromImage(resampled), transform_type, transform_params


def save_nifti(
    path: Path,
    volume_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> None:
    if volume_zyx.ndim == 4:
        volume_xyz = np.transpose(volume_zyx, (3, 2, 1, 0))
    else:
        volume_xyz = to_xyz(volume_zyx)

    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = spacing_xyz[0]
    affine[1, 1] = spacing_xyz[1]
    affine[2, 2] = spacing_xyz[2]

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume_xyz, affine), str(path))


def collect_label_entries(label_roots: List[Path]) -> List[LabelEntry]:
    labels = []
    for root in label_roots:
        if not root.exists():
            continue
        for path in root.glob("**/*.nii*"):
            name = path.stem
            if name.endswith(".nii"):
                name = Path(name).stem
            patient_id = parse_patient_id_from_label_name(name)
            series_id = parse_series_id_from_label_name(name)
            if not patient_id or series_id is None:
                continue
            labels.append(
                LabelEntry(
                    patient_id=patient_id,
                    series_id=series_id,
                    label_path=path,
                    qualifiers=extract_qualifiers(name),
                )
            )
    return labels


def pick_reference_label(label_entries: List[LabelEntry], series_dirs: Dict[int, Path]) -> Optional[LabelEntry]:
    qualifiers_order = {
        "": 0,
        "P": 1,
        "V": 2,
        "P+V": 3,
        "Vesicle": 4,
    }

    candidates = [l for l in label_entries if l.series_id in series_dirs]
    if not candidates:
        return None

    # Pick series_id based on the most common label series number for this patient.
    series_counts: Dict[int, int] = {}
    for entry in candidates:
        series_counts[entry.series_id] = series_counts.get(entry.series_id, 0) + 1
    reference_series_id = sorted(series_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    series_candidates = [l for l in candidates if l.series_id == reference_series_id]
    series_candidates.sort(
        key=lambda l: (
            qualifiers_order.get(l.qualifiers, 99),
            l.label_path.name,
        )
    )
    return series_candidates[0]


def build_reference_label_files(
    label_entries: List[LabelEntry],
    reference_series_id: int,
    ref_shape: Tuple[int, int, int],
    ref_spacing: Tuple[float, float, float],
    final_flip_axis: str,
    out_labels_dir: Path,
    overwrite_existing: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    output_labels: List[str] = []
    label_files: List[str] = []
    label_qualifiers: List[str] = []

    for entry in label_entries:
        if entry.series_id != reference_series_id:
            continue
        label_volume, label_spacing = load_nifti_label(entry.label_path)
        if label_volume.shape != ref_shape:
            print(
                f"Warning: label shape {label_volume.shape} does not match reference {ref_shape} for {entry.label_path}",
                flush=True,
            )
            continue
        # SANNA_FULL labels are Z-flipped.
        label_mask = np.flip(label_volume, axis=0) if FLIP_LABEL_Z else label_volume
        for axis_char in final_flip_axis.lower():
            if axis_char in {"x", "y", "z"}:
                axis_index = {"z": 0, "y": 1, "x": 2}[axis_char]
                label_mask = np.flip(label_mask, axis=axis_index)

        if entry.label_path.name.endswith(".nii.gz"):
            base_name = entry.label_path.name[:-7]
            output_name = f"{normalize_label_name(base_name)}.nii.gz"
        else:
            output_name = f"{normalize_label_name(entry.label_path.stem)}{entry.label_path.suffix}"
        output_path = out_labels_dir / output_name
        if output_path.exists() and not overwrite_existing:
            output_labels.append(str(output_path))
            label_files.append(str(entry.label_path))
            if entry.qualifiers:
                label_qualifiers.append(entry.qualifiers)
            continue
        save_nifti(output_path, label_mask.astype(np.float32), ref_spacing)

        output_labels.append(str(output_path))
        label_files.append(str(entry.label_path))
        if entry.qualifiers:
            label_qualifiers.append(entry.qualifiers)

    label_qualifiers = sorted(set(label_qualifiers))
    return output_labels, label_files, label_qualifiers


def build_reference_union_mask(
    label_entries: List[LabelEntry],
    reference_series_id: int,
    ref_shape: Tuple[int, int, int],
) -> np.ndarray:
    merged = np.zeros(ref_shape, dtype=np.uint8)
    for entry in label_entries:
        if entry.series_id != reference_series_id:
            continue
        label_volume, _ = load_nifti_label(entry.label_path)
        if label_volume.shape != ref_shape:
            continue
        label_mask = (np.flip(label_volume, axis=0) if FLIP_LABEL_Z else label_volume) > 0
        merged = np.maximum(merged, label_mask.astype(np.uint8))
    return merged


def collect_dicom_series(dicom_roots: List[Path]) -> Dict[str, Dict[int, Path]]:
    series_map: Dict[str, Dict[int, Path]] = {}
    for root in dicom_roots:
        if not root.exists():
            continue
        for patient_dir in root.iterdir():
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name
            dicoms_dir = patient_dir / "DICOMS"
            if not dicoms_dir.exists():
                continue
            for study_dir in dicoms_dir.iterdir():
                if not study_dir.is_dir():
                    continue
                for series_dir in study_dir.iterdir():
                    if not series_dir.is_dir():
                        continue
                    match = re.search(r"SER(\d+)", series_dir.name)
                    if not match:
                        continue
                    series_id = int(match.group(1))
                    series_map.setdefault(patient_id, {})[series_id] = series_dir
    return series_map


def find_aligned_reference_image(output_root: Path, patient_id: str, reference_series_id: int) -> Optional[Path]:
    images_dir = output_root / "images" / patient_id
    if not images_dir.exists():
        return None
    candidates = sorted(images_dir.glob(f"{patient_id}_SER{reference_series_id:05d}_*.nii.gz"))
    if not candidates:
        return None
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Align SANNA_FULL phases to label series")
    parser.add_argument(
        "--dicom-roots",
        type=str,
        default="C:\\Projekt_badawczy\\SANNA_FULL\\Liver3D_originals;C:\\Projekt_badawczy\\SANNA_FULL\\tumors\\Liver3D_originals",
    )
    parser.add_argument(
        "--label-roots",
        type=str,
        default="C:\\Projekt_badawczy\\SANNA_FULL\\Liver3D_labels;C:\\Projekt_badawczy\\SANNA_FULL\\tumors\\Liver3D_labels",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="C:\\Projekt_badawczy\\full_data_converted_aligned",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="C:\\Projekt_badawczy\\tomography_segmentation\\generated_helper_csv_files\\unique_series_list_labeled.csv",
    )
    parser.add_argument(
        "--phase-mapping-csv",
        type=str,
        default="C:\\Projekt_badawczy\\tomography_segmentation\\generated_helper_csv_files\\phase_mapping_manual.csv",
        help="CSV with patient_id, series_id, phase to restrict conversion and label outputs.",
    )
    parser.add_argument(
        "--skip-patients-csv",
        type=str,
        default="C:\\Projekt_badawczy\\tomography_segmentation\\generated_helper_csv_files\\double_triple_dicomm.csv",
        help="CSV with patient IDs to skip (single column, no header).",
    )
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--patients", type=str, default=None, help="Comma-separated patient IDs")
    parser.add_argument(
        "--start-from-patient",
        type=str,
        default=None,
        help="Process only patients with ID >= this value in sorted order, e.g. 094 or 186",
    )
    parser.add_argument(
        "--ants-mask",
        choices=["liver", "none"],
        default="liver",
        help="ANTs metric mask: liver (default) or none for whole scan.",
    )
    parser.add_argument(
        "--final-flip-axes",
        type=str,
        default="xyz",
        help="Combination of axes to flip, e.g. x, y, z, xy, xz, yz, xyz",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Backfill labels only using existing aligned images in output-root/images (skip image alignment).",
    )
    parser.add_argument(
        "--labels-overwrite",
        action="store_true",
        help="When --labels-only is used, overwrite already existing output label files.",
    )
    args = parser.parse_args()

    dicom_roots = [Path(p) for p in args.dicom_roots.split(";") if p]
    label_roots = [Path(p) for p in args.label_roots.split(";") if p]
    output_root = Path(args.output_root)
    metadata_csv = Path(args.metadata_csv)
    phase_mapping_csv = Path(args.phase_mapping_csv)
    skip_patients_csv = Path(args.skip_patients_csv)

    log_path = output_root / f"alignment_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


    labels = collect_label_entries(label_roots)
    dicom_map = collect_dicom_series(dicom_roots)

    skip_patients: set[str] = set()
    if skip_patients_csv.exists():
        try:
            skip_df = pd.read_csv(skip_patients_csv, header=None)
            for val in skip_df.iloc[:, 0].tolist():
                try:
                    skip_patients.add(str(int(val)).zfill(3))
                except Exception:
                    continue
        except Exception:
            pass

    phase_map: Dict[Tuple[str, int], str] = {}
    if phase_mapping_csv.exists():
        mapping_df = pd.read_csv(phase_mapping_csv, low_memory=False)
        if {"patient_id", "series_id", "phase"}.issubset(mapping_df.columns):
            for _, row in mapping_df.iterrows():
                pid = str(int(row["patient_id"])).zfill(3)
                sid = int(row["series_id"])
                phase_map[(pid, sid)] = str(row["phase"])
    if not phase_map:
        raise ValueError(f"Phase mapping CSV is missing or empty: {phase_mapping_csv}")


    patient_filter = None
    if args.patients:
        patient_filter = {p.strip().zfill(3) for p in args.patients.split(",")}

    labels_by_patient: Dict[str, List[LabelEntry]] = {}
    for label in labels:
        if patient_filter and label.patient_id not in patient_filter:
            continue
        if label.patient_id in skip_patients:
            continue
        labels_by_patient.setdefault(label.patient_id, []).append(label)

    mapped_patient_ids = sorted({patient_id for patient_id, _ in phase_map.keys()})
    if patient_filter:
        mapped_patient_ids = [pid for pid in mapped_patient_ids if pid in patient_filter]
    elif args.start_from_patient:
        start_pid = str(args.start_from_patient).strip().zfill(3)
        if start_pid:
            mapped_patient_ids = [pid for pid in mapped_patient_ids if pid >= start_pid]
    patient_ids = [pid for pid in mapped_patient_ids if pid not in skip_patients]
    total_patients = len(patient_ids)

    report_rows = []
    processed_patients = 0
    skipped_no_series = 0
    skipped_no_reference_image = 0
    overwrite_existing_labels = (not args.labels_only) or args.labels_overwrite


    for patient_index, patient_id in enumerate(patient_ids, start=1):
        label_entries = sorted(labels_by_patient.get(patient_id, []), key=lambda x: x.label_path.name)
        mapped_series_ids = sorted([series_id for pid, series_id in phase_map.keys() if pid == patient_id])
        if args.labels_only:
            series_dirs = {series_id: Path() for series_id in mapped_series_ids}
        else:
            series_dirs = dicom_map.get(patient_id, {})
            series_dirs = {k: v for k, v in series_dirs.items() if (patient_id, k) in phase_map}
        if not series_dirs:
            skipped_no_series += 1
            continue

        label_entry = pick_reference_label(label_entries, series_dirs) if label_entries else None
        reference_series_id = label_entry.series_id if label_entry is not None else sorted(series_dirs.keys())[0]
        ref_series_dir = series_dirs.get(reference_series_id)
        if ref_series_dir is None:
            skipped_no_series += 1
            continue

        if label_entry is None:
            log(
                f"[{patient_index}/{total_patients}] Patient {patient_id} - reference SER{reference_series_id:05d} (no label reference)"
            )
        else:
            log(f"[{patient_index}/{total_patients}] Patient {patient_id} - reference SER{reference_series_id:05d}")

        if args.labels_only:
            if label_entry is None:
                continue
            ref_image_path = find_aligned_reference_image(output_root, patient_id, reference_series_id)
            if ref_image_path is None:
                skipped_no_reference_image += 1
                log(
                    f"  Labels-only: missing aligned reference image for SER{reference_series_id:05d}; skipping labels"
                )
                continue
            ref_volume, ref_spacing = load_nifti_label(ref_image_path)
            out_labels_dir = output_root / "labels" / patient_id
            output_labels, ref_label_files, ref_label_qualifiers = build_reference_label_files(
                label_entries,
                reference_series_id,
                ref_volume.shape,
                ref_spacing,
                args.final_flip_axes,
                out_labels_dir,
                overwrite_existing=overwrite_existing_labels,
            )
            processed_patients += 1
            log(
                f"  Labels-only: labels={len(output_labels)} from reference image {ref_image_path.name}"
            )
            report_rows.append(
                {
                    "patient_id": patient_id,
                    "reference_series_id": reference_series_id,
                    "reference_image": str(ref_image_path),
                    "label_files": ";".join(ref_label_files),
                    "label_qualifiers": ";".join(ref_label_qualifiers),
                    "output_labels": ";".join(output_labels),
                }
            )
            continue

        ref_volume, ref_spacing, ref_phase = load_dicom_series(ref_series_dir)
        bbox = None

        out_labels_dir = output_root / "labels" / patient_id
        if label_entry is not None:
            output_labels, ref_label_files, ref_label_qualifiers = build_reference_label_files(
                label_entries,
                reference_series_id,
                ref_volume.shape,
                ref_spacing,
                args.final_flip_axes,
                out_labels_dir,
                overwrite_existing=overwrite_existing_labels,
            )
            ref_union_mask = build_reference_union_mask(
                label_entries,
                reference_series_id,
                ref_volume.shape,
            )
            if np.count_nonzero(ref_union_mask) == 0:
                ref_union_mask = None
        else:
            output_labels = []
            ref_label_files = []
            ref_label_qualifiers = []
            ref_union_mask = None
        processed_patients += 1

        series_items = sorted(series_dirs.items())
        total_series = len(series_items)
        for series_index, (series_id, series_dir) in enumerate(series_items, start=1):
            if phase_map and (patient_id, series_id) not in phase_map:
                continue
            series_start = time.time()
            volume, spacing, phase = load_dicom_series(series_dir)

            if spacing != ref_spacing or volume.shape != ref_volume.shape:
                volume = resample_volume_to_ref(volume, spacing, ref_volume.shape, ref_spacing, -1024.0)

            ref_xyz = to_xyz(ref_volume)
            mov_xyz = to_xyz(volume)

            transform_type = "ANTsPy" if ants is not None else "SimpleITK" if sitk is not None else "none"
            transform_params = ""
            ants_mask_xyz = None
            if args.ants_mask == "liver":
                ants_mask_xyz = to_xyz(ref_union_mask) if ref_union_mask is not None else None

            if ants is not None:
                aligned_xyz, transform_type, transform_params = register_antspy(
                    ref_xyz,
                    mov_xyz,
                    ref_spacing,
                    fixed_mask_xyz=ants_mask_xyz,
                )
            elif sitk is not None:
                aligned_xyz, transform_type, transform_params = rigid_register_sitk(
                    ref_xyz,
                    mov_xyz,
                    ref_spacing,
                    fixed_mask_xyz=ants_mask_xyz,
                )
            else:
                aligned_xyz = mov_xyz

            aligned_zyx = to_zyx(aligned_xyz)
            if args.final_flip_axes != "none":
                for axis_char in args.final_flip_axes.lower():
                    if axis_char in {"x", "y", "z"}:
                        axis_index = {"z": 0, "y": 1, "x": 2}[axis_char]
                        aligned_zyx = np.flip(aligned_zyx, axis=axis_index)
            # Cropping disabled to keep full volumes.

            out_images_dir = output_root / "images" / patient_id
            phase_name = phase_map.get((patient_id, series_id), phase or "unknown")
            image_name = f"{patient_id}_SER{series_id:05d}_{phase_name}.nii.gz"

            image_path = out_images_dir / image_name

            save_nifti(image_path, aligned_zyx.astype(np.float32), ref_spacing)

            elapsed = time.time() - series_start
            log(
                f"  Series {series_index}/{len(series_dirs)}: SER{series_id:05d} phase={phase_name} "
                f"transform={transform_type} time={elapsed:.1f}s"
            )

            report_rows.append(
                {
                    "patient_id": patient_id,
                    "series_id": series_id,
                    "phase": phase_name,
                    "reference_series_id": reference_series_id,
                    "label_files": ";".join(ref_label_files),
                    "label_qualifiers": ";".join(ref_label_qualifiers),
                    "spacing_x": spacing[0],
                    "spacing_y": spacing[1],
                    "spacing_z": spacing[2],
                    "aligned_spacing_x": ref_spacing[0],
                    "aligned_spacing_y": ref_spacing[1],
                    "aligned_spacing_z": ref_spacing[2],
                    "transform_type": transform_type,
                    "transform_params": transform_params,
                    "crop_bbox": "none",
                    "output_image": str(image_path),
                    "output_labels": ";".join(output_labels),
                }
            )

    if args.labels_only:
        report_path = output_root / "labels_fill_report.csv"
    else:
        report_path = output_root / "alignment_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    log(
        f"Done. Patients processed: {processed_patients}. "
        f"Skipped (no series): {skipped_no_series}. "
        f"Skipped (no aligned reference image): {skipped_no_reference_image}. "
        f"Series outputs: {len(report_rows)}."
    )
    log(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
