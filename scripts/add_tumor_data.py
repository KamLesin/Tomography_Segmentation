"""  
Add tumor data to existing NPZ dataset - appends to metadata.csv

This script processes tumor-only data from SANNA_FULL/tumors/ and appends it
to the existing Full_data_converted dataset. It continues numbering from where
the previous conversion left off.

Features:
- Continues ID numbering from existing metadata
- Appends to existing metadata.csv (preserves all existing data)
- Uses same conversion logic as original sanna_prepare.py
- Processes DICOM from tumors/Liver3D_originals/
- Processes labels from tumors/Liver3D_labels/
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from dataclasses import dataclass
from collections import defaultdict
import warnings
import re

import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Phase detection patterns (from analyze_all_series.py)
PHASE_PATTERNS = {
    'precontrast': [
        r'\bpre\b', r'\bnative\b', r'\bnon-contrast\b', r'\bplain\b', 
        r'\bwithout\b', r'\bprecontrast\b', r'\bpre-contrast\b',
        r'\bbez kontrastu\b', r'\bnativna\b', r'\bnativ\b', r'\bbez cm\b',
        r'pre\s+contrast', r'\bpremonitoring\b'
    ],
    'arterial': [
        r'\barterial\b', r'\barter\b', r'\bart\s+phase\b', r'\bart\.\b', 
        r'\bearly\b', r'\bhap\b', r'tetnicz', r'tentnicza',
        r'faza tętnicza', r'faza arterial', r'f\.?\s*tetnic',
        r'faza tetnic'
    ],
    'venous': [
        r'\bvenous\b', r'\bven\b', r'\bportal\b', r'\bpvp\b', 
        r'\bportal venous\b', r'\bport\b', r'\bpv\b',
        r'żyln', r'faza żylna', r'faza venous', r'zyln',
        r'faza portal', r'wrotna', r'f\.?\s*zyln'
    ],
    'delayed': [
        r'\bdelayed\b', r'\blate\b', r'\bequilibrium\b', r'\beq\b', 
        r'\bdelay\b', r'późn', r'faza późna', r'pozn',
        r'równowaga', r'faza opóźniona', r'faza pozn'
    ],
    'hepatic': [
        r'\bhepatic\b', r'\bliver\b', r'wątrobowa', r'wątroba'
    ],
}


def detect_phase_from_text(text: str) -> str:
    """Wykrywa fazę badania na podstawie tekstu z DICOM."""
    if not text:
        return 'unknown'
    
    text_lower = text.lower()
    for phase, patterns in PHASE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return phase
    return 'unknown'


@dataclass
class DicomMetadata:
    """Store DICOM series metadata."""
    spacing_x: float
    spacing_y: float
    spacing_z: float
    num_slices: int
    shape: Tuple[int, int, int]  # (Z, H, W)
    affine: np.ndarray
    rescale_slope: float
    rescale_intercept: float
    image_position_patient: Optional[List[float]] = None
    phase: str = 'unknown'


@dataclass
class NiftiMetadata:
    """Store NIfTI label metadata."""
    spacing_x: float
    spacing_y: float
    spacing_z: float
    shape: Tuple[int, int, int]
    affine: np.ndarray
    axis_order: str = 'zyx'


def load_dicom_series(series_dir: str) -> Tuple[np.ndarray, DicomMetadata]:
    """Load DICOM series and convert to HU."""
    dicom_files = []
    for root, dirs, files in os.walk(series_dir):
        for f in files:
            if f.endswith('.dcm') or not '.' in f:
                dicom_files.append(os.path.join(root, f))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {series_dir}")
    
    # Read first file to get metadata
    ds = pydicom.dcmread(dicom_files[0], force=True)
    
    # Get spacing
    pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
    spacing_x, spacing_y = float(pixel_spacing[1]), float(pixel_spacing[0])
    slice_thickness_raw = getattr(ds, 'SliceThickness', None)
    if slice_thickness_raw is None:
        slice_thickness_raw = getattr(ds, 'SpacingBetweenSlices', None)
    try:
        slice_thickness = float(slice_thickness_raw)
    except (TypeError, ValueError):
        slice_thickness = 1.0
    
    # Rescale parameters
    rescale_slope = float(getattr(ds, 'RescaleSlope', 1.0))
    rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    
    # Phase detection
    series_desc = getattr(ds, 'SeriesDescription', '')
    protocol_name = getattr(ds, 'ProtocolName', '')
    phase = detect_phase_from_text(f"{series_desc} {protocol_name}")
    
    # Read all slices
    slices_data = []
    for dcm_path in dicom_files:
        ds = pydicom.dcmread(dcm_path, force=True)
        instance_number = int(getattr(ds, 'InstanceNumber', 0))
        image_position = getattr(ds, 'ImagePositionPatient', None)
        z_pos = float(image_position[2]) if image_position else instance_number
        
        pixel_array = ds.pixel_array.astype(np.float32)
        # Convert to HU
        hu_array = pixel_array * rescale_slope + rescale_intercept
        
        slices_data.append((z_pos, hu_array, image_position))
    
    # Sort by Z position
    slices_data.sort(key=lambda x: x[0])

    # Prefer spacing estimated from actual z positions if available
    z_positions = [s[0] for s in slices_data]
    if len(z_positions) > 1:
        z_diffs = np.diff(np.array(z_positions, dtype=np.float32))
        z_diffs = np.abs(z_diffs[z_diffs != 0])
        if z_diffs.size > 0:
            slice_thickness = float(np.median(z_diffs))
    
    # Stack volume
    volume = np.stack([s[1] for s in slices_data], axis=0)
    first_image_position = slices_data[0][2]
    
    # Create affine matrix (simplified)
    affine = np.eye(4)
    affine[0, 0] = spacing_x
    affine[1, 1] = spacing_y
    affine[2, 2] = slice_thickness
    if first_image_position:
        affine[:3, 3] = first_image_position
    
    metadata = DicomMetadata(
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        spacing_z=slice_thickness,
        num_slices=volume.shape[0],
        shape=volume.shape,
        affine=affine,
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
        image_position_patient=list(first_image_position) if first_image_position else None,
        phase=phase,
    )
    
    return volume, metadata


def load_nifti_volume(nifti_path: str) -> Tuple[np.ndarray, NiftiMetadata]:
    """Load NIfTI label volume."""
    nii = nib.load(nifti_path)
    volume_xyz = nii.get_fdata().astype(np.float32)
    if volume_xyz.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {volume_xyz.shape} for {nifti_path}")

    # NIfTI is usually (X, Y, Z). Convert to (Z, Y, X) to match DICOM stack shape.
    volume = np.transpose(volume_xyz, (2, 1, 0))
    
    # Get spacing from header
    spacing = nii.header.get_zooms()[:3]
    
    metadata = NiftiMetadata(
        spacing_x=float(spacing[0]),
        spacing_y=float(spacing[1]),
        spacing_z=float(spacing[2]),
        shape=volume.shape,
        affine=nii.affine,
        axis_order='zyx',
    )
    
    return volume, metadata


def find_nifti_for_dicom_series(patient_id: str, series_id: str, nifti_base_dir: str) -> Optional[str]:
    """Find matching NIfTI file for DICOM series."""
    patient_dir = os.path.join(nifti_base_dir, patient_id)
    
    if not os.path.exists(patient_dir):
        return None
    
    # Look for .nii or .nii.gz files
    for f in os.listdir(patient_dir):
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            return os.path.join(patient_dir, f)
    
    return None


def detect_z_flip(dicom_volume: np.ndarray, label_volume: np.ndarray, threshold: float = 0.1) -> bool:
    """Detect if label needs Z-flip by comparing correlation."""
    # Downsample for speed, while keeping at least 1 voxel on each axis.
    dicom_target = tuple(max(1, int(round(dim * 0.25))) for dim in dicom_volume.shape)
    dicom_zoom = tuple(dicom_target[i] / max(dicom_volume.shape[i], 1) for i in range(3))
    dicom_ds = zoom(dicom_volume, dicom_zoom, order=1)

    # Resample label downsample to DICOM downsample shape before correlation.
    label_zoom = tuple(dicom_target[i] / max(label_volume.shape[i], 1) for i in range(3))
    label_ds = zoom(label_volume, label_zoom, order=0)

    if dicom_ds.size == 0 or label_ds.size == 0:
        return False
    
    # Normalize DICOM
    dicom_norm = (dicom_ds - dicom_ds.mean()) / (dicom_ds.std() + 1e-8)
    
    # Correlation normal vs flipped
    corr_normal = np.corrcoef(dicom_norm.flatten(), label_ds.flatten())[0, 1]
    corr_flipped = np.corrcoef(dicom_norm.flatten(), np.flip(label_ds, axis=0).flatten())[0, 1]

    if np.isnan(corr_normal):
        corr_normal = -1.0
    if np.isnan(corr_flipped):
        corr_flipped = -1.0

    return corr_flipped > corr_normal


def resample_label_to_dicom_grid(
    label_volume: np.ndarray,
    label_meta: NiftiMetadata,
    target_shape: Tuple[int, int, int],
    dicom_meta: DicomMetadata,
) -> np.ndarray:
    """Resample label to match DICOM grid."""
    # Prefer geometric scaling by spacing, then force exact target shape.
    spacing_scales = (
        label_meta.spacing_z / max(dicom_meta.spacing_z, 1e-6),
        label_meta.spacing_y / max(dicom_meta.spacing_y, 1e-6),
        label_meta.spacing_x / max(dicom_meta.spacing_x, 1e-6),
    )

    approx_shape = tuple(
        max(1, int(round(label_volume.shape[i] * spacing_scales[i])))
        for i in range(3)
    )

    # First pass: spacing-aware approximation.
    first_zoom = tuple(approx_shape[i] / max(label_volume.shape[i], 1) for i in range(3))
    resampled = zoom(label_volume, first_zoom, order=0)

    # Second pass: force exact target shape (still nearest-neighbor).
    final_zoom = tuple(target_shape[i] / max(resampled.shape[i], 1) for i in range(3))
    resampled = zoom(resampled, final_zoom, order=0)

    # Guard exact shape by center crop/pad.
    result = np.zeros(target_shape, dtype=np.float32)

    src_z0 = max(0, (resampled.shape[0] - target_shape[0]) // 2)
    src_y0 = max(0, (resampled.shape[1] - target_shape[1]) // 2)
    src_x0 = max(0, (resampled.shape[2] - target_shape[2]) // 2)

    dst_z0 = max(0, (target_shape[0] - resampled.shape[0]) // 2)
    dst_y0 = max(0, (target_shape[1] - resampled.shape[1]) // 2)
    dst_x0 = max(0, (target_shape[2] - resampled.shape[2]) // 2)

    copy_z = min(resampled.shape[0], target_shape[0])
    copy_y = min(resampled.shape[1], target_shape[1])
    copy_x = min(resampled.shape[2], target_shape[2])

    result[
        dst_z0:dst_z0 + copy_z,
        dst_y0:dst_y0 + copy_y,
        dst_x0:dst_x0 + copy_x,
    ] = resampled[
        src_z0:src_z0 + copy_z,
        src_y0:src_y0 + copy_y,
        src_x0:src_x0 + copy_x,
    ]

    return (result > 0.5).astype(np.float32)


def shift_volume_z(volume: np.ndarray, shift: int) -> np.ndarray:
    """Shift volume along Z axis, keeping shape constant and padding with zeros."""
    shifted = np.zeros_like(volume)
    if shift == 0:
        return volume.copy()
    if shift > 0:
        shifted[shift:] = volume[:-shift]
    else:
        shifted[:shift] = volume[-shift:]
    return shifted


def fit_best_z_shift(
    dicom_volume: np.ndarray,
    label_volume: np.ndarray,
    max_shift: int = 20,
) -> Tuple[np.ndarray, int, float]:
    """Find best Z shift maximizing overlap with plausible soft-tissue HU range."""
    tissue_mask = (dicom_volume > -80) & (dicom_volume < 250)
    label_mask = label_volume > 0

    if label_mask.sum() == 0:
        return label_volume, 0, 0.0

    best_shift = 0
    best_score = -1.0
    best_volume = label_volume

    for shift in range(-max_shift, max_shift + 1):
        shifted = shift_volume_z(label_mask.astype(np.float32), shift) > 0
        overlap = (shifted & tissue_mask).sum()
        score = overlap / max(shifted.sum(), 1)
        if score > best_score:
            best_score = score
            best_shift = shift
            best_volume = shifted.astype(np.float32)

    return best_volume, best_shift, float(best_score)


def validate_alignment(dicom_volume: np.ndarray, label_volume: np.ndarray, min_overlap: float = 0.01) -> Tuple[bool, float]:
    """Validate that DICOM and label are reasonably aligned."""
    # Check if label has any non-zero voxels
    label_mask = label_volume > 0
    n_label_voxels = label_mask.sum()
    
    if n_label_voxels == 0:
        return False, 0.0
    
    # Check overlap with reasonable HU range for liver (-50 to 200)
    dicom_mask = (dicom_volume > -50) & (dicom_volume < 200)
    overlap = (label_mask & dicom_mask).sum()
    overlap_fraction = overlap / n_label_voxels
    
    return overlap_fraction >= min_overlap, overlap_fraction


def save_slice_to_npz(
    output_path: str,
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    dicom_metadata: DicomMetadata,
    slice_index: int,
) -> None:
    """Save image and label slices to separate NPZ files with metadata."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Compute slice Z position in physical coordinates
    if dicom_metadata.image_position_patient:
        z_position = dicom_metadata.image_position_patient[2] + (
            slice_index * dicom_metadata.spacing_z
        )
    else:
        z_position = slice_index * dicom_metadata.spacing_z
    
    np.savez_compressed(
        output_path,
        arr_0=image_slice.astype(np.float32),
        spacing_x=np.float32(dicom_metadata.spacing_x),
        spacing_y=np.float32(dicom_metadata.spacing_y),
        spacing_z=np.float32(dicom_metadata.spacing_z),
        slice_z_position=np.float32(z_position),
        slice_z_index=np.int32(slice_index),
    )


def process_patient_series(
    patient_id: str,
    dicom_series_dir: str,
    nifti_base_dir: str,
    output_base_dir: str,
    series_id: str,
    starting_id: int,
) -> Dict:
    """
    Process one patient-series pair.
    
    Returns dict with metadata for this series (to be added to metadata.csv).
    """
    result = {
        'patient_id': patient_id,
        'series_id': series_id,
        'status': 'success',
        'num_slices': 0,
        'errors': [],
        'rows': [],
    }
    
    try:
        # Load DICOM series
        logger.info(f"Loading DICOM: {patient_id}/{series_id}")
        dicom_volume, dicom_meta = load_dicom_series(dicom_series_dir)
        
        # Find and load NIfTI label
        nifti_path = find_nifti_for_dicom_series(patient_id, series_id, nifti_base_dir)
        if not nifti_path:
            result['status'] = 'skip_no_label'
            result['errors'].append(f"No NIfTI label found for {patient_id}/{series_id}")
            return result
        
        logger.info(f"Loading NIfTI: {nifti_path}")
        label_volume, label_meta = load_nifti_volume(nifti_path)
        
        # Detect Z-flip
        needs_flip = detect_z_flip(dicom_volume, label_volume)
        if needs_flip:
            logger.info(f"Flipping Z-axis for {patient_id}/{series_id}")
            label_volume = np.flip(label_volume, axis=0)
        
        # Resample label to DICOM grid
        logger.info(f"Resampling label to DICOM grid")
        label_volume = resample_label_to_dicom_grid(
            label_volume, label_meta, dicom_volume.shape, dicom_meta
        )

        # Fit best Z-shift after resampling (common issue in SANNA tumor subset)
        label_volume, z_shift, z_fit_score = fit_best_z_shift(dicom_volume, label_volume)
        logger.info(
            f"Best Z-shift for {patient_id}/{series_id}: {z_shift} slices (score={z_fit_score:.4f})"
        )
        
        # Validate alignment
        is_valid, overlap = validate_alignment(dicom_volume, label_volume)
        if not is_valid:
            result['status'] = 'skip_invalid_alignment'
            result['errors'].append(f"Invalid alignment, overlap fraction: {overlap}")
            return result
        
        # Extract qualifiers from NIfTI filename (P, V, P+V, Vesicle)
        nifti_name = Path(nifti_path).stem
        qualifiers = ''
        if 'Vesicle' in nifti_name:
            qualifiers = 'Vesicle'
        elif 'P+V' in nifti_name:
            qualifiers = 'P+V'
        elif 'P' in nifti_name and 'V' not in nifti_name:
            qualifiers = 'P'
        elif 'V' in nifti_name and 'P' not in nifti_name:
            qualifiers = 'V'
        
        # Save each slice
        output_images_dir = os.path.join(output_base_dir, 'images', f"{patient_id}_{series_id}")
        output_labels_dir = os.path.join(output_base_dir, 'labels', f"{patient_id}_{series_id}")
        
        current_id = starting_id
        
        for z_idx in range(dicom_volume.shape[0]):
            image_slice = dicom_volume[z_idx]
            label_slice = label_volume[z_idx]
            
            image_path = os.path.join(output_images_dir, f"slice_{z_idx}.npz")
            label_path = os.path.join(output_labels_dir, f"slice_{z_idx}.npz")
            
            # Save image
            save_slice_to_npz(image_path, image_slice, label_slice, dicom_meta, z_idx)
            
            # Save label (same format for consistency)
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            np.savez_compressed(
                label_path,
                arr_0=label_slice.astype(np.float32),
                spacing_x=np.float32(dicom_meta.spacing_x),
                spacing_y=np.float32(dicom_meta.spacing_y),
                spacing_z=np.float32(dicom_meta.spacing_z),
                slice_z_position=np.float32(dicom_meta.image_position_patient[2] + z_idx * dicom_meta.spacing_z if dicom_meta.image_position_patient else z_idx * dicom_meta.spacing_z),
                slice_z_index=np.int32(z_idx),
            )
            
            # Add to metadata with continuous ID
            image_rel_path = os.path.relpath(image_path, output_base_dir)
            label_rel_path = os.path.relpath(label_path, output_base_dir)
            
            result['rows'].append({
                'id': current_id,
                'slice_id': z_idx,
                'patient_id': int(patient_id),
                'series_id': int(series_id.replace("SER", "") if "SER" in series_id else 0),
                'image_path': image_rel_path.replace('\\', '/'),
                'label_path': label_rel_path.replace('\\', '/'),
                'spacing_x': dicom_meta.spacing_x,
                'spacing_y': dicom_meta.spacing_y,
                'spacing_z': dicom_meta.spacing_z,
                'qualifiers': qualifiers,
                'nifti_z_flipped': needs_flip,
                'nifti_z_shift': z_shift,
                'nifti_source': Path(nifti_path).name,
                'phase': dicom_meta.phase,
            })
            
            current_id += 1
        
        result['num_slices'] = len(result['rows'])
        logger.info(f"Saved {result['num_slices']} slices for {patient_id}/{series_id}")
    
    except Exception as e:
        result['status'] = 'error'
        result['errors'].append(str(e))
        logger.error(f"Error processing {patient_id}/{series_id}: {e}", exc_info=True)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Add tumor data to existing NPZ dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--tumor_dicom_root',
        type=str,
        default=r'C:\Projekt_badawczy\SANNA_FULL\tumors\Liver3D_originals',
        help='Path to tumors Liver3D_originals directory',
    )
    parser.add_argument(
        '--tumor_nifti_root',
        type=str,
        default=r'C:\Projekt_badawczy\SANNA_FULL\tumors\Liver3D_labels',
        help='Path to tumors Liver3D_labels directory',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=r'C:\Projekt_badawczy\Full_data_converted',
        help='Output directory (existing dataset)',
    )
    parser.add_argument(
        '--existing_metadata',
        type=str,
        default=r'C:\Projekt_badawczy\Full_data_converted\metadata.csv',
        help='Path to existing metadata.csv',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )
    parser.add_argument(
        '--patient_ids',
        type=str,
        default='',
        help='Comma-separated patient IDs to process (e.g. 186,187). Empty = all patients.',
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    selected_patient_ids = {
        pid.strip() for pid in args.patient_ids.split(',') if pid.strip()
    }
    if selected_patient_ids:
        logger.info(f"Filtering to patient IDs: {sorted(selected_patient_ids)}")
    
    # Validate inputs
    tumor_dicom_root = Path(args.tumor_dicom_root)
    tumor_nifti_root = Path(args.tumor_nifti_root)
    output_dir = Path(args.output_dir)
    existing_metadata_path = Path(args.existing_metadata)
    
    if not tumor_dicom_root.exists():
        logger.error(f"Tumor DICOM root does not exist: {tumor_dicom_root}")
        sys.exit(1)
    
    if not tumor_nifti_root.exists():
        logger.error(f"Tumor NIfTI root does not exist: {tumor_nifti_root}")
        sys.exit(1)
    
    if not existing_metadata_path.exists():
        logger.error(f"Existing metadata does not exist: {existing_metadata_path}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read existing metadata to get starting ID
    logger.info(f"Reading existing metadata from {existing_metadata_path}")
    existing_df = pd.read_csv(existing_metadata_path)
    starting_id = existing_df['id'].max() + 1
    logger.info(f"Starting ID for new data: {starting_id}")
    
    # Discover all patient directories in tumor data
    logger.info("Discovering tumor patient data...")
    patient_dirs = []
    
    for patient_dir in sorted(tumor_dicom_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name.replace('XXX', '')  # Handle 002XXX format

        if selected_patient_ids and patient_id not in selected_patient_ids:
            continue
        
        # Check if labels exist
        label_dir = tumor_nifti_root / patient_id
        if not label_dir.exists():
            logger.warning(f"No labels found for patient {patient_id}, skipping")
            continue
        
        # For tumor data, typically all DICOMs are in patient dir directly
        # Check if there's a DICOMS subdirectory structure or direct DICOMs
        if (patient_dir / 'DICOMS').exists():
            # Has DICOMS structure, need to find series
            dicoms_dir = patient_dir / 'DICOMS'
            for study_dir in dicoms_dir.iterdir():
                if not study_dir.is_dir():
                    continue
                for series_dir in study_dir.iterdir():
                    if not series_dir.is_dir():
                        continue
                    patient_dirs.append((patient_id, series_dir.name, str(series_dir)))
        else:
            # Direct DICOM files or single series
            # Treat the whole directory as one series
            patient_dirs.append((patient_id, "SER00001", str(patient_dir)))
    
    logger.info(f"Found {len(patient_dirs)} tumor patient-series pairs")
    
    # Process each pair
    all_new_rows = []
    summary_stats = defaultdict(int)
    current_starting_id = starting_id
    
    for patient_id, series_id, series_dir in tqdm(patient_dirs, desc="Processing tumor data"):
        result = process_patient_series(
            patient_id,
            series_dir,
            str(tumor_nifti_root),
            str(output_dir),
            series_id,
            current_starting_id,
        )
        
        summary_stats[result['status']] += 1
        
        if result['rows']:
            all_new_rows.extend(result['rows'])
            current_starting_id += len(result['rows'])
        
        if result['errors']:
            for error in result['errors']:
                logger.warning(f"[{patient_id}/{series_id}] {error}")
    
    # Append to existing metadata CSV
    if all_new_rows:
        new_df = pd.DataFrame(all_new_rows)
        new_df = new_df[
            [
                'id',
                'slice_id',
                'patient_id',
                'series_id',
                'image_path',
                'label_path',
                'spacing_x',
                'spacing_y',
                'spacing_z',
                'qualifiers',
                'nifti_z_flipped',
                'nifti_z_shift',
                'nifti_source',
                'phase',
            ]
        ]
        
        # Backup existing metadata
        backup_path = existing_metadata_path.parent / f"{existing_metadata_path.stem}_backup.csv"
        logger.info(f"Backing up existing metadata to {backup_path}")
        existing_df.to_csv(backup_path, index=False)
        
        # Append new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=False)
        combined_df.to_csv(existing_metadata_path, index=False)
        logger.info(f"Appended {len(new_df)} rows to {existing_metadata_path}")
        logger.info(f"Total rows now: {len(combined_df)}")
    else:
        logger.warning("No new data to append!")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TUMOR DATA ADDITION SUMMARY")
    logger.info("="*60)
    for status, count in sorted(summary_stats.items()):
        logger.info(f"{status}: {count}")
    logger.info(f"Total new metadata rows added: {len(all_new_rows)}")
    logger.info(f"Starting ID: {starting_id}")
    logger.info(f"Ending ID: {current_starting_id - 1}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
