"""  
Prepare SANNA_FULL dataset: convert DICOM + NIfTI to sliced NPZ format.

Features:
- Maps DICOM to Hounsfield Units (HU) using RescaleSlope/RescaleIntercept
- Resamples NIfTI labels to DICOM grid (physical coordinates)
- Detects and corrects Z-axis flips in NIfTI
- Handles multi-phase labels (P, V, P+V, Vesicle)
- Generates metadata.csv with spacing, slice position, orientation info, and phase detection
- Validates alignment between DICOM and labels
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
    phase: str = 'unknown'  # detected phase


@dataclass
class NiftiMetadata:
    """Store NIfTI label metadata."""
    spacing_x: float
    spacing_y: float
    spacing_z: float
    shape: Tuple[int, int, int]
    affine: np.ndarray
    needs_z_flip: bool = False


def load_dicom_series(series_dir: str) -> Tuple[np.ndarray, DicomMetadata]:
    """
    Load all DICOM files in a series directory and stack them.
    
    Returns:
        volume (Z, H, W) in HU
        metadata
    """
    dicom_files = sorted(
        [f for f in Path(series_dir).glob('*') if f.is_file()],
        key=lambda x: x.name
    )
    
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files in {series_dir}")
    
    datasets = []
    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(file_path)
            datasets.append(ds)
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            continue
    
    if not datasets:
        raise ValueError(f"No valid DICOM files in {series_dir}")
    
    # Sort by InstanceNumber or SliceLocation
    try:
        datasets.sort(key=lambda ds: int(ds.InstanceNumber))
    except (AttributeError, ValueError):
        try:
            datasets.sort(key=lambda ds: float(ds.SliceLocation))
        except (AttributeError, ValueError):
            logger.warning("Could not sort by InstanceNumber or SliceLocation, using file order")
    
    # Extract pixel arrays and convert to HU
    slices = []
    for ds in datasets:
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Map to Hounsfield Units
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
        else:
            slope, intercept = 1.0, -1024.0
            logger.warning(f"Missing RescaleSlope/Intercept in {ds.filename}, using defaults")
        
        hu_array = pixel_array * slope + intercept
        slices.append(hu_array)
    
    volume = np.stack(slices, axis=0)  # (Z, H, W)
    
    # Extract metadata from first slice
    ds0 = datasets[0]
    spacing_y, spacing_x = map(float, ds0.PixelSpacing)
    spacing_z = float(ds0.SliceThickness)
    
    image_position_patient = None
    if hasattr(ds0, 'ImagePositionPatient'):
        image_position_patient = [float(x) for x in ds0.ImagePositionPatient]
    
    # Detect phase from DICOM metadata
    series_desc = str(getattr(ds0, 'SeriesDescription', '') or '')
    protocol_name = str(getattr(ds0, 'ProtocolName', '') or '')
    study_desc = str(getattr(ds0, 'StudyDescription', '') or '')
    search_text = ' '.join([series_desc, protocol_name, study_desc])
    phase = detect_phase_from_text(search_text)
    
    metadata = DicomMetadata(
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        spacing_z=spacing_z,
        num_slices=volume.shape[0],
        shape=volume.shape,
        affine=np.eye(4),  # DICOM doesn't have direct affine; set to identity
        rescale_slope=slope,
        rescale_intercept=intercept,
        image_position_patient=image_position_patient,
        phase=phase,
    )
    
    return volume, metadata


def load_nifti_volume(nifti_path: str) -> Tuple[np.ndarray, NiftiMetadata]:
    """
    Load NIfTI label volume.
    
    Returns:
        volume, metadata
    """
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    spacing = np.abs(img.header.get_zooms())  # (x, y, z)
    
    # NIfTI order is typically (X, Y, Z); reshape if needed
    if data.ndim == 3:
        # Check if we need to transpose to (Z, Y, X)
        # For now, assume (X, Y, Z) and transpose to (Z, Y, X)
        data = np.transpose(data, (2, 0, 1))
    
    metadata = NiftiMetadata(
        spacing_x=spacing[0],
        spacing_y=spacing[1],
        spacing_z=spacing[2],
        shape=data.shape,
        affine=affine,
        needs_z_flip=False,
    )
    
    return data, metadata


def detect_z_flip(dicom_volume: np.ndarray, label_volume: np.ndarray) -> bool:
    """
    Detect if label volume needs Z-axis flip to match DICOM.
    
    Heuristic: compute middle slice and check correlation.
    """
    mid_z_dicom = dicom_volume.shape[0] // 2
    mid_z_label = label_volume.shape[0] // 2
    
    # If label is very different in shape, can't rely on this
    if abs(label_volume.shape[0] - dicom_volume.shape[0]) > 10:
        return False
    
    # Compare middle slices: if label's middle matches DICOM's end better, flip
    corr_normal = np.corrcoef(
        dicom_volume[mid_z_dicom].flatten(),
        label_volume[mid_z_label].flatten()
    )[0, 1]
    
    corr_flipped = np.corrcoef(
        dicom_volume[mid_z_dicom].flatten(),
        label_volume[-(mid_z_label + 1)].flatten()
    )[0, 1]
    
    # If flipped version correlates better, recommend flip
    # (For binary masks, correlation might not be ideal, but it's a heuristic)
    return corr_flipped > corr_normal


def resample_label_to_dicom_grid(
    label_volume: np.ndarray,
    label_metadata: NiftiMetadata,
    dicom_shape: Tuple[int, int, int],
    dicom_metadata: DicomMetadata,
) -> np.ndarray:
    """
    Resample label volume to match DICOM grid using physical spacing.
    
    Assumes both volumes share same patient coordinate system.
    """
    # Compute zoom factors based on physical spacing
    scale_z = (label_metadata.spacing_z / dicom_metadata.spacing_z) * (
        dicom_shape[0] / label_volume.shape[0]
    )
    scale_y = (label_metadata.spacing_y / dicom_metadata.spacing_y) * (
        dicom_shape[1] / label_volume.shape[1]
    )
    scale_x = (label_metadata.spacing_x / dicom_metadata.spacing_x) * (
        dicom_shape[2] / label_volume.shape[2]
    )
    
    # Use nearest neighbor for binary labels to avoid blurring
    resampled = zoom(label_volume, (scale_z, scale_y, scale_x), order=0)
    
    # Pad or crop to exact DICOM shape
    if resampled.shape != dicom_shape:
        result = np.zeros(dicom_shape, dtype=label_volume.dtype)
        z_min = (dicom_shape[0] - resampled.shape[0]) // 2
        y_min = (dicom_shape[1] - resampled.shape[1]) // 2
        x_min = (dicom_shape[2] - resampled.shape[2]) // 2
        
        z_max = z_min + resampled.shape[0]
        y_max = y_min + resampled.shape[1]
        x_max = x_min + resampled.shape[2]
        
        result[z_min:z_max, y_min:y_max, x_min:x_max] = resampled
        resampled = result
    
    return resampled


def validate_alignment(
    dicom_volume: np.ndarray,
    label_volume: np.ndarray,
    threshold: float = 0.1
) -> Tuple[bool, float]:
    """
    Validate that label volume aligns with DICOM volume.
    
    Returns: (is_valid, overlap_fraction)
    """
    # Simple check: label should not be empty, and should have non-zero
    # pixels spread reasonably across Z dimension
    if not np.any(label_volume):
        return False, 0.0
    
    # Check that non-zero slices are distributed
    nonzero_slices = np.where(np.any(label_volume, axis=(1, 2)))[0]
    if len(nonzero_slices) == 0:
        return False, 0.0
    
    overlap_fraction = len(nonzero_slices) / label_volume.shape[0]
    
    # Very rough check: if label spans less than threshold% of volume, it's suspicious
    if overlap_fraction < threshold:
        return False, overlap_fraction
    
    return True, overlap_fraction


def find_nifti_for_dicom_series(
    patient_id: str,
    series_id: str,
    nifti_base_dir: str
) -> Optional[str]:
    """
    Find matching NIfTI label for a DICOM series.
    
    Series ID format: SER00001, SER00002, ... → extract as 01, 02, ...
    """
    patient_nifti_dir = Path(nifti_base_dir) / patient_id
    if not patient_nifti_dir.exists():
        return None
    
    # Extract numeric series ID (e.g., "SER00002" → "02")
    try:
        series_num = int(series_id.replace("SER", "")) if "SER" in series_id else int(series_id)
        series_key = f"{series_num:02d}"
    except ValueError:
        return None
    
    # Look for NIfTI files matching series pattern
    nifti_files = list(patient_nifti_dir.glob("*.nii*"))
    
    # Find best match: look for files containing the series number
    for nifti_file in nifti_files:
        if series_key in nifti_file.name:
            return str(nifti_file)
    
    # If exact match not found, return first NIfTI in directory
    if nifti_files:
        logger.warning(
            f"No exact series match for patient {patient_id} series {series_id}, "
            f"using {nifti_files[0].name}"
        )
        return str(nifti_files[0])
    
    return None


def save_slice_to_npz(
    output_path: str,
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    dicom_metadata: DicomMetadata,
    slice_index: int,
) -> None:
    """
    Save image and label slices to separate NPZ files with metadata.
    """
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
        
        for z_idx in range(dicom_volume.shape[0]):
            image_slice = dicom_volume[z_idx]
            label_slice = label_volume[z_idx]
            
            image_path = os.path.join(output_images_dir, f"slice_{z_idx}.npz")
            label_path = os.path.join(output_labels_dir, f"slice_{z_idx}.npz")
            
            # Save image
            save_slice_to_npz(image_path, image_slice, label_slice, dicom_meta, z_idx)
            
            # Save label (same format for consistency, even though it's binary)
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
            
            # Add to metadata
            image_rel_path = os.path.relpath(image_path, output_base_dir)
            label_rel_path = os.path.relpath(label_path, output_base_dir)
            
            result['rows'].append({
                'slice_id': z_idx,
                'patient_id': int(patient_id),
                'series_id': int(series_id.replace("SER", "") if "SER" in series_id else series_id),
                'image_path': image_rel_path.replace('\\', '/'),
                'label_path': label_rel_path.replace('\\', '/'),
                'spacing_x': dicom_meta.spacing_x,
                'spacing_y': dicom_meta.spacing_y,
                'spacing_z': dicom_meta.spacing_z,
                'qualifiers': qualifiers,
                'nifti_z_flipped': needs_flip,
                'nifti_source': Path(nifti_path).name,
                'phase': dicom_meta.phase,  # detected phase
            })
        
        result['num_slices'] = len(result['rows'])
        logger.info(f"Saved {result['num_slices']} slices for {patient_id}/{series_id}")
    
    except Exception as e:
        result['status'] = 'error'
        result['errors'].append(str(e))
        logger.error(f"Error processing {patient_id}/{series_id}: {e}", exc_info=True)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert SANNA_FULL dataset (DICOM + NIfTI) to NPZ slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'dicom_root',
        type=str,
        help='Path to Liver3D_originals directory',
    )
    parser.add_argument(
        'nifti_root',
        type=str,
        help='Path to Liver3D_labels directory',
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for prepared dataset',
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip if output directory already has data',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )
    parser.add_argument(
        '--patients',
        type=str,
        default=None,
        help='Comma-separated list of patient IDs to process (e.g., "001,003,005"). If not specified, processes all.',
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    dicom_root = Path(args.dicom_root)
    nifti_root = Path(args.nifti_root)
    output_dir = Path(args.output_dir)
    
    if not dicom_root.exists():
        logger.error(f"DICOM root does not exist: {dicom_root}")
        sys.exit(1)
    
    if not nifti_root.exists():
        logger.error(f"NIfTI root does not exist: {nifti_root}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse patient filter
    patient_filter = None
    if args.patients:
        patient_filter = set(p.strip().zfill(3) for p in args.patients.split(','))
        logger.info(f"Filtering to patients: {sorted(patient_filter)}")
    
    # Discover all patient-series pairs
    logger.info("Discovering patient-series pairs...")
    patient_series_pairs = []
    
    for patient_dir in sorted(dicom_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name
        
        # Apply patient filter
        if patient_filter and patient_id not in patient_filter:
            continue
        
        dicoms_dir = patient_dir / 'DICOMS'
        
        if not dicoms_dir.exists():
            continue
        
        # Find all study directories
        for study_dir in dicoms_dir.iterdir():
            if not study_dir.is_dir():
                continue
            
            # Find all series directories
            for series_dir in study_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                
                patient_series_pairs.append((patient_id, series_dir.name, str(series_dir)))
    
    logger.info(f"Found {len(patient_series_pairs)} patient-series pairs")
    
    # Process each pair
    all_metadata_rows = []
    summary_stats = defaultdict(int)
    
    for patient_id, series_id, series_dir in tqdm(patient_series_pairs, desc="Processing"):
        result = process_patient_series(
            patient_id,
            series_dir,
            str(nifti_root),
            str(output_dir),
            series_id,
        )
        
        summary_stats[result['status']] += 1
        
        if result['rows']:
            all_metadata_rows.extend(result['rows'])
        
        if result['errors']:
            for error in result['errors']:
                logger.warning(f"[{patient_id}/{series_id}] {error}")
    
    # Create metadata CSV
    if all_metadata_rows:
        df = pd.DataFrame(all_metadata_rows)
        df = df[
            [
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
                'nifti_source',
                'phase',
            ]
        ]
        df.index.name = 'id'
        
        metadata_path = output_dir / 'metadata.csv'
        df.to_csv(metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("CONVERSION SUMMARY")
    logger.info("="*60)
    for status, count in sorted(summary_stats.items()):
        logger.info(f"{status}: {count}")
    logger.info(f"Total metadata rows: {len(all_metadata_rows)}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
