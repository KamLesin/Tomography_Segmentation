"""
Prepare metadata for CV fold generation.

Converts the slice-level metadata.csv to the format expected by split_into_folds.py:
- patient_id, study_id, series_id, labeled, num_slices, slice_thickness, vendor, contrast, label_voxel_count

Since our NPZ metadata doesn't have DICOM vendor/study info directly,
we'll need to infer or use defaults.
"""

import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_npz_and_check_label(npz_path: str) -> int:
    """Load NPZ file and count non-zero label voxels."""
    try:
        data = np.load(npz_path)
        arr = data['arr_0']
        return int(np.sum(arr > 0))
    except Exception as e:
        logger.warning(f"Failed to load {npz_path}: {e}")
        return 0


def prepare_metadata_for_folds(input_csv: str, output_csv: str, check_labels: bool = False):
    """
    Convert slice-level metadata to series-level metadata for fold generation.
    
    Args:
        input_csv: Path to metadata.csv (slice-level)
        output_csv: Output path for series-level metadata
        check_labels: If True, actually load NPZ files to count label voxels (slow!)
    """
    logger.info(f"Loading metadata from {input_csv}")
    df = pd.read_csv(input_csv)
    
    logger.info(f"Total slices: {len(df)}")
    logger.info(f"Unique patients: {df['patient_id'].nunique()}")
    logger.info(f"Unique series: {df.groupby(['patient_id', 'series_id']).ngroups}")
    
    # Get base directory for loading NPZ files
    base_dir = Path(input_csv).parent
    
    # Group by patient and series to create series-level metadata
    logger.info("Aggregating to series level...")
    
    series_records = []
    
    for (patient_id, series_id), group in df.groupby(['patient_id', 'series_id']):
        # Basic info
        num_slices = len(group)
        slice_thickness = group['spacing_z'].median()
        
        # Check if series has any labels
        # We can estimate this by checking if any slice has a label file
        # Or we can assume all our data has labels (since it came from labeled dataset)
        has_label = 1  # Assume all data is labeled
        
        # Count label voxels if requested
        label_voxel_count = 0
        if check_labels and not group['label_path'].isna().all():
            logger.debug(f"Counting label voxels for patient {patient_id}, series {series_id}")
            for _, row in group.iterrows():
                label_path = base_dir / row['label_path']
                if label_path.exists():
                    label_voxel_count += load_npz_and_check_label(str(label_path))
        else:
            # Estimate: assume 10% of slices have labels, each with ~1000 voxels
            # This is just a rough estimate for stratification
            label_voxel_count = int(num_slices * 0.1 * 1000)
        
        # Try to infer vendor from phase or other metadata
        # For now, use placeholder since we don't have vendor in our metadata
        # You could add this by reading DICOM files or using a mapping
        vendor = "SIEMENS"  # Default placeholder
        
        # Study ID: for our data, we can use series_id as study_id
        # or default to 1 if we consider each patient-series as one study
        study_id = 1
        
        # Contrast: infer from phase
        phase = group['phase'].iloc[0] if 'phase' in group.columns else 'unknown'
        contrast = 0 if phase in ['no_contrast', 'precontrast', 'unknown'] else 1
        
        series_records.append({
            'patient_id': int(patient_id),
            'study_id': int(study_id),
            'series_id': int(series_id),
            'labeled': has_label,
            'num_slices': num_slices,
            'slice_thickness': float(slice_thickness),
            'vendor': vendor,
            'contrast': contrast,
            'label_voxel_count': label_voxel_count,
        })
    
    # Create DataFrame
    result_df = pd.DataFrame(series_records)
    
    logger.info(f"Created series-level metadata with {len(result_df)} series")
    logger.info(f"Labeled series: {result_df['labeled'].sum()}")
    logger.info(f"Series with contrast: {result_df['contrast'].sum()}")
    
    # Save
    result_df.to_csv(output_csv, index=False)
    logger.info(f"Saved to {output_csv}")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare metadata for CV fold generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input metadata.csv (slice-level)',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output metadata (series-level)',
    )
    parser.add_argument(
        '--check_labels',
        action='store_true',
        help='Actually load NPZ files to count label voxels (slow!)',
    )
    parser.add_argument(
        '--vendor_csv',
        type=str,
        default=None,
        help='Optional CSV with patient_id,vendor mapping to use real vendor info',
    )
    
    args = parser.parse_args()
    
    # If vendor mapping provided, load it
    vendor_map = {}
    if args.vendor_csv:
        logger.info(f"Loading vendor mapping from {args.vendor_csv}")
        vendor_df = pd.read_csv(args.vendor_csv)
        vendor_map = dict(zip(vendor_df['patient_id'], vendor_df['vendor']))
    
    prepare_metadata_for_folds(args.input, args.output, args.check_labels)


if __name__ == "__main__":
    main()
