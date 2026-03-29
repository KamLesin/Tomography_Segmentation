#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" run_full_pipeline.py \
  --base-dicom-root "${BASE_DICOM_ROOT:-/path/to/SANNA_FULL/Liver3D_originals}" \
  --base-nifti-root "${BASE_NIFTI_ROOT:-/path/to/SANNA_FULL/Liver3D_labels}" \
  --tumor-dicom-root "${TUMOR_DICOM_ROOT:-/path/to/SANNA_FULL/tumors/Liver3D_originals}" \
  --tumor-nifti-root "${TUMOR_NIFTI_ROOT:-/path/to/SANNA_FULL/tumors/Liver3D_labels}" \
  --output-dir "${OUTPUT_DIR:-/path/to/Full_data_converted}" \
  --n-folds "${N_FOLDS:-6}" \
  --verbose
