#!/bin/bash
# Training script for liver segmentation

set -euo pipefail

DATA_ROOT="${SEG_DATA_ROOT:-../Full_data_converted}"
METADATA="${DATA_ROOT}/metadata.csv"
FOLDS="${DATA_ROOT}/cv_folds.csv"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Run training
"${PYTHON_BIN}" src/train.py \
    --metadata "${METADATA}" \
    --folds-csv "${FOLDS}" \
    --data-root "${DATA_ROOT}" \
    --fold 0 \
    --epochs 15 \
    --batch-size 4 \
    --lr 0.0001 \
    --img-size 512 \
    --gpu 0 \
    --use-batchnorm \
    --weighted-loss \
    --experiment "baseline"
