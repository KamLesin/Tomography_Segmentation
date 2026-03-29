#!/bin/bash
# Run k-fold cross validation - train on all 6 folds

set -euo pipefail

DATA_ROOT="${SEG_DATA_ROOT:-../Full_data_converted}"
METADATA="${DATA_ROOT}/metadata.csv"
FOLDS="${DATA_ROOT}/cv_folds.csv"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Train on each fold
for f in 0 1 2 3 4 5; do
    echo
    echo "========================================"
    echo "Training fold $f"
    echo "========================================"
    echo
    
    "${PYTHON_BIN}" src/train.py \
        --metadata "${METADATA}" \
        --folds-csv "${FOLDS}" \
        --data-root "${DATA_ROOT}" \
        --fold $f \
        --epochs 15 \
        --batch-size 4 \
        --lr 0.0001 \
        --img-size 512 \
        --gpu 0 \
        --use-batchnorm \
        --weighted-loss \
        --experiment "baseline"
    
    if [ $? -ne 0 ]; then
        echo "Error in fold $f"
        exit 1
    fi
done

echo
echo "========================================"
echo "All folds completed!"
echo "========================================"
