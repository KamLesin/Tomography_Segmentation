#!/bin/bash
# Run k-fold cross validation on multiple GPUs in parallel
# Each fold runs on a separate GPU simultaneously

set -euo pipefail

DATA_ROOT="${SEG_DATA_ROOT:-../Full_data_converted}"
METADATA="${DATA_ROOT}/metadata.csv"
FOLDS="${DATA_ROOT}/cv_folds.csv"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo
echo "========================================"
echo "Starting 6-fold CV in PARALLEL mode"
echo "Each fold will run on a separate GPU"
echo "========================================"
echo

# Start each fold on a different GPU in background
for f in 0 1 2 3 4 5; do
    echo "Starting fold $f on GPU $f..."
    
    "${PYTHON_BIN}" src/train.py \
        --metadata "${METADATA}" \
        --folds-csv "${FOLDS}" \
        --data-root "${DATA_ROOT}" \
        --fold $f \
        --epochs 15 \
        --batch-size 4 \
        --lr 0.0001 \
        --img-size 512 \
        --gpu $f \
        --use-batchnorm \
        --weighted-loss \
        --experiment "baseline" &
done

echo
echo "========================================"
echo "All 6 folds started in PARALLEL!"
echo "========================================"
echo
echo "Monitor progress:"
echo "- Each fold runs in the background"
echo "- Training Fold 0 - GPU 0"
echo "- Training Fold 1 - GPU 1"
echo "- Training Fold 2 - GPU 2"
echo "- Training Fold 3 - GPU 3"
echo "- Training Fold 4 - GPU 4"
echo "- Training Fold 5 - GPU 5"
echo
echo "Expected completion time: ~1/6 of sequential version"
echo "(if you have 6 GPUs available)"
echo
echo "Use 'jobs' to monitor running processes"
echo "Use 'wait' to wait for all background jobs to complete"
