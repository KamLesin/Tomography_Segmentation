# Liver/Tumor Segmentation - Reproducible Package

This folder is a standalone training package that can be published to GitHub.
It includes:

- training code (`src/`)
- data preparation pipeline (`scripts/`)
- fold generation utilities
- cross-validation launch scripts

The goal is to rebuild train-ready `Full_data_converted/` from raw SANNA data,
then train UNet models with k-fold validation.

## Project Layout

```
new_project/
├── src/
│   ├── model.py
│   ├── dataset.py
│   └── train.py
├── scripts/
│   ├── run_full_pipeline.py
│   ├── sanna_prepare.py
│   ├── add_tumor_data.py
│   ├── prepare_metadata_for_folds.py
│   └── prepare_metadata/
│       └── split_into_folds.py
├── run_training.bat
├── run_training.sh
├── run_all_folds.bat
├── run_all_folds.sh
├── run_all_folds_parallel.bat
├── run_all_folds_parallel.sh
├── aggregate_results.py
└── requirements.txt
```

## Environment Setup

1. Create/activate Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Pipeline (Raw -> Train-ready)

Run full conversion with one command:

```bash
python scripts/run_full_pipeline.py \
    --base-dicom-root "C:/path/to/SANNA_FULL/Liver3D_originals" \
    --base-nifti-root "C:/path/to/SANNA_FULL/Liver3D_labels" \
    --tumor-dicom-root "C:/path/to/SANNA_FULL/tumors/Liver3D_originals" \
    --tumor-nifti-root "C:/path/to/SANNA_FULL/tumors/Liver3D_labels" \
    --output-dir "C:/path/to/Full_data_converted" \
    --n-folds 6 \
    --verbose
```

Pipeline steps:

1. `sanna_prepare.py` converts base SANNA DICOM+NIfTI to per-slice NPZ and creates `metadata.csv`.
2. `add_tumor_data.py` appends tumor subset data to the same `metadata.csv`.
3. `prepare_metadata_for_folds.py` builds series-level metadata for stratification.
4. `split_into_folds.py` creates `cv_folds.csv` and `cv_stats.txt`.

Output expected in data root:

- `metadata.csv`
- `cv_folds.csv`
- `cv_stats.txt`
- `images/*/slice_*.npz`
- `labels/*/slice_*.npz`

## Fold Tuning Utility

To re-tune fold balancing for a different number of folds, use:

```bash
python scripts/tune_folds.py \
    --input-csv "C:/path/to/Full_data_converted/series_metadata_for_folds.csv" \
    --n-folds 16 \
    --output-dir "C:/path/to/Full_data_converted" \
    --canonical-prefix "cv16_best"
```

This performs grid search over patient/series/image balancing weights,
saves all candidate results, and copies the best split to:

- `cv16_best_folds.csv`
- `cv16_best_stats.txt`

You can override the weight grid with:

- `--weights-patients`
- `--weights-series`
- `--weights-images`

## Training

Direct training command:

```bash
python src/train.py \
    --metadata "C:/path/to/Full_data_converted/metadata.csv" \
    --folds-csv "C:/path/to/Full_data_converted/cv_folds.csv" \
    --data-root "C:/path/to/Full_data_converted" \
    --fold 0 \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --img-size 512 \
    --gpu 0 \
    --use-batchnorm \
    --weighted-loss \
    --experiment "baseline"
```

Or use helper scripts:

- sequential single fold: `run_training.bat` / `run_training.sh`
- sequential all folds: `run_all_folds.bat` / `run_all_folds.sh`
- parallel all folds: `run_all_folds_parallel.bat` / `run_all_folds_parallel.sh`

Optional environment override for helper scripts:

- `SEG_DATA_ROOT` (path to `Full_data_converted`)

## Results

Training artifacts are written to:

```
runs/<experiment_name>/fold_<N>/
```

Each fold directory contains checkpoints, best model, and TensorBoard events.

Aggregate metrics after CV:

```bash
python aggregate_results.py --experiment-dir runs/<experiment_name>
```

## Notes

- `dataset.py` filters out entries with `label_path == 'none'`.
- Label conversion is multiclass: `0=background`, `1=liver`, `2=tumor`.
- HU normalization in loader: clip to `[-200, 300]`, then scale to `[0, 1]`.

## License

This project is licensed under the MIT License.
See `LICENSE`.

## CI

GitHub Actions workflow is included at `.github/workflows/ci.yml`.

It runs on every push and pull request and performs:

1. dependency installation from `requirements.txt`
2. Python syntax checks (`compileall`) for `src/` and `scripts/`
3. CLI smoke tests (`--help`) for all major training/data-prep entrypoints
