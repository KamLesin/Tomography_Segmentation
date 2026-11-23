# tomography_segmentation

This repository contains both:
- data-preparation utilities for SANNA/CECT datasets,
- a modular PyTorch training pipeline for multiphase liver CT segmentation.

## Current layout

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── .github/workflows/ci.yml
├── config/
│   └── training/
│       └── multiphase_default.yaml
├── src/
│   └── multiphase_seg/
│       ├── data.py
│       ├── encoders.py
│       ├── fusion.py
│       ├── decoder.py
│       ├── model.py
│       ├── folds.py
│       └── train.py
└── scripts/
	└── preparing_data/
		├── align_full_data_converted.py
		├── build_unique_series_list_labeled.py
		├── check_unique_series_limit.py
		├── count_unique_patients.py
		└── phase_selection_gui.py
	└── training/
		├── make_cv_folds.py
		├── train_cv.py
		├── run_hypothesis.py
		└── launch_cv_multi_gpu.py
```

## Data preparation

The scripts under [scripts/preparing_data](scripts/preparing_data) support:

- aligning already converted data,
- filtering and validating series lists,
- counting patient/series coverage,
- manually selecting phases for ambiguous series.

## Multiphase segmentation pipeline

Implemented model style:
- input: three phases (A, PV, D), each as 2.5D stack of 5 axial slices,
- encoder: three independent branches (late fusion),
- fusion: cross-attention between phases (or concat-only ablation),
- decoder: U-Net-like decoder with skip connections,
- output: binary segmentation mask.

Main training modules:
- [src/multiphase_seg/data.py](src/multiphase_seg/data.py): dataset and patient record building for `cect`, `full`, and `mixed` modes.
- [src/multiphase_seg/encoders.py](src/multiphase_seg/encoders.py): independent ResNet34 encoders for A/PV/D.
- [src/multiphase_seg/fusion.py](src/multiphase_seg/fusion.py): pairwise cross-attention fusion and concat ablation.
- [src/multiphase_seg/decoder.py](src/multiphase_seg/decoder.py): U-Net style decoder.
- [src/multiphase_seg/model.py](src/multiphase_seg/model.py): complete segmentation model.
- [src/multiphase_seg/folds.py](src/multiphase_seg/folds.py): patient-level CV fold generation.
- [src/multiphase_seg/train.py](src/multiphase_seg/train.py): fold training loop.

### 1) Generate CV folds

```bash
python scripts/training/make_cv_folds.py \
	--mode mixed \
	--cect-root ../CECT_data_aligned \
	--full-root ../full_data_converted_aligned \
	--n-folds 64 \
	--output generated_helper_csv_files/cv_multiphase_folds_mixed.csv
```

If `--output` is omitted, the script auto-generates mode-specific names:
- `cv_multiphase_folds_cect.csv` for `--mode cect`
- `cv_multiphase_folds_mixed.csv` for `--mode mixed`
- `cv_multiphase_folds_pg.csv` for `--mode full` (legacy `full` tagged as `pg`)

The fold script also writes balance statistics by default:
- `generated_helper_csv_files/cv_multiphase_folds_stats.csv`
- `generated_helper_csv_files/cv_multiphase_folds_stats_summary.txt`

Stats include per-fold counts such as:
- number of patients,
- total slice count,
- lesion voxel fraction (`lesion_voxel_fraction`),
- optional lesion component count proxy (`--compute-lesion-components`, slower).

### 2) Train one fold

```bash
python scripts/training/train_cv.py \
	--config config/training/multiphase_default.yaml \
	--folds-csv generated_helper_csv_files/cv_multiphase_folds.csv \
	--fold 0 \
	--output-dir runs/multiphase_cv
```

### 3) Launch all folds on 8 GPUs

```bash
python scripts/training/launch_cv_multi_gpu.py \
	--config config/training/multiphase_default.yaml \
	--folds-csv generated_helper_csv_files/cv_multiphase_folds.csv \
	--gpus 0,1,2,3,4,5,6,7 \
	--output-dir runs/multiphase_cv
```

Assignment policy is deterministic: fold `k` goes to GPU index `k % N`.

Current launcher behavior is dynamic: each free GPU pulls the next fold from a shared queue until all folds are finished.

### Hyperparameter tuning on 4 GPUs

The tuning script varies one parameter at a time around a base configuration:
learning rate (and its minimum), weight decay and Dice loss weight. Batch size,
backbone and attention token limit remain fixed as specified in the base
configuration. The learning-rate candidates are 0.0001 and 0.0002.
Gradient accumulation is not implemented.

```bash
python -u scripts/training/hyperparameter_tuning.py \
    --base-config config/training/multiphase_quadro5000_v1.yaml \
    --folds-csv generated_helper_csv_files/cv_multiphase_folds_cect_tumor_32.csv \
    --gpus 0,1,2,3 \
    --max-folds 4 \
    --output-dir runs/tuning_example
```

`--max-folds` limits the folds executed, not the patients in each training
split. Omit it to execute all folds. Each GPU trains a separate fold, so the
batch size is per process and is not multiplied by the number of GPUs. The
four-fold example executes up to 32 trainings, not necessarily one night.
Training aborts on a failed configuration; inspect `tuning_summary.json` and
the corresponding `cv_results/fold_*/launcher.log`. Use a new output directory
for another run; existing runs are not resumed.

The batch size in the Quadro 5000 configuration is an unverified starting
point, not a 13 GB guarantee. Before an extended search, test on the target
server and monitor each GPU using `nvidia-smi`; peak usage depends on the
model, attention settings, data, CUDA and other GPU processes. Even with a
fixed batch size, an OOM can occur. Adjust the base batch size before tuning
if measurement indicates insufficient headroom.

For eight 24 GB GPUs, `config/training/multiphase_quadro6000.yaml` keeps the
same model and 40 epochs but starts with batch size 16 per fold. Verify it on
the target GPUs; increasing batch size changes optimization and is not
guaranteed to fit merely because the GPU has more memory. The cosine learning
rate scheduler uses `train.epochs` as its period, so reducing the tuning
epochs also changes the learning-rate schedule.

After a successful run, use:

```bash
python scripts/training/analyze_tuning_results.py --tuning-dir runs/tuning_example --export-best
```

The exported config is unchanged for transfer to other hardware; do not
automatically scale its batch size or learning rate. The validation folds
used to select the best configuration are not an independent test set.
Use nested cross-validation or a held-out test set to report unbiased results.

To investigate discrepancies between model-resolution and native-resolution
validation for a saved checkpoint, stop competing GPU work and run from the
project root:

```bash
python -u scripts/training/diagnose_validation.py \
    --config runs/hyperparameter_tuning/lr_0.00010/config.yaml \
    --folds-csv generated_helper_csv_files/cv_multiphase_folds_cect_tumor_32.csv \
    --fold 4 --checkpoint runs/hyperparameter_tuning/lr_0.00010/cv_results/fold_04/best.pt \
    --device cuda:0 --batch-size 2
```

The diagnostic reads but does not update the cache; it compares cached and
fresh inputs and reports model-resolution and native-resolution Dice. Run it
on the machine holding the original data and cache, using the same project
version that produced the checkpoint. A partial selection with
`--max-patients` cannot reproduce the original fold-average Dice.

### 4) Run hypothesis experiments from one CLI entrypoint

The script [scripts/training/run_hypothesis.py](scripts/training/run_hypothesis.py) runs predefined experiment sets and saves:
- fold-level metrics in `fold_results.csv`,
- summary statistics in `summary.json`,
- boxplot (`dice_fold_boxplot.png`) with fold-level points overlaid per option.

Example (Hypothesis 1, PV single-phase inference):

```bash
python scripts/training/run_hypothesis.py \
	--hypothesis h1 \
	--config config/training/multiphase_default.yaml \
	--folds-csv generated_helper_csv_files/cv_multiphase_folds.csv \
	--phase PV \
	--output-dir runs/hypotheses
```

Available presets:
- `h1`: baseline single-phase vs multiphase training, both evaluated in single-phase inference mode.
- `h2`: unregistered training roots vs registered/aligned training roots.
- `h3`: currently same training comparison as `h1`, reported with global Dice only (small-lesion stratification pending).
- `h4`: 2x2 setup for backbone size (`small`/`large`) and training regime (`single-phase`/`multiphase`).

Useful options:
- `--device cuda:0` to pin a run to one GPU.
- `--max-folds 2` for a fast smoke test.
- `--run-name my_h1_test` to control output folder naming.
- `--gpus 0,1,2,3,4,5,6,7` to run folds dynamically on 8 GPUs (free GPU takes next available fold).

### Ablation

Set in [config/training/multiphase_default.yaml](config/training/multiphase_default.yaml):
- `model.fusion_mode: cross_attention` (default),
- `model.fusion_mode: concat` for no-attention baseline.

## Main scripts

- [scripts/preparing_data/align_full_data_converted.py](scripts/preparing_data/align_full_data_converted.py): align or correct converted slice data.
- [scripts/preparing_data/build_unique_series_list_labeled.py](scripts/preparing_data/build_unique_series_list_labeled.py): keep only series belonging to labeled patients.
- [scripts/preparing_data/check_unique_series_limit.py](scripts/preparing_data/check_unique_series_limit.py): report patients with too many series.
- [scripts/preparing_data/count_unique_patients.py](scripts/preparing_data/count_unique_patients.py): count unique patients in target folders.
- [scripts/preparing_data/phase_selection_gui.py](scripts/preparing_data/phase_selection_gui.py): GUI for manual phase assignment.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

The dependency set reflects the data-preparation utilities in this repository.

## Generated data

Keep generated datasets and temporary outputs out of git. The existing [.gitignore](.gitignore) already excludes the common ones, including:

- `Full_data_converted/`
- `runs/`
- `legacy runs/`
- `__pycache__/`
- `*.pyc`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
