"""End-to-end data preparation pipeline for training.

This script prepares base SANNA data, appends tumor data, and creates CV folds.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], name: str) -> None:
    print(f"\n=== {name} ===")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed ({name}) with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare dataset from raw SANNA files to train-ready Full_data_converted format"
    )
    parser.add_argument("--base-dicom-root", required=True, help="Path to SANNA_FULL/Liver3D_originals")
    parser.add_argument("--base-nifti-root", required=True, help="Path to SANNA_FULL/Liver3D_labels")
    parser.add_argument("--tumor-dicom-root", required=True, help="Path to SANNA_FULL/tumors/Liver3D_originals")
    parser.add_argument("--tumor-nifti-root", required=True, help="Path to SANNA_FULL/tumors/Liver3D_labels")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory (e.g. Full_data_converted)")
    parser.add_argument("--n-folds", type=int, default=6, help="Number of CV folds")
    parser.add_argument("--tumor-patient-ids", default="", help="Optional comma-separated tumor patient IDs")
    parser.add_argument("--skip-base", action="store_true", help="Skip base SANNA conversion step")
    parser.add_argument("--skip-tumors", action="store_true", help="Skip tumor append step")
    parser.add_argument("--skip-folds", action="store_true", help="Skip fold-generation step")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode in sub-steps")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = project_root / "scripts"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = output_dir / "metadata.csv"

    py = sys.executable

    if not args.skip_base:
        cmd = [
            py,
            str(scripts_dir / "sanna_prepare.py"),
            str(Path(args.base_dicom_root).resolve()),
            str(Path(args.base_nifti_root).resolve()),
            str(output_dir),
        ]
        if args.verbose:
            cmd.append("--verbose")
        run_step(cmd, "Base SANNA conversion")

    if not args.skip_tumors:
        cmd = [
            py,
            str(scripts_dir / "add_tumor_data.py"),
            "--tumor_dicom_root",
            str(Path(args.tumor_dicom_root).resolve()),
            "--tumor_nifti_root",
            str(Path(args.tumor_nifti_root).resolve()),
            "--output_dir",
            str(output_dir),
            "--existing_metadata",
            str(metadata_csv),
        ]
        if args.tumor_patient_ids.strip():
            cmd.extend(["--patient_ids", args.tumor_patient_ids.strip()])
        if args.verbose:
            cmd.append("--verbose")
        run_step(cmd, "Tumor append")

    if not args.skip_folds:
        series_meta_csv = output_dir / "series_metadata_for_folds.csv"
        cmd_prepare = [
            py,
            str(scripts_dir / "prepare_metadata_for_folds.py"),
            "--input",
            str(metadata_csv),
            "--output",
            str(series_meta_csv),
        ]
        run_step(cmd_prepare, "Series metadata aggregation")

        cmd_split = [
            py,
            str(scripts_dir / "prepare_metadata" / "split_into_folds.py"),
            "--input-csv",
            str(series_meta_csv),
            "--n-folds",
            str(args.n_folds),
            "--output-prefix",
            str(output_dir / "cv"),
        ]
        run_step(cmd_split, "Fold split")

    print("\nPipeline finished successfully.")
    print(f"Dataset root: {output_dir}")
    print(f"Metadata: {metadata_csv}")
    print(f"Folds: {output_dir / 'cv_folds.csv'}")


if __name__ == "__main__":
    main()
