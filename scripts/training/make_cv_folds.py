from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multiphase_seg.data import build_patient_records
from multiphase_seg.folds import (
    build_fold_statistics,
    create_cv_folds,
    fold_summary,
    save_folds_csv,
    suggested_fold_counts,
    write_fold_statistics_summary,
)


def _resolve_output_path(path_value: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return ROOT / p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create patient-level CV folds for multiphase segmentation")
    p.add_argument("--mode", choices=["cect", "pg", "mixed"], default="mixed")
    p.add_argument("--cect-root", type=Path, default=Path("../CECT_data_aligned"))
    p.add_argument("--pg-root", type=Path, default=Path("../full_data_converted_aligned"))
    p.add_argument("--n-folds", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--missing-phase-strategy", choices=["drop", "keep"], default="drop")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output path for folds CSV. "
            "If omitted, default is generated_helper_csv_files/cv_multiphase_folds_<mode>.csv, "
            "where mode full is tagged as pg."
        ),
    )
    p.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional CSV path for per-fold balance statistics (default: <output>_stats.csv)",
    )
    p.add_argument(
        "--stats-summary-output",
        type=Path,
        default=None,
        help="Optional TXT path for aggregate balance summary (default: <output>_stats_summary.txt)",
    )
    p.add_argument(
        "--compute-lesion-components",
        action="store_true",
        help="Compute 3D connected components in masks as lesion-count proxy (slower).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mode_tag = "pg" if args.mode == "full" else str(args.mode)
    output_path = (
        _resolve_output_path(Path(args.output))
        if args.output is not None
        else ROOT / "generated_helper_csv_files" / f"cv_multiphase_folds_{mode_tag}.csv"
    )
    stats_output = (
        _resolve_output_path(Path(args.stats_output))
        if args.stats_output is not None
        else output_path.with_name(f"{output_path.stem}_stats.csv")
    )
    stats_summary_output = (
        _resolve_output_path(Path(args.stats_summary_output))
        if args.stats_summary_output is not None
        else output_path.with_name(f"{output_path.stem}_stats_summary.txt")
    )

    records = build_patient_records(
        mode=args.mode,
        cect_root=args.cect_root,
        pg_root=args.pg_root,
        missing_phase_strategy=args.missing_phase_strategy,
    )

    if not records:
        raise RuntimeError("No patient records found. Verify roots and filtering options.")

    if args.n_folds > len(records):
        raise RuntimeError(
            f"Cannot create {args.n_folds} folds from only {len(records)} patients. "
            f"Lower --n-folds or increase dataset size."
        )

    df = create_cv_folds(records, n_folds=args.n_folds, seed=args.seed, balance_by_source=True)
    save_folds_csv(df, output_path)

    stats_df = build_fold_statistics(
        records=records,
        folds_df=df,
        n_folds=args.n_folds,
        compute_lesion_components=bool(args.compute_lesion_components),
    )
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(stats_output, index=False)
    write_fold_statistics_summary(stats_df, stats_summary_output)

    print(f"Saved folds to: {output_path}")
    print(f"Saved fold stats to: {stats_output}")
    print(f"Saved fold stats summary to: {stats_summary_output}")
    print("Fold summary:")
    print(fold_summary(df).to_string(index=False))
    print(f"Total patients: {len(records)}")
    print(f"Suggested fold counts for this size: {suggested_fold_counts(len(records))}")
    if not stats_df.empty:
        print("Balance snapshot:")
        print(
            stats_df[["fold", "patients", "total_slices", "lesion_voxel_fraction"]]
            .to_string(index=False)
        )
        empty_folds = int((stats_df["patients"] == 0).sum())
        print(f"Empty folds: {empty_folds}")


if __name__ == "__main__":
    main()
