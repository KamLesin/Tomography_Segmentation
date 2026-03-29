"""Tune fold-balancing weights for split_into_folds.py.

This utility runs a grid search over patient/series/image weights,
creates candidate splits, scores them by spread, and copies the best
candidate to a canonical output prefix.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_float_list(text: str) -> list[float]:
    values = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        values.append(float(x))
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def run_split(
    python_exe: str,
    splitter_script: Path,
    input_csv: Path,
    n_folds: int,
    out_prefix: Path,
    weight_patients: float,
    weight_series: float,
    weight_images: float,
    swap_iters: int,
    seed: int,
) -> tuple[bool, str]:
    cmd = [
        python_exe,
        str(splitter_script),
        "--input-csv",
        str(input_csv),
        "--n-folds",
        str(n_folds),
        "--output-prefix",
        str(out_prefix),
        "--weight-patients",
        str(weight_patients),
        "--weight-series",
        str(weight_series),
        "--weight-images",
        str(weight_images),
        "--swap-iters",
        str(swap_iters),
        "--seed",
        str(seed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-600:]
        return False, tail
    return True, "ok"


def score_split(
    folds_csv: Path,
    score_weight_patients: float,
    score_weight_series: float,
    score_weight_images: float,
) -> dict:
    df = pd.read_csv(folds_csv)
    grouped = (
        df.groupby("fold")
        .agg(
            patients=("patient_id", "count"),
            total_series=("n_series", "sum"),
            total_images=("total_images", "sum"),
        )
        .reset_index()
        .sort_values("fold")
    )

    p_spread = (grouped["patients"].max() - grouped["patients"].min()) / (
        grouped["patients"].mean() + 1e-12
    )
    s_spread = (grouped["total_series"].max() - grouped["total_series"].min()) / (
        grouped["total_series"].mean() + 1e-12
    )
    i_spread = (grouped["total_images"].max() - grouped["total_images"].min()) / (
        grouped["total_images"].mean() + 1e-12
    )

    score = (
        score_weight_patients * p_spread
        + score_weight_series * s_spread
        + score_weight_images * i_spread
    )

    return {
        "patients_min": int(grouped["patients"].min()),
        "patients_max": int(grouped["patients"].max()),
        "series_min": int(grouped["total_series"].min()),
        "series_max": int(grouped["total_series"].max()),
        "images_min": int(grouped["total_images"].min()),
        "images_max": int(grouped["total_images"].max()),
        "patients_spread": float(p_spread),
        "series_spread": float(s_spread),
        "images_spread": float(i_spread),
        "score": float(score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search fold balancing weights and select the best split"
    )
    parser.add_argument("--input-csv", required=True, help="Series metadata CSV path")
    parser.add_argument("--n-folds", type=int, required=True, help="Number of folds")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated fold files and tuning report",
    )
    parser.add_argument(
        "--canonical-prefix",
        default="cv_best",
        help="Best split copied to <output-dir>/<canonical-prefix>_folds.csv and _stats.txt",
    )
    parser.add_argument(
        "--candidate-prefix",
        default="cv_tuned",
        help="Prefix base for candidate outputs",
    )

    parser.add_argument("--weights-patients", default="8,10,12")
    parser.add_argument("--weights-series", default="3,4")
    parser.add_argument("--weights-images", default="2,3")

    parser.add_argument("--swap-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--score-weight-patients",
        type=float,
        default=5.0,
        help="Scoring weight for patient count spread",
    )
    parser.add_argument(
        "--score-weight-series",
        type=float,
        default=2.0,
        help="Scoring weight for series spread",
    )
    parser.add_argument(
        "--score-weight-images",
        type=float,
        default=1.5,
        help="Scoring weight for image spread",
    )

    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable to use when invoking split script",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    splitter_script = script_dir / "prepare_metadata" / "split_into_folds.py"

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    w_patients = parse_float_list(args.weights_patients)
    w_series = parse_float_list(args.weights_series)
    w_images = parse_float_list(args.weights_images)

    rows = []

    for wp in w_patients:
        for ws in w_series:
            for wi in w_images:
                tag = f"{args.candidate_prefix}_p{wp:g}_s{ws:g}_i{wi:g}".replace(".", "_")
                out_prefix = output_dir / tag
                ok, msg = run_split(
                    python_exe=args.python_exe,
                    splitter_script=splitter_script,
                    input_csv=input_csv,
                    n_folds=args.n_folds,
                    out_prefix=out_prefix,
                    weight_patients=wp,
                    weight_series=ws,
                    weight_images=wi,
                    swap_iters=args.swap_iters,
                    seed=args.seed,
                )

                if not ok:
                    rows.append(
                        {
                            "tag": tag,
                            "ok": False,
                            "weight_patients": wp,
                            "weight_series": ws,
                            "weight_images": wi,
                            "error": msg,
                        }
                    )
                    continue

                score_info = score_split(
                    folds_csv=Path(str(out_prefix) + "_folds.csv"),
                    score_weight_patients=args.score_weight_patients,
                    score_weight_series=args.score_weight_series,
                    score_weight_images=args.score_weight_images,
                )

                rows.append(
                    {
                        "tag": tag,
                        "ok": True,
                        "weight_patients": wp,
                        "weight_series": ws,
                        "weight_images": wi,
                        **score_info,
                        "error": "",
                    }
                )

    results_df = pd.DataFrame(rows)
    results_csv = output_dir / f"{args.candidate_prefix}_weight_search_results.csv"
    results_df.to_csv(results_csv, index=False)

    ok_df = results_df[results_df["ok"] == True].copy()
    if ok_df.empty:
        raise RuntimeError(f"No successful candidates. See {results_csv}")

    ok_df = ok_df.sort_values("score")
    best = ok_df.iloc[0]

    best_prefix = output_dir / str(best["tag"])
    canonical_prefix = output_dir / args.canonical_prefix

    shutil.copyfile(str(best_prefix) + "_folds.csv", str(canonical_prefix) + "_folds.csv")
    shutil.copyfile(str(best_prefix) + "_stats.txt", str(canonical_prefix) + "_stats.txt")

    print("Tuning complete")
    print(f"Results: {results_csv}")
    print(
        "Best:",
        f"tag={best['tag']}",
        f"weights=(patients={best['weight_patients']}, series={best['weight_series']}, images={best['weight_images']})",
        f"score={best['score']:.6f}",
    )
    print(f"Canonical folds: {canonical_prefix}_folds.csv")
    print(f"Canonical stats: {canonical_prefix}_stats.txt")


if __name__ == "__main__":
    main()
