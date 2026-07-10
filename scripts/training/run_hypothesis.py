from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multiphase_seg.train import run_fold_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run predefined hypothesis experiments for multiphase segmentation")
    p.add_argument("--hypothesis", choices=["h1", "h2", "h3", "h4"], required=True)
    p.add_argument("--config", type=Path, default=Path("config/training/multiphase_default.yaml"))
    p.add_argument("--folds-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/hypotheses"))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs for dynamic multi-GPU fold execution")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable for subprocess launchers")
    p.add_argument("--phase", choices=["A", "PV", "D"], default="PV")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--small-backbone", type=str, default="resnet18")
    p.add_argument("--large-backbone", type=str, default="resnet34")
    p.add_argument("--aligned-cect-root", type=str, default=None)
    p.add_argument("--aligned-full-root", type=str, default=None)
    p.add_argument("--unaligned-cect-root", type=str, default=None)
    p.add_argument("--unaligned-full-root", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_project_path(path: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def dump_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _set_nested(cfg: Dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _guess_unaligned_root(configured_root: Optional[str]) -> Optional[str]:
    if not configured_root:
        return None

    raw = str(configured_root)
    candidates = [
        raw.replace("_aligned_fixcheck", ""),
        raw.replace("_aligned", ""),
        raw.replace("full_data_converted", "Full_data_converted"),
    ]

    for candidate in candidates:
        p = (ROOT / candidate).resolve()
        if p.exists():
            try:
                return str(p.relative_to(ROOT)).replace("\\", "/")
            except ValueError:
                return str(p)

    # Best effort fallback if path cannot be validated at parse-time.
    return candidates[1] if len(candidates) > 1 else candidates[0]


def _discover_folds(folds_csv: Path, max_folds: Optional[int]) -> List[int]:
    df = pd.read_csv(folds_csv)
    folds = sorted(df["fold"].unique().tolist())
    if max_folds is not None:
        folds = folds[: max(0, int(max_folds))]
    if not folds:
        raise ValueError("No folds found for execution")
    return [int(x) for x in folds]


def _build_h1_or_h3_arms(base_cfg: Dict[str, Any], phase: str, hypothesis: str) -> List[Tuple[str, Dict[str, Any]]]:
    baseline = copy.deepcopy(base_cfg)
    _set_nested(baseline, "data.force_phase_input", phase)
    _set_nested(baseline, "train.eval_phase_override", phase)

    multiphase = copy.deepcopy(base_cfg)
    _set_nested(multiphase, "data.force_phase_input", None)
    _set_nested(multiphase, "train.eval_phase_override", phase)

    _set_nested(baseline, "experiment_name", f"{hypothesis}_baseline_single_train_single_infer_{phase.lower()}")
    _set_nested(multiphase, "experiment_name", f"{hypothesis}_multiphase_train_single_infer_{phase.lower()}")

    return [
        ("baseline_single_phase", baseline),
        ("multiphase_train_single_infer", multiphase),
    ]


def _build_h2_arms(base_cfg: Dict[str, Any], args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    aligned = copy.deepcopy(base_cfg)
    unaligned = copy.deepcopy(base_cfg)

    mode = str(base_cfg.get("data", {}).get("mode", "mixed")).lower()
    data_cfg = base_cfg.get("data", {})

    if mode in {"cect", "mixed"}:
        aligned_cect = args.aligned_cect_root or data_cfg.get("cect_root")
        unaligned_cect = args.unaligned_cect_root or _guess_unaligned_root(aligned_cect)
        _set_nested(aligned, "data.cect_root", aligned_cect)
        _set_nested(unaligned, "data.cect_root", unaligned_cect)

    if mode in {"full", "mixed"}:
        aligned_full = args.aligned_full_root or data_cfg.get("full_root")
        unaligned_full = args.unaligned_full_root or _guess_unaligned_root(aligned_full)
        _set_nested(aligned, "data.full_root", aligned_full)
        _set_nested(unaligned, "data.full_root", unaligned_full)

    _set_nested(aligned, "experiment_name", "h2_registered_training")
    _set_nested(unaligned, "experiment_name", "h2_unregistered_training")

    return [
        ("baseline_unregistered", unaligned),
        ("registered_training", aligned),
    ]


def _build_h4_arms(
    base_cfg: Dict[str, Any],
    phase: str,
    small_backbone: str,
    large_backbone: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    arms: List[Tuple[str, Dict[str, Any]]] = []

    for model_size, backbone in (("large", large_backbone), ("small", small_backbone)):
        cfg_single = copy.deepcopy(base_cfg)
        _set_nested(cfg_single, "model.encoder_backbone", backbone)
        _set_nested(cfg_single, "data.force_phase_input", phase)
        _set_nested(cfg_single, "train.eval_phase_override", phase)
        _set_nested(cfg_single, "experiment_name", f"h4_{model_size}_single_phase_{phase.lower()}")
        arms.append((f"{model_size}_single_phase", cfg_single))

        cfg_multi = copy.deepcopy(base_cfg)
        _set_nested(cfg_multi, "model.encoder_backbone", backbone)
        _set_nested(cfg_multi, "data.force_phase_input", None)
        _set_nested(cfg_multi, "train.eval_phase_override", phase)
        _set_nested(cfg_multi, "experiment_name", f"h4_{model_size}_multiphase_train_{phase.lower()}")
        arms.append((f"{model_size}_multiphase", cfg_multi))

    return arms


def build_arms(base_cfg: Dict[str, Any], args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    if args.hypothesis == "h1":
        return _build_h1_or_h3_arms(base_cfg, phase=args.phase, hypothesis="h1")
    if args.hypothesis == "h2":
        return _build_h2_arms(base_cfg, args=args)
    if args.hypothesis == "h3":
        return _build_h1_or_h3_arms(base_cfg, phase=args.phase, hypothesis="h3")
    if args.hypothesis == "h4":
        return _build_h4_arms(
            base_cfg,
            phase=args.phase,
            small_backbone=args.small_backbone,
            large_backbone=args.large_backbone,
        )
    raise ValueError(f"Unsupported hypothesis: {args.hypothesis}")


def _run_arm(
    arm_name: str,
    cfg: Dict[str, Any],
    folds_csv: Path,
    folds: List[int],
    run_dir: Path,
    device: Optional[str],
    gpus: Optional[str],
    python_exe: str,
) -> List[Dict[str, Any]]:
    cfg_path = run_dir / "configs" / f"{arm_name}.yaml"
    dump_yaml(cfg_path, cfg)

    arm_output = run_dir / "arms" / arm_name
    arm_output.mkdir(parents=True, exist_ok=True)

    if gpus:
        folds_df = pd.read_csv(folds_csv)
        folds_subset_csv = run_dir / "folds" / f"{arm_name}_folds.csv"
        folds_subset_csv.parent.mkdir(parents=True, exist_ok=True)
        folds_df[folds_df["fold"].isin(folds)].to_csv(folds_subset_csv, index=False)

        cmd = [
            python_exe,
            str(ROOT / "scripts" / "training" / "launch_cv_multi_gpu.py"),
            "--folds-csv",
            str(folds_subset_csv),
            "--config",
            str(cfg_path),
            "--output-dir",
            str(arm_output),
            "--gpus",
            str(gpus),
            "--python",
            str(python_exe),
        ]
        print(f"[{arm_name}] launching multi-GPU folds on gpus={gpus}")
        proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Multi-GPU fold execution failed for arm={arm_name} with code {proc.returncode}")

        arm_rows: List[Dict[str, Any]] = []
        for fold in folds:
            out_fold = arm_output / f"fold_{int(fold):02d}"
            result_path = out_fold / "result.json"
            if not result_path.exists():
                raise FileNotFoundError(f"Missing result file for arm={arm_name}, fold={fold}: {result_path}")
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            arm_rows.append(
                {
                    "arm": arm_name,
                    "fold": int(fold),
                    "best_val_dice": float(result["best_val_dice"]),
                    "result_json": str(result_path),
                    "history_json": str(result.get("history_path", out_fold / "history.json")),
                }
            )
            print(f"[{arm_name}] fold={fold} done, best_val_dice={float(result['best_val_dice']):.5f}")

        return arm_rows

    arm_rows: List[Dict[str, Any]] = []
    for fold in folds:
        print(f"[{arm_name}] fold={fold} start")
        result = run_fold_training(
            config_path=cfg_path,
            folds_csv=folds_csv,
            fold=fold,
            output_dir=arm_output,
            device_override=device,
        )
        arm_rows.append(
            {
                "arm": arm_name,
                "fold": int(fold),
                "best_val_dice": float(result["best_val_dice"]),
                "result_json": str(Path(result["output_dir"]) / "result.json"),
                "history_json": str(result["history_path"]),
            }
        )
        print(f"[{arm_name}] fold={fold} done, best_val_dice={result['best_val_dice']:.5f}")

    return arm_rows


def _plot_fold_boxplot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate hypothesis plots") from exc

    arms = list(df["arm"].drop_duplicates())
    rng = np.random.default_rng(42)
    values_per_arm = [
        df.loc[df["arm"] == arm, "best_val_dice"].to_numpy(dtype=np.float32)
        for arm in arms
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(1, len(arms) + 1)
    ax.boxplot(
        values_per_arm,
        labels=arms,
        positions=positions,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.8},
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
        },
        boxprops={"facecolor": "#b8d8ea", "alpha": 0.9},
        whiskerprops={"linewidth": 1.3},
        capprops={"linewidth": 1.3},
    )

    # Keep per-fold visibility by overlaying jittered points on top of each box.
    for idx, ys in enumerate(values_per_arm, start=1):
        xs = np.full(len(ys), idx, dtype=np.float32) + rng.uniform(-0.09, 0.09, size=len(ys))
        ax.scatter(xs, ys, s=34, alpha=0.85, color="#1f4e79", edgecolor="white", linewidth=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(arms, rotation=15, ha="right")
    ax.set_ylabel("Best validation Dice")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _aggregate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    for arm, sub in df.groupby("arm"):
        values = sub["best_val_dice"].astype(float).to_list()
        summary[arm] = {
            "n_folds": len(values),
            "mean_best_val_dice": float(np.mean(values)),
            "std_best_val_dice": float(np.std(values)),
            "fold_scores": values,
        }
    return summary


def main() -> None:
    args = parse_args()
    config_path = _resolve_project_path(args.config)
    folds_csv_path = _resolve_project_path(args.folds_csv)
    output_root = _resolve_project_path(args.output_dir)

    base_cfg = load_yaml(config_path)
    folds = _discover_folds(folds_csv_path, args.max_folds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.hypothesis}_{timestamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    arms = build_arms(base_cfg, args)

    all_rows: List[Dict[str, Any]] = []
    for arm_name, cfg in arms:
        rows = _run_arm(
            arm_name=arm_name,
            cfg=cfg,
            folds_csv=folds_csv_path,
            folds=folds,
            run_dir=run_dir,
            device=args.device,
            gpus=args.gpus,
            python_exe=args.python,
        )
        all_rows.extend(rows)

    result_df = pd.DataFrame(all_rows).sort_values(["arm", "fold"]).reset_index(drop=True)
    result_csv = run_dir / "fold_results.csv"
    result_df.to_csv(result_csv, index=False)

    title = f"{args.hypothesis.upper()} fold-level Dice (metric: best val Dice)"
    plot_path = run_dir / "dice_fold_boxplot.png"
    _plot_fold_boxplot(result_df, plot_path, title=title)

    summary = {
        "hypothesis": args.hypothesis,
        "phase_for_single_phase_tests": args.phase,
        "folds": folds,
        "config": str(config_path),
        "folds_csv": str(folds_csv_path),
        "arms": _aggregate_summary(result_df),
        "note_h3": (
            "H3 currently uses global Dice only. Small-lesion stratification is not included yet."
            if args.hypothesis == "h3"
            else ""
        ),
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(f"  fold results: {result_csv}")
    print(f"  plot: {plot_path}")
    print(f"  summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
