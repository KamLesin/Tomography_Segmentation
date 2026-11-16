from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import yaml


def collect_fold_results(config_dir: Path, expected_folds: list[int]) -> list[float]:
    scores = []
    for fold in expected_folds:
        path = config_dir / "cv_results" / f"fold_{fold:02d}" / "result.json"
        with path.open(encoding="utf-8") as f:
            result = json.load(f)
        if result["fold"] != fold:
            raise ValueError(f"Incorrect fold ID in {path}")
        score = float(result["best_val_dice"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid best_val_dice in {path}: {score}")
        scores.append(score)
    if not scores:
        raise ValueError(f"No expected folds for {config_dir}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank completed tuning configurations by mean patient validation Dice")
    parser.add_argument("--tuning-dir", type=Path, required=True)
    parser.add_argument("--export-best", action="store_true")
    parser.add_argument("--target-config", type=Path, default=Path("config/training/multiphase_production.yaml"))
    args = parser.parse_args()

    with (args.tuning_dir / "tuning_summary.json").open(encoding="utf-8") as f:
        entries = json.load(f)
    ranked = []
    reference_folds = None
    for entry in entries:
        if entry["status"] != "success":
            print(f"Skipping {entry['config_name']}: {entry['status']}")
            continue
        folds = sorted(entry["folds"])
        if reference_folds is None:
            reference_folds = folds
        elif folds != reference_folds:
            raise ValueError("Configurations have different validation folds and cannot be ranked fairly")
        config_dir = args.tuning_dir / entry["config_name"]
        scores = collect_fold_results(config_dir, folds)
        ranked.append((statistics.mean(scores), statistics.stdev(scores) if len(scores) > 1 else 0.0, entry["config_name"], config_dir))
    if not ranked:
        raise ValueError("No complete successful configurations found")
    ranked.sort(key=lambda result: result[0], reverse=True)
    print("config                         mean Dice   std Dice   folds")
    for mean, std, name, _ in ranked:
        print(f"{name:30} {mean:9.4f} {std:10.4f} {len(reference_folds):7}")

    if args.export_best:
        if args.target_config.exists():
            raise FileExistsError(f"Refusing to overwrite {args.target_config}")
        with (ranked[0][3] / "config.yaml").open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        args.target_config.parent.mkdir(parents=True, exist_ok=True)
        with args.target_config.open("x", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"Best configuration saved unchanged to {args.target_config}")
    print("Transfer: retain batch size, learning rate and architecture initially; validate on the target hardware.")
    print("These validation folds were used for selection; use independent evaluation or nested CV for unbiased reporting.")


if __name__ == "__main__":
    main()
