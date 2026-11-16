"""
Hyperparameter tuning runner for multiphase segmentation.

This script performs systematic hyperparameter grid search with cross-validation
across all folds. Single command execution suitable for overnight runs.

Usage:
    python scripts/training/hyperparameter_tuning.py \
        --folds-csv generated_helper_csv_files/cv_multiphase_folds_cect_tumor_32.csv \
        --gpus 0,1,2,3 \
        --output-dir runs/hyperparameter_tuning \
        [--max-folds 4]  # Optional: limit to N folds per config for quick testing
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """Save configuration to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def create_tuning_grid() -> List[Dict[str, Any]]:
    """
    Define hyperparameter search grid.
    
    Returns:
        List of configuration dictionaries to test
    """
    # Base configuration overrides for tuning
    grid = []
    
    # GRID 1: Learning rate sweep
    for lr in [0.00005, 0.0001, 0.0002]:
        grid.append({
            "name": f"lr_{lr:.5f}",
            "train.lr": lr,
            "train.min_lr": lr / 100,
        })
    
    # GRID 2: Batch size (accounting for available VRAM)
    for batch_size, accum_steps in [(10, 1), (8, 2), (6, 3)]:
        grid.append({
            "name": f"batch_{batch_size}_accum_{accum_steps}",
            "train.batch_size": batch_size,
            "train.accumulation_steps": accum_steps,
        })
    
    # GRID 3: Weight decay
    for wd in [0.00005, 0.0001, 0.0005]:
        grid.append({
            "name": f"wd_{wd:.5f}",
            "train.weight_decay": wd,
        })
    
    # GRID 4: Dice weight in loss
    for dice_w in [0.3, 0.5, 0.7]:
        grid.append({
            "name": f"dice_{dice_w:.1f}",
            "train.dice_weight": dice_w,
        })
    
    # GRID 5: Model capacity (attention tokens)
    for attn_tokens in [1024, 1536, 2048]:
        grid.append({
            "name": f"attn_{attn_tokens}",
            "model.attention_max_tokens": attn_tokens,
        })
    
    # GRID 6: Encoder backbone
    for backbone in ["resnet18", "resnet34"]:
        grid.append({
            "name": f"backbone_{backbone}",
            "model.encoder_backbone": backbone,
        })
    
    return grid


def apply_config_overrides(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply dot-notation overrides to base configuration.
    
    Example: "train.lr" -> 0.0002 sets config["train"]["lr"] = 0.0002
    """
    config = copy.deepcopy(base_config)
    
    for key, value in overrides.items():
        if key == "name":
            continue
        
        # Navigate nested dict using dot notation
        parts = key.split(".")
        current = config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run hyperparameter tuning grid search for multiphase segmentation"
    )
    p.add_argument(
        "--folds-csv",
        type=Path,
        required=True,
        help="Path to CV folds CSV manifest",
    )
    p.add_argument(
        "--base-config",
        type=Path,
        default=Path("config/training/multiphase_quadro5000_v1.yaml"),
        help="Base configuration to build overrides from",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/hyperparameter_tuning"),
        help="Root output directory for all tuning runs",
    )
    p.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU IDs for parallel fold execution",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable",
    )
    p.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Limit to N folds per config (for quick testing)",
    )
    p.add_argument(
        "--skip-grids",
        type=str,
        default="",
        help="Comma-separated grid names to skip (lr, batch, wd, dice, attn, backbone)",
    )
    p.add_argument(
        "--only-grids",
        type=str,
        default="",
        help="Only run specified grids (comma-separated)",
    )
    return p.parse_args()


def run_tuning(args: argparse.Namespace) -> None:
    """Execute hyperparameter tuning grid search."""
    
    # Load base configuration
    if not args.base_config.exists():
        raise FileNotFoundError(f"Base config not found: {args.base_config}")
    
    base_config = load_yaml(args.base_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load folds CSV to get fold count
    folds_df = pd.read_csv(args.folds_csv)
    all_folds = sorted(folds_df["fold"].unique())
    if args.max_folds:
        all_folds = all_folds[:args.max_folds]
    
    print(f"[HPT] Loaded {len(all_folds)} folds from {args.folds_csv}")
    print(f"[HPT] Base config: {args.base_config}")
    print(f"[HPT] GPUs: {args.gpus}")
    
    # Generate tuning grid
    grid = create_tuning_grid()
    
    # Filter by skip_grids or only_grids
    if args.skip_grids:
        skip = set(args.skip_grids.split(","))
        grid = [g for g in grid if not any(skip_name in g["name"] for skip_name in skip)]
    
    if args.only_grids:
        only = set(args.only_grids.split(","))
        grid = [g for g in grid if any(only_name in g["name"] for only_name in only)]
    
    print(f"[HPT] Tuning grid size: {len(grid)} configurations")
    
    # Tracking
    summary_file = output_dir / "tuning_summary.json"
    results = []
    start_time = datetime.now()
    
    # Iterate through grid
    for grid_idx, grid_config in enumerate(grid, 1):
        config_name = grid_config["name"]
        config_dir = output_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create config file
        tuned_config = apply_config_overrides(base_config, grid_config)
        config_path = config_dir / "config.yaml"
        save_yaml(tuned_config, config_path)
        
        print(f"\n{'='*80}")
        print(f"[HPT] [{grid_idx}/{len(grid)}] Starting: {config_name}")
        print(f"[HPT] Overrides: {', '.join(f'{k}={v}' for k, v in grid_config.items() if k != 'name')}")
        print(f"[HPT] Output: {config_dir}")
        print(f"{'='*80}\n")
        
        # Launch multi-GPU fold execution
        cmd = [
            args.python,
            str(ROOT / "scripts" / "training" / "launch_cv_multi_gpu.py"),
            "--folds-csv", str(args.folds_csv),
            "--config", str(config_path),
            "--output-dir", str(config_dir / "cv_results"),
            "--gpus", args.gpus,
            "--python", args.python,
        ]
        
        try:
            start_config_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=False,
                text=True,
                timeout=None,  # No timeout for overnight runs
            )
            elapsed_config_time = time.time() - start_config_time
            
            status = "success" if result.returncode == 0 else "failed"
            print(f"\n[HPT] {config_name}: {status.upper()} (took {elapsed_config_time/3600:.1f}h)")
            
            results.append({
                "grid_idx": grid_idx,
                "config_name": config_name,
                "status": status,
                "elapsed_seconds": elapsed_config_time,
                "overrides": {k: v for k, v in grid_config.items() if k != "name"},
            })
            
        except subprocess.TimeoutExpired:
            print(f"\n[HPT] {config_name}: TIMEOUT")
            results.append({
                "grid_idx": grid_idx,
                "config_name": config_name,
                "status": "timeout",
                "overrides": {k: v for k, v in grid_config.items() if k != "name"},
            })
        except Exception as e:
            print(f"\n[HPT] {config_name}: ERROR - {e}")
            results.append({
                "grid_idx": grid_idx,
                "config_name": config_name,
                "status": "error",
                "error": str(e),
                "overrides": {k: v for k, v in grid_config.items() if k != "name"},
            })
        
        # Save intermediate results
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    
    # Final summary
    total_time = datetime.now() - start_time
    print(f"\n{'='*80}")
    print(f"[HPT] TUNING COMPLETE")
    print(f"[HPT] Total time: {total_time}")
    print(f"[HPT] Results saved to: {summary_file}")
    print(f"[HPT] Results directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Print summary table
    df_results = pd.DataFrame(results)
    print(df_results[["config_name", "status"]].to_string())
    print()


if __name__ == "__main__":
    args = parse_args()
    run_tuning(args)
