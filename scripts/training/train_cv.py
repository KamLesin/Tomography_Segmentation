from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multiphase_seg.train import run_fold_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one CV fold for multiphase segmentation")
    p.add_argument("--config", type=Path, default=Path("config/training/multiphase_default.yaml"))
    p.add_argument("--folds-csv", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/multiphase_cv"))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-batch-progress", action="store_true", help="Disable per-batch tqdm progress bars")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_fold_training(
        config_path=args.config,
        folds_csv=args.folds_csv,
        fold=args.fold,
        output_dir=args.output_dir,
        device_override=args.device,
        show_batch_progress=not args.no_batch_progress,
    )
    print(result)


if __name__ == "__main__":
    main()
