from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch CV fold training across multiple GPUs")
    p.add_argument("--folds-csv", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config/training/multiphase_default.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/multiphase_cv"))
    p.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def _run_folds_for_gpu(gpu_id: str, fold_queue: List[int], lock: Lock, args: argparse.Namespace) -> None:
    while True:
        with lock:
            if not fold_queue:
                return
            fold = fold_queue.pop(0)

        cmd = [
            args.python,
            str(ROOT / "scripts" / "training" / "train_cv.py"),
            "--config",
            str(args.config),
            "--folds-csv",
            str(args.folds_csv),
            "--fold",
            str(fold),
            "--output-dir",
            str(args.output_dir),
            "--device",
            "cuda:0",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        print(f"[GPU {gpu_id}] Starting fold {fold}")
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Fold {fold} failed on GPU {gpu_id} with code {proc.returncode}")
        print(f"[GPU {gpu_id}] Finished fold {fold}")



def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.folds_csv)
    folds = sorted(df["fold"].unique().tolist())
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("No GPUs specified")

    # Dynamic scheduling: every free GPU pulls the next fold from a shared queue.
    fold_queue = list(folds)
    lock = Lock()

    print(f"Folds queue: {fold_queue}")
    print(f"GPUs: {gpu_ids}")

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(_run_folds_for_gpu, gpu, fold_queue, lock, args) for gpu in gpu_ids]
        for fut in futures:
            fut.result()

    print("All assigned folds finished successfully.")


if __name__ == "__main__":
    main()
