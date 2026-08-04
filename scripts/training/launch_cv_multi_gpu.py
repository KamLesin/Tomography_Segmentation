from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import traceback
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _should_show_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("%"):
        return False
    if stripped.startswith("[GPU "):
        return False
    return True


def _prefix_line(line: str, prefix: str) -> str:
    return f"{prefix} {line}" if line.endswith("\n") else f"{prefix} {line}\n"


def _stream_process_output(proc: subprocess.Popen[str], log_path: Path, prefix: str) -> None:
    assert proc.stdout is not None
    with open(log_path, "a", encoding="utf-8") as log_file:
        for line in proc.stdout:
            log_file.write(line)
            if _should_show_line(line):
                print(_prefix_line(line.rstrip("\n"), prefix), end="")


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

        fold_dir = Path(args.output_dir) / f"fold_{int(fold):02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        launcher_log_path = fold_dir / "launcher.log"

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
            "--no-batch-progress",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        print(f"[GPU {gpu_id}] Starting fold {fold}")
        with open(launcher_log_path, "w", encoding="utf-8") as f:
            f.write(f"gpu_id={gpu_id}\n")
            f.write("command=\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("=== STREAMED OUTPUT ===\n")

        prefix = f"[GPU {gpu_id} fold {fold:02d}]"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        print(f"{prefix} started")
        _stream_process_output(proc, launcher_log_path, prefix)
        proc.wait()

        with open(launcher_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\nreturncode={proc.returncode}\n")

        if proc.returncode != 0:
            raise RuntimeError(
                f"Fold {fold} failed on GPU {gpu_id} with code {proc.returncode}. See {launcher_log_path}"
            )
        print(f"{prefix} finished")



def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.folds_csv)
    if df.empty:
        raise ValueError(
            f"Folds CSV has no rows: {args.folds_csv}. Generate folds first with scripts/training/make_cv_folds.py."
        )
    if "fold" not in df.columns:
        raise ValueError(f"Missing required 'fold' column in folds CSV: {args.folds_csv}")
    folds = sorted(df["fold"].unique().tolist())
    if not folds:
        raise ValueError(
            f"No unique fold IDs found in {args.folds_csv}. Check the CSV content and generation step."
        )
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("No GPUs specified")

    # Dynamic scheduling: every free GPU pulls the next fold from a shared queue.
    fold_queue = list(folds)
    lock = Lock()

    print(f"Folds queue ({len(fold_queue)}): {fold_queue}")
    print(f"GPUs: {gpu_ids}")

    try:
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = [pool.submit(_run_folds_for_gpu, gpu, fold_queue, lock, args) for gpu in gpu_ids]
            for fut in futures:
                fut.result()
    except Exception:
        print("Launcher failed. Traceback:")
        print(traceback.format_exc())
        raise

    print("All assigned folds finished successfully.")


if __name__ == "__main__":
    main()
