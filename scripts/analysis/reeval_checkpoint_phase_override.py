"""Recompute validation Dice for an already-trained arm's saved checkpoints under a
different `train.eval_phase_override`, without retraining.

Use this to make a multiphase arm (e.g. H2's baseline_unregistered, trained with no
phase restriction) comparable to a single-phase arm (e.g. H1's baseline_single_phase,
evaluated with eval_phase_override=PV): both must be scored on the same phase-
restricted validation target for the comparison to be meaningful.

Example:
    python scripts/analysis/reeval_checkpoint_phase_override.py \\
        --run-dir ../results_from_apl19/h2_effb0_32fold_8gpu_corrected_rerun \\
        --arm baseline_unregistered \\
        --eval-phase-override PV \\
        --output results/h2_baseline_unregistered_pv_eval/fold_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multiphase_seg.model import MultiphaseLateFusionUNet
from multiphase_seg.train import _build_dataloaders, _run_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, required=True, help="Hypothesis run dir containing arms/<arm>/fold_XX/best.pt")
    p.add_argument("--arm", type=str, required=True)
    p.add_argument("--eval-phase-override", type=str, required=True, help="A | PV | D | all")
    p.add_argument("--folds-csv", type=str, default=None, help="Defaults to the folds_csv recorded in run-dir/summary.json.")
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def main() -> None:
    args = parse_args()
    run_dir = _resolve(args.run_dir)

    config_path = run_dir / "configs" / f"{args.arm}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("train", {})["eval_phase_override"] = args.eval_phase_override

    folds_csv_path = _resolve(args.folds_csv) if args.folds_csv else None
    if folds_csv_path is None:
        with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        folds_csv_path = _resolve(summary["folds_csv"])
    folds_df = pd.read_csv(folds_csv_path)

    fold_values = sorted(int(v) for v in folds_df["fold"].unique().tolist())
    if args.folds:
        fold_values = [f for f in fold_values if f in args.folds]

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    rows = []
    for fold in fold_values:
        ckpt_path = run_dir / "arms" / args.arm / f"fold_{fold:02d}" / "best.pt"
        if not ckpt_path.exists():
            print(f"[skip] missing checkpoint arm={args.arm} fold={fold}")
            continue

        model = MultiphaseLateFusionUNet(
            in_channels_per_phase=int(model_cfg.get("in_channels_per_phase", 5)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            pretrained_encoder=bool(model_cfg.get("pretrained_encoder", False)),
            encoder_backbone=str(model_cfg.get("encoder_backbone", "resnet34")),
            fusion_mode=str(model_cfg.get("fusion_mode", "cross_attention")),
            attention_heads=int(model_cfg.get("attention_heads", 8)),
            attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
            attention_max_tokens=(
                None if model_cfg.get("attention_max_tokens") is None else int(model_cfg.get("attention_max_tokens", 4096))
            ),
        ).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()

        _, dl_val = _build_dataloaders(cfg, fold, folds_df)
        with torch.no_grad():
            val_loss, val_dice, val_cache_hit_rate = _run_epoch(
                model,
                dl_val,
                optimizer=None,
                device=device,
                dice_weight=float(train_cfg.get("dice_weight", 0.5)),
                amp_enabled=amp_enabled,
                eval_phase_override=args.eval_phase_override,
                show_progress=True,
                progress_label=f"reeval fold {fold:02d}",
            )
        print(f"[fold={fold}] val_dice({args.eval_phase_override})={val_dice:.4f}")
        rows.append(
            {
                "arm": args.arm,
                "fold": fold,
                "best_val_dice": val_dice,
                "val_loss": val_loss,
                "eval_phase_override": args.eval_phase_override,
                "source_checkpoint": str(ckpt_path),
            }
        )
        del model

    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
