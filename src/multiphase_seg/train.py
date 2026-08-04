from __future__ import annotations

import json
import random
from pathlib import Path
import traceback
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .data import MultiphaseSliceDataset, build_patient_records
from .losses import bce_dice_loss
from .metrics import dice_coefficient
from .model import MultiphaseLateFusionUNet


PHASE_TO_INDEX = {"A": 0, "PV": 1, "D": 2}


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_dataloaders(cfg: Dict[str, Any], fold: int, folds_df: pd.DataFrame):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    records = build_patient_records(
        mode=data_cfg["mode"],
        cect_root=Path(data_cfg["cect_root"]) if data_cfg.get("cect_root") else None,
        full_root=Path(data_cfg["full_root"]) if data_cfg.get("full_root") else None,
        missing_phase_strategy=data_cfg.get("missing_phase_strategy", "drop"),
    )

    val_uids = set(folds_df.loc[folds_df["fold"] == fold, "patient_uid"].tolist())
    train_uids = set(folds_df.loc[folds_df["fold"] != fold, "patient_uid"].tolist())

    ds_train = MultiphaseSliceDataset(
        records,
        patient_uids=train_uids,
        image_size=tuple(data_cfg.get("image_size", [320, 320])),
        context_slices=int(data_cfg.get("context_slices", 2)),
        hu_window=tuple(data_cfg.get("hu_window", [-200.0, 300.0])),
        max_slices_per_patient=data_cfg.get("max_train_slices_per_patient"),
        cache_items=int(data_cfg.get("cache_items", 8)),
        cache_enabled=bool(data_cfg.get("training_cache_enabled", False)),
        cache_root=Path(data_cfg["training_cache_root"]) if data_cfg.get("training_cache_root") else None,
        cache_dtype=str(data_cfg.get("training_cache_dtype", "float32")),
        cache_version=str(data_cfg.get("training_cache_version", "v1")),
        rebuild_cache=bool(data_cfg.get("training_cache_rebuild", False)),
        force_phase_input=data_cfg.get("force_phase_input"),
    )

    ds_val = MultiphaseSliceDataset(
        records,
        patient_uids=val_uids,
        image_size=tuple(data_cfg.get("image_size", [320, 320])),
        context_slices=int(data_cfg.get("context_slices", 2)),
        hu_window=tuple(data_cfg.get("hu_window", [-200.0, 300.0])),
        max_slices_per_patient=data_cfg.get("max_val_slices_per_patient"),
        cache_items=int(data_cfg.get("cache_items", 8)),
        cache_enabled=bool(data_cfg.get("training_cache_enabled", False)),
        cache_root=Path(data_cfg["training_cache_root"]) if data_cfg.get("training_cache_root") else None,
        cache_dtype=str(data_cfg.get("training_cache_dtype", "float32")),
        cache_version=str(data_cfg.get("training_cache_version", "v1")),
        rebuild_cache=bool(data_cfg.get("training_cache_rebuild", False)),
        force_phase_input=data_cfg.get("force_phase_input"),
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
    )

    return dl_train, dl_val


def _apply_eval_phase_override(phase_present: torch.Tensor, eval_phase_override: Optional[str]) -> torch.Tensor:
    if eval_phase_override is None:
        return phase_present

    key = str(eval_phase_override).strip().upper()
    if not key or key == "ALL":
        return phase_present
    if key not in PHASE_TO_INDEX:
        raise ValueError("train.eval_phase_override must be one of: A, PV, D, all")

    out = torch.zeros_like(phase_present, dtype=torch.bool)
    idx = PHASE_TO_INDEX[key]
    out[:, idx] = phase_present[:, idx]
    return out


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    dice_weight: float,
    amp_enabled: bool,
    eval_phase_override: Optional[str] = None,
    show_progress: bool = True,
    progress_label: Optional[str] = None,
):
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    dices = []
    cache_hits = []

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and is_train)

    iterator = tqdm(loader, leave=False, disable=not show_progress, desc=progress_label)
    for batch in iterator:
        x = batch["phases"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        phase_present = batch["phase_present"].to(device, non_blocking=True) > 0.5
        if not is_train:
            phase_present = _apply_eval_phase_override(phase_present, eval_phase_override)
        cache_hit = batch.get("cache_hit")
        if cache_hit is not None:
            cache_hits.extend(cache_hit.detach().cpu().view(-1).tolist())

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(x, phase_present=phase_present)
            loss = bce_dice_loss(logits, y, dice_weight=dice_weight)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        dice = dice_coefficient(logits.detach(), y)
        losses.append(float(loss.detach().cpu()))
        dices.append(float(dice.detach().cpu()))

    cache_hit_rate = float(np.mean(cache_hits)) if cache_hits else float("nan")
    return (
        float(np.mean(losses)) if losses else float("nan"),
        float(np.mean(dices)) if dices else float("nan"),
        cache_hit_rate,
    )


def run_fold_training(
    config_path: Path,
    folds_csv: Path,
    fold: int,
    output_dir: Path,
    device_override: Optional[str] = None,
    show_batch_progress: bool = True,
    epoch_log_interval: int = 1,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    seed_everything(int(cfg.get("seed", 42)))

    folds_df = pd.read_csv(folds_csv)

    out_fold = Path(output_dir) / f"fold_{fold:02d}"
    out_fold.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_override) if device_override else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    try:
        model = MultiphaseLateFusionUNet(
            in_channels_per_phase=int(model_cfg.get("in_channels_per_phase", 5)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            pretrained_encoder=bool(model_cfg.get("pretrained_encoder", False)),
            encoder_backbone=str(model_cfg.get("encoder_backbone", "resnet34")),
            fusion_mode=str(model_cfg.get("fusion_mode", "cross_attention")),
            attention_heads=int(model_cfg.get("attention_heads", 8)),
            attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
            attention_max_tokens=(
                None
                if model_cfg.get("attention_max_tokens") is None
                else int(model_cfg.get("attention_max_tokens", 4096))
            ),
        ).to(device)

        dl_train, dl_val = _build_dataloaders(cfg, fold, folds_df)
        train_size = len(dl_train.dataset)
        val_size = len(dl_val.dataset)
        if train_size == 0 or val_size == 0:
            raise ValueError(
                "Empty dataset split after fold filtering. "
                f"train_samples={train_size}, val_samples={val_size}, fold={fold}. "
                "Check: folds CSV patient_uids, data roots in config, and missing_phase_strategy/force_phase_input settings."
            )
        print(f"[fold={fold}] train_samples={train_size}, val_samples={val_size}", flush=True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get("epochs", 40)),
            eta_min=float(train_cfg.get("min_lr", 1e-6)),
        )

        best_val_dice = -1.0
        best_train_dice = -1.0
        history = []

        epochs = int(train_cfg.get("epochs", 40))
        dice_weight = float(train_cfg.get("dice_weight", 0.5))
        amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
        eval_phase_override = train_cfg.get("eval_phase_override")

        for epoch in range(1, epochs + 1):
            train_loss, train_dice, train_cache_hit_rate = _run_epoch(
                model,
                dl_train,
                optimizer,
                device,
                dice_weight=dice_weight,
                amp_enabled=amp_enabled,
                eval_phase_override=None,
                show_progress=show_batch_progress,
                progress_label=f"train fold {fold:02d} epoch {epoch:03d}/{epochs}",
            )

            with torch.no_grad():
                val_loss, val_dice, val_cache_hit_rate = _run_epoch(
                    model,
                    dl_val,
                    optimizer=None,
                    device=device,
                    dice_weight=dice_weight,
                    amp_enabled=amp_enabled,
                    eval_phase_override=eval_phase_override,
                    show_progress=show_batch_progress,
                    progress_label=f"val fold {fold:02d} epoch {epoch:03d}/{epochs}",
                )

            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_cache_hit_rate": train_cache_hit_rate,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_cache_hit_rate": val_cache_hit_rate,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)

            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row}, out_fold / "last.pt")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row}, out_fold / "best.pt")

            if train_dice > best_train_dice:
                best_train_dice = train_dice

            if epoch_log_interval > 0 and (epoch % epoch_log_interval == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"[fold={fold}] epoch={epoch:03d}/{epochs} "
                    f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} best_train_dice={best_train_dice:.4f} "
                    f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} best_val_dice={best_val_dice:.4f} "
                    f"train_cache_hit_rate={train_cache_hit_rate:.3f} val_cache_hit_rate={val_cache_hit_rate:.3f}"
                , flush=True)

        history_path = out_fold / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        result = {
            "fold": fold,
            "best_val_dice": best_val_dice,
            "history_path": str(history_path),
            "output_dir": str(out_fold),
        }

        with open(out_fold / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result
    except Exception as exc:
        error_payload = {
            "fold": int(fold),
            "config_path": str(config_path),
            "folds_csv": str(folds_csv),
            "output_dir": str(out_fold),
            "device": str(device),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        with open(out_fold / "error.json", "w", encoding="utf-8") as f:
            json.dump(error_payload, f, indent=2)
        raise
