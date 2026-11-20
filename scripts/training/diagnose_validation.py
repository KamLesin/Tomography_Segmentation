"""Compare cached and fresh validation inputs and native predictions for a saved fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from multiphase_seg.data import MultiphaseSliceDataset, _first_existing_path, build_patient_records
from multiphase_seg.metrics import confusion_counts, dice_from_counts
from multiphase_seg.model import MultiphaseLateFusionUNet
from multiphase_seg.train import _apply_eval_phase_override


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--folds-csv", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-patients", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_patients is not None and args.max_patients < 1):
        parser.error("--batch-size and --max-patients must be positive")

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    train = cfg["train"]
    device = torch.device(args.device)
    manifest = pd.read_csv(args.folds_csv)
    val_uids = {str(uid).strip().casefold() for uid in manifest.loc[manifest["fold"] == args.fold, "patient_uid"]}
    if not val_uids:
        raise ValueError(f"No patients for fold {args.fold}")
    records = build_patient_records(
        mode=data["mode"],
        cect_root=Path(data["cect_root"]) if data.get("cect_root") else None,
        full_root=Path(data["full_root"]) if data.get("full_root") else None,
        missing_phase_strategy=data.get("missing_phase_strategy", "drop"),
    )
    selected = sorted((r for r in records if r.patient_uid.strip().casefold() in val_uids), key=lambda r: r.patient_uid)
    missing = val_uids - {r.patient_uid.strip().casefold() for r in selected}
    if missing:
        raise ValueError(f"Patients missing from data roots: {sorted(missing)}")
    if args.max_patients is not None:
        selected = selected[:args.max_patients]
    cache_enabled = bool(data.get("training_cache_enabled", False))
    cache_root = Path(data.get("training_cache_root") or "runs/cache_npz")
    if cache_enabled and not cache_root.is_dir():
        raise FileNotFoundError(f"Configured cache root does not exist: {cache_root}")

    model = MultiphaseLateFusionUNet.from_config(cfg["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    amp = bool(train.get("amp", True)) and device.type == "cuda"
    override = train.get("eval_phase_override")
    print(
        f"checkpoint_epoch={checkpoint.get('epoch')} "
        f"checkpoint_val_dice={checkpoint.get('metrics', {}).get('val_dice')} "
        f"patients={len(selected)} device={device}",
        flush=True,
    )
    fresh_scores = []
    native_scores = []
    cached_scores = []

    with torch.inference_mode():
        for rec in selected:
            ds = MultiphaseSliceDataset(
                [rec], image_size=tuple(data.get("image_size", [320, 320])),
                context_slices=int(data.get("context_slices", 2)),
                hu_window=tuple(data.get("hu_window", [-200.0, 300.0])),
                cache_enabled=False, cache_dtype=str(data.get("training_cache_dtype", "float32")),
                cache_version=str(data.get("training_cache_version", "v1")),
                force_phase_input=data.get("force_phase_input"),
            )
            if not len(ds):
                raise ValueError(f"No slices available for {rec.patient_uid}")
            ds.cache_root = cache_root
            mask_path = _first_existing_path([rec.mask_paths.get(p) for p in ("PV", "A", "D")])
            if mask_path is None:
                raise FileNotFoundError(f"No mask for {rec.patient_uid}")
            gt = ds._load_volume(mask_path, is_mask=True) > 0
            gt_total = int(gt.sum())
            counts = np.zeros(3, dtype=np.float64)
            cached_counts = np.zeros(3, dtype=np.float64)
            native_tp = native_pred = cached_slices = missing_cache = invalid_meta = phase_mismatch = 0
            mask_mismatch = phase_array_mismatch = 0
            max_image_diff = 0.0
            offset = 0

            for batch in DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0):
                x = batch["phases"].to(device)
                phases = _apply_eval_phase_override(batch["phase_present"].to(device) > 0.5, override)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(x, phase_present=phases)
                counts += np.array([v.sum().item() for v in confusion_counts(logits.float(), batch["mask"].to(device))])
                probs = F.interpolate(torch.sigmoid(logits.float()), size=gt.shape[:2], mode="bilinear", align_corners=False)
                masks = (probs >= 0.5).squeeze(1).cpu().numpy()
                for mask, z in zip(masks, batch["z_index"].tolist()):
                    native_pred += int(mask.sum())
                    native_tp += int(np.logical_and(mask, gt[:, :, int(z)]).sum())

                if cache_enabled:
                    for j, z in enumerate(batch["z_index"].tolist()):
                        path = ds._cache_path(rec, int(z))
                        if not path.exists():
                            missing_cache += 1
                            continue
                        with np.load(path, allow_pickle=False) as payload:
                            meta = json.loads(str(payload["meta"].item()))
                            cached_x = torch.from_numpy(payload["phases"].astype(np.float32))
                            cached_y = torch.from_numpy(payload["mask"].astype(np.float32))
                            cached_phase = torch.from_numpy(payload["phase_present"].astype(np.float32))
                        invalid_meta += int(not ds._validate_cache_meta(meta, rec, int(z)))
                        phase_mismatch += int(meta.get("force_phase_input") != ds.force_phase_input)
                        mask_mismatch += int(not torch.equal(cached_y, batch["mask"][j]))
                        phase_array_mismatch += int(not torch.equal(cached_phase, batch["phase_present"][j]))
                        max_image_diff = max(max_image_diff, float((cached_x - batch["phases"][j]).abs().max()))
                        cached_slices += 1
                        cached_phase = _apply_eval_phase_override(cached_phase.unsqueeze(0).to(device) > 0.5, override)
                        with torch.autocast(device_type=device.type, enabled=amp):
                            cached_logits = model(cached_x.unsqueeze(0).to(device), phase_present=cached_phase)
                        cached_counts += np.array([
                            v.sum().item() for v in confusion_counts(cached_logits.float(), cached_y.unsqueeze(0).to(device))
                        ])
                offset += len(batch["z_index"])
            native_dice = dice_from_counts(native_tp, native_pred - native_tp, gt_total - native_tp)
            cached_dice = dice_from_counts(*cached_counts) if cached_slices == offset else None
            fresh_dice = dice_from_counts(*counts)
            fresh_scores.append(fresh_dice)
            native_scores.append(native_dice)
            if cached_dice is not None:
                cached_scores.append(cached_dice)
            print(
                f"patient={rec.patient_uid} slices={offset} fresh_model_dice={fresh_dice:.6f} "
                f"native_dice={native_dice:.6f} native_gt_voxels={gt_total} native_pred_voxels={native_pred} "
                f"cache_model_dice={cached_dice} cached_slices={cached_slices} missing_cache={missing_cache} "
                f"invalid_meta={invalid_meta} phase_meta_mismatch={phase_mismatch} "
                f"phase_array_mismatch={phase_array_mismatch} mask_mismatch={mask_mismatch} "
                f"max_input_diff={max_image_diff:.6g}",
                flush=True,
            )
    print(
        f"mean_fresh_model_dice={np.mean(fresh_scores):.6f} mean_native_dice={np.mean(native_scores):.6f} "
        f"mean_cache_model_dice={np.mean(cached_scores) if len(cached_scores) == len(selected) else None}",
        flush=True,
    )


if __name__ == "__main__":
    main()
