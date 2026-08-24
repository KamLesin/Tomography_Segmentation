"""Re-run inference for a completed hypothesis run to get per-patient metrics.

The training pipeline (run_hypothesis.py / train.py) only stores a fold-level
aggregate Dice. This script reloads each fold's saved `best.pt` checkpoint and
evaluates it on that fold's held-out validation patients to produce:
  - one row per (arm, fold, patient) with a Dice score, and
  - (optionally) predicted tumor masks as NIfTI files, in the original CT
    geometry, so they can be opened next to the ground truth in ITK-SNAP.

This is required for Hypothesis 3 (lesion-size stratification) and for any
qualitative mask visualization, since the training loop never saved
per-patient results.

Example:
    python scripts/analysis/run_inference.py \\
        --run-dir ../results_from_apl19/h1_effb0_32fold_8gpu \\
        --arms baseline_single_phase multiphase_train_single_infer \\
        --save-masks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import build_model_from_cfg, load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, required=True, help="Hypothesis run dir, e.g. ../results_from_apl19/h1_effb0_32fold_8gpu")
    p.add_argument("--arms", type=str, nargs="+", default=None, help="Subset of arms to evaluate (default: all under run-dir/arms).")
    p.add_argument("--folds-csv", type=str, default=None, help="Defaults to the folds_csv recorded in run-dir/summary.json.")
    p.add_argument("--folds", type=int, nargs="+", default=None, help="Subset of fold indices to evaluate.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--cect-root-override", type=str, default=None)
    p.add_argument("--full-root-override", type=str, default=None)
    p.add_argument("--save-masks", action="store_true", help="Also write predicted masks as NIfTI in original CT geometry.")
    p.add_argument("--masks-output-dir", type=str, default=None)
    p.add_argument("--max-patients-per-fold", type=int, default=None, help="Cap patients per fold (smoke test).")
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def _first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def run_patient_inference(
    model: torch.nn.Module,
    rec: Any,
    cfg: Dict[str, Any],
    device: torch.device,
    threshold: float,
    batch_size: int,
    save_mask: bool,
    masks_dir: Optional[Path],
    arm_name: str,
    fold: int,
) -> Optional[Dict[str, Any]]:
    from multiphase_seg.data import MultiphaseSliceDataset
    from multiphase_seg.train import _apply_eval_phase_override

    data_cfg = cfg.get("data", {})
    ds = MultiphaseSliceDataset(
        [rec],
        patient_uids=[rec.patient_uid],
        image_size=tuple(data_cfg.get("image_size", [320, 320])),
        context_slices=int(data_cfg.get("context_slices", 2)),
        hu_window=tuple(data_cfg.get("hu_window", [-200.0, 300.0])),
        max_slices_per_patient=None,
        cache_enabled=False,
        force_phase_input=data_cfg.get("force_phase_input"),
    )
    if len(ds) == 0:
        return None

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    eval_phase_override = cfg.get("train", {}).get("eval_phase_override")

    ct_path = _first_existing([rec.image_paths.get("PV"), rec.image_paths.get("A"), rec.image_paths.get("D")])
    gt_mask_path = _first_existing([rec.mask_paths.get("PV"), rec.mask_paths.get("A"), rec.mask_paths.get("D")])

    tp = fp = fn = 0.0
    native_pred_vol = None
    native_gt = None
    native_affine = None
    native_shape = None

    if save_mask and ct_path is not None:
        ref_img = nib.load(str(ct_path))
        native_affine = ref_img.affine
        native_shape = np.squeeze(np.asarray(ref_img.dataobj)).shape
        native_pred_vol = np.zeros(native_shape, dtype=np.uint8)
        if gt_mask_path is not None:
            native_gt = (np.squeeze(np.asarray(nib.load(str(gt_mask_path)).dataobj)) > 0).astype(np.uint8)

    with torch.no_grad():
        for batch in loader:
            x = batch["phases"].to(device)
            y = batch["mask"].to(device)
            phase_present = batch["phase_present"].to(device) > 0.5
            phase_present = _apply_eval_phase_override(phase_present, eval_phase_override)

            logits = model(x, phase_present=phase_present)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            tp += float((preds * y).sum().item())
            fp += float((preds * (1 - y)).sum().item())
            fn += float(((1 - preds) * y).sum().item())

            if native_pred_vol is not None:
                z_indices = batch["z_index"].tolist()
                preds_np = preds.squeeze(1).cpu().numpy()  # [B, H, W] at resized resolution
                for i, z in enumerate(z_indices):
                    resized = torch.from_numpy(preds_np[i])[None, None]
                    native_slice = F.interpolate(resized, size=native_shape[:2], mode="nearest")
                    native_pred_vol[:, :, int(z)] = native_slice.squeeze().numpy().astype(np.uint8)

    denom = 2 * tp + fp + fn
    dice_resized = 1.0 if denom == 0 else (2 * tp) / denom

    result: Dict[str, Any] = {
        "dice_resized": dice_resized,
        "gt_voxels_resized": tp + fn,
        "pred_voxels_resized": tp + fp,
        "ct_path": str(ct_path) if ct_path else "",
        "gt_mask_path": str(gt_mask_path) if gt_mask_path else "",
    }

    if native_pred_vol is not None:
        if native_gt is not None:
            tp_n = int(np.logical_and(native_pred_vol, native_gt).sum())
            fp_n = int(np.logical_and(native_pred_vol, np.logical_not(native_gt)).sum())
            fn_n = int(np.logical_and(np.logical_not(native_pred_vol), native_gt).sum())
            denom_n = 2 * tp_n + fp_n + fn_n
            result["dice_native"] = 1.0 if denom_n == 0 else (2 * tp_n) / denom_n
            result["gt_voxels_native"] = int(native_gt.sum())

        pred_path = masks_dir / arm_name / f"fold_{int(fold):02d}" / f"{rec.patient_id}_pred_mask.nii.gz"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(native_pred_vol.astype(np.float32), native_affine), str(pred_path))
        result["pred_mask_path"] = str(pred_path)

    return result


def main() -> None:
    args = parse_args()
    from multiphase_seg.data import build_patient_records

    run_dir = resolve_path(args.run_dir)
    run_name = run_dir.name

    folds_csv_path = resolve_path(args.folds_csv) if args.folds_csv else None
    if folds_csv_path is None:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise SystemExit("Could not infer folds CSV: pass --folds-csv explicitly (no summary.json found).")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        folds_csv_path = resolve_path(summary["folds_csv"])

    folds_df = pd.read_csv(folds_csv_path)
    fold_values = sorted(int(v) for v in folds_df["fold"].unique().tolist())
    if args.folds:
        fold_values = [f for f in fold_values if f in args.folds]

    arms = args.arms or sorted(p.name for p in (run_dir / "arms").iterdir() if p.is_dir())

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks_dir = resolve_path(args.masks_output_dir) if args.masks_output_dir else resolve_path(f"results/{run_name}/predicted_masks")
    output_path = resolve_path(args.output) if args.output else resolve_path(f"results/{run_name}/per_patient_metrics.csv")

    all_rows: List[Dict[str, Any]] = []
    for arm in arms:
        config_path = run_dir / "configs" / f"{arm}.yaml"
        if not config_path.exists():
            print(f"[skip] no saved config for arm={arm}")
            continue
        cfg = load_yaml(config_path)

        data_cfg = cfg.setdefault("data", {})
        if args.cect_root_override:
            data_cfg["cect_root"] = args.cect_root_override
        if args.full_root_override:
            data_cfg["full_root"] = args.full_root_override

        cect_root = resolve_path(data_cfg["cect_root"]) if data_cfg.get("cect_root") else None
        full_root = resolve_path(data_cfg["full_root"]) if data_cfg.get("full_root") else None
        records = build_patient_records(
            mode=str(data_cfg.get("mode", "mixed")),
            cect_root=cect_root,
            full_root=full_root,
            missing_phase_strategy=str(data_cfg.get("missing_phase_strategy", "drop")),
        )
        rec_by_uid = {r.patient_uid: r for r in records}
        print(f"[arm={arm}] discovered {len(records)} patient records")

        for fold in fold_values:
            ckpt_path = run_dir / "arms" / arm / f"fold_{fold:02d}" / "best.pt"
            if not ckpt_path.exists():
                print(f"[skip] missing checkpoint arm={arm} fold={fold}")
                continue

            model = build_model_from_cfg(cfg).to(device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model"])
            model.eval()

            val_uids = folds_df.loc[folds_df["fold"] == fold, "patient_uid"].tolist()
            if args.max_patients_per_fold:
                val_uids = val_uids[: args.max_patients_per_fold]

            n_done = 0
            for uid in val_uids:
                rec = rec_by_uid.get(uid)
                if rec is None:
                    continue
                result = run_patient_inference(
                    model=model,
                    rec=rec,
                    cfg=cfg,
                    device=device,
                    threshold=args.threshold,
                    batch_size=args.batch_size,
                    save_mask=args.save_masks,
                    masks_dir=masks_dir,
                    arm_name=arm,
                    fold=fold,
                )
                if result is None:
                    continue
                row = {"arm": arm, "fold": fold, "patient_uid": uid, "patient_id": rec.patient_id, "source": rec.source}
                row.update(result)
                all_rows.append(row)
                n_done += 1

            del model
            print(f"[done] arm={arm} fold={fold} patients_evaluated={n_done}/{len(val_uids)}")

    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved per-patient metrics: {output_path} ({len(df)} rows)")
    if args.save_masks:
        print(f"Saved predicted masks under: {masks_dir}")


if __name__ == "__main__":
    main()
