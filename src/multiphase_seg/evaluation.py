from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

from .data import MultiphaseSliceDataset, PatientRecord, _first_existing_path
from .metrics import dice_from_counts


def hd95_mm(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]) -> float:
    """95th percentile symmetric surface distance in mm; NaN if either mask is empty."""
    if not pred.any() or not gt.any():
        return float("nan")

    def surface(m: np.ndarray) -> np.ndarray:
        return m & ~ndi.binary_erosion(m)

    sp, sg = surface(pred), surface(gt)
    d_to_gt = ndi.distance_transform_edt(~sg, sampling=spacing)[sp]
    d_to_pred = ndi.distance_transform_edt(~sp, sampling=spacing)[sg]
    return float(np.percentile(np.concatenate([d_to_gt, d_to_pred]), 95))


def lesion_detection(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Lesion-wise recall/precision using 26-connected 3D components and any-overlap matching."""
    structure = ndi.generate_binary_structure(3, 3)
    gt_lab, n_gt = ndi.label(gt, structure)
    pred_lab, n_pred = ndi.label(pred, structure)

    hit_gt = len(np.unique(gt_lab[pred])) - (1 if (gt_lab[pred] == 0).any() else 0)
    hit_pred = len(np.unique(pred_lab[gt])) - (1 if (pred_lab[gt] == 0).any() else 0)
    return {
        "gt_lesions": float(n_gt),
        "pred_lesions": float(n_pred),
        "lesion_recall": hit_gt / n_gt if n_gt else float("nan"),
        "lesion_precision": hit_pred / n_pred if n_pred else float("nan"),
    }


@torch.no_grad()
def evaluate_patient_native(
    model: torch.nn.Module,
    rec: PatientRecord,
    data_cfg: Dict[str, Any],
    device: torch.device,
    eval_phase_override: Optional[str] = None,
    batch_size: int = 16,
    threshold: float = 0.5,
    amp_enabled: bool = False,
) -> Optional[Dict[str, Any]]:
    """Slice-wise 2.5D inference over the whole patient, metrics computed on the native-resolution volume.

    Slices the dataset does not generate (first/last `context_slices`) count as empty predictions.
    """
    from .train import _apply_eval_phase_override

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

    ct_path = _first_existing_path([rec.image_paths.get("PV"), rec.image_paths.get("A"), rec.image_paths.get("D")])
    gt_path = _first_existing_path([rec.mask_paths.get("PV"), rec.mask_paths.get("A"), rec.mask_paths.get("D")])
    if ct_path is None or gt_path is None:
        return None

    gt = ds._load_volume(gt_path, is_mask=True) > 0
    pred = np.zeros(gt.shape, dtype=bool)
    spacing = tuple(float(s) for s in nib.load(str(ct_path)).header.get_zooms()[:3])

    model.eval()
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
        x = batch["phases"].to(device)
        phase_present = _apply_eval_phase_override(batch["phase_present"].to(device) > 0.5, eval_phase_override)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x, phase_present=phase_present)
        probs = F.interpolate(torch.sigmoid(logits.float()), size=gt.shape[:2], mode="bilinear", align_corners=False)
        masks = (probs >= threshold).squeeze(1).cpu().numpy()
        for m, z in zip(masks, batch["z_index"].tolist()):
            pred[:, :, int(z)] = m

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    return {
        "patient_uid": rec.patient_uid,
        "patient_id": rec.patient_id,
        "source": rec.source,
        "dice_native": dice_from_counts(tp, fp, fn),
        "hd95_mm": hd95_mm(pred, gt, spacing),
        "gt_voxels": int(gt.sum()),
        "pred_voxels": int(pred.sum()),
        **lesion_detection(pred, gt),
    }


def evaluate_patients_native(
    model: torch.nn.Module,
    records: Sequence[PatientRecord],
    data_cfg: Dict[str, Any],
    device: torch.device,
    **kwargs: Any,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        row = evaluate_patient_native(model, rec, data_cfg, device, **kwargs)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)
