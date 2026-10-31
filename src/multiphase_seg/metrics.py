from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def confusion_counts(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample (tp, fp, fn) pixel counts, each of shape [B]."""
    preds = (torch.sigmoid(logits) >= threshold).float().flatten(1)
    t = targets.flatten(1)
    tp = (preds * t).sum(dim=1)
    fp = (preds * (1.0 - t)).sum(dim=1)
    fn = ((1.0 - preds) * t).sum(dim=1)
    return tp, fp, fn


def dice_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = 2.0 * tp + fp + fn
    return 1.0 if denom == 0 else float(2.0 * tp / denom)


class PatientDiceAccumulator:
    """Pools pixel counts over all slices of a patient and returns one Dice per patient."""

    def __init__(self) -> None:
        self._counts: Dict[str, np.ndarray] = {}

    def update(self, patient_uids, logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> None:
        tp, fp, fn = confusion_counts(logits, targets, threshold)
        stacked = torch.stack([tp, fp, fn], dim=1).double().cpu().numpy()
        for uid, row in zip(patient_uids, stacked):
            acc = self._counts.setdefault(str(uid), np.zeros(3, dtype=np.float64))
            acc += row

    def dice_per_patient(self) -> Dict[str, float]:
        return {uid: dice_from_counts(*c) for uid, c in self._counts.items()}
