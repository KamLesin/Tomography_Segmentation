from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, dice_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = soft_dice_loss(logits, targets)
    return (1.0 - dice_weight) * bce + dice_weight * dice
