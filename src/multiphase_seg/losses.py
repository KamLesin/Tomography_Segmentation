from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    # Pooled over the whole batch so empty slices do not each count as a full Dice term.
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    return 1.0 - (2.0 * intersection + smooth) / (union + smooth)


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, dice_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = soft_dice_loss(logits, targets)
    return (1.0 - dice_weight) * bce + dice_weight * dice
