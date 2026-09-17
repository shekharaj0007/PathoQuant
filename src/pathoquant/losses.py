"""Dice + BCE for class-imbalanced nuclei masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    dice = 1.0 - (2.0 * intersection + smooth) / (probs.sum(dim=dims) + targets.sum(dim=dims) + smooth)
    return 0.5 * bce + 0.5 * dice.mean()
