"""
Binary Dice Loss with optional ignore_index support.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    L_dice = 1 - (2 * |P ∩ T| + smooth) / (|P| + |T| + smooth)
    """

    def __init__(
        self,
        smooth:       float    = 1.0,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(
        self,
        logits:  torch.Tensor,   # (B, 1, H, W)
        targets: torch.Tensor,   # (B, 1, H, W) float {0.0, 1.0, [255.0]}
    ) -> torch.Tensor:

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
        else:
            valid = torch.ones_like(targets, dtype=torch.bool)

        if not valid.any():
            return logits.sum() * 0.0

        probs   = torch.sigmoid(logits)
        probs_  = probs[valid]
        targets_= targets[valid].float()

        intersection = (probs_ * targets_).sum()
        denominator  = probs_.sum() + targets_.sum()

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice
