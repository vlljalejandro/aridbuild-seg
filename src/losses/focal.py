"""
Binary focal loss with optional ignore_index.

Defaults to None, since the pseudo-labels here are binary with no ignore
region. Keep this in step with the mask format: an earlier variant wrote 255
for low-confidence pixels, and mismatching the two either excludes nothing or
lets 255 be read as a positive target, which drives the Dice term negative.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    L_focal = -alpha * (1 - p)^gamma * log(p)      for positives
              -(1-alpha) * p^gamma * log(1-p)      for negatives

    alpha balances the positive class, gamma is the focusing exponent.
    """

    def __init__(
        self,
        alpha:        float = 0.25,
        gamma:        float = 2.0,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(
        self,
        logits:  torch.Tensor,   # (B, 1, H, W) — pre-sigmoid
        targets: torch.Tensor,   # (B, 1, H, W) — float {0.0, 1.0, [255.0]}
    ) -> torch.Tensor:

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
        else:
            valid = torch.ones_like(targets, dtype=torch.bool)

        if not valid.any():
            return logits.sum() * 0.0   # zero loss, keeps graph alive

        logits_  = logits[valid]
        targets_ = targets[valid].float()

        bce    = F.binary_cross_entropy_with_logits(logits_, targets_, reduction="none")
        probs  = torch.sigmoid(logits_)
        pt     = torch.where(targets_ == 1, probs, 1 - probs)
        alpha  = torch.where(targets_ == 1,
                             torch.full_like(pt, self.alpha),
                             torch.full_like(pt, 1 - self.alpha))
        focal  = alpha * (1 - pt) ** self.gamma * bce

        return focal.mean()
