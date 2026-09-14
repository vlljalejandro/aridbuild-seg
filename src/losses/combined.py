"""
Combined loss for the dual-head models.

    total = focal_weight * Focal(seg_logits, seg_mask)
          + dice_weight  * Dice(seg_logits, seg_mask)
          + dist_weight  * MSE(dist_logits, dist_target)

The MSE term is auxiliary: unused at inference, no part in instance separation
(the watershed floods the probability map, not this head), and no reported
metric depends on it. dist_weight stays at 0.5 because every released
checkpoint was trained with it. Set it to 0 for new work.

An earlier revision approximated the EDT by iterated erosion and returned
all-zero targets, leaving the head untrained. The assertion in the smoke test
below catches that.

    criterion = CombinedLoss(focal_weight=1.0, dice_weight=1.0, dist_weight=0.5)
    dist_target = compute_distance_target(masks)
    loss, loss_dict = criterion(seg_logits, dist_logits, masks, dist_target)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal import BinaryFocalLoss
from .dice  import BinaryDiceLoss


# ─────────────────────────────────────────────────────────────────────────────
# Distance transform target
# ─────────────────────────────────────────────────────────────────────────────

def compute_distance_target(mask: torch.Tensor) -> torch.Tensor:
    """
    Normalised Euclidean distance transform of a binary (B,1,H,W) mask, using
    scipy's exact EDT on CPU.

    Each sample is divided by its own maximum, so the target is always in
    [0, 1] and dist_weight does not need retuning for datasets with different
    building sizes.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    device  = mask.device
    mask_np = mask.detach().cpu().numpy()   # (B, 1, H, W)
    B       = mask_np.shape[0]
    out     = np.zeros_like(mask_np, dtype=np.float32)

    for b in range(B):
        edt = distance_transform_edt(mask_np[b, 0]).astype(np.float32)
        mx  = edt.max()
        if mx > 0:
            edt /= mx
        out[b, 0] = edt

    return torch.from_numpy(out).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """L_total = w_focal * L_focal + w_dice * L_dice + w_dist * L_mse"""

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight:  float = 1.0,
        dist_weight:  float = 0.5,
        focal_alpha:  float = 0.25,
        focal_gamma:  float = 2.0,
        dice_smooth:  float = 1.0,
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight
        self.dist_weight  = dist_weight

        self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                                     ignore_index=ignore_index)
        self.dice  = BinaryDiceLoss(smooth=dice_smooth,
                                    ignore_index=ignore_index)

    def forward(
        self,
        seg_logits:   torch.Tensor,
        dist_logits:  torch.Tensor,
        seg_targets:  torch.Tensor,
        dist_targets: torch.Tensor | None = None,
    ):
        """
        All tensors are (B, 1, H, W); logits are pre-sigmoid and seg_targets is
        binary {0.0, 1.0}. dist_targets is computed here if not supplied.

        Returns (total, {'focal', 'dice', 'dist', 'total'}).
        """
        l_focal = self.focal(seg_logits, seg_targets)
        l_dice  = self.dice(seg_logits,  seg_targets)

        if dist_targets is None:
            dist_targets = compute_distance_target(seg_targets)
        dist_pred = torch.sigmoid(dist_logits)
        l_dist    = F.mse_loss(dist_pred, dist_targets.to(dist_pred.dtype))

        total = (
            self.focal_weight * l_focal
            + self.dice_weight  * l_dice
            + self.dist_weight  * l_dist
        )

        return total, {
            "focal": l_focal.item(),
            "dice":  l_dice.item(),
            "dist":  l_dist.item(),
            "total": total.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, H, W = 4, 512, 512

    seg_logits  = torch.randn(B, 1, H, W)
    dist_logits = torch.randn(B, 1, H, W)
    masks       = (torch.rand(B, 1, H, W) > 0.80).float()

    dist_targets = compute_distance_target(masks)
    print(f"dist_target range: [{dist_targets.min():.3f}, {dist_targets.max():.3f}]")
    assert dist_targets.max() > 0, "EDT returned all zeros!"

    criterion = CombinedLoss(focal_weight=1.0, dice_weight=1.0, dist_weight=0.5)
    total, loss_dict = criterion(seg_logits, dist_logits, masks, dist_targets)

    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k:<6s}: {v:.4f}")

    assert total.item() > 0, "Total loss should be positive!"
    total.backward()
    print("\n✓ CombinedLoss smoke test passed")
