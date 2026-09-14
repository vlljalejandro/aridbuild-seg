"""
src.losses — loss functions for the building-segmentation pipeline.

    BinaryFocalLoss          pixel-level class-balanced focal loss
    BinaryDiceLoss           region-level soft Dice loss
    CombinedLoss             Focal + Dice + auxiliary distance MSE
    compute_distance_target  normalised EDT from binary masks -> [0,1] target

The distance term is auxiliary: unused at inference, absent from every
reported metric, retained only so released checkpoints reproduce exactly.
"""

from .focal import BinaryFocalLoss
from .dice import BinaryDiceLoss
from .combined import CombinedLoss, compute_distance_target

__all__ = [
    "BinaryFocalLoss",
    "BinaryDiceLoss",
    "CombinedLoss",
    "compute_distance_target",
]
