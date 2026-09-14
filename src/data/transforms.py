"""
Albumentations pipelines applying the same spatial transform to image and mask.

Keep the albumentations pin in environment.yml: A.GaussNoise took `std_range`
from 1.4.21 and A.Affine's argument names changed in the same series, so an
unpinned install will fail at import or silently augment differently.

    out = get_train_transforms()(image=np_img_hwc, mask=np_mask_hw)
    img, mask = out["image"], out["mask"]
"""

from __future__ import annotations

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────────────────────────────────────
# Channel statistics of the target domain (D3)
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SIZE = 512
_MEAN       = (0.2186, 0.2315, 0.4944)
_STD        = (0.2308, 0.2263, 0.2268)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms (image + mask)
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(
    target_size: int   = TARGET_SIZE,
    mean:        tuple = _MEAN,
    std:         tuple = _STD,
) -> A.Compose:
    """
    Training augmentation: reflect-pad to 512 (a no-op for D2/D3), flips,
    90-degree rotations, a mild affine, photometric jitter, occasional noise
    or blur, then normalise to float32 tensors.
    """
    return A.Compose([
        # ── geometry ─────────────────────────────────────────────────────────
        A.PadIfNeeded(
            min_height  = target_size,
            min_width   = target_size,
            border_mode = cv2.BORDER_REFLECT_101,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent = {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale             = (0.90, 1.10),
            rotate            = (-15, 15),
            border_mode       = cv2.BORDER_REFLECT_101,
            p                 = 0.4,
        ),
        # ── photometric (image only, mask unchanged) ──────────────────────────
        A.OneOf([
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=1.0,
            ),
        ], p=0.6),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.11), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        # ── normalise + to tensor ─────────────────────────────────────────────
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
        # image : (3, H, W) float32   mask : (H, W) uint8
    ])


def get_val_transforms(
    target_size: int   = TARGET_SIZE,
    mean:        tuple = _MEAN,
    std:         tuple = _STD,
) -> A.Compose:
    """Validation and test: padding and normalisation only."""
    return A.Compose([
        A.PadIfNeeded(
            min_height  = target_size,
            min_width   = target_size,
            border_mode = cv2.BORDER_REFLECT_101,
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])