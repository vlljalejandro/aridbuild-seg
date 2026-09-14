"""
Watershed-based instance separation for building segmentation.

A topographic surface is flooded from its local maxima, constrained to the
foreground mask, giving one basin per building instance.

The surface is the model's **probability map**, not the auxiliary distance
head. The `dist_map` parameter name is historical; every caller in this
repository passes sigmoid(seg_logits). The distance head plays no part in
instance separation, and no reported number depends on it.

Related: Bai & Urtasun, "Deep Watershed Transform for Instance Segmentation",
CVPR 2017 — which regresses a distance field with a dedicated branch. This
stage introduces no learned component beyond the segmentation head.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def apply_watershed(
    seg_mask:     np.ndarray,      # (H, W) uint8 or bool  — binary foreground
    dist_map:     np.ndarray,      # (H, W) float32 in [0,1] — flooding surface
    min_distance: int   = 8,       # minimum pixel separation between seeds
    min_area_px:  int   = 30,      # drop instances smaller than this (pixels)
    seed_thresh:  float = 0.2,     # minimum dist_map value to be a valid seed
) -> np.ndarray:
    """
    Separate touching buildings into instances. Returns (H, W) uint16, 0 for
    background and 1..N for instances.

    min_distance controls how aggressively adjacent buildings are split
    (smaller means more splits; 10 px is ~2.7 m at 0.27 m/px). seed_thresh
    keeps spurious seeds out of thin or noisy regions.
    """
    mask = seg_mask.astype(bool)

    if not mask.any():
        return np.zeros(seg_mask.shape, dtype=np.uint16)

    # ── seed detection ────────────────────────────────────────────────────────
    dist_fg = dist_map * mask.astype(np.float32)   # seeds never land in background

    peaks = peak_local_max(
        dist_fg,
        min_distance  = min_distance,
        threshold_abs = seed_thresh,
        labels        = mask,         # restrict peaks to foreground
    )

    if len(peaks) == 0:
        # no distinct peaks: each connected component is one instance
        labeled, _ = ndi.label(mask)
        return labeled.astype(np.uint16)

    # ── marker image ──────────────────────────────────────────────────────────
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks) + 1)

    # dilate so seeds are not single pixels
    markers = ndi.label(ndi.binary_dilation(markers > 0, iterations=2))[0]

    # ── watershed ─────────────────────────────────────────────────────────────
    # negated so flooding runs from the peaks outward
    labels = watershed(-dist_fg, markers=markers, mask=mask)

    # ── remove small instances ────────────────────────────────────────────────
    ids, counts = np.unique(labels, return_counts=True)
    small = ids[(ids != 0) & (counts < min_area_px)]
    if small.size:
        labels[np.isin(labels, small)] = 0

    # ── compact IDs to 1..N ───────────────────────────────────────────────────
    # Do NOT use ndi.label(labels > 0) here. Watershed partitions the mask with
    # no gap between basins, so connected components on their union merges
    # every split back together and silently undoes the separation.
    ids   = np.unique(labels)
    ids   = ids[ids != 0]
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    labels = remap[labels]

    return labels.astype(np.uint16)


# ─────────────────────────────────────────────────────────────────────────────
# Batch wrapper
# ─────────────────────────────────────────────────────────────────────────────

def batch_watershed(
    seg_masks: np.ndarray,   # (B, H, W) uint8
    dist_maps: np.ndarray,   # (B, H, W) float32
    **kwargs,
) -> np.ndarray:
    """Apply apply_watershed to a batch. Returns (B, H, W) uint16."""
    B = seg_masks.shape[0]
    out = np.zeros_like(seg_masks, dtype=np.uint16)
    for i in range(B):
        out[i] = apply_watershed(seg_masks[i], dist_maps[i], **kwargs)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scipy.ndimage import distance_transform_edt

    H, W = 512, 512
    # two touching rectangles
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[100:250, 100:280] = 1
    mask[150:300, 220:380] = 1

    edt = distance_transform_edt(mask).astype(np.float32)
    dist = edt / (edt.max() + 1e-6)

    labels = apply_watershed(mask, dist, min_distance=8, min_area_px=30)
    n_instances = labels.max()

    print(f"Foreground pixels : {mask.sum():,}")
    print(f"Instances found   : {n_instances}")
    print(f"Label dtype       : {labels.dtype}  shape={labels.shape}")
    assert n_instances >= 2, f"Expected ≥2 instances for two buildings, got {n_instances}"
    print("✓ Watershed smoke test passed")
