"""
Dataset classes for the building-segmentation pipeline. All labeled datasets
return (image, mask) as (3,512,512) and (1,512,512) float32 tensors.

    SegmentationDataset  COCO-format tiles: D1 (CrowdAI) and D3 (target)
    InriaDataset         512x512 crops cut on the fly from Inria TIF pairs (D2)
    LMDBDataset          unlabeled LMDB tiles, image only; not used in training
    PseudoLabelDataset   LMDB tiles paired with pseudo-label masks (D4)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import lmdb
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# COCO helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_coco_index(
    json_path: Path,
) -> Tuple[Dict[int, dict], Dict[int, List[dict]]]:
    """
    Parse a COCO-format JSON and return:
        image_index : {image_id → image_info dict}
        ann_index   : {image_id → [annotation, ...]}
    Only the 'images' and 'annotations' keys are required.
    """
    with open(json_path) as f:
        coco = json.load(f)

    image_index: Dict[int, dict]       = {img["id"]: img for img in coco["images"]}
    ann_index:   Dict[int, List[dict]] = {img_id: [] for img_id in image_index}
    for ann in coco.get("annotations", []):
        ann_index[ann["image_id"]].append(ann)

    return image_index, ann_index


def _coco_anns_to_mask(
    anns:   List[dict],
    height: int,
    width:  int,
) -> np.ndarray:
    """
    Rasterise COCO polygon / RLE annotations into a binary uint8 mask
    (1 = building, 0 = background) of shape (H, W).
    Requires pycocotools.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if not anns:
        return mask

    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError(
            "pycocotools is required for mask decoding.  "
            "Install with:  pip install pycocotools"
        )

    for ann in anns:
        seg = ann.get("segmentation")
        if not seg:
            continue
        if isinstance(seg, list):                    # polygon format
            rle = coco_mask.frPyObjects(seg, height, width)
            m   = coco_mask.decode(rle)
            if m.ndim == 3:
                m = m.any(axis=2)
        else:                                        # RLE format
            m = coco_mask.decode(seg)
        mask = np.maximum(mask, m.astype(np.uint8))

    return mask


def _collect_samples(sources: List[dict]) -> List[dict]:
    """
    Flatten a list of {"image_dir": str, "ann_file": str} source dicts
    into a flat list of sample records.

    Each record contains:
        image_path : Path
        anns       : list[dict]   COCO annotation dicts
        height     : int
        width      : int
    """
    samples: List[dict] = []

    for source in sources:
        img_dir  = Path(source["image_dir"])
        ann_file = Path(source["ann_file"])

        image_index, ann_index = _load_coco_index(ann_file)

        for img_id, img_info in image_index.items():
            file_name  = img_info["file_name"]
            image_path = img_dir / file_name

            if not image_path.exists():
                image_path = img_dir / Path(file_name).name

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image '{file_name}' not found under '{img_dir}'. "
                    "Verify that image_dir matches the prefix in file_name."
                )

            samples.append({
                "image_path": image_path,
                "anns":       ann_index[img_id],
                "height":     img_info.get("height", 0),
                "width":      img_info.get("width",  0),
            })

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# SegmentationDataset — D1 (CrowdAI) and D3 (Saudi Arabia)
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    """
    Labeled tiles from COCO-format annotations: D1 (300x300) and D3 (512x512).
    Padding to 512x512 is handled by the transform, so it is required.

    Parameters
    ----------
    samples   : sample dicts from ``_collect_samples``.
    transform : albumentations Compose accepting image + mask.
    """

    def __init__(
        self,
        samples:   List[dict],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples   = samples
        self.transform = transform

        if transform is None:
            raise ValueError(
                "A transform is required.  "
                "Pass get_val_transforms() as the minimum for inference."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        img = np.array(Image.open(sample["image_path"]).convert("RGB"))

        h, w = sample["height"], sample["width"]
        if h == 0 or w == 0:
            h, w = img.shape[:2]

        mask = _coco_anns_to_mask(sample["anns"], h, w)

        out         = self.transform(image=img, mask=mask)
        img_tensor  = out["image"]
        mask_tensor = out["mask"].unsqueeze(0).float()

        return img_tensor, mask_tensor


# ─────────────────────────────────────────────────────────────────────────────
# InriaDataset — D2 (Inria Aerial Image Dataset)
# ─────────────────────────────────────────────────────────────────────────────

_INRIA_SIZE   = 5000
_CROP_SIZE    = 512
_INRIA_STRIDE = 498
_N_CROPS      = 10


def _inria_positions() -> List[int]:
    pos = [i * _INRIA_STRIDE for i in range(_N_CROPS - 1)]
    pos.append(_INRIA_SIZE - _CROP_SIZE)   # 4488 — anchored at end
    return pos


_POSITIONS = _inria_positions()


class InriaDataset(Dataset):
    """
    Inria Aerial Image Dataset (D2). Cuts 512x512 crops on the fly from
    5000x5000 TIF pairs on a fixed 10x10 grid, stride 498 px, giving
    180 x 100 = 18,000 samples. Ground-truth uint8 {0,255} maps to {0.0,1.0}.

    Parameters
    ----------
    samples   : dicts from ``InriaDataset.collect_samples()``.
    transform : albumentations Compose accepting image + mask.
    """

    def __init__(
        self,
        samples:   List[dict],
        transform: Optional[Callable] = None,
    ) -> None:
        if transform is None:
            raise ValueError(
                "A transform is required. "
                "Pass get_val_transforms() as the minimum."
            )
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample   = self.samples[idx]
        row, col = sample["row"], sample["col"]
        y0       = _POSITIONS[row]
        x0       = _POSITIONS[col]

        img  = np.array(Image.open(sample["image_path"]).convert("RGB"))
        mask = np.array(Image.open(sample["gt_path"]).convert("L"))

        img  = img [y0:y0 + _CROP_SIZE, x0:x0 + _CROP_SIZE]
        mask = mask[y0:y0 + _CROP_SIZE, x0:x0 + _CROP_SIZE]
        mask = (mask > 0).astype(np.uint8)

        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].unsqueeze(0).float()

    @staticmethod
    def collect_samples(train_dir: str | Path) -> List[dict]:
        """
        Build a flat list of crop records from the Inria train directory.

        Parameters
        ----------
        train_dir : path to AerialImageDataset/train/
                    Must contain images/ and gt/ subdirectories.
        """
        train_dir  = Path(train_dir)
        images_dir = train_dir / "images"
        gt_dir     = train_dir / "gt"

        tif_paths = sorted(images_dir.glob("*.tif")) + \
                    sorted(images_dir.glob("*.TIF"))

        samples = []
        missing = []

        for img_path in tif_paths:
            gt_path = gt_dir / img_path.name
            if not gt_path.exists():
                gt_path = gt_dir / (img_path.stem + ".TIF")
            if not gt_path.exists():
                missing.append(img_path.name)
                continue

            for row in range(_N_CROPS):
                for col in range(_N_CROPS):
                    samples.append({
                        "image_path": img_path,
                        "gt_path":    gt_path,
                        "row":        row,
                        "col":        col,
                    })

        if missing:
            print(f"[InriaDataset] Warning: {len(missing)} TIFs have no GT — skipped.")

        print(
            f"[InriaDataset] {len(tif_paths) - len(missing)} TIF pairs  "
            f"→ {len(samples):,} crops  ({_N_CROPS}×{_N_CROPS} per TIF)"
        )
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# LMDBDataset — unlabeled tile reader (not used in training)
# ─────────────────────────────────────────────────────────────────────────────

class LMDBDataset(Dataset):
    """
    Unlabeled JPEG tiles from an LMDB store, iterated in key order. The
    environment is opened lazily inside each worker to avoid pickling issues.

    Not used by the training pipeline, which uses PseudoLabelDataset. Kept for
    inspecting an LMDB directly.
    """

    def __init__(
        self,
        lmdb_path: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self.transform = transform
        self._env: Optional[lmdb.Environment] = None

        env = self._make_env()
        with env.begin(write=False) as txn:
            self._keys: List[bytes] = [k for k, _ in txn.cursor()]
        env.close()

    def _make_env(self) -> lmdb.Environment:
        return lmdb.open(
            self.lmdb_path,
            readonly    = True,
            lock        = False,
            readahead   = False,
            meminit     = False,
        )

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = self._make_env()
        return self._env

    def __del__(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __len__(self) -> int:
        return len(self._keys)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        env = self._get_env()
        with env.begin(write=False) as txn:
            jpeg_bytes = txn.get(self._keys[idx])

        img = np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img


# ─────────────────────────────────────────────────────────────────────────────
# PseudoLabelDataset — self-training on D4
# ─────────────────────────────────────────────────────────────────────────────

class PseudoLabelDataset(Dataset):
    """
    LMDB tiles paired with pseudo-label masks from scripts/pseudo_label.py.
    Masks are uint8 PNG, 0 = background and 1 = building, with no ignore
    region, so the student trains with the same loss as the supervised stages.

    Parameters
    ----------
    lmdb_path    : LMDB of unlabeled target tiles
    mask_dir     : directory of {x18}_{y18}.png masks
    manifest     : manifest.csv (x18, y18, pos_frac)
    transform    : albumentations Compose accepting image + mask
    min_pos_frac : skip tiles below this positive fraction
    """

    def __init__(
        self,
        lmdb_path:    str | Path,
        mask_dir:     str | Path,
        manifest:     str | Path,
        transform:    Callable,
        min_pos_frac: float = 0.02,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform
        self._env      = None

        df = pd.read_csv(manifest)
        df = df[df["pos_frac"] >= min_pos_frac].reset_index(drop=True)
        self.samples = list(zip(df["x18"].tolist(), df["y18"].tolist()))

        print(
            f"[PseudoLabelDataset] {len(self.samples):,} tiles  "
            f"(min_pos_frac={min_pos_frac})"
        )

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False,
                readahead=False, meminit=False,
            )
        return self._env

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x18, y18 = self.samples[idx]

        key = f"{x18}_{y18}".encode()
        with self._get_env().begin(write=False) as txn:
            val = txn.get(key)
        img = np.array(Image.open(io.BytesIO(val)).convert("RGB"))

        mask_path = self.mask_dir / f"{x18}_{y18}.png"
        mask      = np.array(Image.open(mask_path))   # uint8: 0 or 1

        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].unsqueeze(0).float()