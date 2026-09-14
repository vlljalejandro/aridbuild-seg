"""
DataLoader factories, one per training stage.

    get_supervised_loaders              D1+D2 (CrowdAI + Inria)
    get_finetune_loaders                D3 (labeled target tiles)
    get_pseudolabel_supervised_loaders  D1+D2+pseudo-labels

Each returns (train, val, test). At val_split = test_split = 0.10 this gives
66,009 / 8,251 / 8,251 for D1+D2 and 1,810 / 226 / 226 for D3.

Splits are deterministic: D1 order comes from the COCO JSON, D2 from
sorted(glob(...)), and both are partitioned by torch.random_split under an
explicit generator seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from .dataset    import InriaDataset, PseudoLabelDataset, SegmentationDataset, _collect_samples
from .transforms import get_train_transforms, get_val_transforms

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = (0.2186, 0.2315, 0.4944)
_STD  = (0.2308, 0.2263, 0.2268)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(subset) -> List[dict]:
    """Extract the underlying sample list from a ``random_split`` Subset."""
    return [subset.dataset[i] for i in subset.indices]


def _make_loader(
    dataset,
    *,
    batch_size:  int,
    shuffle:     bool,
    num_workers: int,
    pin_memory:  bool,
    drop_last:   bool = False,
) -> DataLoader:
    extra = (
        dict(persistent_workers=True, prefetch_factor=4)
        if num_workers > 0 else {}
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = drop_last,
        **extra,
    )


def _split(
    samples:   List[dict],
    val_frac:  float,
    test_frac: float,
    seed:      int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Reproducibly split a sample list into train / val / test."""
    n      = len(samples)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    n_tr   = n - n_val - n_test

    gen = torch.Generator().manual_seed(seed)
    tr, val, te = random_split(samples, [n_tr, n_val, n_test], generator=gen)
    return _unwrap(tr), _unwrap(val), _unwrap(te)


# ─────────────────────────────────────────────────────────────────────────────
# Supervised training — D1 + D2 (CrowdAI + Inria)
# ─────────────────────────────────────────────────────────────────────────────

def get_supervised_loaders(
    cfg:  dict,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Combined D1 (CrowdAI COCO) + D2 (Inria TIF) supervised loaders.

    cfg keys: supervised_sources (D1), inria_sources (D2), val_split,
    test_split, batch_size, num_workers, pin_memory, and optionally
    img_mean / img_std.
    """
    mean = cfg.get("img_mean", _MEAN)
    std  = cfg.get("img_std",  _STD)
    val_frac  = cfg["val_split"]
    test_frac = cfg["test_split"]

    train_tfm = get_train_transforms(mean=mean, std=std)
    val_tfm   = get_val_transforms(mean=mean, std=std)

    # ── D1: CrowdAI COCO ─────────────────────────────────────────────────────
    d1_samples = _collect_samples(cfg["supervised_sources"])
    d1_tr, d1_val, d1_te = _split(d1_samples, val_frac, test_frac, seed)

    d1_train_ds = SegmentationDataset(d1_tr,  transform=train_tfm)
    d1_val_ds   = SegmentationDataset(d1_val, transform=val_tfm)
    d1_test_ds  = SegmentationDataset(d1_te,  transform=val_tfm)

    src_names = [Path(s["image_dir"]).name for s in cfg["supervised_sources"]]
    print(
        f"[supervised] D1 sources={src_names}\n"
        f"[supervised] D1 train={len(d1_train_ds):>7,}  "
        f"val={len(d1_val_ds):>5,}  test={len(d1_test_ds):>5,}"
    )

    # ── D2: Inria TIF crops ───────────────────────────────────────────────────
    inria_sources     = cfg.get("inria_sources", [])
    d2_train_datasets = []
    d2_val_datasets   = []
    d2_test_datasets  = []

    for source in inria_sources:
        inria_samples = InriaDataset.collect_samples(source["train_dir"])
        d2_tr, d2_val, d2_te = _split(inria_samples, val_frac, test_frac, seed)

        d2_train_datasets.append(InriaDataset(d2_tr,  transform=train_tfm))
        d2_val_datasets.append(  InriaDataset(d2_val, transform=val_tfm))
        d2_test_datasets.append( InriaDataset(d2_te,  transform=val_tfm))

        print(
            f"[supervised] D2 source={Path(source['train_dir']).parent.name}\n"
            f"[supervised] D2 train={len(d2_tr):>7,}  "
            f"val={len(d2_val):>5,}  test={len(d2_te):>5,}"
        )

    # ── Combine D1 + D2 ───────────────────────────────────────────────────────
    all_train = [d1_train_ds] + d2_train_datasets
    all_val   = [d1_val_ds]   + d2_val_datasets
    all_test  = [d1_test_ds]  + d2_test_datasets

    train_ds = ConcatDataset(all_train) if len(all_train) > 1 else all_train[0]
    val_ds   = ConcatDataset(all_val)   if len(all_val)   > 1 else all_val[0]
    test_ds  = ConcatDataset(all_test)  if len(all_test)  > 1 else all_test[0]

    print(
        f"[supervised] Combined  train={len(train_ds):>7,}  "
        f"val={len(val_ds):>5,}  test={len(test_ds):>5,}"
    )

    kw = dict(
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
        pin_memory  = cfg["pin_memory"],
    )
    return (
        _make_loader(train_ds, shuffle=True,  drop_last=True,  **kw),
        _make_loader(val_ds,   shuffle=False, drop_last=False, **kw),
        _make_loader(test_ds,  shuffle=False, drop_last=False, **kw),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning — D3 (2K Saudi Arabia)
# ─────────────────────────────────────────────────────────────────────────────

def get_finetune_loaders(
    cfg:  dict,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    D3 fine-tuning loaders (labeled target tiles, COCO format).

    cfg keys: finetune_sources, val_split, test_split, batch_size,
    num_workers, pin_memory, and optionally img_mean / img_std.
    """
    mean = cfg.get("img_mean", _MEAN)
    std  = cfg.get("img_std",  _STD)

    samples               = _collect_samples(cfg["finetune_sources"])
    tr_samples, val_s, te = _split(samples, cfg["val_split"], cfg["test_split"], seed)

    train_ds = SegmentationDataset(tr_samples, transform=get_train_transforms(mean=mean, std=std))
    val_ds   = SegmentationDataset(val_s,      transform=get_val_transforms(mean=mean,   std=std))
    test_ds  = SegmentationDataset(te,         transform=get_val_transforms(mean=mean,   std=std))

    src_names = [Path(s["image_dir"]).name for s in cfg["finetune_sources"]]
    print(
        f"[finetune]   D3 sources={src_names}\n"
        f"[finetune]   train={len(train_ds):>7,}  "
        f"val={len(val_ds):>5,}  test={len(test_ds):>5,}"
    )

    kw = dict(
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
        pin_memory  = cfg["pin_memory"],
    )
    return (
        _make_loader(train_ds, shuffle=True,  drop_last=True,  **kw),
        _make_loader(val_ds,   shuffle=False, drop_last=False, **kw),
        _make_loader(test_ds,  shuffle=False, drop_last=False, **kw),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-training — D1 + D2 + pseudo-labeled Saudi tiles
# ─────────────────────────────────────────────────────────────────────────────

def get_pseudolabel_supervised_loaders(
    cfg:  dict,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Combined D1 + D2 + pseudo-labels for self-training.

    Pseudo-labels go into the TRAIN split only. Val and test stay clean D1+D2,
    so checkpoint selection runs against the 8,251-tile source validation
    split and never touches target data.

    cfg keys beyond the supervised ones: pseudo_lmdb_path, pseudo_mask_dir,
    pseudo_manifest, pseudo_min_pos_frac (default 0.02).
    """
    mean = cfg.get("img_mean", _MEAN)
    std  = cfg.get("img_std",  _STD)
    val_frac  = cfg["val_split"]
    test_frac = cfg["test_split"]

    train_tfm = get_train_transforms(mean=mean, std=std)
    val_tfm   = get_val_transforms(mean=mean, std=std)

    # ── D1: CrowdAI ───────────────────────────────────────────────────────────
    d1_samples = _collect_samples(cfg["supervised_sources"])
    d1_tr, d1_val, d1_te = _split(d1_samples, val_frac, test_frac, seed)

    d1_train_ds = SegmentationDataset(d1_tr,  transform=train_tfm)
    d1_val_ds   = SegmentationDataset(d1_val, transform=val_tfm)
    d1_test_ds  = SegmentationDataset(d1_te,  transform=val_tfm)

    # ── D2: Inria ─────────────────────────────────────────────────────────────
    d2_train_ds_list, d2_val_ds_list, d2_test_ds_list = [], [], []
    for source in cfg.get("inria_sources", []):
        inria_samples = InriaDataset.collect_samples(source["train_dir"])
        d2_tr, d2_val, d2_te = _split(inria_samples, val_frac, test_frac, seed)
        d2_train_ds_list.append(InriaDataset(d2_tr,  transform=train_tfm))
        d2_val_ds_list.append(  InriaDataset(d2_val, transform=val_tfm))
        d2_test_ds_list.append( InriaDataset(d2_te,  transform=val_tfm))

    # ── Pseudo-labels (train only) ────────────────────────────────────────────
    pseudo_ds = PseudoLabelDataset(
        lmdb_path    = cfg["pseudo_lmdb_path"],
        mask_dir     = cfg["pseudo_mask_dir"],
        manifest     = cfg["pseudo_manifest"],
        transform    = train_tfm,
        min_pos_frac = cfg.get("pseudo_min_pos_frac", 0.02),
    )

    # ── Combine ───────────────────────────────────────────────────────────────
    train_ds = ConcatDataset([d1_train_ds] + d2_train_ds_list + [pseudo_ds])
    val_ds   = ConcatDataset([d1_val_ds]   + d2_val_ds_list)
    test_ds  = ConcatDataset([d1_test_ds]  + d2_test_ds_list)

    print(
        f"[pseudolabel_supervised] D1+D2+pseudo  "
        f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}\n"
        f"  D1={len(d1_train_ds):,}  "
        f"D2={sum(len(d) for d in d2_train_ds_list):,}  "
        f"Pseudo={len(pseudo_ds):,}"
    )

    kw = dict(
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
        pin_memory  = cfg["pin_memory"],
    )
    return (
        _make_loader(train_ds, shuffle=True,  drop_last=True,  **kw),
        _make_loader(val_ds,   shuffle=False, drop_last=False, **kw),
        _make_loader(test_ds,  shuffle=False, drop_last=False, **kw),
    )