"""
Everything the paper's tables and figures need, from cached per-tile counts.
No imagery, weights or GPU.

per_tile_counts.npz holds five length-226 int arrays per label: inter, union,
tp, pp (predicted positive), ap (actual positive). Micro-averaged metrics are
ratios of sums, so a bootstrap replicate is a resample of tile indices
followed by those sums, which is why the whole analysis fits in a few hundred
kilobytes.
"""

from pathlib import Path

import numpy as np

ARCHES = {
    "b1": "B1 U-Net/ResNet-50",
    "b2": "B2 SegFormer-B2",
    "b3": "B3 Swin-T/FPN",
}

CONDITIONS = {
    "src":        "Src",
    "tgt":        "Tgt",
    "src-tgt":    "Src->Tgt",
    "src-pl":     "Src+PL",
    "src-pl-tgt": "Src+PL->Tgt",
}

CONDITION_NOTES = {
    "src":        "source supervised training only, evaluated zero-shot",
    "tgt":        "target fine-tuning only, no source data",
    "src-tgt":    "conventional transfer learning",
    "src-pl":     "source data plus pseudo-labels, no labeled target tiles",
    "src-pl-tgt": "full pipeline",
}

# Table III, in paper order. Each entry reports condition_a - condition_b.
COMPARISONS = [
    ("Pseudo-labels alone, no target labels",      "src-pl",     "src"),
    ("Target labels alone, no pseudo-labels",      "src-tgt",    "src"),
    ("Target labels added after pseudo-labels",    "src-pl-tgt", "src-pl"),
    ("Out-of-domain pretraining",                  "src-tgt",    "tgt"),
    ("Self-training vs. the conventional baseline", "src-pl-tgt", "src-tgt"),
    ("No target labels vs. the conventional baseline", "src-pl",  "src-tgt"),
]

N_BOOT = 10_000
SEED = 42


def load(npz_path):
    """Returns ({label: {inter, union, tp, pp, ap}}, tile_ids)."""
    z = np.load(Path(npz_path), allow_pickle=True)
    stems = z["stems"] if "stems" in z else None
    out = {}
    for key in z.files:
        if "/" not in key:
            continue
        label, field = key.split("/", 1)
        out.setdefault(label, {})[field] = z[key]
    n = {len(v["union"]) for v in out.values()}
    if len(n) != 1:
        raise ValueError(f"inconsistent tile counts across labels: {n}")
    return out, stems


def micro(counts, idx=None):
    """Micro-averaged IoU, F1, precision, recall over the given tile indices."""
    sel = slice(None) if idx is None else idx
    inter = counts["inter"][sel].sum()
    union = counts["union"][sel].sum()
    tp = counts["tp"][sel].sum()
    pp = counts["pp"][sel].sum()
    ap = counts["ap"][sel].sum()
    iou = inter / union if union else np.nan
    prec = tp / pp if pp else np.nan
    rec = tp / ap if ap else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else np.nan
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec}


def replicate_indices(n_tiles, n_boot=N_BOOT, seed=SEED):
    """
    One fixed set of resampled index arrays, reused across every model.

    This is what makes the intervals *paired*: applying identical tile
    resamples to both models cancels per-tile difficulty in the difference.
    Drawing fresh indices per model would inflate every interval.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_tiles, size=(n_boot, n_tiles))


def boot_iou(counts, idx_matrix):
    """Vectorised micro-IoU across all replicates."""
    inter = counts["inter"][idx_matrix].sum(axis=1)
    union = counts["union"][idx_matrix].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, np.nan)


def ci(values, alpha=0.05):
    """Percentile interval."""
    lo, hi = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_delta(data, label_a, label_b, idx_matrix):
    """Point estimate and percentile interval for IoU(a) - IoU(b)."""
    a = boot_iou(data[label_a], idx_matrix)
    b = boot_iou(data[label_b], idx_matrix)
    d = a - b
    point = micro(data[label_a])["iou"] - micro(data[label_b])["iou"]
    lo, hi = ci(d)
    return {
        "delta": float(point),
        "lo": lo,
        "hi": hi,
        "excludes_zero": (lo > 0) or (hi < 0),
    }
