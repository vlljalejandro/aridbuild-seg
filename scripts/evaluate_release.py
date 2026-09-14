"""
Runs all 15 released checkpoints over the 226-tile target test split and
rewrites release/per_tile_counts.npz, from which the tables and figures are
regenerated.

Paths derive from this file's location. Needs release/splits.csv and
release/masks/ (in git), data/d3/images/ (you fetch) and data/checkpoints/.

Inference settings: RGB channel order, element 0 of the model's output tuple,
sigmoid on the single channel, and img_mean / img_std / threshold taken from
each checkpoint's own cfg rather than from any YAML.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SPLITS_CSV = REPO / "release" / "splits.csv"
GPKG = REPO / "release" / "d3_saudi_buildings.gpkg"
IMAGES = REPO / "data" / "d3" / "images"
MASKS = REPO / "release" / "masks"
CKPT_DIR = REPO / "data" / "checkpoints"
OUT_NPZ = REPO / "release" / "per_tile_counts.npz"

SPLIT = "test"
HEAD = 0          # segmentation head; element 1 is the aux distance head
BGR = False       # verified RGB
BATCH = 8
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# published test IoU, Table II
PAPER = {
    "b1-src":        0.5988, "b2-src":        0.6069, "b3-src":        0.5616,
    "b1-tgt":        0.7075, "b2-tgt":        0.7090, "b3-tgt":        0.7302,
    "b1-src-tgt":    0.7582, "b2-src-tgt":    0.7437, "b3-src-tgt":    0.7526,
    "b1-src-pl":     0.7733, "b2-src-pl":     0.7656, "b3-src-pl":     0.7738,
    "b1-src-pl-tgt": 0.7715, "b2-src-pl-tgt": 0.7674, "b3-src-pl-tgt": 0.7793,
}


def require(path, what, hint):
    if not path.exists():
        raise SystemExit(f"{what} not found at {path}\n  {hint}")


def test_stems():
    """
    Split membership from splits.csv, falling back to the GeoPackage. Never
    falls back to all tiles: scoring 2,262 instead of 226 gives a
    plausible-looking number that is wrong.
    """
    if SPLITS_CSV.exists():
        with open(SPLITS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        col = "tile_id" if "tile_id" in rows[0] else "new_stem"
        stems = [r[col] for r in rows if r["split"] == SPLIT]
    elif GPKG.exists():
        import geopandas as gpd
        t = gpd.read_file(GPKG, layer="tiles")
        col = "tile_id" if "tile_id" in t.columns else "new_stem"
        stems = sorted(t.loc[t["split"] == SPLIT, col])
    else:
        raise SystemExit(f"need {SPLITS_CSV} or {GPKG}")
    if len(stems) != 226:
        raise SystemExit(f"expected 226 {SPLIT} tiles, got {len(stems)}")
    return stems


def load_data(stems):
    imgs, msks = [], []
    for s in stems:
        ip, mp = IMAGES / f"{s}.png", MASKS / f"{s}.png"
        if not ip.exists():
            raise SystemExit(f"missing image {ip}")
        if not mp.exists():
            raise SystemExit(f"missing mask {mp}")
        imgs.append(np.asarray(Image.open(ip).convert("RGB"),
                               dtype=np.float32) / 255.0)
        m = np.asarray(Image.open(mp), dtype=np.uint8)
        msks.append((m[..., 0] if m.ndim == 3 else m) > 127)
    return np.stack(imgs), np.stack(msks)


def evaluate(ckpt_path, imgs, msks):
    from src.models import build_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    mean = np.array(cfg["data"]["img_mean"], dtype=np.float32)
    std = np.array(cfg["data"]["img_std"], dtype=np.float32)
    thr = float(cfg.get("eval", {}).get("threshold", cfg.get("threshold", 0.5)))

    mcfg = dict(cfg["model"])
    mcfg["pretrained"] = False     # the strict load overwrites them anyway,
                                   # and this avoids needing network access
    model = build_model(mcfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(DEVICE)

    x = imgs[..., ::-1] if BGR else imgs
    x = np.ascontiguousarray(((x - mean) / std).transpose(0, 3, 1, 2))

    n = len(x)
    inter = np.zeros(n, dtype=np.int64)
    union = np.zeros(n, dtype=np.int64)
    tp = np.zeros(n, dtype=np.int64)
    pp = np.zeros(n, dtype=np.int64)
    ap = np.zeros(n, dtype=np.int64)

    with torch.no_grad():
        for s in range(0, n, BATCH):
            xb = torch.from_numpy(x[s:s + BATCH]).to(DEVICE)
            out = model(xb)
            logit = out[HEAD] if isinstance(out, (tuple, list)) else out
            pred = (torch.sigmoid(logit)[:, 0] > thr).cpu().numpy()
            gt = msks[s:s + BATCH]
            for j in range(len(pred)):
                i = s + j
                inter[i] = np.logical_and(pred[j], gt[j]).sum()
                union[i] = np.logical_or(pred[j], gt[j]).sum()
                tp[i] = inter[i]
                pp[i] = pred[j].sum()
                ap[i] = gt[j].sum()

    del model
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
    return inter, union, tp, pp, ap


def main():
    require(IMAGES, "imagery",
            "imagery is not redistributed -- fetch zoom-18 tiles using the "
            "x18/y18 columns of release/d3_saudi_buildings.gpkg and save each "
            "as data/d3/images/<tile_id>.png")
    require(MASKS, "masks", "release/masks/ ships with the repository")
    require(CKPT_DIR, "checkpoints",
            "download data/checkpoints/ from the HF model repo")

    stems = test_stems()
    imgs, msks = load_data(stems)
    print(f"{len(stems)} tiles on {DEVICE}, "
          f"mask positive fraction {msks.mean():.4f}\n")

    labels = sorted(PAPER, key=lambda k: (k.split("-", 1)[1], k))
    store = {"stems": np.array(stems)}
    deltas = []

    print(f"{'label':<16}{'IoU':>9}{'paper':>9}{'delta':>10}"
          f"{'prec':>8}{'rec':>8}")
    print("-" * 60)
    for label in labels:
        path = CKPT_DIR / f"{label}.pt"
        if not path.exists():
            print(f"{label:<16}  MISSING")
            continue
        inter, union, tp, pp, ap = evaluate(path, imgs, msks)
        iou = inter.sum() / union.sum()
        prec = tp.sum() / pp.sum() if pp.sum() else float("nan")
        rec = tp.sum() / ap.sum() if ap.sum() else float("nan")
        d = iou - PAPER[label]
        deltas.append((label, d))
        for k, v in zip(("inter", "union", "tp", "pp", "ap"),
                        (inter, union, tp, pp, ap)):
            store[f"{label}/{k}"] = v
        print(f"{label:<16}{iou:>9.4f}{PAPER[label]:>9.4f}{d:>+10.4f}"
              f"{prec:>8.4f}{rec:>8.4f}")

    np.savez_compressed(OUT_NPZ, **store)
    print(f"\nwrote {OUT_NPZ}")

    if deltas:
        a = np.array([d for _, d in deltas])
        print(f"\ndeltas: mean {a.mean():+.4f}  sd {a.std():.4f}  "
              f"range [{a.min():+.4f}, {a.max():+.4f}]")

        # Sign is the wrong test: a uniform ground-truth difference helps
        # recall-leaning models and hurts precision-leaning ones, so both signs
        # are legitimate. What must hold is that the three architectures move
        # together within a condition, since a ground-truth property cannot be
        # architecture-specific but an evaluator bug would be.
        by_cond = {}
        for label, d in deltas:
            by_cond.setdefault(label.split("-", 1)[1], []).append(d)
        worst = max(max(v) - min(v) for v in by_cond.values())
        print("\n  condition        spread across b1/b2/b3")
        for cond, ds in by_cond.items():
            print(f"  {cond:<16} {max(ds) - min(ds):.4f}")
        if worst < 0.0015 and np.abs(a).max() < 0.005:
            print("\n  Architectures agree within every condition and no delta")
            print("  exceeds tolerance. Reproduction verified.")
        else:
            print("\n  Architectures disagree within a condition, or a delta")
            print("  exceeds tolerance. Investigate before accepting.")


if __name__ == "__main__":
    main()
