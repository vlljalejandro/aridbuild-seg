"""
The teacher's predicted-to-true building area ratio on the 226-tile target
test split, at the reporting threshold (0.50) and at the threshold
pseudo-labels were generated with (0.20).

Backs the calibration figures in the Discussion.
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
MASKS = REPO / "release" / "masks"
IMAGES = REPO / "data" / "d3" / "images"
TEACHER = REPO / "data" / "checkpoints" / "b1-src-tgt.pt"

THRESHOLDS = [0.50, 0.20]
BATCH = 8
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    from src.models import build_model

    with open(SPLITS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    col = "tile_id" if "tile_id" in rows[0] else "new_stem"
    stems = [r[col] for r in rows if r["split"] == "test"]
    assert len(stems) == 226, len(stems)

    imgs, msks = [], []
    for s in stems:
        imgs.append(np.asarray(Image.open(IMAGES / f"{s}.png").convert("RGB"),
                               dtype=np.float32) / 255.0)
        m = np.asarray(Image.open(MASKS / f"{s}.png"), dtype=np.uint8)
        msks.append((m[..., 0] if m.ndim == 3 else m) > 127)
    imgs, msks = np.stack(imgs), np.stack(msks)

    ckpt = torch.load(TEACHER, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    mean = np.array(cfg["data"]["img_mean"], dtype=np.float32)
    std = np.array(cfg["data"]["img_std"], dtype=np.float32)
    mcfg = dict(cfg["model"])
    mcfg["pretrained"] = False
    model = build_model(mcfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(DEVICE)

    x = np.ascontiguousarray(((imgs - mean) / std).transpose(0, 3, 1, 2))

    probs = []
    with torch.no_grad():
        for s in range(0, len(x), BATCH):
            out = model(torch.from_numpy(x[s:s + BATCH]).to(DEVICE))
            logit = out[0] if isinstance(out, (tuple, list)) else out
            probs.append(torch.sigmoid(logit)[:, 0].cpu().numpy())
    probs = np.concatenate(probs)

    true_area = int(msks.sum())
    print(f"teacher   : {TEACHER.name} ({mcfg['name']})")
    print(f"tiles     : {len(stems)}")
    print(f"true area : {true_area:,} px "
          f"({msks.mean():.4f} of all pixels)\n")

    for thr in THRESHOLDS:
        pred = probs > thr
        pa = int(pred.sum())
        ratio = pa / true_area
        inter = int(np.logical_and(pred, msks).sum())
        union = int(np.logical_or(pred, msks).sum())
        print(f"threshold {thr:.2f}")
        print(f"  predicted area   {pa:,} px")
        print(f"  pred/true ratio  {ratio:.4f}  "
              f"({(ratio - 1) * 100:+.1f}% vs ground truth)")
        print(f"  IoU              {inter / union:.4f}")
        print(f"  precision        {inter / pa:.4f}")
        print(f"  recall           {inter / true_area:.4f}\n")


if __name__ == "__main__":
    main()
