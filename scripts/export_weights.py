"""
Export training checkpoints for release.

Strips optimizer, scheduler and scaler state, which cuts each file by roughly
two thirds and makes it loadable with ``weights_only=True``.

Released files are renamed from internal run names ("main-b3-v2", "b1-warm")
to the paper's configuration names, so a checkpoint maps to a table row without
knowing the project's chronology.

    python scripts/export_weights.py --out data/checkpoints --dry-run
    python scripts/export_weights.py --out data/checkpoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = next(
    p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    if (p / "src").is_dir()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Release name -> training run directory. Recovered by matching each
# checkpoint's stored epoch/best_iou against the training logs; encoder_name
# disambiguates across backbones. Four superseded v1 runs (main-b2-supervised,
# main-b2-finetune, main-b3-supervised, main-b3-finetune) are not released.
RUNS = {
    "b1-src":        "b1-supervised",
    "b2-src":        "b2-supervised",
    "b3-src":        "b3-supervised",
    "b1-tgt":        "b1-finetune",
    "b2-tgt":        "b2-finetune",
    "b3-tgt":        "b3-finetune",
    "b1-src-tgt":    "b1-warm-finetune",
    "b2-src-tgt":    "b2-warm-finetune",
    "b3-src-tgt":    "b3-warm-finetune",
    "b1-src-pl":     "main-b1-supervised",
    "b2-src-pl":     "main-b2-v2-supervised",
    "b3-src-pl":     "main-b3-v2-supervised",
    "b1-src-pl-tgt": "main-b1-finetune",
    "b2-src-pl-tgt": "main-b2-v2-finetune",
    "b3-src-pl-tgt": "main-b3-v2-finetune",
}

CKPT_ROOT = Path("outputs/checkpoints")

# Everything else in the stored config is paths and schedule detail that would
# only mislead someone loading the weights.
_KEEP_CFG = ("model", "loss", "eval", "seed")


def export(name: str, src: Path, dst_dir: Path, dry_run: bool) -> tuple[int, int]:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})

    slim = {
        "model": ckpt["model"],
        "cfg": {k: cfg[k] for k in _KEEP_CFG if k in cfg},
        "epoch": ckpt.get("epoch"),
        "best_iou": ckpt.get("best_iou"),
        "release_name": name,
        "source_run": src.parent.name,
    }

    dst = dst_dir / f"{name}.pt"
    before = src.stat().st_size
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        torch.save(slim, dst)
        after = dst.stat().st_size
        # fail here rather than shipping something that cannot be loaded safely
        torch.load(dst, map_location="cpu", weights_only=True)
    else:
        after = 0
    return before, after


def main():
    ap = argparse.ArgumentParser(description="Strip and rename checkpoints.")
    ap.add_argument("--out", default="data/checkpoints")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written without writing it")
    a = ap.parse_args()

    dst_dir = Path(a.out)
    total_before = total_after = 0
    written = missing = 0

    for name, run in RUNS.items():
        src = CKPT_ROOT / run / "best.pt"
        if not src.exists():
            print(f"[skip] {name:<16} not found: {src}")
            missing += 1
            continue
        before, after = export(name, src, dst_dir, a.dry_run)
        total_before += before
        total_after += after
        written += 1
        if a.dry_run:
            print(f"[dry ] {name:<16} <- {src}  ({before/1e6:.0f} MB)")
        else:
            print(f"[ok  ] {name:<16} {before/1e6:>6.0f} -> {after/1e6:>5.0f} MB")

    print(f"\n{written} exported, {missing} missing")
    if not a.dry_run and written:
        print(f"total {total_before/1e6:.0f} -> {total_after/1e6:.0f} MB "
              f"({100 * (1 - total_after / total_before):.0f}% smaller)")
        print(f"written to {dst_dir}")
        print("\nAll files verified loadable with weights_only=True.")


if __name__ == "__main__":
    main()
