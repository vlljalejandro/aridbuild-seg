"""
Table II: target-domain test results, 226 tiles, threshold 0.50, with 95%
bootstrap intervals on IoU. Reads per_tile_counts.npz only.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.release.counts import (ARCHES, CONDITION_NOTES, CONDITIONS, boot_iou,
                                ci, load, micro, replicate_indices)

NPZ = REPO / "release" / "per_tile_counts.npz"
OUT_TEX = REPO / "release" / "table_ii.tex"

# short labels for the single-column table
SHORT = {"b1": "B1 U-Net-R50", "b2": "B2 SegFormer", "b3": "B3 Swin-FPN"}


def main():
    data, stems = load(NPZ)
    n_tiles = len(next(iter(data.values()))["union"])
    idx = replicate_indices(n_tiles)

    print(f"{n_tiles} tiles\n")
    print(f"{'':<22}{'IoU':>8}{'95% CI':>20}{'F1':>8}"
          f"{'Prec':>8}{'Rec':>8}")

    lines = []
    for cond, cond_name in CONDITIONS.items():
        header = f"{cond_name} --- {CONDITION_NOTES[cond]}"
        print(f"\n{header}")
        lines.append("\\midrule\n\\multicolumn{5}{@{}l}{\\textit{"
                     f"{cond_name}" "} --- " f"{CONDITION_NOTES[cond]}" "}\\\\")
        for arch, arch_name in ARCHES.items():
            label = f"{arch}-{cond}"
            if label not in data:
                print(f"  {arch_name:<20}  MISSING")
                continue
            m = micro(data[label])
            lo, hi = ci(boot_iou(data[label], idx))
            print(f"  {arch_name:<20}{m['iou']:>8.4f}"
                  f"   [{lo:.3f}, {hi:.3f}]"
                  f"{m['f1']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}")
            lines.append(
                f"{SHORT[arch]} & {m['iou']:.4f} [{lo:.3f}, {hi:.3f}] & "
                f"{m['f1']:.4f} & {m['precision']:.4f} & {m['recall']:.4f} "
                "\\\\")

    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_TEX}")

if __name__ == "__main__":
    main()