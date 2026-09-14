"""
Table III: paired bootstrap comparisons, 10,000 replicates on the same 226
tiles. Reads per_tile_counts.npz only (no checkpoints, imagery or GPU).
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.release.counts import (ARCHES, COMPARISONS, N_BOOT, SEED, load,
                                micro, paired_delta, replicate_indices)

NPZ = REPO / "release" / "per_tile_counts.npz"
OUT_TEX = REPO / "release" / "table_iv.tex"


def main():
    data, stems = load(NPZ)
    n_tiles = len(next(iter(data.values()))["union"])
    print(f"{len(data)} models, {n_tiles} tiles, "
          f"{N_BOOT} replicates, seed {SEED}\n")

    idx = replicate_indices(n_tiles)

    rows = []
    for title, cond_a, cond_b in COMPARISONS:
        print(title)
        for arch in ARCHES:
            la, lb = f"{arch}-{cond_a}", f"{arch}-{cond_b}"
            if la not in data or lb not in data:
                print(f"  {arch.upper()}: missing {la} or {lb}")
                continue
            r = paired_delta(data, la, lb, idx)
            mark = "" if r["excludes_zero"] else "   (interval contains zero)"
            print(f"  {arch.upper()}  {r['delta']:+.4f}  "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}]{mark}")
            rows.append((title, arch.upper(), r))
        print()

    n_excl = sum(1 for _, _, r in rows if r["excludes_zero"])
    print(f"{len(rows)} comparisons, {n_excl} with intervals excluding zero")
    print("No multiplicity correction applied.")

    with open(OUT_TEX, "w") as f:
        last = None
        for title, arch, r in rows:
            if title != last:
                f.write("\\midrule\n\\multicolumn{4}{@{}l}{\\textit{"
                        f"{title}" "}}\\\\\n")
                last = title
            f.write(f"{arch} & ${r['delta']:+.4f}$ & "
                    f"$[{r['lo']:+.4f}, {r['hi']:+.4f}]$ \\\\\n")
    print(f"\nwrote {OUT_TEX}")


if __name__ == "__main__":
    main()
