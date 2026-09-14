"""
Figure 3: target-domain test IoU by configuration, from cached counts. Reads
release/per_tile_counts.npz only (no imagery, weights or GPU).

Figures 1 and 2 are static diagrams; Figure 4 comes from
scripts/make_qualitative.py.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.release.counts import (ARCHES, CONDITIONS, boot_iou, ci, load, micro,
                                replicate_indices)

NPZ = REPO / "release" / "per_tile_counts.npz"
OUT = REPO / "figures" / "figure_3_results.pdf"

# shaded: configurations using no labeled target tile
NO_TARGET_LABELS = {"src", "src-pl"}
COLORS = {"b1": "#3B6EA5", "b2": "#C1663B", "b3": "#4A8A5C"}
WIDTH = 0.24


def main():
    OUT.parent.mkdir(exist_ok=True)
    data, _ = load(NPZ)
    idx = replicate_indices(len(next(iter(data.values()))["union"]))
    conds = list(CONDITIONS)

    fig, ax = plt.subplots(figsize=(3.45, 2.5))

    for i, cond in enumerate(conds):
        if cond in NO_TARGET_LABELS:
            ax.axvspan(i - 0.5, i + 0.5, color="0.93", zorder=0, lw=0)

    for k, (arch, arch_name) in enumerate(ARCHES.items()):
        xs, ys, lo, hi = [], [], [], []
        for i, cond in enumerate(conds):
            label = f"{arch}-{cond}"
            if label not in data:
                continue
            v = micro(data[label])["iou"]
            a, b = ci(boot_iou(data[label], idx))
            xs.append(i + (k - 1) * WIDTH)
            ys.append(v)
            lo.append(v - a)
            hi.append(b - v)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=3.8, capsize=2,
                    lw=1.0, elinewidth=1.0, color=COLORS[arch],
                    label=arch_name, zorder=3)

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([CONDITIONS[c] for c in conds], fontsize=7)
    ax.set_ylabel("Target-domain test IoU", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-0.5, len(conds) - 0.5)
    ax.grid(axis="y", lw=0.35, color="0.88", zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=6.8, loc="lower right",
              handletextpad=0.3, borderaxespad=0.2)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()