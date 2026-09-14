# arid-building-segmentation

Building footprint segmentation in arid environments, with pseudo-label
self-training as a substitute for target-domain annotation. Code, annotations
and the 15 trained checkpoints behind the paper.

We share the annotations, masks and tile identifiers. Every number in the
paper reproduces with no imagery and no GPU (see Tier 1 below).

![Input imagery, predicted building probability, and instances after
probability-seeded watershed separation.](figures/figure_4_qualitative.png)

---

## What's here

```
arid-building-segmentation/
├── release/                      in git, ~20 MB
│   ├── d3_saudi_buildings.gpkg   2,262 tiles + 15,635 polygons, WGS84
│   ├── splits.csv                tile_id,split — 1,810 / 226 / 226
│   ├── masks/                    2,262 PNG — the ground truth
│   ├── tile_index.csv.gz         D4: 200,000 labeled, 156,790 used
│   ├── d4_coverage.gpkg          D4 extent, zoom-11 cells
│   ├── per_tile_counts.npz       per-tile counts, all 15 models
│   └── WEIGHTS.md
│
├── data/                         gitignored
│   ├── checkpoints/              15 × <label>.pt  ] HF model repo
│   └── d3/images/                you fetch these
│
├── config/  scripts/  src/  tests/
└── environment.yml
```

The `tiles` layer carries `tile_id`, `quadkey`, `x18`, `y18`, `split` and
`n_bldg`; the `buildings` layer carries `ann_id`, `tile_id`, `area_m2`.
`tile_id` is the `arid_XXXX` stem (it names the mask, the `splits.csv` row,
and joins the two layers).

Labels are `b{1,2,3}-{src,tgt,src-tgt,src-pl,src-pl-tgt}`. B1 is
U-Net/ResNet-50, B2 SegFormer-B2, B3 Swin-T/FPN.

---

## Tier 1 — reproduce every number in the paper (no imagery, no GPU)

`per_tile_counts.npz` holds per-tile intersection, union and confusion counts
for all 15 models on the 226 test tiles, which is enough to recompute every
table and the full paired bootstrap.

```bash
conda env create -f environment.yml && conda activate aridbuild-seg

python scripts/make_paper_table.py    # Table II
python scripts/bootstrap_ci.py        # Table III, 18 comparisons
python scripts/make_figs.py           # Figure 3
```

Seconds, on a laptop.

Figures 1 and 2 are static assets. Figure 4 comes from
`scripts/make_qualitative.py`, which needs imagery and a checkpoint.

## Tier 2 — reproduce from the checkpoints (needs imagery + GPU)

### 1. Fetch the imagery

Read `x18`, `y18` (or `quadkey`) and `tile_id` from the gpkg's `tiles` layer,
fetch each zoom-18 tile from a web-Mercator source, and save it as
`data/d3/images/<tile_id>.png`. Naming by `tile_id` is what links imagery to
`release/masks/` and `splits.csv`.

Tiles are 512×512, ~0.27 m/px at these latitudes.

### 2. Run the evaluator

Paths derive from the repo root; nothing to edit.

```bash
python scripts/evaluate_release.py
```

It rewrites `per_tile_counts.npz`, so Tier 1 then regenerates everything from
your own inference.

### 3. Retrain from scratch (optional)

Needs D1, D2 and D4 as well (see `data/README.md`). Roughly two weeks on one
GPU: benchmarks ~20 h/architecture, pseudo-label generation ~2 h, students
~120 h/architecture.

---

## Notes that will save you time

**Model naming.** The B3 checkpoints store `Mask2FormerSwinT`, but the model is
a Swin-T encoder with an FPN decoder and dense convolutional heads (no queries,
no mask classification). The old string is kept as an alias so the checkpoints
load; the paper calls it Swin-T/FPN.

**Output is a 2-tuple.** `forward()` returns `(seg_logit, dist_logit)`, both
`(N,1,512,512)`. Element 1 is a train-only auxiliary distance head, kept so the
checkpoints load exactly. Reading it instead of element 0 yields a mask that
thresholds to something plausible and scores around 0.09 IoU.

**Normalization.** Take `img_mean` and `img_std` from the checkpoint's own
`cfg`, not from the YAML files. Images are loaded in RGB order.

**Config drift.** The `cfg` in each checkpoint is a training-time snapshot with
stale paths. `config/` is authoritative for everything except normalization and
threshold.

**Single teacher.** All pseudo-labels come from `b1-src-tgt`, and all three
students distil from it. Cross-architecture agreement shows the pseudo-labels
transfer; it is not independent replication.

**Bootstrap pairing.** `bootstrap_ci.py` draws one index matrix and reuses it
for every model. That is what makes the intervals paired. Drawing fresh
indices per model inflates every interval.

**The `resume_from` guard.** An empty `resume_from` falls through to a derived
path, which once silently trained benchmarks from ImageNet instead of the
source checkpoint. The guard now hard-fails; leave it in place.

---

## Known limitations

Single seed per configuration. The test split is 226 tiles, so effects below
roughly 2 pp are hard to resolve. The bootstrap resamples tiles independently,
but tiles from the same scene are not exchangeable (the released `x18`/`y18`
make a scene-level block bootstrap straightforward). Instance separation runs
at inference but is scored only at pixel level.

## Citation

```
[blank]
```
