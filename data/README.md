# Data

Four sources. D1 and D2 are out-of-domain supervision; D3 is the labeled
target set and the sole evaluation basis; D4 is the unlabeled target pool.

| | Source | Available | Used in training | Tile size |
|---|---|---|---|---|
| D1 | CrowdAI Mapping Challenge | 64,511 | 51,609 | 300×300 |
| D2 | Inria Aerial Image Labeling | 18,000 | 14,400 | 512×512 |
| D3 | Saudi, labeled | 2,262 | 1,810 | 512×512 |
| D4 | Saudi, unlabeled | 200,000 | 156,790 | 512×512 |

"Used in training" is after splitting and filtering. For D3 the remaining 452
tiles are the validation and test splits.

## D1, D2

Obtain from the original sources. Pool them and split 80/10/10 at seed 42 to
get the 66,009 / 8,251 / 8,251 used throughout. D1 tiles are reflection-padded
from 300×300 to 512×512 at load time.

## D3 — labeled target set

```
release/d3_saudi_buildings.gpkg    2,262 tiles + 15,635 polygons, WGS84
release/splits.csv                 tile_id,split — 1,810 / 226 / 226
release/masks/                     2,262 PNG — the ground truth
```

We share the annotations and masks. For imagery, fetch zoom-18 tiles using
`x18`/`y18` and save each as `<tile_id>.png`.

`tiles` layer: `tile_id`, `quadkey`, `x18`, `y18`, `split`, `n_bldg`,
`geometry`. `buildings` layer: `ann_id`, `tile_id`, `area_m2`, `geometry`.

`tile_id` is the `arid_XXXX` stem and the only identifier in the release (it
names the mask and the `splits.csv` row, and joins the two GeoPackage layers).
`x18`/`y18` are the web-Mercator address, for retrieval and spatial analysis.

Masks are the canonical ground truth (every number in the paper is computed
against them). Polygon counts are taken after multipart footprints are split
into single parts, and `n_bldg` is recomputed from the geometry on every
rebuild.

## D4 — tile identifiers

```
release/tile_index.csv.gz    tile_id, x18, y18, pos_frac, used
release/d4_coverage.gpkg     zoom-11 coverage cells with tile counts
```

200,000 rows; `used` marks the 156,790 whose pseudo-label positive fraction
reached 2% and which therefore entered student training. `pos_frac` is the
teacher's positive-pixel fraction for that tile.

## Regenerating

```bash
python scripts/check_roundtrip.py      # gpkg polygons -> masks/
python scripts/tile_index_geom.py      # tile_index.csv.gz -> coverage gpkg
```
