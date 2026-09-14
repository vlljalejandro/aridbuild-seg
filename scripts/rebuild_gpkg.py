"""
Builds the released GeoPackage from the working one: folds in the release
identifiers, trims redundant columns, recomputes derived ones.

Provenance record. It reads working files that are not part of the release, so
it does not run from a clone; it is kept to document how the shipped
GeoPackage was produced. Edit the paths below to re-run it.

Tiles are named arid_XXXX, and that string is `tile_id` in both layers. The
old x18_y18 tile_id is dropped, since x18 and y18 are already columns.
file_name, coco_image_id, image_id and geo_qc are dropped as duplication or
dead flags. n_bldg is recomputed from the buildings layer on every run rather
than carried forward, which is what produced the 15,610-vs-15,635
disagreement.

Writes:
  d3_saudi_buildings_public.gpkg    release names only
  d3_saudi_buildings_private.gpkg   keeps the original->release mapping
  splits.csv                        tile_id,split
"""

import csv
import json
from pathlib import Path

import geopandas as gpd

RELEASE = Path(r"C:\Users\vallejad\Desktop\repro")
GPKG_IN = RELEASE / "d3_saudi_buildings.gpkg"
MANIFEST = RELEASE / "arid_test" / "manifest.csv"
ANNOT = RELEASE / "arid_test" / "annotations.json"

GPKG_PUB = RELEASE / "d3_saudi_buildings_public.gpkg"
GPKG_PRIV = RELEASE / "d3_saudi_buildings_private.gpkg"
SPLITS_CSV = RELEASE / "splits.csv"

PUBLIC_COLUMNS = {"tile_id", "quadkey", "x18", "y18", "split", "n_bldg",
                  "geometry"}


def write_layers(path, tiles, buildings):
    """
    The second layer is appended explicitly and both are verified afterwards:
    on some geopandas/pyogrio versions a second to_file with the default mode
    truncates the file and drops the first layer.
    """
    import fiona

    if path.exists():
        path.unlink()
    tiles.to_file(path, layer="tiles", driver="GPKG")
    buildings.to_file(path, layer="buildings", driver="GPKG", mode="a")
    layers = set(fiona.listlayers(str(path)))
    if layers != {"tiles", "buildings"}:
        raise SystemExit(f"{path.name} has layers {sorted(layers)} -- "
                         "the second write clobbered the first")


def main():
    tiles = gpd.read_file(GPKG_IN, layer="tiles")
    buildings = gpd.read_file(GPKG_IN, layer="buildings")

    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    stem_to_new = {Path(r["original_file_name"]).stem: r["new_stem"]
                   for r in rows}

    keys = tiles["file_name"].map(lambda v: Path(str(v)).stem)
    missing = int((~keys.isin(stem_to_new)).sum())
    if missing:
        raise SystemExit(f"{missing} gpkg tiles absent from manifest -- aborting")
    tiles["new_id"] = keys.map(stem_to_new)
    if tiles["new_id"].duplicated().any():
        raise SystemExit("duplicate release identifiers -- aborting")

    # -- every id must exist in the COCO file ---------------------------------
    with open(ANNOT) as f:
        coco = json.load(f)
    coco_stems = {Path(im["file_name"]).stem for im in coco["images"]}
    unmapped = sorted(set(tiles["new_id"]) - coco_stems)
    if unmapped:
        raise SystemExit(f"{len(unmapped)} ids absent from annotations.json, "
                         f"e.g. {unmapped[:5]} -- aborting")
    print(f"annotations.json: all {len(tiles)} ids present")

    # -- repoint the buildings layer onto the release identifier --------------
    old_to_new = dict(zip(tiles["tile_id"], tiles["new_id"]))
    orphans = sorted(set(buildings["tile_id"]) - set(old_to_new))
    if orphans:
        raise SystemExit(f"{len(orphans)} building tile_ids absent from the "
                         f"tiles layer, e.g. {orphans[:5]} -- aborting")
    buildings["tile_id"] = buildings["tile_id"].map(old_to_new)
    print(f"buildings: {len(buildings)} rows repointed onto release ids")

    # -- one identifier, tile_id, in both layers ------------------------------
    tiles = tiles.drop(columns=["tile_id"]).rename(columns={"new_id": "tile_id"})

    # -- recompute n_bldg from the shipped geometry ---------------------------
    counts = buildings.groupby("tile_id").size()
    new_counts = tiles["tile_id"].map(counts).fillna(0).astype(int)
    changed = int((tiles["n_bldg"] != new_counts).sum())
    tiles["n_bldg"] = new_counts
    if int(new_counts.sum()) != len(buildings):
        raise SystemExit("recomputed n_bldg disagrees with the buildings layer")
    print(f"n_bldg: recomputed, {changed} tiles changed, "
          f"sum {int(new_counts.sum())}, {int((new_counts == 0).sum())} empty")

    # -- private copy: keeps provenance --------------------------------------
    priv = tiles.rename(columns={"file_name": "original_file_name"}).copy()
    priv = priv.drop(columns=[c for c in ("image_id", "coco_image_id", "geo_qc")
                              if c in priv.columns])
    write_layers(GPKG_PRIV, priv, buildings)
    print(f"wrote {GPKG_PRIV.name}  ({sorted(priv.columns)})")

    # -- public copy ----------------------------------------------------------
    pub = priv.drop(columns=["original_file_name"]).copy()
    write_layers(GPKG_PUB, pub, buildings)

    leaks = [c for c in pub.columns if "original" in c.lower()]
    if leaks:
        raise SystemExit(f"public copy still carries {leaks} -- aborting")
    if set(pub.columns) != PUBLIC_COLUMNS:
        extra = sorted(set(pub.columns) - PUBLIC_COLUMNS)
        gone = sorted(PUBLIC_COLUMNS - set(pub.columns))
        raise SystemExit(f"public columns wrong -- extra {extra}, missing {gone}")
    print(f"wrote {GPKG_PUB.name}   ({sorted(pub.columns)})")
    print(f"  buildings layer: {sorted(buildings.columns)}")

    # -- split file -----------------------------------------------------------
    with open(SPLITS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_id", "split"])
        for tid, sp in sorted(zip(pub["tile_id"], pub["split"])):
            w.writerow([tid, sp])
    print(f"wrote {SPLITS_CSV.name}")

    counts = pub["split"].value_counts().to_dict()
    print(f"\nsplit counts: {counts}")
    if counts.get("train") != 1810 or counts.get("test") != 226 \
            or counts.get("val") != 226:
        raise SystemExit("split counts wrong -- do not ship")


if __name__ == "__main__":
    main()
