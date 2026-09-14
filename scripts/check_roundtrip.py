"""
Checks that release/masks/ matches the polygons in
release/d3_saudi_buildings.gpkg.

The masks are the canonical ground truth but are derived from the polygons,
and the two ship separately. Expect ~1.0; below 0.99 means the masks were not
built from this GeoPackage.
"""

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GPKG = REPO / "release" / "d3_saudi_buildings.gpkg"
MASKS = REPO / "release" / "masks"

IMG_SIZE = 512
SAMPLE = 0          # 0 = every tile; otherwise check this many
WARN_BELOW = 0.99


def bbox(x18, y18):
    """WGS84 bounds of a z18 tile (same formula as pseudo_label._z18_to_bbox)."""
    n = 2 ** 18
    lon0 = x18 / n * 360.0 - 180.0
    lon1 = (x18 + 1) / n * 360.0 - 180.0
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y18 / n))))
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y18 + 1) / n))))
    return lon0, lat0, lon1, lat1


def to_pixels(geom, bb, size):
    lon0, lat0, lon1, lat1 = bb
    dx, dy = lon1 - lon0, lat1 - lat0
    return [((lon - lon0) / dx * size, (lat1 - lat) / dy * size)
            for lon, lat in geom.exterior.coords]


def rasterize(geoms, bb, size):
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    for g in geoms:
        parts = [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))
        for part in parts:
            px = to_pixels(part, bb, size)
            if len(px) >= 3:
                draw.polygon(px, fill=1)
    return np.array(canvas, dtype=np.uint8)


def main():
    import geopandas as gpd

    if not GPKG.exists():
        raise SystemExit(f"not found: {GPKG}")
    if not MASKS.exists():
        raise SystemExit(f"not found: {MASKS}")

    tiles = gpd.read_file(GPKG, layer="tiles")
    buildings = gpd.read_file(GPKG, layer="buildings")
    print(f"{len(tiles):,} tiles, {len(buildings):,} polygons")

    by_tile = {tid: g for tid, g in buildings.groupby("tile_id")["geometry"]}

    rows = tiles if not SAMPLE else tiles.head(SAMPLE)
    ious, empty_ok, missing = [], 0, []

    for i, r in enumerate(rows.itertuples()):
        mp = MASKS / f"{r.tile_id}.png"
        if not mp.exists():
            missing.append(r.tile_id)
            continue
        m = np.asarray(Image.open(mp), dtype=np.uint8)
        if m.ndim == 3:
            m = m[..., 0]
        gt = m > 127

        geoms = by_tile.get(r.tile_id, [])
        pred = rasterize(list(geoms), bbox(int(r.x18), int(r.y18)),
                         IMG_SIZE) > 0

        union = np.logical_or(pred, gt).sum()
        if union == 0:
            empty_ok += 1          # both empty: IoU undefined
            continue
        ious.append(np.logical_and(pred, gt).sum() / union)

        if i and i % 500 == 0:
            print(f"    {i}/{len(rows)}")

    if missing:
        print(f"\n  {len(missing)} masks missing, e.g. {missing[:5]}")

    a = np.array(ious)
    print(f"\n  compared: {len(a):,} tiles with content, "
          f"{empty_ok} empty on both sides")
    print(f"  mean IoU   {a.mean():.4f}")
    print(f"  median     {np.median(a):.4f}")
    print(f"  min        {a.min():.4f}")
    print(f"  below 0.95 {int((a < 0.95).sum())} tiles")

    if a.mean() < WARN_BELOW:
        raise SystemExit(
            f"\n  mean {a.mean():.4f} is below {WARN_BELOW}: the shipped masks "
            "were not rasterized from this GeoPackage.")
    print("\n  masks agree with the polygons.")


if __name__ == "__main__":
    main()
