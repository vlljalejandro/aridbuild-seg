"""
Turns release/tile_index.csv.gz into geometry.

  coverage   zoom-11 cells containing at least one used tile (~900 features).
             Shipped: it shows where the pseudo-label corpus is, which the CSV
             cannot convey.
  full       every tile as its own z18 rectangle (~25 MB). Not shipped, since
             it is derivable from x18/y18. Set WRITE_FULL to generate it.
"""

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "release" / "tile_index.csv.gz"
OUT_COVERAGE = REPO / "release" / "d4_coverage.gpkg"
OUT_FULL = REPO / "outputs" / "d4_tiles_full.gpkg"

USED_ONLY = True     # restrict to the 156,790 tiles actually trained on
WRITE_FULL = False   # also emit the per-tile grid


def bbox(x, y, z):
    """WGS84 bounds of a web-Mercator tile."""
    n = 2 ** z
    lon0 = x / n * 360.0 - 180.0
    lon1 = (x + 1) / n * 360.0 - 180.0
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return box(lon0, lat0, lon1, lat1)


def main():
    df = pd.read_csv(INDEX)
    print(f"{INDEX.name}: {len(df):,} rows")
    if USED_ONLY:
        df = df[df["used"]].reset_index(drop=True)
        print(f"  used only: {len(df):,}")

    # -- coverage at zoom 11 (z18 >> 7) ---------------------------------------
    x11 = df["x18"].values >> 7
    y11 = df["y18"].values >> 7
    cells = pd.DataFrame({"x11": x11, "y11": y11})
    agg = (cells.groupby(["x11", "y11"]).size()
           .reset_index(name="n_tiles"))
    print(f"\ncoverage: {len(agg):,} zoom-11 cells, "
          f"{agg.n_tiles.min()}-{agg.n_tiles.max()} tiles each")

    cov = gpd.GeoDataFrame(
        agg,
        geometry=[bbox(int(a), int(b), 11)
                  for a, b in zip(agg.x11, agg.y11)],
        crs="EPSG:4326",
    )
    if OUT_COVERAGE.exists():
        OUT_COVERAGE.unlink()
    cov.to_file(OUT_COVERAGE, layer="d4_coverage", driver="GPKG")
    print(f"wrote {OUT_COVERAGE.name}  "
          f"{OUT_COVERAGE.stat().st_size / 1024:.0f} KB")

    if not WRITE_FULL:
        print("\nper-tile grid not written (WRITE_FULL = False)")
        return

    # -- full grid ------------------------------------------------------------
    full = gpd.GeoDataFrame(
        df[["tile_id", "x18", "y18", "pos_frac", "used"]],
        geometry=[bbox(int(a), int(b), 18)
                  for a, b in zip(df.x18, df.y18)],
        crs="EPSG:4326",
    )
    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FULL.exists():
        OUT_FULL.unlink()
    full.to_file(OUT_FULL, layer="d4_tiles", driver="GPKG")
    print(f"wrote {OUT_FULL.name}  "
          f"{OUT_FULL.stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
