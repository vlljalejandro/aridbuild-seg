"""
Convert the uint16 instance label maps from watershed.py into georeferenced
shapely polygons: trace contours, transform pixel coordinates to WGS84 using
the tile bbox, simplify, filter by minimum area, and return dicts ready for a
GeoDataFrame.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import mapping, shape
from shapely.validation import make_valid

try:
    import rasterio.features
    import rasterio.errors
    from rasterio.errors import NotGeoreferencedWarning
    import warnings
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    raise ImportError(
        "rasterio is required for polygonization. "
        "Install with: pip install rasterio"
    )

CROP_PX = 512   # tile size in pixels


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transform
# ─────────────────────────────────────────────────────────────────────────────

def _pixel_to_geo_transform(
    bbox: Tuple[float, float, float, float],
    tile_size: int = CROP_PX,
) -> List[float]:
    """
    Coefficients for shapely affine_transform, pixel -> WGS84.

    [a, b, d, e, xoff, yoff] gives x' = a*x + b*y + xoff and
    y' = d*x + e*y + yoff. The pixel origin is top-left and geographic y
    increases northward, so y is flipped.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_scale = (lon_max - lon_min) / tile_size      # degrees per pixel (x)
    lat_scale = (lat_max - lat_min) / tile_size      # degrees per pixel (y, positive)
    return [
        lon_scale,   0.0,         # a, b
        0.0,         -lat_scale,  # d, e  (negative: pixel y → south)
        lon_min,     lat_max,     # xoff, yoff (origin = top-left → NW corner)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Area approximation
# ─────────────────────────────────────────────────────────────────────────────

def _approx_area_m2(polygon, lat_centre: float) -> float:
    """
    Polygon area in square metres, flat-earth approximation:
    1 deg lon ~ 111,320 * cos(lat) m, 1 deg lat ~ 110,540 m. Error is under 1%
    at tile scale.
    """
    import math
    lat_rad   = math.radians(lat_centre)
    m_per_deg_lon = 111_320.0 * math.cos(lat_rad)
    m_per_deg_lat = 110_540.0

    # Shapely area is in degrees² — convert
    return polygon.area * m_per_deg_lon * m_per_deg_lat


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def instance_labels_to_polygons(
    instance_labels: np.ndarray,           # (H, W) uint16
    bbox:            Tuple[float, float, float, float],  # (lon_min, lat_min, lon_max, lat_max)
    filename:        str           = "",
    min_area_m2:     float         = 10.0,  # drop buildings smaller than this
    simplify_tol:    float         = 5e-6,  # degrees — ~0.5m at Saudi latitudes
    tile_size:       int           = CROP_PX,
) -> List[Dict]:
    """
    Convert a watershed instance label map to georeferenced polygon dicts with
    keys: geometry (WGS84 Polygon), filename, label_id, area_m2, and edge (a
    comma-separated list of tile edges the polygon touches, for edge merging).

    simplify_tol is a Douglas-Peucker tolerance in degrees.
    """
    if instance_labels.max() == 0:
        return []

    lon_min, lat_min, lon_max, lat_max = bbox
    lat_centre = (lat_min + lat_max) / 2.0
    transform  = _pixel_to_geo_transform(bbox, tile_size)

    # 5 px (~1.5 m): masks reach the edge, this covers simplification pull-in
    lon_per_px = (lon_max - lon_min) / tile_size
    lat_per_px = (lat_max - lat_min) / tile_size
    edge_tol_lon = 5 * lon_per_px
    edge_tol_lat = 5 * lat_per_px

    results = []

    import os, sys
    os.environ.setdefault("GDAL_PAM_ENABLED", "NO")

    # GDAL writes deprecation warnings straight to stderr, bypassing the
    # warnings module, so stderr is redirected for this one call.
    _devnull = open(os.devnull, "w")
    _old_stderr, sys.stderr = sys.stderr, _devnull
    try:
        shapes = list(rasterio.features.shapes(
            instance_labels.astype(np.int32),
            mask=(instance_labels > 0).astype(np.uint8),
            connectivity=8,
        ))
    finally:
        sys.stderr = _old_stderr
        _devnull.close()

    for geom_dict, label_id in shapes:
        if label_id == 0:
            continue

        poly = shape(geom_dict)           # pixel coordinates
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            continue

        poly_geo = affine_transform(poly, transform)
        if not poly_geo.is_valid:
            poly_geo = make_valid(poly_geo)

        poly_geo = poly_geo.simplify(simplify_tol, preserve_topology=True)

        if poly_geo.is_empty:
            continue

        area_m2 = _approx_area_m2(poly_geo, lat_centre)
        if area_m2 < min_area_m2:
            continue

        # which tile edges this polygon touches, for edge merging
        bounds  = poly_geo.bounds   # (minx, miny, maxx, maxy) in lon/lat
        touched = []
        if bounds[0] <= lon_min + edge_tol_lon:  touched.append("left")
        if bounds[2] >= lon_max - edge_tol_lon:  touched.append("right")
        if bounds[3] >= lat_max - edge_tol_lat:  touched.append("top")
        if bounds[1] <= lat_min + edge_tol_lat:  touched.append("bottom")

        results.append({
            "geometry":  poly_geo,
            "filename":  filename,
            "label_id":  int(label_id),
            "area_m2":   round(area_m2, 1),
            "edge":      ",".join(touched),
        })

    return results