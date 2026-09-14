"""
Pseudo-label generation over the unlabeled target pool (D4).

The teacher's probability map is thresholded low, split into instances by a
probability-seeded watershed, polygonised with Douglas-Peucker simplification
and buffer smoothing, then rasterised back to a binary {0,1} pixel mask. The
polygon round-trip removes staircase artifacts and produces labels with no
ignore region, so the student trains with the same loss configuration as the
supervised stages.

Tile selection
--------------
Which tiles get labeled is read from the released tile index, not re-derived.
The original cut cannot be reproduced (see the note in ``run``).

Every tile of the labeled target set (D3) is excluded, including its training
split, or held-out evaluation tiles could enter the student's training corpus.
The index was built after exclusion, so the check in ``run`` is an assertion
rather than a filter.

Usage
-----
    python scripts/pseudo_label.py --cfg config/pseudo_label.yaml

Masks already on disk are skipped, so an interrupted run restarts with the
same command.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

ROOT = next(
    p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    if (p / "src").is_dir()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.data.transforms import get_val_transforms
from src.postprocess import apply_watershed
from src.postprocess.polygonize import instance_labels_to_polygons


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """
    Apply dot-notation KEY=VALUE overrides.

    Scalars only — bool, int, float, str. List-valued fields must be set in
    the YAML; there is no syntax for null.
    """
    def _cast(v: str):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        if v.lstrip("-").isdigit():
            return int(v)
        try:
            f = float(v)
            if "." in v or "e" in v.lower():
                return f
        except ValueError:
            pass
        return v

    for item in overrides:
        key, _, val = item.partition("=")
        parts = key.strip().split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _cast(val.strip())
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# LMDB dataset
# ─────────────────────────────────────────────────────────────────────────────

class LMDBTileDataset(Dataset):
    """Reads JPEG-encoded 512x512 tiles from LMDB, keyed by ``{x18}_{y18}``."""

    def __init__(self, lmdb_path: str, keys: list, transform):
        self.lmdb_path = lmdb_path
        self.keys = keys
        self.transform = transform
        self._env = None

    def _env_(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False,
                readahead=False, meminit=False,
            )
        return self._env

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        x18, y18 = self.keys[idx]
        key = f"{x18}_{y18}".encode()
        with self._env_().begin(write=False) as txn:
            val = txn.get(key)
        if val is None:
            raise KeyError(f"Tile {x18}_{y18} not present in {self.lmdb_path}")
        img = np.array(Image.open(io.BytesIO(val)).convert("RGB"))
        return self.transform(image=img)["image"], x18, y18


def _collate(batch):
    imgs, xs, ys = zip(*batch)
    return torch.stack(imgs), list(xs), list(ys)


# ─────────────────────────────────────────────────────────────────────────────
# Tile exclusion
# ─────────────────────────────────────────────────────────────────────────────

def _load_excluded_tiles(gpkg_path: str, layer: str = "tiles") -> set:
    """
    Return {(x18, y18)} for every labeled target tile.

    All splits are excluded, not just val and test. Training tiles carry no
    leakage risk, but excluding them keeps the pseudo-label corpus disjoint
    from the labeled corpus, which makes the Src+PL configuration exactly
    "no labeled target tile" rather than "no labeled target annotation".
    """
    import geopandas as gpd

    tiles = gpd.read_file(gpkg_path, layer=layer)
    for col in ("x18", "y18"):
        if col not in tiles.columns:
            raise KeyError(
                f"Layer '{layer}' of {gpkg_path} has no '{col}' column; "
                f"found {list(tiles.columns)}."
            )
    return set(zip(tiles["x18"].astype(int), tiles["y18"].astype(int)))


# ─────────────────────────────────────────────────────────────────────────────
# Georeferencing
# ─────────────────────────────────────────────────────────────────────────────

def _z18_to_bbox(x18: int, y18: int):
    """Return (lon_min, lat_min, lon_max, lat_max) in WGS84 for a z18 tile."""
    n = 2 ** 18
    lon_min = x18 / n * 360.0 - 180.0
    lon_max = (x18 + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y18 / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y18 + 1) / n))))
    return lon_min, lat_min, lon_max, lat_max


# ─────────────────────────────────────────────────────────────────────────────
# Teacher
# ─────────────────────────────────────────────────────────────────────────────

def load_teacher(checkpoint: str, device: torch.device) -> torch.nn.Module:
    """
    Load a teacher from a training checkpoint.

    The architecture is read from the checkpoint's stored config, so the
    teacher cannot silently be constructed as the wrong model. Checkpoints
    predating the SwinTFPN rename record "Mask2FormerSwinT"; the registry
    accepts both.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = dict(ckpt.get("cfg", {}).get("model", {}))
    if "name" not in model_cfg:
        raise KeyError(
            f"Checkpoint {checkpoint} has no cfg.model.name; cannot determine "
            "the architecture. Pass a checkpoint written by scripts/train.py."
        )
    model_cfg["pretrained"] = False

    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print(
        f"[pseudo_label] Teacher: {model_cfg['name']}  "
        f"epoch={ckpt.get('epoch', '?')}  best_iou={ckpt.get('best_iou', float('nan')):.4f}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Mask generation
# ─────────────────────────────────────────────────────────────────────────────

def _poly_to_pixels(polygon, bbox: tuple, img_size: int = 512) -> list:
    """Convert a shapely Polygon from WGS84 degrees to pixel coordinates."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    coords = []
    for lon, lat in polygon.exterior.coords:
        px = (lon - lon_min) / lon_range * img_size
        py = (lat_max - lat) / lat_range * img_size
        coords.append((px, py))
    return coords


def make_mask_polygon(
    prob_map: np.ndarray,
    x18: int,
    y18: int,
    seg_threshold: float = 0.20,
    min_distance: int = 10,
    min_area_px: int = 30,
    min_area_m2: float = 5.0,
    simplify_tol: float = 3.0e-6,
    post_buffer_smooth: float = 3.0e-6,
    img_size: int = 512,
) -> np.ndarray:
    """
    Polygon-smoothed pseudo-label generation.

    1. Low-threshold binary mask — captures the full footprint including edges
    2. Watershed seeded from local maxima of the probability map -> instances
    3. Polygonise with Douglas-Peucker simplification
    4. Buffer in/out smoothing — rounds staircase artifacts
    5. Rasterise back to a binary pixel mask

    Returns a uint8 mask: 0 = background, 1 = building. No ignore region.
    """
    from shapely.validation import make_valid

    bbox = _z18_to_bbox(x18, y18)
    seg_mask = (prob_map >= seg_threshold).astype(np.uint8)

    empty = np.zeros((img_size, img_size), dtype=np.uint8)
    if seg_mask.sum() == 0:
        return empty

    inst_labels = apply_watershed(
        seg_mask, prob_map,
        min_distance=min_distance,
        min_area_px=min_area_px,
    )
    if inst_labels.max() == 0:
        return empty

    polys = instance_labels_to_polygons(
        inst_labels, bbox,
        filename=f"{x18}_{y18}.JPEG",
        min_area_m2=min_area_m2,
        simplify_tol=simplify_tol,
    )
    if not polys:
        return empty

    smoothed = []
    for p in polys:
        geom = p["geometry"]
        if post_buffer_smooth > 0:
            geom = geom.buffer(post_buffer_smooth).buffer(-post_buffer_smooth)
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            smoothed.append(geom)
        elif geom.geom_type == "MultiPolygon":
            smoothed.extend(list(geom.geoms))

    canvas = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(canvas)
    for geom in smoothed:
        pixels = _poly_to_pixels(geom, bbox, img_size)
        if len(pixels) >= 3:
            draw.polygon(pixels, fill=1)

    return np.array(canvas, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tile_index = Path(cfg["tile_index"])
    lmdb_path = cfg["lmdb_path"]
    out_dir = Path(cfg["out_dir"])
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    batch_size = cfg.get("batch_size", 16)

    print(f"[pseudo_label] Teacher : {cfg['checkpoint']}")

    # ── candidate tiles ───────────────────────────────────────────────────────
    # The tile index IS the selection, not an input to re-derive. The original
    # run took the top 200,000 of 1,835,221 candidates scoring >= 0.95, but the
    # classifier saturates (all 200,000 scored exactly 1.0) and the sort was not
    # stable, so re-running it would select a different 200,000.
    print(f"[pseudo_label] Loading tile index: {tile_index}")
    df = pd.read_csv(tile_index)
    for col in ("x18", "y18"):
        if col not in df.columns:
            raise KeyError(f"{tile_index} has no '{col}' column; "
                           f"found {list(df.columns)}")
    print(f"[pseudo_label] Index carries {len(df):,} labeled tiles")

    # ── exclude the labeled target set ────────────────────────────────────────
    # Normally drops nothing (the index was built after exclusion). Kept as an
    # assertion: a D3 tile here would leak evaluation data into training.
    excluded = _load_excluded_tiles(
        cfg["exclude_tiles_gpkg"], cfg.get("exclude_tiles_layer", "tiles")
    )
    keep = ~pd.Series(
        [(x, y) in excluded for x, y in zip(df["x18"], df["y18"])], index=df.index
    )
    n_dropped = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    if n_dropped:
        print(f"[pseudo_label] WARNING: dropped {n_dropped:,} labeled target "
              f"tiles that should not have been in the index")
    print(f"[pseudo_label] {len(df):,} tiles to label "
          f"({len(excluded):,} in the exclusion set)")

    overlap = excluded & set(zip(df["x18"].astype(int), df["y18"].astype(int)))
    assert not overlap, (
        f"{len(overlap)} labeled target tiles survived exclusion, e.g. "
        f"{sorted(overlap)[:5]}. Refusing to generate pseudo-labels."
    )

    # ── resume ────────────────────────────────────────────────────────────────
    todo = [
        (int(r.x18), int(r.y18))
        for r in df.itertuples()
        if not (mask_dir / f"{int(r.x18)}_{int(r.y18)}.png").exists()
    ]
    n_existing = len(df) - len(todo)
    print(f"[pseudo_label] {n_existing:,} already done — {len(todo):,} remaining")

    # Previous rows are carried forward so a resumed run does not truncate the
    # manifest.
    prior_rows = []
    if manifest_path.exists() and n_existing:
        prior = pd.read_csv(manifest_path)
        done = {(int(x), int(y)) for x, y in zip(prior["x18"], prior["y18"])}
        prior_rows = prior[
            [(x, y) not in set(todo) for x, y in zip(prior["x18"], prior["y18"])]
        ].to_dict("records")
        print(f"[pseudo_label] Carried {len(prior_rows):,} rows from existing manifest")
        missing = n_existing - len(done & set(zip(df["x18"], df["y18"])))
        if missing > 0:
            print(
                f"[pseudo_label] Warning: {missing:,} masks on disk have no "
                "manifest row and will be absent from the output manifest. "
                "Delete the mask directory to regenerate cleanly."
            )

    if not todo:
        print("[pseudo_label] Nothing to do.")
        return

    # ── model + transform ─────────────────────────────────────────────────────
    model = load_teacher(cfg["checkpoint"], device)
    transform = get_val_transforms(
        mean=tuple(cfg.get("img_mean", [0.2186, 0.2315, 0.4944])),
        std=tuple(cfg.get("img_std", [0.2308, 0.2263, 0.2268])),
    )

    loader = DataLoader(
        LMDBTileDataset(lmdb_path, todo, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,       # LMDB env is not fork-safe here; see README
        collate_fn=_collate,
    )

    amp_ctx = (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )

    params = dict(
        seg_threshold=cfg.get("seg_threshold", 0.20),
        min_distance=cfg.get("min_distance", 10),
        min_area_px=cfg.get("min_area_px", 30),
        min_area_m2=cfg.get("min_area_m2", 5.0),
        simplify_tol=cfg.get("simplify_tol", 3.0e-6),
        post_buffer_smooth=cfg.get("post_buffer_smooth", 3.0e-6),
    )

    # ── inference ─────────────────────────────────────────────────────────────
    rows = list(prior_rows)
    n_done = 0
    t_start = time.perf_counter()

    for imgs, x18s, y18s in loader:
        imgs = imgs.to(device, non_blocking=True)
        with amp_ctx:
            seg_logits, _ = model(imgs)
        probs = torch.sigmoid(seg_logits.float()).cpu().numpy()   # (B,1,H,W)

        for i in range(len(x18s)):
            x18, y18 = int(x18s[i]), int(y18s[i])
            mask = make_mask_polygon(probs[i, 0], x18, y18, **params)

            Image.fromarray(mask).save(mask_dir / f"{x18}_{y18}.png")
            rows.append({
                "x18": x18,
                "y18": y18,
                "pos_frac": round(float(mask.mean()), 4),
            })
            n_done += 1

        if n_done % 1000 == 0 or n_done == len(todo):
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / max(n_done, 1)) * (len(todo) - n_done)
            print(
                f"  {n_done:>7,}/{len(todo):,}  "
                f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min"
            )

    # ── manifest ──────────────────────────────────────────────────────────────
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)

    elapsed = time.perf_counter() - t_start
    min_pos = cfg.get("report_min_pos_frac", 0.02)
    print(
        f"\n[pseudo_label] Done."
        f"\n  Masks written this run : {n_done:,}"
        f"\n  Manifest rows          : {len(manifest):,}"
        f"\n  Mean pos_frac          : {manifest['pos_frac'].mean():.3f}"
        f"\n  Above min_pos_frac     : {int((manifest['pos_frac'] >= min_pos).sum()):,}"
        f"  (>= {min_pos}, the training-time filter)"
        f"\n  Output                 : {out_dir}"
        f"\n  Elapsed                : {elapsed/60:.1f} min"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate pseudo-labels over D4.")
    p.add_argument("--cfg", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def main():
    args = parse_args()
    run(apply_overrides(load_cfg(args.cfg), args.set))


if __name__ == "__main__":
    main()
