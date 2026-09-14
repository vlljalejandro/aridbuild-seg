"""
Fast checks that run without imagery, weights or a GPU.

    pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── post-processing ──────────────────────────────────────────────────────────
#
# apply_watershed seeds from local maxima of `dist_map`, which in production is
# the probability map. Fixtures use smooth bumps rather than flat plateaus: a
# flat region has no local-maximum structure, so a constant blob seeds on the
# min_distance grid and over-segments, testing the grid rather than the
# algorithm.


def _bump(size, cy, cx, r):
    """Smooth radial peak at (cy, cx), falling to zero by radius r."""
    y, x = np.ogrid[:size, :size]
    d = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return np.clip(1.0 - d / r, 0.0, 1.0).astype(np.float32)


def test_watershed_splits_touching_buildings():
    """
    Two footprints sharing an edge must yield two instances.

    The regression test that matters here: a basin merge is invisible in pixel
    IoU, the only metric reported, so nothing else would catch it.
    """
    from src.postprocess import apply_watershed

    size = 128
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[40:80, 24:64] = 1          # two 40x40 footprints, touching
    mask[40:80, 64:104] = 1
    dist = np.maximum(_bump(size, 60, 44, 22), _bump(size, 60, 84, 22))

    labels = apply_watershed(mask > 0, dist, min_distance=10,
                             min_area_px=30, seed_thresh=0.20)
    n = len(set(np.unique(labels)) - {0})
    assert n == 2, f"expected 2 instances, got {n}"


def test_watershed_keeps_single_building_single():
    from src.postprocess import apply_watershed

    size = 128
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[40:90, 40:90] = 1
    dist = _bump(size, 65, 65, 28)

    labels = apply_watershed(mask > 0, dist, min_distance=10,
                             min_area_px=30, seed_thresh=0.20)
    n = len(set(np.unique(labels)) - {0})
    assert n == 1, f"expected 1 instance, got {n}"


def test_watershed_drops_tiny_instances():
    """Nothing below min_area_px survives."""
    from src.postprocess import apply_watershed

    size = 128
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[10:13, 10:13] = 1          # 9 px
    dist = _bump(size, 11, 11, 3)

    labels = apply_watershed(mask > 0, dist, min_distance=10,
                             min_area_px=30, seed_thresh=0.20)
    assert len(set(np.unique(labels)) - {0}) == 0


# ── cached counts: the statistical path ──────────────────────────────────────

def test_micro_iou_is_ratio_of_sums():
    from src.release.counts import micro

    counts = {"inter": np.array([10, 20]), "union": np.array([20, 20]),
              "tp": np.array([10, 20]), "pp": np.array([15, 20]),
              "ap": np.array([15, 25])}
    m = micro(counts)
    assert m["iou"] == pytest.approx(30 / 40)
    assert m["precision"] == pytest.approx(30 / 35)
    assert m["recall"] == pytest.approx(30 / 40)


def test_bootstrap_indices_are_deterministic_and_shared():
    """One index matrix, reused for every model: that is what makes the
    intervals paired. Fresh indices per model would inflate all of them."""
    from src.release.counts import replicate_indices

    a = replicate_indices(226, n_boot=50, seed=42)
    b = replicate_indices(226, n_boot=50, seed=42)
    assert a.shape == (50, 226)
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < 226


def test_paired_delta_of_a_model_with_itself_is_zero():
    from src.release.counts import paired_delta, replicate_indices

    rng = np.random.default_rng(0)
    c = {"inter": rng.integers(0, 100, 226), "union": rng.integers(100, 200, 226),
         "tp": rng.integers(0, 100, 226), "pp": rng.integers(100, 200, 226),
         "ap": rng.integers(100, 200, 226)}
    data = {"m": c}
    idx = replicate_indices(226, n_boot=200, seed=1)
    r = paired_delta(data, "m", "m", idx)
    assert r["delta"] == pytest.approx(0.0)
    assert r["lo"] == pytest.approx(0.0) and r["hi"] == pytest.approx(0.0)


# ── released files, when present ─────────────────────────────────────────────

NPZ = ROOT / "release" / "per_tile_counts.npz"
SPLITS = ROOT / "release" / "splits.csv"
INDEX = ROOT / "release" / "tile_index.csv.gz"


@pytest.mark.skipif(not NPZ.exists(), reason="per_tile_counts.npz not present")
def test_cached_counts_cover_all_fifteen_models():
    from src.release.counts import load

    data, stems = load(NPZ)
    assert len(data) == 15, sorted(data)
    for label, c in data.items():
        assert len(c["union"]) == 226, label
        assert (c["inter"] <= c["union"]).all(), label


@pytest.mark.skipif(not SPLITS.exists(), reason="splits.csv not present")
def test_split_sizes():
    import csv

    with open(SPLITS, newline="") as f:
        rows = list(csv.DictReader(f))
    counts = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    assert counts == {"train": 1810, "val": 226, "test": 226}


@pytest.mark.skipif(not INDEX.exists(), reason="tile_index.csv.gz not present")
def test_tile_index_matches_the_paper():
    import pandas as pd

    d = pd.read_csv(INDEX)
    assert len(d) == 200_000
    assert int(d["used"].sum()) == 156_790
    assert d.loc[d["used"], "pos_frac"].mean() == pytest.approx(0.1558, abs=5e-4)


# ── models ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,encoder", [
    ("UNetResNet50", "resnet50.a1_in1k"),
    ("SegFormerB2", "nvidia/mit-b2"),
    ("SwinTFPN", "swin_tiny_patch4_window7_224"),
    ("Mask2FormerSwinT", "swin_tiny_patch4_window7_224"),   # legacy alias
])
def test_model_builds_and_returns_two_heads(name, encoder):
    """
    forward() returns (seg_logit, dist_logit). Reading element 1 as the
    segmentation output scores about 0.09 IoU, so the contract is worth
    pinning.
    """
    import torch
    from src.models import build_model

    try:
        model = build_model({"name": name, "encoder_name": encoder,
                             "pretrained": False, "fpn_dim": 256,
                             "embed_dim": 256,
                             "decoder_channels": [256, 128, 64, 32, 16]})
    except Exception as e:                      # offline CI, no HF config
        pytest.skip(f"{name} could not be built here: {e}")

    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 512, 512))

    assert isinstance(out, (tuple, list)) and len(out) == 2, type(out)
    for t in out:
        assert t.shape == (1, 1, 512, 512), t.shape
