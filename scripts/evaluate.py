"""
Test-set evaluation during development. For reproducing the paper from the
released checkpoints, use scripts/evaluate_release.py instead.

Runs inference on the held-out D3 test split and reports IoU / F1 / precision /
recall, plus a CSV.

The paper reports the fixed 0.50 threshold. The threshold sweep is diagnostic
only: it selects on the test split, so the swept optimum is not a reportable
result.

Usage
-----
python scripts/evaluate.py --cfg config/benchmark.yaml --all
python scripts/evaluate.py --cfg config/benchmark.yaml \\
    --checkpoint outputs/checkpoints/b1-supervised/best.pt --name "B1 Src"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

ROOT = next(
    p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    if (p / "src").is_dir()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models              import build_model
from src.data.dataloader     import get_finetune_loaders
from src.losses.combined     import compute_distance_target
from src.metrics             import SegmentationMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list) -> dict:
    def _cast(v):
        if v.lower() in ("true", "false"): return v.lower() == "true"
        try: return int(v)
        except ValueError: pass
        try: return float(v)
        except ValueError: pass
        return v
    for item in overrides:
        key, _, val = item.partition("=")
        parts = key.strip().split(".")
        node  = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _cast(val.strip())
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    """Build from the checkpoint's own stored config. Checkpoints predating the
    SwinTFPN rename say "Mask2FormerSwinT"; the registry accepts both."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = dict(ckpt.get("cfg", {}).get("model", {}))
    if "name" not in model_cfg:
        raise KeyError(f"{checkpoint} has no cfg.model.name")
    model_cfg["pretrained"] = False

    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print(f"  Loaded {model_cfg['name']} — epoch={ckpt.get('epoch', '?')}  "
          f"val_iou={ckpt.get('best_iou', float('nan')):.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:      torch.nn.Module,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    threshold:  float = 0.5,
) -> dict:
    """Metrics at `threshold`, plus the best threshold found by sweeping
    [0.1, 0.9]. The sweep is diagnostic only."""
    from src.losses import CombinedLoss
    criterion = CombinedLoss()

    metrics = SegmentationMetrics(threshold=threshold, device=device)
    all_logits, all_masks = [], []
    total_loss = 0.0

    for imgs, masks in loader:
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        dist_targets = compute_distance_target(masks)

        seg_logits, dist_logits = model(imgs)
        _, loss_dict = criterion(seg_logits, dist_logits, masks, dist_targets)

        metrics.update(seg_logits, masks)
        all_logits.append(seg_logits.cpu())
        all_masks.append(masks.cpu())
        total_loss += loss_dict["total"]

    scores = metrics.compute()

    cat_logits = torch.cat(all_logits)
    cat_masks  = torch.cat(all_masks)
    thresholds = np.arange(0.1, 0.95, 0.05).round(2)
    sweep      = metrics.compute_threshold_sweep(cat_logits, cat_masks, thresholds)
    best_t     = thresholds[int(np.argmax([sweep[t]["iou"] for t in thresholds]))]
    best_scores = sweep[best_t]

    return {
        "threshold":          threshold,
        "iou":                scores["iou"],
        "f1":                 scores["f1"],
        "precision":          scores["precision"],
        "recall":             scores["recall"],
        "loss":               total_loss / len(loader),
        "best_threshold":     float(best_t),
        "best_iou":           best_scores["iou"],
        "best_f1":            best_scores["f1"],
        "best_precision":     best_scores["precision"],
        "best_recall":        best_scores["recall"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 110)
    print(f"{'Model':<30}  {'Threshold':>9}  {'IoU':>7}  {'F1':>7}  {'Precision':>9}  {'Recall':>7}  "
          f"{'Best_t':>6}  {'Best_IoU':>8}  {'Best_F1':>7}")
    print("=" * 110)
    for r in results:
        print(
            f"{r['name']:<30}  "
            f"{r['threshold']:>9.2f}  "
            f"{r['iou']:>7.4f}  "
            f"{r['f1']:>7.4f}  "
            f"{r['precision']:>9.4f}  "
            f"{r['recall']:>7.4f}  "
            f"{r['best_threshold']:>6.2f}  "
            f"{r['best_iou']:>8.4f}  "
            f"{r['best_f1']:>7.4f}"
        )
    print("=" * 110)


def save_csv(results: list[dict], out_path: Path) -> None:
    import csv
    keys = ["name", "threshold", "iou", "f1", "precision", "recall",
            "best_threshold", "best_iou", "best_f1", "best_precision", "best_recall", "loss"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

# The fifteen reported models, under their release names.
DEFAULT_MODELS = [
    {"name": f"{arch} {label}", "checkpoint": f"data/checkpoints/{key}.pt"}
    for arch, key_prefix in (
        ("B1 U-Net/ResNet-50", "b1"),
        ("B2 SegFormer-B2", "b2"),
        ("B3 Swin-T/FPN", "b3"),
    )
    for label, key in (
        ("Src", f"{key_prefix}-src"),
        ("Tgt", f"{key_prefix}-tgt"),
        ("Src->Tgt", f"{key_prefix}-src-tgt"),
        ("Src+PL", f"{key_prefix}-src-pl"),
        ("Src+PL->Tgt", f"{key_prefix}-src-pl-tgt"),
    )
]


def parse_args():
    p = argparse.ArgumentParser(description="Test-set evaluation across models.")
    p.add_argument("--cfg",        required=True,  help="Config supplying the data/ and loader/ blocks.")
    p.add_argument("--checkpoint", default=None,   help="Single checkpoint to evaluate.")
    p.add_argument("--name",       default=None,   help="Display name for the model.")
    p.add_argument("--all",        action="store_true",
                   help="Evaluate everything in DEFAULT_MODELS.")
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="Binary threshold. The paper reports 0.5.")
    p.add_argument("--out_csv",    default="outputs/results/evaluation.csv",
                   help="Output CSV path.")
    p.add_argument("--set",        nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = apply_overrides(load_cfg(args.cfg), args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device={device}  threshold={args.threshold}")

    # ── data: always the D3 test split ───────────────────────────────────────
    loader_cfg = {**cfg["data"], **cfg["loader"]}
    _, _, test_loader = get_finetune_loaders(loader_cfg, seed=cfg.get("seed", 42))
    print(f"[evaluate] Test samples: {len(test_loader.dataset):,}")

    # ── model list ───────────────────────────────────────────────────────────
    if args.all:
        to_eval = DEFAULT_MODELS
    elif args.checkpoint:
        to_eval = [{"name": args.name or Path(args.checkpoint).parent.name,
                    "checkpoint": args.checkpoint}]
    else:
        print("Specify --checkpoint <path> or --all")
        return

    # ── evaluate ──────────────────────────────────────────────────────────────
    results = []
    for entry in to_eval:
        ckpt = Path(entry["checkpoint"])
        if not ckpt.exists():
            print(f"\n[skip] {entry['name']} — checkpoint not found: {ckpt}")
            continue

        print(f"\n[evaluate] {entry['name']}")
        model  = load_model(ckpt, device)
        scores = evaluate(model, test_loader, device, threshold=args.threshold)
        scores["name"] = entry["name"]
        results.append(scores)

        print(f"  IoU={scores['iou']:.4f}  F1={scores['f1']:.4f}  "
              f"Prec={scores['precision']:.4f}  Rec={scores['recall']:.4f}  "
              f"[t={args.threshold}]")
        print(f"  Best t={scores['best_threshold']}  "
              f"IoU={scores['best_iou']:.4f}  F1={scores['best_f1']:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── report ────────────────────────────────────────────────────────────────
    if results:
        print_table(results)
        save_csv(results, Path(args.out_csv))


if __name__ == "__main__":
    main()
