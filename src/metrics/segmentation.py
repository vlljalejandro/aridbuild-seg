"""
Pixel-level segmentation metrics.

Counts are summed across the epoch and the metrics computed once at the end,
rather than averaging per-batch scores: building coverage varies widely
between tiles, and averaging batch scores over-weights the sparse ones.

    metrics = SegmentationMetrics(threshold=0.5, device=device)
    metrics.reset()
    for imgs, masks in val_loader:
        seg_logits, _ = model(imgs)
        metrics.update(seg_logits, masks)
    scores = metrics.compute()   # iou, f1, precision, recall, acc

Only the segmentation head is scored. The auxiliary distance head is not
evaluated anywhere, and every number in the paper comes from the segmentation
head alone.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SegmentationMetrics:
    """
    Accumulates TP / FP / FN / TN across batches and computes IoU, F1,
    precision, recall and accuracy at epoch end.

    Counters are int64 on `device`, which avoids floating-point accumulation
    error over millions of pixels. Match `device` to the model to avoid
    host-device transfers in the loop.
    """

    def __init__(
        self,
        threshold: float        = 0.5,
        device:    torch.device = torch.device("cpu"),
    ) -> None:
        self.threshold = threshold
        self.device    = device
        self._tp = torch.tensor(0, dtype=torch.int64, device=device)
        self._fp = torch.tensor(0, dtype=torch.int64, device=device)
        self._fn = torch.tensor(0, dtype=torch.int64, device=device)
        self._tn = torch.tensor(0, dtype=torch.int64, device=device)

    # ── state management ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Zero all accumulators. Call at the start of each epoch."""
        self._tp.zero_()
        self._fp.zero_()
        self._fn.zero_()
        self._tn.zero_()

    # ── batch update ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def update(
        self,
        logits:  torch.Tensor,  # (B, 1, H, W)  raw seg logits
        targets: torch.Tensor,  # (B, 1, H, W)  binary float {0, 1} or 255 (ignore)
    ) -> None:
        """
        Threshold the logits (sigmoid applied internally) and accumulate pixel
        counts. Targets of 255 are treated as ignore and excluded.
        """
        preds = (torch.sigmoid(logits) >= self.threshold).long()
        tgts  = targets.long()
        valid = (tgts != 255)   # exclude ignore pixels
        preds = preds[valid]
        tgts  = tgts[valid]

        self._tp += (preds *       tgts ).sum()
        self._fp += (preds * (1 - tgts) ).sum()
        self._fn += ((1 - preds) * tgts ).sum()
        self._tn += ((1 - preds) * (1 - tgts)).sum()

    # ── epoch-level computation ───────────────────────────────────────────────

    def compute(self, prefix: str = "") -> Dict[str, float]:
        """
        Metrics from the accumulated counts, as {prefix}iou / f1 / precision /
        recall / acc. `prefix` ("val/", "test/") keeps W&B keys from colliding.
        """
        tp = self._tp.item()
        fp = self._fp.item()
        fn = self._fn.item()
        tn = self._tn.item()

        eps = 1e-7   # guards an all-negative batch

        iou       = tp / (tp + fp + fn + eps)
        f1        = (2 * tp) / (2 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        acc       = (tp + tn) / (tp + fp + fn + tn + eps)

        return {
            f"{prefix}iou":       round(iou,       4),
            f"{prefix}f1":        round(f1,         4),
            f"{prefix}precision": round(precision,  4),
            f"{prefix}recall":    round(recall,     4),
            f"{prefix}acc":       round(acc,        4),
        }

    # ── convenience ───────────────────────────────────────────────────────────

    def compute_threshold_sweep(
        self,
        logits:   torch.Tensor,
        targets:  torch.Tensor,
        thresholds: tuple = (0.3, 0.4, 0.5, 0.6, 0.7),
    ) -> Dict[float, Dict[str, float]]:
        """
        Metrics at several thresholds over one tensor, so the threshold can be
        explored without re-running the val loop. Returns
        {threshold: {"iou": ..., "f1": ..., ...}}.

        Diagnostic only: selecting a threshold on the split you report is
        selection on the test set. The paper reports the fixed 0.50.
        """
        results = {}
        for t in thresholds:
            tmp = SegmentationMetrics(threshold=t, device=logits.device)
            tmp.update(logits, targets)
            results[t] = tmp.compute()
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 4, 512, 512

    # perfect predictions: IoU and F1 should be 1.0
    masks       = (torch.rand(B, 1, H, W) > 0.8).float()
    perfect_log = masks * 10 - (1 - masks) * 10   # large pos logit where mask=1

    m = SegmentationMetrics(threshold=0.5)
    m.update(perfect_log, masks)
    scores = m.compute()
    print("Perfect predictions:")
    for k, v in scores.items():
        print(f"  {k:<12s}: {v:.4f}")
    assert scores["iou"] > 0.999, f"Expected IoU ≈ 1.0, got {scores['iou']}"
    assert scores["f1"]  > 0.999, f"Expected F1  ≈ 1.0, got {scores['f1']}"

    # random predictions
    m.reset()
    random_log = torch.randn(B, 1, H, W)
    m.update(random_log, masks)
    scores_rand = m.compute()
    print("\nRandom logits:")
    for k, v in scores_rand.items():
        print(f"  {k:<12s}: {v:.4f}")

    # two identical batches must give the same result as one
    m.reset()
    m.update(perfect_log, masks)
    m.update(perfect_log, masks)
    scores_multi = m.compute()
    assert abs(scores_multi["iou"] - scores["iou"]) < 1e-4, \
        "Accumulation changed the result — counts must be summed, not averaged"
    print("\n✓ Multi-batch accumulation consistent")

    sweep = m.compute_threshold_sweep(perfect_log, masks)
    print("\nThreshold sweep (perfect logits):")
    for t, s in sweep.items():
        print(f"  t={t}  iou={s['iou']:.4f}  f1={s['f1']:.4f}")

    print("\n✓ SegmentationMetrics smoke test passed")
