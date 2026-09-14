"""
Concrete trainer for the supervised (D1+D2, optionally plus pseudo-labels) and
fine-tune (D3) stages.

The two stages are structurally identical — forward, combined loss, backward,
metrics, validate. They differ only in which loaders are passed in and whether
the encoder is frozen for the first few epochs.

    trainer = SegmentationTrainer(model, optimizer, scheduler, criterion,
                                  train_loader, val_loader, cfg, device)
    trainer.fit(resume_from="outputs/checkpoints/b1-supervised/best.pt")
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from src.losses.combined import CombinedLoss, compute_distance_target
from src.metrics.segmentation import SegmentationMetrics


class SegmentationTrainer(BaseTrainer):
    """
    Trainer for the supervised and fine-tune stages.

    ``freeze_encoder_epochs`` freezes the encoder for that many epochs at the
    start (0 = never), letting the heads stabilise before the encoder moves.

    The auxiliary distance target comes from compute_distance_target(). That
    head is unused at inference and contributes to no reported metric.
    """

    def __init__(
        self,
        model:                 nn.Module,
        optimizer:             torch.optim.Optimizer,
        scheduler,
        criterion:             CombinedLoss,
        train_loader:          DataLoader,
        val_loader:            DataLoader,
        cfg:                   dict,
        device:                torch.device,
        run_name:              Optional[str] = None,
        freeze_encoder_epochs: int           = 0,
    ) -> None:
        super().__init__(model, optimizer, scheduler, cfg, device, run_name)
        self.criterion             = criterion
        self.train_loader          = train_loader
        self.val_loader            = val_loader
        self.freeze_encoder_epochs = freeze_encoder_epochs

        # one per split, reset each epoch
        self.train_metrics = SegmentationMetrics(
            threshold=cfg.get("threshold", 0.5), device=device
        )
        self.val_metrics = SegmentationMetrics(
            threshold=cfg.get("threshold", 0.5), device=device
        )

        self._loss_keys = ["focal", "dice", "dist", "total"]

    # ── encoder freeze / unfreeze ─────────────────────────────────────────────

    def _maybe_toggle_encoder(self, epoch: int) -> None:
        """Freeze encoder for the first N epochs, then unfreeze."""
        if self.freeze_encoder_epochs == 0:
            return
        if epoch == 1:
            self.model.freeze_encoder()
            print(f"[trainer] Encoder frozen for {self.freeze_encoder_epochs} epoch(s)")
        elif epoch == self.freeze_encoder_epochs + 1:
            self.model.unfreeze_encoder()
            print(f"[trainer] Encoder unfrozen at epoch {epoch}")

    # ── training epoch ────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self._maybe_toggle_encoder(epoch)
        self.train_metrics.reset()
        running = {k: 0.0 for k in self._loss_keys}
        n_steps = len(self.train_loader)

        for step, (imgs, masks) in enumerate(self.train_loader, start=1):
            imgs  = imgs.to(self.device,  non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            dist_targets = compute_distance_target(masks)

            with self._autocast():
                seg_logits, dist_logits = self.model(imgs)
                loss, loss_dict = self.criterion(
                    seg_logits, dist_logits, masks, dist_targets
                )

            self._step(loss)

            for k in self._loss_keys:
                running[k] += loss_dict[k]
            self.train_metrics.update(seg_logits.detach(), masks)

            # ── per-step W&B log ──────────────────────────────────────────────
            log_every = self.cfg.get("log_interval", 50)
            if step % log_every == 0 or step == n_steps:
                global_step = (epoch - 1) * n_steps + step
                wandb.log({
                    "train/loss_step":  loss_dict["total"],
                    "train/focal_step": loss_dict["focal"],
                    "train/dice_step":  loss_dict["dice"],
                    "train/dist_step":  loss_dict["dist"],
                    "global_step":      global_step,
                })

        avg_loss = {f"train/{k}_loss": v / n_steps for k, v in running.items()}
        seg_scores = self.train_metrics.compute(prefix="train/")
        return {**avg_loss, **seg_scores}

    # ── validation epoch ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.val_metrics.reset()
        running = {k: 0.0 for k in self._loss_keys}
        n_steps = len(self.val_loader)

        for imgs, masks in self.val_loader:
            imgs  = imgs.to(self.device,  non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            dist_targets = compute_distance_target(masks)

            with self._autocast():
                seg_logits, dist_logits = self.model(imgs)
                _, loss_dict = self.criterion(
                    seg_logits, dist_logits, masks, dist_targets
                )

            for k in self._loss_keys:
                running[k] += loss_dict[k]

            self.val_metrics.update(seg_logits, masks)

        avg_loss   = {f"val/{k}_loss": v / n_steps for k, v in running.items()}
        seg_scores = self.val_metrics.compute(prefix="val/")
        return {**avg_loss, **seg_scores}