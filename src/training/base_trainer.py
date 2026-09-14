"""
Abstract base trainer: mixed precision, W&B logging, checkpoint save/resume,
gradient clipping and seeding, shared by every training stage.

Subclasses implement ``_train_epoch(epoch)`` and ``_val_epoch(epoch)``, both
returning metric dicts. ``_val_epoch`` must include "val/iou", which is what
selects the best checkpoint.
"""

from __future__ import annotations

import abc
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic ops: a small speed cost, worth it for debugging
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _get_amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    """bfloat16 where CUDA supports it, else float16; None on CPU."""
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# Base trainer
# ─────────────────────────────────────────────────────────────────────────────

class BaseTrainer(abc.ABC):
    """``cfg`` is the full config dict; ``run_name`` defaults to
    cfg["run_name"] and names both the W&B run and the checkpoint directory."""

    def __init__(
        self,
        model:      nn.Module,
        optimizer:  torch.optim.Optimizer,
        scheduler,
        cfg:        dict,
        device:     torch.device,
        run_name:   Optional[str] = None,
    ) -> None:
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg       = cfg
        self.device    = device
        self.run_name  = run_name or cfg.get("run_name", "run")

        # ── reproducibility ──────────────────────────────────────────────────
        seed_everything(cfg.get("seed", 42))

        # ── mixed precision ──────────────────────────────────────────────────
        self._amp_dtype = _get_amp_dtype(device)
        self._scaler    = (
            torch.amp.GradScaler("cuda")
            if self._amp_dtype == torch.float16 else None
        )
        # bfloat16 needs no loss scaler: its exponent range matches fp32

        # ── checkpoint dir ───────────────────────────────────────────────────
        self.ckpt_dir = Path(cfg.get("checkpoint_dir", "outputs/checkpoints")) / self.run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── state ────────────────────────────────────────────────────────────
        self._best_iou    = 0.0
        self._start_epoch = 1

    # ── W&B lifecycle ─────────────────────────────────────────────────────────

    def _init_wandb(self) -> None:
        wandb.init(
            project = self.cfg.get("wandb_project", "building-seg"),
            name    = self.run_name,
            config  = self.cfg,
            resume  = "allow",
        )
        # log="gradients" installs backward hooks that hang Swin-T and can
        # corrupt W&B state. Off by default; only set wandb_watch in the config
        # if you need it and are not training B3.
        watch = self.cfg.get("wandb_watch", None)
        wandb.watch(self.model, log=watch, log_freq=100)

    # ── AMP context ───────────────────────────────────────────────────────────

    def _autocast(self):
        """Autocast context for the current device."""
        if self._amp_dtype is None:
            return torch.autocast("cpu", enabled=False)
        return torch.autocast(self.device.type, dtype=self._amp_dtype)

    # ── optimizer step ────────────────────────────────────────────────────────

    def _step(self, loss: torch.Tensor) -> None:
        """Backward and optimizer step, scaled for FP16 and unscaled
        otherwise."""
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.get("grad_clip", 1.0)
            )
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.get("grad_clip", 1.0)
            )
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ── checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, tag: str = "latest") -> Path:
        """
        Save model, optimizer, scheduler and trainer state. "latest" is
        overwritten each epoch; "best" is kept.
        """
        state = {
            "epoch":      epoch,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict() if self.scheduler else None,
            "scaler":     self._scaler.state_dict()   if self._scaler   else None,
            "best_iou":   self._best_iou,
            "cfg":        self.cfg,
        }
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(state, path)
        return path

    def _load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model"])

        if not getattr(self, "_reset_on_load", False):
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.scheduler and ckpt.get("scheduler"):
                self.scheduler.load_state_dict(ckpt["scheduler"])
            if self._scaler and ckpt.get("scaler"):
                self._scaler.load_state_dict(ckpt["scaler"])

        self._best_iou    = ckpt.get("best_iou", 0.0)
        self._start_epoch = ckpt["epoch"] + 1
        print(
            f"[trainer] Resumed from '{path}'  "
            f"epoch={ckpt['epoch']}  best_iou={self._best_iou:.4f}"
        )

    # ── abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """One training epoch. Returns epoch-level metrics; log per-step
        metrics inside via ``wandb.log()``."""

    @abc.abstractmethod
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        """One validation epoch. Must return at least ``"val/iou"``."""

    # ── main fit loop ─────────────────────────────────────────────────────────

    def fit(self, resume_from=None, reset_epoch=False):
        """
        Full training loop.

        ``reset_epoch=True`` restarts the epoch counter and best-IoU tracker
        after loading, which is what a new stage needs when starting from the
        previous stage's checkpoint. False resumes the same stage in place.
        """
        if resume_from:
            self._reset_on_load = reset_epoch
            self._load_checkpoint(resume_from)
            self._reset_on_load = False
            if reset_epoch:
                self._start_epoch = 1
                self._best_iou    = 0.0
                print(f"[trainer] Epoch counter and best IoU reset for new stage")

        self._init_wandb()

        n_epochs = self.cfg["epochs"]
        print(
            f"\n[trainer] Starting '{self.run_name}'  "
            f"epochs={n_epochs}  device={self.device}  "
            f"amp={self._amp_dtype or 'disabled'}\n"
        )

        for epoch in range(self._start_epoch, n_epochs + 1):
            t_epoch = time.perf_counter()

            # ── train ─────────────────────────────────────────────────────────
            self.model.train()
            train_metrics = self._train_epoch(epoch)

            # ── validate ──────────────────────────────────────────────────────
            self.model.eval()
            val_metrics = self._val_epoch(epoch)

            # ── LR scheduler ─────────────────────────────────────────────────
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics.get("val/iou", 0.0))
                else:
                    self.scheduler.step()

            # ── logging ───────────────────────────────────────────────────────
            epoch_time = time.perf_counter() - t_epoch
            current_lr = self.optimizer.param_groups[0]["lr"]

            log_dict = {
                **train_metrics,
                **val_metrics,
                "epoch":      epoch,
                "lr":         current_lr,
                "epoch_time": epoch_time,
            }
            wandb.log(log_dict)

            # ── console summary ───────────────────────────────────────────────
            val_iou = val_metrics.get("val/iou", 0.0)
            val_f1  = val_metrics.get("val/f1",  0.0)
            print(
                f"[{epoch:03d}/{n_epochs}]  "
                f"loss={train_metrics.get('train/total_loss', 0):.4f}  "
                f"val_iou={val_iou:.4f}  val_f1={val_f1:.4f}  "
                f"lr={current_lr:.2e}  t={epoch_time:.1f}s"
            )

            # ── checkpointing ─────────────────────────────────────────────────
            self._save_checkpoint(epoch, tag="latest")

            if val_iou > self._best_iou:
                self._best_iou = val_iou
                best_path = self._save_checkpoint(epoch, tag="best")
                wandb.run.summary["best_val_iou"]   = self._best_iou
                wandb.run.summary["best_val_iou_ep"] = epoch
                print(f"           ✓ New best IoU {self._best_iou:.4f} → {best_path}")

        wandb.finish()
        print(f"\n[trainer] Done.  Best val IoU: {self._best_iou:.4f}")