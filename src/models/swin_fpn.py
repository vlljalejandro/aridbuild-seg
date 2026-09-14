"""
B3 — Swin-T encoder with an FPN pixel decoder.

Swin-T from timm gives four feature maps at strides [4, 8, 16, 32]; an FPN
fuses them top-down into one stride-4 map, and two heads upsample to full
resolution.

    seg_logits, dist_logits = model(x)   # both (B, 1, 512, 512)

This class was previously called ``Mask2FormerSwinT``, which was a misnomer:
there is no mask classification and no masked-attention decoder, only a Swin-T
backbone with a standard FPN. The alias is kept in ``src/models/__init__.py``
so checkpoints recording the old string still load.

``dist_head`` is auxiliary. It is unused at inference and contributes nothing
to any reported metric, but it is present in every released checkpoint, so it
is kept for exact loading. See ``src/losses/combined.py``.
"""

from __future__ import annotations

from typing import List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    """Conv2d → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class LateralBlock(nn.Module):
    """1x1 projection of an encoder feature, summed with the upsampled
    coarser feature."""

    def __init__(self, in_ch: int, fpn_dim: int) -> None:
        super().__init__()
        self.lateral = nn.Conv2d(in_ch, fpn_dim, kernel_size=1, bias=False)

    def forward(
        self,
        x:    torch.Tensor,    # current (coarser) FPN feature
        skip: torch.Tensor,    # encoder lateral connection
    ) -> torch.Tensor:
        skip = self.lateral(skip)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return x + skip


# ─────────────────────────────────────────────────────────────────────────────
# FPN pixel decoder
# ─────────────────────────────────────────────────────────────────────────────

class FPNDecoder(nn.Module):
    """
    Merges encoder features top-down, stride 32 -> 4. All scales are projected
    to ``fpn_dim`` channels; the finest is refined with a Conv3x3+BN+ReLU.
    """

    def __init__(self, enc_channels: List[int], fpn_dim: int = 256) -> None:
        super().__init__()

        self.top_proj = nn.Conv2d(enc_channels[-1], fpn_dim,
                                  kernel_size=1, bias=False)

        self.laterals = nn.ModuleList([
            LateralBlock(c, fpn_dim)
            for c in reversed(enc_channels[:-1])   # 384, 192, 96
        ])

        self.refine = ConvBnRelu(fpn_dim, fpn_dim, kernel_size=3)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """features: [stride-4, 8, 16, 32] -> (B, fpn_dim, H/4, W/4)"""
        x = self.top_proj(features[-1])

        for lateral, feat in zip(self.laterals, reversed(features[:-1])):
            x = lateral(x, feat)

        return self.refine(x)              # (B, fpn_dim, H/4, W/4)


# ─────────────────────────────────────────────────────────────────────────────
# Output head
# ─────────────────────────────────────────────────────────────────────────────

class OutputHead(nn.Sequential):
    """Conv3×3+BN+ReLU → Conv1×1, then upsample to full resolution."""

    def __init__(self, in_ch: int) -> None:
        super().__init__(
            ConvBnRelu(in_ch, in_ch, kernel_size=3),
            nn.Conv2d(in_ch, 1, kernel_size=1),
        )

    def forward(                                        # type: ignore[override]
        self,
        x:           torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        x = super().forward(x)
        return F.interpolate(x, size=target_size,
                             mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class SwinTFPN(nn.Module):
    """Swin-T encoder with an FPN pixel decoder and dual output heads."""

    # Swin-T feature channels at strides [4, 8, 16, 32]
    _ENC_CHANNELS = [96, 192, 384, 768]
    _ENC_INDICES  = (0, 1, 2, 3)

    def __init__(
        self,
        encoder_name: str  = "swin_tiny_patch4_window7_224",
        pretrained:   bool = True,
        fpn_dim:      int  = 256,
    ) -> None:
        super().__init__()

        # ── encoder ──────────────────────────────────────────────────────────
        self.encoder = timm.create_model(
            encoder_name,
            pretrained    = pretrained,
            features_only = True,
            out_indices   = self._ENC_INDICES,
            img_size      = 512,    # override default 224 for our tile size
        )

        enc_ch = self.encoder.feature_info.channels()

        # ── FPN pixel decoder ─────────────────────────────────────────────────
        self.decoder = FPNDecoder(enc_channels=enc_ch, fpn_dim=fpn_dim)

        # ── fusion conv ───────────────────────────────────────────────────────
        self.fuse = ConvBnRelu(fpn_dim, fpn_dim, kernel_size=3)

        # ── dual output heads ─────────────────────────────────────────────────
        self.seg_head  = OutputHead(fpn_dim)
        self.dist_head = OutputHead(fpn_dim)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, H, W) -> seg_logits, dist_logits, both (B, 1, H, W)."""
        H, W = x.shape[-2:]

        # Swin-T returns channels-last (B, H, W, C); permute to (B, C, H, W)
        features = [
            f.permute(0, 3, 1, 2).contiguous()
            for f in self.encoder(x)
        ]

        fused = self.fuse(self.decoder(features))   # (B, fpn_dim, H/4, W/4)
        seg_logits  = self.seg_head(fused,  (H, W))
        dist_logits = self.dist_head(fused, (H, W))

        return seg_logits, dist_logits

    # ── encoder freeze / unfreeze ─────────────────────────────────────────────

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters (used during early fine-tuning)."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters for full end-to-end training."""
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    # ── parameter groups ──────────────────────────────────────────────────────

    def param_groups(self, lr: float, encoder_lr_scale: float = 0.1) -> List[dict]:
        """Parameter groups with a lower LR for the pretrained encoder."""
        return [
            {"params": self.encoder.parameters(),   "lr": lr * encoder_lr_scale},
            {"params": self.decoder.parameters(),   "lr": lr},
            {"params": self.fuse.parameters(),      "lr": lr},
            {"params": self.seg_head.parameters(),  "lr": lr},
            {"params": self.dist_head.parameters(), "lr": lr},
        ]

    # ── convenience ───────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        """Return trainable parameter counts per sub-module."""
        def _count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "encoder":    _count(self.encoder),
            "decoder":    _count(self.decoder),
            "fuse":       _count(self.fuse),
            "seg_head":   _count(self.seg_head),
            "dist_head":  _count(self.dist_head),
            "total":      _count(self),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = SwinTFPN(pretrained=False).eval()

    counts = model.count_parameters()
    print("Parameter counts")
    for k, v in counts.items():
        print(f"  {k:<12s}: {v:>10,}")

    x = torch.randn(2, 3, 512, 512)
    seg, dist = model(x)

    assert seg.shape  == (2, 1, 512, 512), f"seg shape mismatch:  {seg.shape}"
    assert dist.shape == (2, 1, 512, 512), f"dist shape mismatch: {dist.shape}"
    print(f"\nseg  : {tuple(seg.shape)}  min={seg.min():.3f}  max={seg.max():.3f}")
    print(f"dist : {tuple(dist.shape)}  min={dist.min():.3f}  max={dist.max():.3f}")
    print("\n✓ SwinTFPN smoke test passed")