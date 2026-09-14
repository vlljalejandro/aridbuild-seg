"""
B1 — U-Net with a ResNet-50 encoder from timm.

Five encoder feature maps at strides [2, 4, 8, 16, 32]; five decoder blocks
(bilinear x2 upsample, skip concat, two ConvBnReLU) return to full resolution,
then a shared ConvBnReLU feeds both heads.

    seg_logits, dist_logits = model(x)   # both (B, 1, 512, 512)

``dist_head`` is auxiliary: unused at inference, contributing nothing to any
reported metric, kept so released checkpoints load exactly.
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
    """3x3 Conv -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """
    One decoder step: bilinear x2 upsample, concat the encoder skip if present,
    then two ConvBnReLU. skip_ch=0 means no skip.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch,          out_ch),
        )

    def forward(
        self,
        x:    torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # guards a +/-1 px mismatch on odd-dimension inputs
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class OutputHead(nn.Sequential):
    """3x3 ConvBnReLU -> 1x1 Conv, raw logits."""

    def __init__(self, in_ch: int) -> None:
        super().__init__(
            ConvBnRelu(in_ch, in_ch, kernel_size=3),
            nn.Conv2d(in_ch, 1, kernel_size=1),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class UNetResNet50(nn.Module):
    """
    U-Net with a ResNet-50 encoder and dual output heads.

    ``encoder_name`` accepts any timm string whose backbone produces five
    feature maps at strides [2, 4, 8, 16, 32]; the architecture adapts to its
    channel widths. ``decoder_channels`` runs coarse to fine.
    """

    # Encoder feature indices → strides [2, 4, 8, 16, 32]
    _ENC_INDICES = (0, 1, 2, 3, 4)

    def __init__(
        self,
        encoder_name:     str   = "resnet50.a1_in1k",
        pretrained:       bool  = True,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
    ) -> None:
        super().__init__()

        # ── encoder ──────────────────────────────────────────────────────────
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self._ENC_INDICES,
        )
        enc_ch: List[int] = self.encoder.feature_info.channels()
        # [64, 256, 512, 1024, 2048] for ResNet-50

        # ── decoder ──────────────────────────────────────────────────────────
        # stride 32 -> 16 -> 8 -> 4 -> 2 -> 1, each block fusing one skip
        dec = decoder_channels
        self.decoder = nn.ModuleList([
            DecoderBlock(enc_ch[4], enc_ch[3], dec[0]),
            DecoderBlock(dec[0],    enc_ch[2], dec[1]),
            DecoderBlock(dec[1],    enc_ch[1], dec[2]),
            DecoderBlock(dec[2],    enc_ch[0], dec[3]),
            DecoderBlock(dec[3],    0,         dec[4]),   # no skip at stride-1
        ])

        # ── fusion conv ───────────────────────────────────────────────────────
        self.fuse = ConvBnRelu(dec[4], dec[4])

        # ── dual output heads ─────────────────────────────────────────────────
        self.seg_head  = OutputHead(dec[4])   # raw logits  → BCEWithLogits / Focal
        self.dist_head = OutputHead(dec[4])   # raw logits  → MSE vs normalised distance

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, H, W) -> seg_logits, dist_logits, both (B, 1, H, W)."""
        feats = self.encoder(x)

        d = feats[4]
        skip_indices = [3, 2, 1, 0, None]
        for block, skip_idx in zip(self.decoder, skip_indices):
            skip = feats[skip_idx] if skip_idx is not None else None
            d = block(d, skip)

        fused = self.fuse(d)          # (B, dec[-1], H, W)

        seg_logits  = self.seg_head(fused)
        dist_logits = self.dist_head(fused)

        return seg_logits, dist_logits

    # ── encoder freeze / unfreeze ─────────────────────────────────────────────

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (useful during early fine-tuning)."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters for full end-to-end training."""
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    # ── parameter groups ──────────────────────────────────────────────────────

    def param_groups(self, lr: float, encoder_lr_scale: float = 0.1) -> List[dict]:
        """Parameter groups with a lower LR for the pretrained encoder:
        encoder gets lr * encoder_lr_scale, everything else lr."""
        return [
            {"params": self.encoder.parameters(),  "lr": lr * encoder_lr_scale},
            {"params": self.decoder.parameters(),  "lr": lr},
            {"params": self.fuse.parameters(),     "lr": lr},
            {"params": self.seg_head.parameters(), "lr": lr},
            {"params": self.dist_head.parameters(),"lr": lr},
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
    model = UNetResNet50(pretrained=False).eval()

    counts = model.count_parameters()
    print("Parameter counts")
    for k, v in counts.items():
        print(f"  {k:<12s}: {v:>10,}")

    x   = torch.randn(2, 3, 512, 512)
    seg, dist = model(x)

    assert seg.shape  == (2, 1, 512, 512), f"seg shape mismatch:  {seg.shape}"
    assert dist.shape == (2, 1, 512, 512), f"dist shape mismatch: {dist.shape}"
    print(f"\nseg  : {tuple(seg.shape)}  min={seg.min():.3f}  max={seg.max():.3f}")
    print(f"dist : {tuple(dist.shape)}  min={dist.min():.3f}  max={dist.max():.3f}")
    print("\n✓ UNetResNet50 smoke test passed")
