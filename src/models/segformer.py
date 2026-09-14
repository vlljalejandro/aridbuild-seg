"""
B2 — SegFormer with a MiT-B2 encoder.

The encoder comes from HuggingFace transformers (``nvidia/mit-b2``), not timm,
and gives four feature maps at strides [4, 8, 16, 32] with channels
[64, 128, 320, 512]. The All-MLP decoder projects each scale to a shared
embedding dimension, upsamples to stride 4, concatenates and fuses.

    seg_logits, dist_logits = model(x)   # both (B, 1, 512, 512)

``dist_head`` is auxiliary: unused at inference, contributing nothing to any
reported metric, kept so released checkpoints load exactly.

Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation
with Transformers", NeurIPS 2021.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig


# ─────────────────────────────────────────────────────────────────────────────
# Decoder building blocks
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Sequential):
    """Projects one encoder scale to the shared embedding dimension."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )


class SegFormerDecoder(nn.Module):
    """
    All-MLP decoder: project each scale to ``embed_dim``, upsample all to
    stride 4, concatenate, then fuse with a 1x1 Conv+BN+ReLU.
    """

    def __init__(self, enc_channels: List[int], embed_dim: int = 256) -> None:
        super().__init__()
        self.linear_layers = nn.ModuleList([
            MLP(c, embed_dim) for c in enc_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(len(enc_channels) * embed_dim, embed_dim,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """features: finest (stride 4) first -> (B, embed_dim, H/4, W/4)"""
        target_h, target_w = features[0].shape[-2:]

        projected = []
        for feat, mlp in zip(features, self.linear_layers):
            x = mlp(feat)
            if x.shape[-2:] != (target_h, target_w):
                x = F.interpolate(
                    x, size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                )
            projected.append(x)

        return self.fuse(torch.cat(projected, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Output head
# ─────────────────────────────────────────────────────────────────────────────

class OutputHead(nn.Sequential):
    """Conv3x3+BN+ReLU -> Conv1x1, upsampled to full resolution."""

    def __init__(self, in_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        x = super().forward(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class SegFormerB2(nn.Module):
    """
    SegFormer-B2 with dual output heads. ``encoder_name`` is a HuggingFace
    model id; "nvidia/mit-b2" is the ImageNet-1K pretrained encoder.
    """

    # MiT-B2 feature channels at strides [4, 8, 16, 32]
    _ENC_CHANNELS = [64, 128, 320, 512]

    def __init__(
        self,
        encoder_name: str  = "nvidia/mit-b2",
        pretrained:   bool = True,
        embed_dim:    int  = 256,
    ) -> None:
        super().__init__()

        # ── encoder ──────────────────────────────────────────────────────────
        if pretrained:
            import logging
            logging.getLogger("transformers").setLevel(logging.ERROR)
            self.encoder = SegformerModel.from_pretrained(
                encoder_name,
                output_hidden_states=True,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            )
            logging.getLogger("transformers").setLevel(logging.WARNING)
        else:
            config = SegformerConfig.from_pretrained(encoder_name)
            self.encoder = SegformerModel(config)
            self.encoder.config.output_hidden_states = True

        # ── decoder ──────────────────────────────────────────────────────────
        self.decoder = SegFormerDecoder(
            enc_channels=self._ENC_CHANNELS,
            embed_dim=embed_dim,
        )

        # ── dual output heads ─────────────────────────────────────────────────
        self.seg_head  = OutputHead(embed_dim)
        self.dist_head = OutputHead(embed_dim)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, H, W) -> seg_logits, dist_logits, both (B, 1, H, W)."""
        H, W = x.shape[-2:]

        # hidden_states: four tensors at strides [4, 8, 16, 32]
        outputs = self.encoder(pixel_values=x, output_hidden_states=True)
        fused = self.decoder(list(outputs.hidden_states))

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
            "seg_head":   _count(self.seg_head),
            "dist_head":  _count(self.dist_head),
            "total":      _count(self),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = SegFormerB2(pretrained=False).eval()

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
    print("\n✓ SegFormerB2 smoke test passed")