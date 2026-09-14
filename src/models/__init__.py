"""
src.models — model definitions for the building-segmentation pipeline.

Architectures
-------------
    UNetResNet50   B1 — U-Net decoder over a ResNet-50 encoder (timm)
    SegFormerB2    B2 — All-MLP decoder over a MiT-B2 encoder (HuggingFace)
    SwinTFPN       B3 — FPN decoder over a Swin-T encoder (timm)

All three share the same forward contract:

    seg_logits, dist_logits = model(x)   # (B,1,H,W), (B,1,H,W)

``dist_logits`` is an auxiliary output that is not used at inference and does
not contribute to any reported metric. It is retained so released checkpoints
load exactly as trained. See ``src/losses/combined.py``.

Dispatch
--------
Use ``build_model(model_cfg)`` rather than importing classes directly. It
reads a config dict and constructs the right architecture, so ``train.py``,
``evaluate.py`` and ``pseudo_label.py`` cannot drift apart.

``SwinTFPN`` was previously named ``Mask2FormerSwinT``. Checkpoints trained
before the rename store the old string in their config, so REGISTRY accepts
both keys and the old name is exported as an alias. Do not remove either
without re-exporting every B3 checkpoint.
"""

from __future__ import annotations

from typing import Any, Dict

from .unet import UNetResNet50
from .segformer import SegFormerB2
from .swin_fpn import SwinTFPN

# Backward-compatible alias — checkpoint configs written before the rename
# record "Mask2FormerSwinT". Keep this.
Mask2FormerSwinT = SwinTFPN

REGISTRY: Dict[str, Any] = {
    "UNetResNet50": UNetResNet50,
    "SegFormerB2": SegFormerB2,
    "SwinTFPN": SwinTFPN,
    "Mask2FormerSwinT": SwinTFPN,  # legacy checkpoints
}

# Per-architecture constructor arguments, with defaults. Anything not listed
# here is ignored, so a single config block can carry keys for all three
# architectures (as the benchmark config does) without erroring.
_ARGS: Dict[str, Dict[str, Any]] = {
    "UNetResNet50": {
        "encoder_name": "resnet50.a1_in1k",
        "decoder_channels": (256, 128, 64, 32, 16),
    },
    "SegFormerB2": {
        "encoder_name": "nvidia/mit-b2",
        "embed_dim": 256,
    },
    "SwinTFPN": {
        "encoder_name": "swin_tiny_patch4_window7_224",
        "fpn_dim": 256,
    },
}


def build_model(model_cfg: Dict[str, Any]):
    """
    Construct a model from a config dict.

    Parameters
    ----------
    model_cfg : dict with a "name" key, optionally "pretrained" and any of the
                architecture-specific keys in ``_ARGS``.

    Returns
    -------
    nn.Module

    Examples
    --------
    >>> build_model({"name": "SwinTFPN", "pretrained": False})
    """
    name = model_cfg["name"]
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Known: {sorted(REGISTRY)}. "
            "Add it to REGISTRY and _ARGS in src/models/__init__.py."
        )

    cls = REGISTRY[name]
    canonical = "SwinTFPN" if cls is SwinTFPN else name

    kwargs: Dict[str, Any] = {"pretrained": model_cfg.get("pretrained", True)}
    for key, default in _ARGS[canonical].items():
        value = model_cfg.get(key, default)
        if key == "decoder_channels":
            value = tuple(value)
        kwargs[key] = value

    return cls(**kwargs)


__all__ = [
    "UNetResNet50",
    "SegFormerB2",
    "SwinTFPN",
    "Mask2FormerSwinT",
    "REGISTRY",
    "build_model",
]
