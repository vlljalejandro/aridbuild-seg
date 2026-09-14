"""
Training entry point for the supervised (D1+D2) and fine-tune (D3) stages.

Stages: `supervised`, `finetune`, `all`, and `pseudolabel` when the config
supplies pseudo-labels. `--set` overrides any config key without editing the
YAML. One GPU per run.

    python scripts/train.py --cfg config/benchmark.yaml --stage all
    python scripts/train.py --cfg config/benchmark.yaml --stage supervised \\
        --set loader.batch_size=4 run_name=b1
"""

from __future__ import annotations
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"]       = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import argparse
import sys
from pathlib import Path

import torch
import yaml

# ── make the project root importable from any launch directory ───────────────
ROOT = next(
    p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    if (p / "src").is_dir()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader     import get_finetune_loaders, get_supervised_loaders, get_pseudolabel_supervised_loaders
from src.losses.combined     import CombinedLoss
from src.models              import build_model
from src.training            import SegmentationTrainer, seed_everything


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply dot-notation overrides from the CLI to the loaded config dict.

    Example
    -------
        --set loader.batch_size=4 supervised.epochs=20
        → cfg["loader"]["batch_size"] = 4
           cfg["supervised"]["epochs"] = 20

    Scalars only — bool, int, float, str, inferred from the text. List-valued
    fields such as model.decoder_channels must be set in the YAML, and there
    is no syntax for null (--set k=null yields the string "null").
    """
    def _cast(v: str):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
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
# Optimizer / scheduler factories
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, opt_cfg: dict) -> torch.optim.Optimizer:
    name = opt_cfg["name"]
    lr   = opt_cfg["lr"]

    # differential LR when the model exposes param_groups()
    if hasattr(model, "param_groups"):
        params = model.param_groups(
            lr=lr,
            encoder_lr_scale=opt_cfg.get("encoder_lr_scale", 0.1),
        )
    else:
        params = model.parameters()

    if name == "AdamW":
        return torch.optim.AdamW(
            params,
            lr           = lr,
            weight_decay = opt_cfg.get("weight_decay", 1e-4),
            betas        = tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
    if name == "SGD":
        return torch.optim.SGD(
            params,
            lr       = lr,
            momentum = opt_cfg.get("momentum", 0.9),
            weight_decay = opt_cfg.get("weight_decay", 1e-4),
        )
    raise ValueError(f"Unknown optimizer '{name}'.")


def build_scheduler(optimizer, sched_cfg: dict, epochs: int):
    name = sched_cfg["name"]

    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = sched_cfg.get("T_max", epochs),
            eta_min = sched_cfg.get("eta_min", 1e-6),
        )
    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = "max",      # monitor val IoU
            patience = sched_cfg.get("patience", 5),
            factor   = sched_cfg.get("factor", 0.5),
        )
    if name == "OneCycleLR":
        # caller must set T_max to total steps, not epochs
        raise ValueError(
            "OneCycleLR requires steps_per_epoch. "
            "Build it manually in scripts/train.py after constructing the loader."
        )
    if name is None or name == "none":
        return None
    raise ValueError(f"Unknown scheduler '{name}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Stage runners
# ─────────────────────────────────────────────────────────────────────────────

def run_pseudolabel_supervised(cfg: dict, device: torch.device) -> None:
    """Train on D1 + D2 + pseudo-labeled Saudi tiles."""
    stage_cfg = cfg["supervised"]
    seed_everything(cfg.get("seed", 42))

    loader_cfg = {**cfg["data"], **cfg["loader"]}
    train_loader, val_loader, _ = get_pseudolabel_supervised_loaders(loader_cfg, seed=cfg.get("seed", 42))

    model     = build_model(cfg["model"])
    optimizer = build_optimizer(model, stage_cfg["optimizer"])
    scheduler = build_scheduler(optimizer, stage_cfg["scheduler"], stage_cfg["epochs"])

    loss_cfg  = cfg["loss"]
    criterion = CombinedLoss(
        focal_weight = loss_cfg["focal_weight"],
        dice_weight  = loss_cfg["dice_weight"],
        dist_weight  = loss_cfg["dist_weight"],
        focal_alpha  = loss_cfg["focal_alpha"],
        focal_gamma  = loss_cfg["focal_gamma"],
        dice_smooth  = loss_cfg["dice_smooth"],
        ignore_index = loss_cfg.get("ignore_index", None),
    )

    flat_cfg = {
        **cfg,
        "epochs":         stage_cfg["epochs"],
        "grad_clip":      stage_cfg.get("grad_clip", 1.0),
        "log_interval":   stage_cfg.get("log_interval", 50),
        "threshold":      cfg["eval"]["threshold"],
        "checkpoint_dir": cfg["checkpoint_dir"],
        "wandb_project":  cfg["wandb_project"],
        "run_name":       f"{cfg['run_name']}-supervised",
    }

    trainer = SegmentationTrainer(
        model        = model,
        optimizer    = optimizer,
        scheduler    = scheduler,
        criterion    = criterion,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = flat_cfg,
        device       = device,
    )
    trainer.fit()


def run_supervised(cfg: dict, device: torch.device) -> None:
    """Train on the out-of-domain corpus, D1 (CrowdAI) + D2 (Inria)."""
    stage_cfg = cfg["supervised"]
    seed_everything(cfg.get("seed", 42))

    # ── flat loader cfg ───────────────────────────────────────────────────────
    loader_cfg = {
        **cfg["data"],
        **cfg["loader"],
    }

    train_loader, val_loader, _ = get_supervised_loaders(loader_cfg, seed=cfg.get("seed", 42))

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg["model"])

    # ── optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model, stage_cfg["optimizer"])
    scheduler = build_scheduler(optimizer, stage_cfg["scheduler"], stage_cfg["epochs"])

    # ── loss ──────────────────────────────────────────────────────────────────
    loss_cfg  = cfg["loss"]
    criterion = CombinedLoss(
        focal_weight = loss_cfg["focal_weight"],
        dice_weight  = loss_cfg["dice_weight"],
        dist_weight  = loss_cfg["dist_weight"],
        focal_alpha  = loss_cfg["focal_alpha"],
        focal_gamma  = loss_cfg["focal_gamma"],
        dice_smooth  = loss_cfg["dice_smooth"],
        ignore_index = loss_cfg.get("ignore_index", None),
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    flat_cfg = {
        **cfg,
        "epochs":          stage_cfg["epochs"],
        "grad_clip":       stage_cfg.get("grad_clip", 1.0),
        "log_interval":    stage_cfg.get("log_interval", 50),
        "threshold":       cfg["eval"]["threshold"],
        "checkpoint_dir":  cfg["checkpoint_dir"],
        "wandb_project":   cfg["wandb_project"],
        "run_name":        f"{cfg['run_name']}-supervised",
    }

    trainer = SegmentationTrainer(
        model        = model,
        optimizer    = optimizer,
        scheduler    = scheduler,
        criterion    = criterion,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = flat_cfg,
        device       = device,
    )
    trainer.fit()


def run_finetune(cfg: dict, device: torch.device) -> None:
    """Fine-tune on D3 (1,810 labeled target tiles), resuming from the supervised checkpoint."""
    stage_cfg = cfg["finetune"]
    seed_everything(cfg.get("seed", 42))

    loader_cfg = {
        **cfg["data"],
        **cfg["loader"],
    }

    train_loader, val_loader, _ = get_finetune_loaders(loader_cfg, seed=cfg.get("seed", 42))

    model     = build_model(cfg["model"])
    optimizer = build_optimizer(model, stage_cfg["optimizer"])
    scheduler = build_scheduler(optimizer, stage_cfg["scheduler"], stage_cfg["epochs"])

    loss_cfg  = {**cfg["loss"], **stage_cfg.get("loss", {})}
    criterion = CombinedLoss(
        focal_weight = loss_cfg["focal_weight"],
        dice_weight  = loss_cfg["dice_weight"],
        dist_weight  = loss_cfg["dist_weight"],
        focal_alpha  = loss_cfg["focal_alpha"],
        focal_gamma  = loss_cfg["focal_gamma"],
        dice_smooth  = loss_cfg["dice_smooth"],
        ignore_index = loss_cfg.get("ignore_index", None),
    )

    flat_cfg = {
        **cfg,
        "epochs":          stage_cfg["epochs"],
        "grad_clip":       stage_cfg.get("grad_clip", 1.0),
        "log_interval":    stage_cfg.get("log_interval", 10),
        "threshold":       cfg["eval"]["threshold"],
        "checkpoint_dir":  cfg["checkpoint_dir"],
        "wandb_project":   cfg["wandb_project"],
        "run_name":        f"{cfg['run_name']}-finetune",
    }

    trainer = SegmentationTrainer(
        model                 = model,
        optimizer             = optimizer,
        scheduler             = scheduler,
        criterion             = criterion,
        train_loader          = train_loader,
        val_loader            = val_loader,
        cfg                   = flat_cfg,
        device                = device,
        freeze_encoder_epochs = stage_cfg.get("freeze_encoder_epochs", 0),
    )
    # An empty resume_from falls through to the derived path, and the check
    # below hard-fails if it is missing. Do not replace this with a bare
    # `if resume_from:` that skips loading: an empty string once let every
    # benchmark train from ImageNet instead of the supervised checkpoint, with
    # no visible error.
    resume = stage_cfg.get("resume_from") or (
        Path(cfg["checkpoint_dir"]) / f"{cfg['run_name']}-supervised" / "best.pt"
    )
    resume = Path(resume)
    if not resume.exists():
        raise FileNotFoundError(
            f"Fine-tune stage needs a supervised checkpoint but none found at {resume}. "
            f"Set finetune.resume_from explicitly, or run --stage supervised first."
        )
    print(f"[train.py] Fine-tune resuming from: {resume}")
    trainer.fit(resume_from=str(resume), reset_epoch=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train building segmentation model.")
    p.add_argument("--cfg",   required=True,  help="Path to config YAML.")
    p.add_argument("--stage", required=True,
                   choices=["supervised", "finetune", "all", "pseudolabel"],
                   help="Training stage. 'pseudolabel' uses D1+D2+pseudo-labels for supervised.")
    p.add_argument("--set",   nargs="*", default=[],
                   metavar="KEY=VALUE",
                   help="Override config keys (dot notation, e.g. loader.batch_size=4).")
    p.add_argument("--device", default=None,
                   help="Device string, e.g. 'cuda:0' or 'cpu'. "
                        "Defaults to cuda if available.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    cfg    = load_cfg(args.cfg)
    cfg    = apply_overrides(cfg, args.set or [])

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[train.py] stage={args.stage}  device={device}")
    print(f"[train.py] run_name={cfg['run_name']}  project={cfg['wandb_project']}")

    if args.stage == "supervised":
        run_supervised(cfg, device)
    elif args.stage == "finetune":
        run_finetune(cfg, device)
    elif args.stage == "pseudolabel":
        print("[train.py] Running pseudolabel-supervised then finetune sequentially.")
        run_pseudolabel_supervised(cfg, device)
        print("[train.py] Pseudolabel supervised complete. Starting finetune.")
        run_finetune(cfg, device)
    elif args.stage == "all":
        print("[train.py] Running supervised then finetune sequentially.")
        run_supervised(cfg, device)
        print("[train.py] Supervised complete. Starting finetune.")
        run_finetune(cfg, device)


if __name__ == "__main__":
    main()