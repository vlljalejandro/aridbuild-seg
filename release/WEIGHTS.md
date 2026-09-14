# Model weights

Fifteen checkpoints: three architectures × five training configurations. All
are evaluated on the same 226-tile held-out target split at threshold 0.50.

## Architectures

| | Encoder | Decoder | Params |
|---|---|---|---|
| B1 | ResNet-50 (`resnet50.a1_in1k`) | U-Net, skip connections | ~32 M |
| B2 | MiT-B2 (`nvidia/mit-b2`) | All-MLP fusion | ~25 M |
| B3 | Swin-T (`swin_tiny_patch4_window7_224`) | FPN, top-down fusion | ~28 M |

The segmentation head, loss and post-processing are identical across all three.

## Configurations

| | Source data | Labeled target tiles | Pseudo-labels |
|---|---|---|---|
| `src` | yes | no | no |
| `tgt` | no | yes | no |
| `src-tgt` | yes | yes | no |
| `src-pl` | yes | **no** | yes |
| `src-pl-tgt` | yes | yes | yes |

## Test IoU

| Checkpoint | IoU | Precision | Recall |
|---|---|---|---|
| `b1-src` | 0.5988 | 0.8026 | 0.7022 |
| `b2-src` | 0.6069 | 0.7384 | 0.7732 |
| `b3-src` | 0.5616 | 0.7602 | 0.6825 |
| `b1-tgt` | 0.7075 | 0.8137 | 0.8443 |
| `b2-tgt` | 0.7090 | 0.8247 | 0.8348 |
| `b3-tgt` | 0.7302 | 0.8488 | 0.8394 |
| `b1-src-tgt` † | 0.7582 | 0.8657 | 0.8593 |
| `b2-src-tgt` | 0.7437 | 0.8486 | 0.8575 |
| `b3-src-tgt` | 0.7526 | 0.8496 | 0.8682 |
| `b1-src-pl` | 0.7733 | 0.8524 | 0.8929 |
| `b2-src-pl` | 0.7656 | 0.8444 | 0.8914 |
| `b3-src-pl` | 0.7738 | 0.8536 | 0.8921 |
| `b1-src-pl-tgt` | 0.7715 | 0.8863 | 0.8562 |
| `b2-src-pl-tgt` | 0.7674 | 0.8682 | 0.8686 |
| `b3-src-pl-tgt` | 0.7793 | 0.8918 | 0.8606 |

† Teacher (see below).

Reproduce with `python scripts/evaluate_release.py`.

## Loading

From a local clone:

```python
import torch
from src.models import build_model

ckpt = torch.load("data/checkpoints/b3-src-pl-tgt.pt", weights_only=False)
cfg = dict(ckpt["cfg"]["model"])
cfg["pretrained"] = False          # the strict load overwrites them anyway
model = build_model(cfg)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()
```

Or straight from the Hub:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download("vlljalejandro/aridbuild-seg", "b3-src-pl-tgt.pt")
ckpt = torch.load(path, weights_only=False)
```

Each checkpoint stores `model`, `epoch`, `best_iou` and `cfg`. Optimizer,
scaler and scheduler state were stripped on export.

`best_iou` is **not** the test IoU above. For `src` runs it is source-domain
validation (8,251 tiles); for fine-tuned runs it is target validation (226
tiles). Neither is the test split.

## Four things that will trip you up

**`forward()` returns a 2-tuple.** `(seg_logit, dist_logit)`, both
`(N,1,512,512)`. Element 1 is an auxiliary distance head: trained with an MSE
term at weight 0.5, unused at inference, contributing nothing to any reported
metric, retained so the checkpoints reproduce exactly. It is present in every
checkpoint, so `strict=True` succeeds (use it). Reading element 1 instead of
element 0 gives a mask that thresholds to something plausible and scores about
0.09 IoU.

**Normalization lives in the checkpoint.** Take `img_mean` and `img_std` from
`ckpt["cfg"]["data"]`, not from the YAML files. Images are loaded in RGB order.

**B3 checkpoints say `Mask2FormerSwinT`.** The model is a Swin-T encoder with
an FPN decoder and dense convolutional heads (no queries, no mask
classification, nothing to einsum over). `src/models/__init__.py` keeps the old
string as an alias so the checkpoints load; the paper calls it Swin-T/FPN.

**The stored `cfg` has stale paths.** It is a training-time snapshot: dataset
directories, LMDB names and the W&B project do not match this repository. The
files in `config/` are authoritative for everything except normalization and
threshold, which come from the checkpoint.

## One teacher

`b1-src-tgt` generated every pseudo-label. All three `src-pl` students distil
from that single teacher.

So cross-architecture agreement among the students shows the pseudo-labels
transfer across architectures. It is **not** independent replication, and
should not be read as three models arriving at the same answer separately.

The teacher was chosen deliberately: it is not the strongest available model,
and it is a CNN, so the two transformer students cannot be explained by
self-distillation of a shared inductive bias. On the test split it predicts
0.7% less building area than ground truth at threshold 0.50, and 3.3% more at
the 0.20 threshold used for label generation (close to area-calibrated). A stronger teacher (0.7975 target validation IoU against
0.7679) produced measurably *weaker* students in preliminary work.

## Training

AdamW, base LR 1e-4 supervised and 2e-5 fine-tune, encoder LR scaled 0.5 and
1.0, weight decay 1e-4, cosine annealing, gradient clipping 1.0, batch size 8,
bfloat16 on a single RTX 4090. Supervised stages 50 epochs, fine-tuning 30.
Loss is Focal + Dice with γ=2 and α set per dataset (0.25 source, 0.07
target), plus the auxiliary distance term at weight 0.5.

## Training data

The checkpoints were trained on the CrowdAI Mapping Challenge and the Inria
Aerial Image Labeling benchmark (out-of-domain supervision), and on our own
building annotations over Saudi Arabian zoom-18 tiles (target domain).

CrowdAI and Inria carry their own terms, and Inria's are research-use. Check
them before putting these weights to a use their licences would not cover.

## License

Weights: see `LICENSE`. Annotations and derived data: see `LICENSE-DATA`.
Neither covers the third-party training datasets named above.
