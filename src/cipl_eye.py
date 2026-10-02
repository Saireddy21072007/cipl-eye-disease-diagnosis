"""
cipl_eye.py — unified experiment harness for the CIPL-Eye project
=================================================================
ONE file that runs every configuration in the paper, on ANY of the supported
fundus datasets, with a protocol that survives peer review.

What this fixes relative to the notebook versions (dl1 / dl30)
--------------------------------------------------------------
1. PATIENT-LEVEL GROUP SPLIT.  The old code used train_test_split on images.
   ODIR labels are issued per PATIENT across two eyes, so a random image split
   puts the same patient's left and right eye on both sides -> leaked, inflated
   scores.  We now split on the patient ID with GroupShuffleSplit.

2. THREE-WAY SPLIT.  Old code selected the checkpoint on val and reported on
   val.  Now: train / val / test.  Checkpoint AND thresholds are chosen on VAL;
   every reported number comes from TEST, which is touched exactly once.

3. THRESHOLDS TUNED ON VAL, NOT TRAIN.  Old code tuned per-class thresholds on
   TRAIN predictions, which the model has already fit -> optimistic thresholds.

4. THE COSINE-COLLAPSE BUG.  Old add-on head ended in nn.Sigmoid() and
   prototypes were initialised with torch.rand.  Both put every vector in the
   non-negative orthant of R^256, so after L2 normalisation EVERY cosine
   similarity lands in a narrow band (measured: [0.72, 0.86], std 0.016).
   The prototype layer had almost no dynamic range -> every class produced a
   near-identical map at ~0.51 probability.  We drop the sigmoid and use a
   zero-centred embedding + randn prototype init + a learnable temperature.
   This is the single biggest quality fix in this file.

5. MULTI-SEED.  Every config runs over N seeds; we report mean +- std.

6. INTERPRETABILITY IS MEASURED, NOT JUST SHOWN.
      - deletion / insertion faithfulness  (needs no annotations)
      - prototype disentanglement IoU      (needs no annotations)
      - Dice / IoU vs. real lesion masks   (if a mask dataset is provided)

Usage
-----
    # single run
    python cipl_eye.py --dataset odir --model custom_proto --epochs 30

    # the full paper grid (all models x all seeds), then emit LaTeX tables
    python cipl_eye.py --run-all

On Kaggle, drop this file in and call run_all() from a cell, or use
    !python cipl_eye.py --run-all

Requires: torch, torchvision, numpy, pandas, scikit-learn, matplotlib, pillow.
opencv is optional (used only for prettier heat-map overlays).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass, field, asdict
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             multilabel_confusion_matrix, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import cv2
except ImportError:
    cv2 = None

warnings.filterwarnings("ignore")

# numpy 2.0 renamed trapz -> trapezoid; support both so this runs on any Kaggle image
_trapz = getattr(np, "trapezoid", None) or np.trapz


# =====================================================================
# 0.  CONFIG
# =====================================================================
@dataclass
class Config:
    # --- what to run -------------------------------------------------
    dataset: str = "odir"            # "odir" | "rfmid"
    model: str = "custom_proto"      # "custom_proto" | "custom_plain" | "resnet_cbam"
    seed: int = 42

    # --- data --------------------------------------------------------
    data_root: str = ""              # auto-detected if empty
    img_size: int = 224
    batch: int = 32
    workers: int = 2
    min_pos: int = 50                # RFMiD: drop labels with fewer positives

    # --- optimisation ------------------------------------------------
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 10               # early stop on val mAUC
    amp: bool = True                 # mixed precision (~2x faster on Kaggle GPUs)

    # --- prototype head ----------------------------------------------
    protos: int = 10                 # M prototypes per class
    feat_dim: int = 256
    w_clst: float = 0.1
    w_sep: float = 0.1

    # --- interpretability eval ---------------------------------------
    eval_faithfulness: bool = True
    faith_images: int = 200          # test images used for deletion/insertion
    faith_steps: int = 20

    # --- bookkeeping -------------------------------------------------
    out_dir: str = "runs"
    tag: str = ""

    def run_name(self) -> str:
        return self.tag or f"{self.dataset}_{self.model}_s{self.seed}"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Make a run reproducible. cudnn.deterministic costs a little speed but
    means the numbers in the paper can actually be regenerated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# 1.  DATASETS
# =====================================================================
# Every loader returns the same 5 things, so the rest of the file never
# needs to know which dataset it is looking at:
#     img_dir     str
#     files       (N,)  filenames relative to img_dir
#     labels      (N,C) float32 multi-hot
#     groups      (N,)  patient id for group-aware splitting (or arange)
#     class_names list[str]

ODIR_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]
ODIR_NAMES = ["Normal", "Diabetes", "Glaucoma", "Cataract",
              "AMD", "Hypertension", "Myopia", "Other"]


def _first_existing(root: str, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.exists(p):
            return p
    return None


def _autodetect_root(dataset: str, given: str) -> str:
    """Find the dataset directory. Fails LOUDLY — a wrong path must crash, not
    silently produce black images (this bit us once already)."""
    if given and os.path.isdir(given):
        return given
    guesses = {
        "odir": ["/kaggle/input/ocular-disease-recognition-odir5k",
                 "/kaggle/input/datasets/andrewmvd/ocular-disease-recognition-odir5k",
                 "./data/odir"],
        "rfmid": ["/kaggle/input/retinal-disease-classification",
                  "/kaggle/input/datasets/andrewmvd/retinal-disease-classification",
                  "./data/rfmid"],
    }[dataset]
    for g in guesses:
        if os.path.isdir(g):
            return g
    raise FileNotFoundError(
        f"Could not locate the '{dataset}' dataset. Tried: {guesses}\n"
        f"Pass --data-root explicitly.")


def load_odir(cfg: Config):
    root = _autodetect_root("odir", cfg.data_root)
    img_dir = _first_existing(root, ["preprocessed_images",
                                     "ODIR-5K/ODIR-5K/Training Images",
                                     "ODIR-5K/Training Images"])
    csv_path = _first_existing(root, ["full_df.csv", "data.csv"])
    if img_dir is None or csv_path is None:
        raise FileNotFoundError(f"ODIR layout unexpected under {root}: "
                                f"{os.listdir(root)[:10]}")
    df = pd.read_csv(csv_path)

    labels = df[ODIR_COLS].values.astype(np.float32)
    files = df["filename"].astype(str).values
    # 'ID' is the PATIENT id — both eyes of one patient share it. This is the
    # column that makes the split honest.
    groups = df["ID"].values if "ID" in df.columns else np.arange(len(df))

    keep = np.array([os.path.isfile(os.path.join(img_dir, f)) for f in files])
    if keep.sum() == 0:
        raise FileNotFoundError(f"None of the {len(files)} ODIR images exist in {img_dir}")
    if (~keep).sum():
        print(f"[data] WARNING: {(~keep).sum()} missing image files dropped")
    return img_dir, files[keep], labels[keep], groups[keep], list(ODIR_NAMES)


def load_rfmid(cfg: Config):
    """RFMiD — Retinal Fundus Multi-Disease Image Dataset.

    Ships with OFFICIAL train/val/test splits (1920/640/640), which we keep:
    using the official protocol is what lets us compare against published
    numbers later. 46 label columns; most are extremely rare, so we keep only
    labels with >= cfg.min_pos positives in the training split.
    """
    root = _autodetect_root("rfmid", cfg.data_root)

    csvs = {k: None for k in ["train", "val", "test"]}
    for p in glob(os.path.join(root, "**", "*.csv"), recursive=True):
        low = os.path.basename(p).lower()
        if "train" in low:
            csvs["train"] = p
        elif "valid" in low or "eval" in low:
            csvs["val"] = p
        elif "test" in low:
            csvs["test"] = p
    if any(v is None for v in csvs.values()):
        raise FileNotFoundError(f"RFMiD label CSVs not found under {root}. "
                                f"Found: {glob(os.path.join(root, '**', '*.csv'), recursive=True)[:8]}")

    # image directories: match a split's images by looking for the dir holding
    # the most .png files whose name matches the split keyword
    def find_img_dir(keyword: str) -> str:
        best, best_n = None, 0
        for d, _, fs in os.walk(root):
            n = sum(1 for f in fs if f.lower().endswith((".png", ".jpg", ".jpeg")))
            if n > best_n and keyword in d.lower().replace("\\", "/"):
                best, best_n = d, n
        if best is None:
            raise FileNotFoundError(f"No RFMiD image dir for '{keyword}' under {root}")
        return best

    dirs = {"train": find_img_dir("train"),
            "val": find_img_dir("valid") if any("valid" in d.lower() for d, _, _ in os.walk(root)) else find_img_dir("eval"),
            "test": find_img_dir("test")}

    frames = {k: pd.read_csv(v) for k, v in csvs.items()}
    id_col = "ID" if "ID" in frames["train"].columns else frames["train"].columns[0]
    drop = {id_col, "Disease_Risk"}
    label_cols = [c for c in frames["train"].columns if c not in drop]

    pos = frames["train"][label_cols].sum(0)
    label_cols = [c for c in label_cols if pos[c] >= cfg.min_pos]
    if not label_cols:
        raise ValueError(f"No RFMiD label has >= {cfg.min_pos} positives; lower --min-pos")
    print(f"[data] RFMiD: keeping {len(label_cols)} labels with "
          f">= {cfg.min_pos} positives: {label_cols}")

    # Return the three splits pre-made; ODIR-style group splitting is not
    # needed because RFMiD is one image per eye with an official split.
    out = {}
    for k in ["train", "val", "test"]:
        d = frames[k]
        files = d[id_col].astype(str).values
        # filenames in RFMiD CSVs are bare integers; find the real extension
        exts = {os.path.splitext(f)[0]: f for f in os.listdir(dirs[k])}
        files = np.array([exts.get(f, f + ".png") for f in files])
        out[k] = (dirs[k], files, d[label_cols].values.astype(np.float32))
    return out, label_cols


class FundusDataset(Dataset):
    def __init__(self, img_dir, files, labels, size, train=False):
        self.img_dir, self.files, self.labels = img_dir, files, labels
        aug = [transforms.RandomHorizontalFlip(),
               transforms.RandomRotation(10),
               transforms.ColorJitter(0.1, 0.1, 0.1)] if train else []
        self.tf = transforms.Compose(aug + [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # deliberately no try/except: a bad path must crash loudly
        img = Image.open(os.path.join(self.img_dir, self.files[i])).convert("RGB")
        return self.tf(img), torch.tensor(self.labels[i])


def make_loaders(cfg: Config):
    """Returns train/val/test loaders + class names + a handle on the test set
    for the qualitative demo."""
    if cfg.dataset == "rfmid":
        splits, names = load_rfmid(cfg)
        ds = {k: FundusDataset(*splits[k], cfg.img_size, train=(k == "train"))
              for k in splits}
    else:
        img_dir, files, labels, groups, names = load_odir(cfg)
        n = len(files)

        # --- THE FIX: group-aware, three-way split ---------------------
        # 20% test, then 12.5% of the remainder as val -> 70/10/20 overall.
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=cfg.seed)
        trval, test = next(gss.split(np.arange(n), groups=groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=cfg.seed)
        tr_rel, va_rel = next(gss2.split(trval, groups=groups[trval]))
        train, val = trval[tr_rel], trval[va_rel]

        # sanity: no patient may appear in two splits
        for a, b, an, bn in [(train, val, "train", "val"),
                             (train, test, "train", "test"),
                             (val, test, "val", "test")]:
            overlap = set(groups[a]) & set(groups[b])
            assert not overlap, f"patient leak between {an} and {bn}: {list(overlap)[:5]}"

        ds = {"train": FundusDataset(img_dir, files[train], labels[train], cfg.img_size, True),
              "val": FundusDataset(img_dir, files[val], labels[val], cfg.img_size),
              "test": FundusDataset(img_dir, files[test], labels[test], cfg.img_size)}

    print(f"[data] {cfg.dataset}: train={len(ds['train'])} "
          f"val={len(ds['val'])} test={len(ds['test'])} classes={len(names)}")

    loaders = {k: DataLoader(v, cfg.batch, shuffle=(k == "train"),
                             num_workers=cfg.workers, pin_memory=True)
               for k, v in ds.items()}
    return loaders, names, ds["test"]


# =====================================================================
# 2.  MODELS
# =====================================================================
class CustomCNN(nn.Module):
    """Our own backbone, trained from scratch. No pretrained weights, no
    ResNet/DenseNet/VGG. 5 blocks of (Conv-BN-ReLU) x2 + MaxPool:
    224 -> 112 -> 56 -> 28 -> 14 -> 7.  Output [B, 512, 7, 7]."""

    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        self.net = nn.Sequential(block(3, 64), block(64, 128), block(128, 256),
                                 block(256, 512), block(512, 512))
        self.out_channels = 512

    def forward(self, x):
        return self.net(x)


class PrototypeHead(nn.Module):
    """Cosine-similarity prototype layer (CIPL-lite).

    THE FIX vs. the notebook version
    --------------------------------
    old:  add_on ended in nn.Sigmoid()  -> features in [0,1]^256
          prototypes = torch.rand(...)  -> prototypes in [0,1]^256
          Both live in the non-negative orthant, so every cosine similarity is
          squeezed into ~[0.72, 0.86] (std 0.016) no matter what the image
          shows. The layer cannot express "this does NOT look like that".

    new:  no output non-linearity (embedding is zero-centred and unbounded),
          prototypes ~ N(0, 1) so they cover the whole unit sphere, and a
          learnable temperature so the classifier can sharpen the similarities
          it actually finds useful.
    """

    def __init__(self, in_ch, num_classes, m=10, feat_dim=256):
        super().__init__()
        self.add_on = nn.Sequential(
            nn.Conv2d(in_ch, feat_dim, 1), nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 1))          # <- NO Sigmoid
        self.C, self.M, self.N = num_classes, m, num_classes * m
        self.prototypes = nn.Parameter(torch.randn(self.N, feat_dim) * 0.1)
        self.log_temp = nn.Parameter(torch.tensor(2.5))  # ~ exp(2.5) = 12
        self.classifier = nn.Linear(self.N, num_classes)

    def forward(self, feat):
        f = self.add_on(feat)                                # [B,D,h,w]
        B, D, h, w = f.shape
        fn = F.normalize(f, dim=1)
        pn = F.normalize(self.prototypes, dim=1)
        sim = torch.einsum("bdhw,nd->bnhw", fn, pn)          # [B,N,h,w] cosine
        scores = F.adaptive_max_pool2d(sim, 1).flatten(1)    # [B,N]
        logits = self.classifier(scores * self.log_temp.exp())
        cls_maps = sim.view(B, self.C, self.M, h, w).max(2).values  # [B,C,h,w]
        return logits, cls_maps, scores


class PlainHead(nn.Module):
    """Ablation: same backbone, ordinary GAP + linear classifier. Needed to
    show whether the prototype layer earns its place."""

    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, 1)   # keeps a CAM available
        self.num_classes = num_classes

    def forward(self, feat):
        maps = self.conv(feat)                          # [B,C,h,w]
        logits = F.adaptive_avg_pool2d(maps, 1).flatten(1)
        return logits, maps, logits


class CBAM(nn.Module):
    """Convolutional Block Attention Module — channel attention then spatial
    attention. Used only by the pretrained comparison branch."""

    def __init__(self, ch, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ch, ch // r), nn.ReLU(inplace=True),
                                 nn.Linear(ch // r, ch))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        b, c, _, _ = x.shape
        ca = torch.sigmoid(self.mlp(x.mean((2, 3))) +
                           self.mlp(x.amax((2, 3)))).view(b, c, 1, 1)
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat(
            [x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


class ResNetCBAM(nn.Module):
    """ImageNet-pretrained ResNet-18 + CBAM + CAM head.

    NOTE: this VIOLATES the 'design your own network' constraint on purpose.
    It exists only to quantify what that constraint costs — it is the
    comparison row in the paper, never the headline method."""

    def __init__(self, num_classes):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(*list(m.children())[:-2])   # -> [B,512,7,7]
        self.cbam = CBAM(512)
        self.drop = nn.Dropout2d(0.2)
        self.head = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        f = self.drop(self.cbam(self.stem(x)))
        maps = self.head(f)
        logits = F.adaptive_avg_pool2d(maps, 1).flatten(1)
        return logits, maps, logits


def build_model(cfg: Config, num_classes: int) -> nn.Module:
    if cfg.model == "resnet_cbam":
        return ResNetCBAM(num_classes)

    class Net(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.backbone = CustomCNN()
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x))

    if cfg.model == "custom_proto":
        return Net(PrototypeHead(512, num_classes, cfg.protos, cfg.feat_dim))
    if cfg.model == "custom_plain":
        return Net(PlainHead(512, num_classes))
    raise ValueError(f"unknown model '{cfg.model}'")


# =====================================================================
# 3.  LOSS
# =====================================================================
def pos_weight_from(labels: np.ndarray) -> torch.Tensor:
    """#neg / #pos per class. Without this, plain BCE on ODIR collapses to the
    all-negative solution (macro-F1 = 0.000 at 0.70 AUC — config C1)."""
    pos = labels.sum(0)
    neg = len(labels) - pos
    return torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32, device=DEVICE)


def cipl_lite_loss(bce, logits, scores, y, cfg: Config, C: int):
    """BCE + clustering + separation.

    Separation now uses a margin: we only penalise a negative class whose best
    prototype similarity exceeds -0.1, instead of penalising relu(sim) which,
    with the old positive-orthant embedding, penalised literally everything.
    """
    loss = bce(logits, y)
    if scores.shape[1] != C * cfg.protos:      # plain / CBAM heads: BCE only
        return loss
    B = y.shape[0]
    best = scores.view(B, C, cfg.protos).max(2).values      # [B,C]
    pos, neg = y, 1.0 - y
    clst = ((1 - best) * pos).sum() / (pos.sum() + 1e-8)
    sep = (F.relu(best + 0.1) * neg).sum() / (neg.sum() + 1e-8)
    return loss + cfg.w_clst * clst + cfg.w_sep * sep


# =====================================================================
# 4.  TRAIN / PREDICT
# =====================================================================
@torch.no_grad()
def predict(model, loader):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        logits, _, _ = model(x.to(DEVICE, non_blocking=True))
        P.append(torch.sigmoid(logits).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def mean_auc(y_true, y_prob):
    aucs = [roc_auc_score(y_true[:, c], y_prob[:, c])
            if len(np.unique(y_true[:, c])) > 1 else np.nan
            for c in range(y_true.shape[1])]
    return float(np.nanmean(aucs)), aucs


def tune_thresholds(y_true, y_prob):
    """Per-class threshold maximising F1 — computed on VALIDATION, never on
    train (the model has already fit train) and never on test (that would be
    tuning on the number you report)."""
    ths = np.full(y_true.shape[1], 0.5)
    for c in range(y_true.shape[1]):
        best = -1.0
        for t in np.linspace(0.05, 0.95, 37):
            f1 = f1_score(y_true[:, c], (y_prob[:, c] > t).astype(int), zero_division=0)
            if f1 > best:
                best, ths[c] = f1, t
    return ths


def train(model, loaders, cfg: Config, ckpt: str):
    labels = loaders["train"].dataset.labels
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    C = labels.shape[1]

    use_amp = bool(cfg.amp and DEVICE == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[train] mixed precision ON")

    hist = {"loss": [], "val_auc": []}
    best_auc, bad = -1.0, 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _, scores = model(x)
                loss = cipl_lite_loss(bce, logits, scores, y, cfg, C)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        sched.step()

        ep_loss = running / len(loaders["train"].dataset)
        vp, vy = predict(model, loaders["val"])
        vauc, _ = mean_auc(vy, vp)
        hist["loss"].append(ep_loss)
        hist["val_auc"].append(vauc)
        print(f"epoch {ep:02d}/{cfg.epochs}  loss={ep_loss:.4f}  val_mAUC={vauc:.4f}")

        if vauc > best_auc:
            best_auc, bad = vauc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[train] early stop at epoch {ep} (no val gain for {cfg.patience})")
                break

    print(f"[train] best val mAUC = {best_auc:.4f}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return hist, best_auc


# =====================================================================
# 5.  INTERPRETABILITY METRICS
# =====================================================================
def _upsample(map2d: torch.Tensor, size: int) -> np.ndarray:
    m = map2d[None, None].float()
    m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)
    m = m[0, 0].detach().cpu().numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


@torch.no_grad()
def faithfulness(model, dataset, cfg: Config, n_images: int, steps: int):
    """Deletion / Insertion (Petsiuk et al., RISE).

    Deletion: progressively blank the most-activated pixels; a FAITHFUL map
              makes the class probability fall fast -> LOW AUC is good.
    Insertion: start from a blurred image and reveal the most-activated pixels;
              a faithful map makes probability rise fast -> HIGH AUC is good.

    Needs no annotations at all, which is why it is the cheapest way to turn
    'our maps look nice' into a number a reviewer accepts.
    """
    model.eval()
    idx = np.linspace(0, len(dataset) - 1, min(n_images, len(dataset))).astype(int)
    dels, inss = [], []

    for i in idx:
        x, y = dataset[int(i)]
        x = x.unsqueeze(0).to(DEVICE)
        logits, maps, _ = model(x)
        c = int(torch.sigmoid(logits)[0].argmax())          # explain top class
        sal = _upsample(maps[0, c], cfg.img_size).ravel()
        order = np.argsort(-sal)                            # most salient first
        n_px = order.size
        chunk = max(1, n_px // steps)

        base_blur = F.avg_pool2d(x, 17, stride=1, padding=8)  # neutral image

        d_curve, i_curve = [], []
        xd, xi = x.clone(), base_blur.clone()
        flat_d = xd.view(1, 3, -1)
        flat_i = xi.view(1, 3, -1)
        src = x.view(1, 3, -1)
        blur = base_blur.view(1, 3, -1)

        for s in range(steps + 1):
            d_curve.append(float(torch.sigmoid(model(flat_d.view_as(x))[0])[0, c]))
            i_curve.append(float(torch.sigmoid(model(flat_i.view_as(x))[0])[0, c]))
            if s == steps:
                break
            sel = order[s * chunk:(s + 1) * chunk]
            flat_d[:, :, sel] = blur[:, :, sel]     # delete -> blur it out
            flat_i[:, :, sel] = src[:, :, sel]      # insert -> reveal original

        dels.append(_trapz(d_curve, dx=1.0 / steps))
        inss.append(_trapz(i_curve, dx=1.0 / steps))

    return {"deletion_auc": float(np.mean(dels)),      # lower is better
            "insertion_auc": float(np.mean(inss))}     # higher is better


@torch.no_grad()
def disentanglement(model, dataset, n_images: int = 200, pct: float = 80.0):
    """Mean pairwise IoU between different classes' localisation maps.

    This is the metric that directly measures the failure we saw earlier —
    every disease lighting up the same region. LOW is good (each disease has
    its own evidence); near 1.0 means the prototypes are entangled.
    """
    model.eval()
    idx = np.linspace(0, len(dataset) - 1, min(n_images, len(dataset))).astype(int)
    ious = []
    for i in idx:
        x, _ = dataset[int(i)]
        _, maps, _ = model(x.unsqueeze(0).to(DEVICE))
        m = maps[0]                                        # [C,h,w]
        C = m.shape[0]
        binm = []
        for c in range(C):
            a = m[c].detach().cpu().numpy()
            binm.append(a >= np.percentile(a, pct))
        for a in range(C):
            for b in range(a + 1, C):
                inter = np.logical_and(binm[a], binm[b]).sum()
                union = np.logical_or(binm[a], binm[b]).sum()
                if union:
                    ious.append(inter / union)
    return {"prototype_iou": float(np.mean(ious)) if ious else float("nan")}


def dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Dice and IoU for the OPTIONAL lesion-mask evaluation (IDRiD etc.).
    Both inputs are boolean arrays of the same shape."""
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    return {"dice": float(2 * inter / denom) if denom else float("nan"),
            "iou": float(inter / union) if union else float("nan")}


@torch.no_grad()
def evaluate_localisation_idrid(model, idrid_root: str, cfg: Config,
                                class_names: list, disease: str = "Diabetes",
                                pct: float = 90.0):
    """CROSS-DATASET weakly-supervised localisation check.

    Train on ODIR with IMAGE-LEVEL labels only, then ask: do the localisation
    maps actually land on real lesions?  IDRiD ships pixel-level masks for
    microaneurysms, haemorrhages, and hard/soft exudates, so we can score the
    map we never trained to produce.

    This is the experiment that turns "our maps look plausible" into Dice/IoU
    numbers, which is the single biggest gap in the current paper.

    Absolute Dice will be LOW — a 7x7 map upsampled to 224 cannot trace a
    microaneurysm.  Report it against a Grad-CAM / plain-CAM baseline run
    through this same function; the RELATIVE ordering is the result.
    """
    imgs = {}
    for p_ in glob(os.path.join(idrid_root, "**", "*.jpg"), recursive=True):
        if "origin" in p_.lower() or "image" in p_.lower():
            imgs[os.path.splitext(os.path.basename(p_))[0]] = p_
    if not imgs:
        raise FileNotFoundError(f"No IDRiD images found under {idrid_root}")

    # group every lesion mask (.tif) by the image id it belongs to
    masks: dict = {}
    for p_ in glob(os.path.join(idrid_root, "**", "*.tif"), recursive=True):
        stem = os.path.basename(p_)
        key = stem.split("_")[0] + "_" + stem.split("_")[1]      # IDRiD_01
        masks.setdefault(key, []).append(p_)
    if not masks:
        raise FileNotFoundError(f"No IDRiD .tif lesion masks under {idrid_root}")

    c = class_names.index(disease)
    tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    model.eval()
    scores = []
    for key, mask_paths in sorted(masks.items()):
        if key not in imgs:
            continue
        pil = Image.open(imgs[key]).convert("RGB")
        x = tf(pil).unsqueeze(0).to(DEVICE)
        _, maps, _ = model(x)
        sal = _upsample(maps[0, c], cfg.img_size)
        pred = sal >= np.percentile(sal, pct)

        # union of every lesion type = "somewhere a lesion exists"
        gt = None
        for mp in mask_paths:
            m = np.array(Image.open(mp).convert("L").resize(
                (cfg.img_size, cfg.img_size), Image.NEAREST)) > 0
            gt = m if gt is None else np.logical_or(gt, m)
        if gt is None or gt.sum() == 0:
            continue
        scores.append(dice_iou(pred, gt))

    if not scores:
        raise RuntimeError("IDRiD: no image/mask pairs matched — check the layout")
    return {"idrid_n": len(scores),
            "idrid_dice": float(np.mean([s["dice"] for s in scores])),
            "idrid_iou": float(np.mean([s["iou"] for s in scores]))}


# =====================================================================
# 6.  TEXT REPORT
# =====================================================================
def peak_region(map2d: torch.Tensor) -> str:
    m = map2d.detach().cpu().numpy()
    y, x = np.unravel_index(m.argmax(), m.shape)
    H, W = m.shape
    v = "upper" if y < H / 3 else ("lower" if y > 2 * H / 3 else "central")
    h = "left" if x < W / 3 else ("right" if x > 2 * W / 3 else "central")
    return "central (macular) region" if v == h == "central" else f"{v}-{h} region"


def make_report(findings, maps, probs, names) -> str:
    if not findings:
        return "Findings: none above threshold.\nImpression: within normal limits."
    lines = ["Findings:"]
    for c in findings:
        lines.append(f"  - {names[c]} (conf {probs[c]:.2f}): "
                     f"activation in the {peak_region(maps[0, c])}.")
    lines.append(f"\nImpression: consistent with "
                 f"{', '.join(names[c] for c in findings)}. Correlate clinically.")
    return "\n".join(lines)


# =====================================================================
# 7.  FIGURES
# =====================================================================
def save_figures(hist, y_true, y_pred, aucs, names, out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["loss"], marker="o"); ax[0].set_title("Training loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[1].plot(hist["val_auc"], marker="o", color="green")
    ax[1].set_title("Validation mean AUC"); ax[1].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(f"{out}/curves.png", dpi=130); plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(names, [0 if np.isnan(a) else a for a in aucs], color="steelblue")
    plt.axhline(np.nanmean(aucs), color="red", ls="--",
                label=f"mean={np.nanmean(aucs):.3f}")
    plt.ylim(0, 1); plt.ylabel("AUC"); plt.title("Per-class AUC (test)")
    plt.xticks(rotation=45, ha="right"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/auc_bar.png", dpi=130); plt.close()

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    C = len(names)
    cols = min(4, C)
    rows = int(np.ceil(C / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    for c, ax in enumerate(np.atleast_1d(axes).ravel()):
        if c >= C:
            ax.axis("off"); continue
        ax.imshow(mcm[c], cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, mcm[c][i, j], ha="center", va="center")
        ax.set_title(names[c]); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"]); ax.set_yticklabels(["Neg", "Pos"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle("Per-class confusion matrices (test)")
    plt.tight_layout(); plt.savefig(f"{out}/confusion.png", dpi=130); plt.close()


# =====================================================================
# 8.  ONE EXPERIMENT
# =====================================================================
def run_experiment(cfg: Config) -> dict:
    set_seed(cfg.seed)
    out = os.path.join(cfg.out_dir, cfg.run_name())
    os.makedirs(out, exist_ok=True)
    print(f"\n{'=' * 70}\n[run] {cfg.run_name()}  device={DEVICE}\n{'=' * 70}")

    loaders, names, test_ds = make_loaders(cfg)
    model = build_model(cfg, len(names)).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {cfg.model}: {n_par:,} trainable parameters")

    hist, best_val = train(model, loaders, cfg, os.path.join(out, "best.pth"))

    # thresholds from VAL, applied unchanged to TEST
    vp, vy = predict(model, loaders["val"])
    ths = tune_thresholds(vy, vp)

    tp, ty = predict(model, loaders["test"])
    pred = (tp > ths[None, :]).astype(int)
    auc, aucs = mean_auc(ty, tp)

    res = {
        "config": asdict(cfg),
        "n_params": int(n_par),
        "best_val_mAUC": float(best_val),
        "test_mAUC": float(auc),
        "test_macro_f1": float(f1_score(ty, pred, average="macro", zero_division=0)),
        "test_micro_f1": float(f1_score(ty, pred, average="micro", zero_division=0)),
        "test_hamming_acc": float((pred == ty).mean()),
        "test_exact_match": float(accuracy_score(ty, pred)),
        "per_class_auc": {n: (None if np.isnan(a) else float(a))
                          for n, a in zip(names, aucs)},
        "thresholds": ths.tolist(),
        "class_names": names,
    }

    print(f"\n[test] mAUC={res['test_mAUC']:.4f}  macro-F1={res['test_macro_f1']:.4f}  "
          f"micro-F1={res['test_micro_f1']:.4f}  hamming={res['test_hamming_acc']:.4f}")
    print(classification_report(ty, pred, target_names=names, zero_division=0))

    if cfg.eval_faithfulness:
        print("[interp] deletion / insertion ...")
        res.update(faithfulness(model, test_ds, cfg, cfg.faith_images, cfg.faith_steps))
        print("[interp] prototype disentanglement ...")
        res.update(disentanglement(model, test_ds))
        print(f"[interp] deletion={res['deletion_auc']:.4f} (lower better)  "
              f"insertion={res['insertion_auc']:.4f} (higher better)  "
              f"proto-IoU={res['prototype_iou']:.4f} (lower better)")

    save_figures(hist, ty, pred, aucs, names, out)

    # qualitative demo on the first diseased test image
    # pick a test image that actually carries a disease, so localisation has
    # something to point at. "Normal" is a label in ODIR but not in RFMiD.
    skip = [i for i, n in enumerate(names) if n.lower() in ("normal", "disease_risk")]
    dcols = [i for i in range(len(names)) if i not in skip]
    dis = np.where(test_ds.labels[:, dcols].sum(1) > 0)[0]
    i = int(dis[0]) if len(dis) else 0
    x, _ = test_ds[i]
    with torch.no_grad():
        logits, maps, _ = model(x.unsqueeze(0).to(DEVICE))
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    finds = [c for c in range(len(names)) if probs[c] > ths[c] and names[c] != "Normal"]
    res["demo_report"] = make_report(finds, maps, probs, names)
    print("\n=== DEMO REPORT ===\n" + res["demo_report"])

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[done] -> {out}")
    return res


# =====================================================================
# 9.  THE PAPER GRID
# =====================================================================
def aggregate(results: list[dict]) -> pd.DataFrame:
    """mean +- std across seeds, one row per (dataset, model)."""
    rows = []
    for r in results:
        rows.append({"dataset": r["config"]["dataset"],
                     "model": r["config"]["model"],
                     "seed": r["config"]["seed"],
                     "mAUC": r["test_mAUC"],
                     "macroF1": r["test_macro_f1"],
                     "microF1": r["test_micro_f1"],
                     "hamming": r["test_hamming_acc"],
                     "deletion": r.get("deletion_auc", np.nan),
                     "insertion": r.get("insertion_auc", np.nan),
                     "protoIoU": r.get("prototype_iou", np.nan)})
    df = pd.DataFrame(rows)
    return df.groupby(["dataset", "model"]).agg(["mean", "std"]).round(4)


def to_latex(agg: pd.DataFrame, path: str) -> None:
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Dataset & Model & mAUC & Macro-F1 & Deletion $\downarrow$ & Insertion $\uparrow$ \\",
             r"\midrule"]
    for (ds, mdl), row in agg.iterrows():
        def cell(k):
            m, s = row[(k, "mean")], row[(k, "std")]
            return "---" if pd.isna(m) else (f"{m:.3f}" if pd.isna(s) else f"{m:.3f} $\\pm$ {s:.3f}")
        lines.append(f"{ds} & {mdl.replace('_', ' ')} & {cell('mAUC')} & "
                     f"{cell('macroF1')} & {cell('deletion')} & {cell('insertion')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[latex] wrote {path}")


def run_all(datasets=("odir",), models=("custom_proto", "custom_plain", "resnet_cbam"),
            seeds=(0, 1, 2), **overrides):
    results = []
    for ds in datasets:
        for mdl in models:
            for sd in seeds:
                cfg = Config(dataset=ds, model=mdl, seed=sd, **overrides)
                try:
                    results.append(run_experiment(cfg))
                except Exception as e:                      # keep the grid going
                    print(f"[FAIL] {cfg.run_name()}: {type(e).__name__}: {e}")
    if results:
        os.makedirs("runs", exist_ok=True)
        with open("runs/all_results.json", "w") as f:
            json.dump(results, f, indent=2)
        agg = aggregate(results)
        print("\n" + "=" * 70 + "\nAGGREGATE (mean +- std over seeds)\n" + "=" * 70)
        print(agg.to_string())
        agg.to_csv("runs/aggregate.csv")
        to_latex(agg, "runs/table_main.tex")
    return results


# =====================================================================
# 10. CLI
# =====================================================================
def main():
    p = argparse.ArgumentParser(description="CIPL-Eye experiment harness")
    p.add_argument("--dataset", default="odir", choices=["odir", "rfmid"])
    p.add_argument("--model", default="custom_proto",
                   choices=["custom_proto", "custom_plain", "resnet_cbam"])
    p.add_argument("--data-root", default="")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-pos", type=int, default=50)
    p.add_argument("--no-faithfulness", action="store_true")
    p.add_argument("--no-amp", action="store_true",
                   help="disable mixed precision (slower, bit-for-bit fp32)")
    p.add_argument("--run-all", action="store_true")
    a = p.parse_args()

    if a.run_all:
        run_all(epochs=a.epochs, batch=a.batch, lr=a.lr, amp=not a.no_amp,
                eval_faithfulness=not a.no_faithfulness)
        return

    run_experiment(Config(dataset=a.dataset, model=a.model, data_root=a.data_root,
                          epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed,
                          min_pos=a.min_pos, amp=not a.no_amp,
                          eval_faithfulness=not a.no_faithfulness))


if __name__ == "__main__":
    main()
