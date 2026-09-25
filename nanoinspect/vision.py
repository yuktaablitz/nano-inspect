"""Per-category ResNet-18 anomaly classifiers: training, evaluation, and Grad-CAM."""
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ResNet18_Weights

from . import defects
from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR, SEED, TEXTURE_CATEGORIES, CATEGORIES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def to_tensor(arr):
    """uint8 HxWx3 array -> normalized float tensor 3xHxW."""
    t = torch.tensor(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255
    return (t - _MEAN) / _STD


def denormalize(t):
    return (t.cpu() * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


class TrainSet(Dataset):
    """Each epoch: every good image once (half of them turned into synthetic defects), plus
    (variant C) every real split-A defect twice. Randomness comes from a per-worker generator."""

    def __init__(self, category, variant, goods, real_defects, donors, foreign_pool):
        self.category, self.variant = category, variant
        self.goods, self.real_defects = goods, real_defects
        self.donors, self.foreign_pool = donors, foreign_pool
        self.items = [("good", i) for i in range(len(goods))]
        if variant == "C_real_defects":
            self.items += [("real", i) for i in range(len(real_defects))] * 2
        self._rng = None
        self._fg = [None] * len(goods)

    def __len__(self):
        return len(self.items)

    def _synth(self, img, i, rng):
        if self._fg[i] is None:
            self._fg[i] = defects.foreground_mask(img, self.category)
        fg = self._fg[i]
        if self.variant == "A_simple_synthetic":
            return defects.simple_defect(img, rng)[0]
        if self.variant == "C_real_defects" and self.donors and rng.random() < 0.5:
            d_img, d_mask = self.donors[int(rng.integers(len(self.donors)))]
            return defects.transplant_defect(img, rng, d_img, d_mask, self.category)[0]
        return defects.realistic_defect(img, rng, fg, self.foreign_pool)[0]

    def __getitem__(self, idx):
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        rng = self._rng
        kind, i = self.items[idx]
        if kind == "real":
            img, label = self.real_defects[i], 1
        else:
            img, label = self.goods[i], 0
            if rng.random() < 0.5:
                img, label = self._synth(img, i, rng), 1
        if rng.random() < 0.5: img = img[:, ::-1]
        if self.category in TEXTURE_CATEGORIES and rng.random() < 0.5: img = img[::-1]
        return to_tensor(img), label


def fixed_validation(category, val_goods, foreign_pool, seed=SEED):
    """Held-out good images, each once clean and once with a seeded realistic defect."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for img in val_goods:
        fg = defects.foreground_mask(img, category)
        xs += [img, defects.realistic_defect(img, rng, fg, foreign_pool)[0]]; ys += [0, 1]
    return xs, np.array(ys)


def build_model(pretrained=True):
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith(("layer3", "layer4"))
    m.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(m.fc.in_features, 2))
    return m.to(DEVICE)


@torch.no_grad()
def score_arrays(model, arrays, batch_size=64, amp=False):
    model.eval(); out = []
    for i in range(0, len(arrays), batch_size):
        x = torch.stack([to_tensor(a) for a in arrays[i:i + batch_size]]).to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and DEVICE.type == "cuda"):
            out.append(torch.softmax(model(x).float(), 1)[:, 1].cpu())
    return torch.cat(out).numpy() if out else np.array([])


def model_path(category, variant):
    return MODELS_DIR / "resnet" / variant / f"{category}.pt"


def train_category(category, variant, splits, images, masks, foreign_pool, epochs=12, batch_size=32,
                   lr=3e-4, workers=6, force=False, log=print):
    """Train one classifier. Checkpoint = best validation ROC-AUC; ties (synthetic validation
    defects are often all caught) are broken by the score margin between defects and good parts."""
    path = model_path(category, variant)
    if path.exists() and not force:
        return torch.load(path, map_location="cpu", weights_only=False)["meta"]
    torch.manual_seed(SEED); np.random.seed(SEED)
    s = splits[category]
    goods = [images[p] for p in s["fit_good"]]
    real = [images[d["path"]] for d in s["defect_fit"]]
    donors = [(images[d["path"]], masks[d["mask"]]) for d in s["defect_fit"]]
    ds = TrainSet(category, variant, goods, real, donors, foreign_pool)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=True, drop_last=True)
    val_x, val_y = fixed_validation(category, [images[p] for p in s["val_good"]], foreign_pool)

    model = build_model()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best = ((-1.0, -1.0), None, None, 0); history = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train(); tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        val_scores = score_arrays(model, val_x)
        auc = roc_auc_score(val_y, val_scores)
        margin = float(val_scores[val_y == 1].mean() - val_scores[val_y == 0].mean())
        history.append({"epoch": epoch + 1, "loss": tot / n, "val_auc": auc, "val_margin": margin})
        if (round(auc, 3), margin) > best[0]:
            best = ((round(auc, 3), margin), {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    val_scores[val_y == 0], epoch + 1)
    train_s = time.perf_counter() - t0
    del loader; gc.collect()          # shut worker processes down before the next fork
    meta = {"category": category, "variant": variant, "epochs": epochs, "best_epoch": best[3],
            "val_auc": float(best[0][0]), "val_margin": round(best[0][1], 3), "train_seconds": round(train_s, 1), "train_samples_per_epoch": len(ds),
            "review_at": float(np.clip(np.quantile(best[2], 0.95), 0.15, 0.5)), "history": history}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best[1], "meta": meta}, path)
    log(f"{category:<11} {variant:<22} val AUC {best[0][0]:.3f}  margin {best[0][1]:.2f}  epoch {best[3]:>2}  {train_s:5.1f}s")
    return meta


def load_model(category, variant):
    ckpt = torch.load(model_path(category, variant), map_location="cpu", weights_only=False)
    m = build_model(pretrained=False); m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m, ckpt["meta"]


def evaluate(category, variant, splits, images):
    """Metrics on split B (real defects never used for training) + all test/good images."""
    s = splits[category]
    model, meta = load_model(category, variant)
    items = [(p, 0, "good") for p in s["eval_good"]] + [(d["path"], 1, d["type"]) for d in s["defect_eval"]]
    scores = score_arrays(model, [images[p] for p, _, _ in items])
    y = np.array([l for _, l, _ in items]); pred = scores >= 0.5
    row = {"category": category, "variant": variant, "roc_auc": roc_auc_score(y, scores),
           "recall": pred[y == 1].mean(), "false_positive_rate": pred[y == 0].mean(),
           "n_good": int((y == 0).sum()), "n_defect": int(y.sum()), "val_auc": meta["val_auc"],
           "review_at": meta["review_at"], "train_seconds": meta["train_seconds"]}
    preds = pd.DataFrame({"category": category, "variant": variant, "path": [p for p, _, _ in items],
                          "label": y, "defect_type": [t for _, _, t in items], "score": scores})
    return row, preds


class GradCAM:
    def __init__(self, model):
        self.model = model; self.acts = self.grads = None
        layer = model.layer4[-1]
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        x = x.unsqueeze(0).to(DEVICE).requires_grad_(True)
        logits = self.model(x); logits[0, 1].backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        return float(torch.softmax(logits.detach(), 1)[0, 1]), (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def foreign_texture_pool(images_by_category, splits, category, per_category=4, seed=SEED):
    """Good training images from *other* categories, used as foreign textures for Perlin defects."""
    rng = np.random.default_rng(seed); pool = []
    for other, imgs in images_by_category.items():
        if other == category: continue
        keys = splits[other]["fit_good"]
        pool += [imgs[keys[i]] for i in rng.choice(len(keys), per_category, replace=False)]
    return pool
