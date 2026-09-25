"""The explicit "delta" between a part and known-good parts.

For each product we keep a memory bank of local feature vectors (patches) from ~40 known-good training
images, taken from an ImageNet-pretrained ResNet-18 (layers 2 and 3, no training). For a new part,
every patch is compared with its nearest good patch:

    delta(patch) = min over good patches || f(patch) - f(good patch) ||_2

The per-patch deltas form a heatmap of *where* the part differs from every good part we have seen,
and the largest delta is the part's delta score. This is the training-free PatchCore idea
(Roth et al., CVPR 2022) in its simplest form. It needs only good images, which is what a new
product line has, and it gives the operator and the escalation crop an exact location.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .config import IMAGENET_MEAN, IMAGENET_STD
from .data import load_pil

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = 256


class DeltaMap:
    def __init__(self, splits, n_good=40, bank_per_image=400, seed=0):
        m = resnet18(weights=ResNet18_Weights.DEFAULT).eval().to(DEVICE)
        self.body = torch.nn.ModuleDict({k: getattr(m, k) for k in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3")})
        self.mean = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
        g = torch.Generator().manual_seed(seed)
        self.banks = {}
        for cat, s in splits.items():
            feats = [self.features(load_pil(p)) for p in s["fit_good"][:n_good]]
            bank = []
            for f in feats:
                f = f.reshape(f.shape[0], -1).T
                bank.append(f[torch.randperm(len(f), generator=g)[:bank_per_image].to(f.device)])
            self.banks[cat] = torch.cat(bank)
        # calibrate: the delta score of held-out good images defines "normal" for each product
        self.good_ref = {}
        for cat, s in splits.items():
            sc = [self.score(load_pil(p), cat)[0] for p in s["val_good"][:20]]
            self.good_ref[cat] = (float(np.mean(sc)), float(np.std(sc) + 1e-6))

    @torch.inference_mode()
    def features(self, pil):
        x = torch.from_numpy(np.asarray(pil.convert("RGB").resize((SIZE, SIZE)), dtype=np.float32) / 255).permute(2, 0, 1)[None].to(DEVICE)
        x = (x - self.mean) / self.std
        b = self.body
        x = b["maxpool"](b["relu"](b["bn1"](b["conv1"](x)))); x = b["layer1"](x); f2 = b["layer2"](x); f3 = b["layer3"](f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)
        return F.avg_pool2d(f, 3, 1, 1)[0]                   # C x 32 x 32, locally smoothed

    @torch.inference_mode()
    def score(self, pil, cat):
        """Returns (delta score, heatmap in [0,1] at 256x256, z-score vs held-out good parts)."""
        f = self.features(pil); c, h, w = f.shape
        q = f.reshape(c, -1).T
        d = torch.cdist(q, self.banks[cat]).min(1).values.reshape(1, 1, h, w)
        hm = F.interpolate(d, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        s = float(d.max())
        mu, sd = self.good_ref.get(cat, (s, 1.0))
        return s, hm, (s - mu) / sd

    @staticmethod
    def peak_box(hm, w, h, frac=0.3):
        """Box (in original image pixels) centred on the largest delta."""
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        cx, cy, side = x / hm.shape[1] * w, y / hm.shape[0] * h, int(min(w, h) * frac)
        l = int(np.clip(cx - side / 2, 0, w - side)); t = int(np.clip(cy - side / 2, 0, h - side))
        return (l, t, l + side, t + side)
