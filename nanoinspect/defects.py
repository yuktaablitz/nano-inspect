"""Synthetic defect generators. Every function takes and returns uint8 RGB arrays (H, W, 3)
and also returns the binary mask of the pixels it changed.

simple_defect        the original v1 generator: flat rectangles, white lines, grey noise
perlin_defect        DRAEM-style (Zavrtanik et al., ICCV 2021): an irregular Perlin-noise-shaped
                     region blended with a foreign texture; looks like stains, wear, contamination
cutpaste_scar        CutPaste (Li et al., CVPR 2021): a thin strip cut from the same image,
                     colour-jittered, rotated, pasted elsewhere; looks like scratches and cracks
transplant_defect    a real defect (pixels under its ground-truth mask) from split A copied onto a
                     different good part; the most realistic, but needs real defect examples
"""
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .config import TEXTURE_CATEGORIES


def foreground_mask(img, category):
    """Rough object mask so defects land on the part, not on the background."""
    h, w, _ = img.shape
    if category in TEXTURE_CATEGORIES:
        return np.ones((h, w), np.uint8)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.float32)
    dist = np.linalg.norm(img.astype(np.float32) - np.median(border, 0), axis=2)
    m = Image.fromarray(((dist > 40) * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(9))
    m = (np.asarray(m) > 0).astype(np.uint8)
    return m if 0.03 < m.mean() < 0.97 else np.ones((h, w), np.uint8)


def _jitter(pil, rng):
    pil = ImageEnhance.Brightness(pil).enhance(rng.uniform(0.6, 1.4))
    pil = ImageEnhance.Contrast(pil).enhance(rng.uniform(0.6, 1.4))
    return ImageEnhance.Color(pil).enhance(rng.uniform(0.5, 1.5))


def simple_defect(img, rng, fg=None):
    pil = Image.fromarray(img).copy(); w, h = pil.size
    draw = ImageDraw.Draw(pil, "RGBA")
    x1, y1 = int(rng.integers(0, int(w * .65))), int(rng.integers(0, int(h * .65)))
    x2, y2 = int(rng.integers(x1 + 4, min(w, x1 + int(w * .35)))), int(rng.integers(y1 + 4, min(h, y1 + int(h * .35))))
    kind = rng.choice(["rectangle", "scratch", "noise"])
    if kind == "rectangle":
        draw.rectangle([x1, y1, x2, y2], fill=[(255, 30, 30, 190), (20, 20, 20, 210), (255, 220, 0, 180)][rng.integers(3)])
    elif kind == "scratch":
        for _ in range(int(rng.integers(2, 6))):
            draw.line([x1, y1 + int(rng.integers(0, max(1, y2 - y1))), x2, y1 + int(rng.integers(0, max(1, y2 - y1)))],
                      fill=(255, 255, 255, 230), width=int(rng.integers(1, 3)))
    else:
        patch = Image.fromarray(rng.integers(0, 255, (y2 - y1, x2 - x1), dtype=np.uint8)).convert("RGBA")
        patch.putalpha(170); pil.paste(patch, (x1, y1), patch)
    out = np.asarray(pil.convert("RGB"))
    return out, (np.abs(out.astype(int) - img.astype(int)).sum(2) > 0).astype(np.uint8)


def perlin_noise(shape, res, rng):
    """Classic 2D gradient (Perlin) noise; shape must be divisible by res."""
    def fade(t): return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
    d = (shape[0] // res[0], shape[1] // res[1])
    gy, gx = np.meshgrid(np.arange(shape[0]) * res[0] / shape[0], np.arange(shape[1]) * res[1] / shape[1], indexing="ij")
    grid = np.stack([gy % 1, gx % 1], -1)
    angles = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))
    grads = np.dstack([np.cos(angles), np.sin(angles)])
    def tile(g): return g.repeat(d[0], 0).repeat(d[1], 1)
    g00, g10, g01, g11 = tile(grads[:-1, :-1]), tile(grads[1:, :-1]), tile(grads[:-1, 1:]), tile(grads[1:, 1:])
    n00 = (grid * g00).sum(2)
    n10 = (np.dstack([grid[..., 0] - 1, grid[..., 1]]) * g10).sum(2)
    n01 = (np.dstack([grid[..., 0], grid[..., 1] - 1]) * g01).sum(2)
    n11 = (np.dstack([grid[..., 0] - 1, grid[..., 1] - 1]) * g11).sum(2)
    t = fade(grid)
    n0 = n00 * (1 - t[..., 0]) + t[..., 0] * n10
    n1 = n01 * (1 - t[..., 0]) + t[..., 0] * n11
    return np.sqrt(2) * ((1 - t[..., 1]) * n0 + t[..., 1] * n1)


def perlin_defect(img, rng, fg, foreign_pool=()):
    h, w, _ = img.shape
    for _ in range(6):
        res = (2 ** int(rng.integers(1, 5)), 2 ** int(rng.integers(1, 5)))
        noise = perlin_noise((h, w), res, rng)
        mask = ((noise > rng.uniform(0.35, 0.6)) & (fg > 0)).astype(np.uint8)
        if 30 < mask.sum() < 0.35 * max(fg.sum(), 1):
            break
    else:
        return cutpaste_scar(img, rng, fg)
    if len(foreign_pool) and rng.random() < 0.7:
        tex = foreign_pool[int(rng.integers(len(foreign_pool)))]
    else:
        tex = np.rot90(img, int(rng.integers(1, 4))).copy()
    tex = np.asarray(_jitter(Image.fromarray(tex).resize((w, h)), rng)).astype(np.float32)
    beta = rng.uniform(0.0, 0.7)
    m = mask[..., None].astype(np.float32)
    out = img * (1 - m) + (1 - beta) * tex * m + beta * img * m
    return np.clip(out, 0, 255).astype(np.uint8), mask


def cutpaste_scar(img, rng, fg):
    h, w, _ = img.shape
    pw, ph = int(rng.integers(2, 17)), int(rng.integers(10, 26))
    sx, sy = int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))
    patch = _jitter(Image.fromarray(img).crop((sx, sy, sx + pw, sy + ph)), rng).convert("RGBA")
    patch = patch.rotate(rng.uniform(-45, 45), expand=True, resample=Image.BILINEAR)
    ys, xs = np.nonzero(fg)
    i = int(rng.integers(len(xs)))
    x = int(np.clip(xs[i] - patch.width // 2, 0, w - patch.width)); y = int(np.clip(ys[i] - patch.height // 2, 0, h - patch.height))
    pil = Image.fromarray(img).copy(); pil.paste(patch, (x, y), patch)
    mask = np.zeros((h, w), np.uint8)
    mask[y:y + patch.height, x:x + patch.width] = (np.asarray(patch)[..., 3] > 0)
    return np.asarray(pil.convert("RGB")), mask


def realistic_defect(img, rng, fg, foreign_pool=()):
    if rng.random() < 0.5:
        return perlin_defect(img, rng, fg, foreign_pool)
    return cutpaste_scar(img, rng, fg)


def transplant_defect(img, rng, donor_img, donor_mask, category):
    """Copy a real defect onto a good part. Objects in MVTec are roughly aligned, so the
    defect keeps its position; texture defects are shifted to a random place."""
    if category in TEXTURE_CATEGORIES:
        dy, dx = int(rng.integers(img.shape[0])), int(rng.integers(img.shape[1]))
        donor_img = np.roll(donor_img, (dy, dx), (0, 1)); donor_mask = np.roll(donor_mask, (dy, dx), (0, 1))
    alpha = np.asarray(Image.fromarray(donor_mask * 255).filter(ImageFilter.GaussianBlur(1.5)), np.float32)[..., None] / 255
    out = img * (1 - alpha) + donor_img * alpha
    return np.clip(out, 0, 255).astype(np.uint8), (alpha[..., 0] > 0.5).astype(np.uint8)
