"""Dataset indexing, reproducible splits, and a decoded-image cache."""
import hashlib
import json
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from .config import (CACHE_DIR, CATEGORIES, DATA_ROOT, DEFECT_FIT_FRACTION, IMAGE_SIZE,
                     LOCATIONS, SEED, VAL_FRACTION, ARTIFACTS, ensure_dirs)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
SPLITS_PATH = ARTIFACTS / "splits.json"


def _images(folder):
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def rel(path):
    return str(path.relative_to(DATA_ROOT))


def defect_types(category):
    return sorted(d.name for d in (DATA_ROOT / category / "test").iterdir() if d.is_dir() and d.name != "good")


def mask_path(category, defect_type, image_path):
    return DATA_ROOT / category / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"


def make_splits(seed=SEED, force=False):
    """Per category:
    fit_good    train/good used for training
    val_good    train/good held out for checkpoint selection and thresholds
    defect_fit  split A: half of the real test defects (stratified by type), allowed in training
    defect_eval split B: the other half, never used for training
    eval_good   all test/good images, never used for training
    """
    ensure_dirs()
    if SPLITS_PATH.exists() and not force:
        return json.loads(SPLITS_PATH.read_text())
    splits = {}
    for cat in CATEGORIES:
        rng = random.Random(f"{seed}-{cat}")
        good = [rel(p) for p in _images(DATA_ROOT / cat / "train" / "good")]
        rng.shuffle(good)
        n_val = max(4, int(len(good) * VAL_FRACTION))
        fit_defects, eval_defects = [], []
        for dtype in defect_types(cat):
            items = [{"path": rel(p), "type": dtype, "mask": rel(mask_path(cat, dtype, p))}
                     for p in _images(DATA_ROOT / cat / "test" / dtype)]
            rng.shuffle(items)
            k = int(round(len(items) * DEFECT_FIT_FRACTION))
            fit_defects += items[:k]; eval_defects += items[k:]
        splits[cat] = {
            "fit_good": sorted(good[n_val:]), "val_good": sorted(good[:n_val]),
            "defect_fit": sorted(fit_defects, key=lambda d: d["path"]),
            "defect_eval": sorted(eval_defects, key=lambda d: d["path"]),
            "eval_good": [rel(p) for p in _images(DATA_ROOT / cat / "test" / "good")],
        }
    SPLITS_PATH.write_text(json.dumps(splits, indent=1))
    return splits


def split_summary(splits):
    rows = []
    for cat, s in splits.items():
        rows.append({"category": cat, "defect_types": len(defect_types(cat)),
                     "fit_good": len(s["fit_good"]), "val_good": len(s["val_good"]),
                     "defect_fit (A)": len(s["defect_fit"]), "defect_eval (B)": len(s["defect_eval"]),
                     "eval_good": len(s["eval_good"])})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = ["TOTAL", df.defect_types.sum()] + [df[c].sum() for c in df.columns[2:]]
    return df


def leakage_check(splits):
    """Content-hash check: no evaluation image may appear anywhere in the training data."""
    def md5(r): return hashlib.md5((DATA_ROOT / r).read_bytes()).hexdigest()
    problems = []
    for cat, s in splits.items():
        train = {md5(r) for r in s["fit_good"] + s["val_good"]} | {md5(d["path"]) for d in s["defect_fit"]}
        evals = {md5(r) for r in s["eval_good"]} | {md5(d["path"]) for d in s["defect_eval"]}
        if train & evals:
            problems.append((cat, len(train & evals)))
    return problems


def location_from_mask(mask):
    """Name the 3x3 grid cell that holds the centre of mass of a defect mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return "none"
    h, w = mask.shape
    col = min(2, int(xs.mean() / w * 3)); row = min(2, int(ys.mean() / h * 3))
    return LOCATIONS[row * 3 + col]


# ---------------------------------------------------------------- decoding + cache
def _decode(args):
    relpath, size, is_mask = args
    im = Image.open(DATA_ROOT / relpath)
    if is_mask:
        return relpath, (np.asarray(im.convert("L").resize((size, size), Image.NEAREST)) > 127).astype(np.uint8)
    return relpath, np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))


def decode_many(jobs, workers):
    if workers <= 1:
        return dict(map(_decode, jobs))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        return dict(ex.map(_decode, jobs, chunksize=8))


def decode_benchmark(category="bottle", worker_counts=(0, 2, 4, 8, 16), size=IMAGE_SIZE):
    """Time full-resolution PNG decode + resize for one category with different worker counts."""
    paths = [rel(p) for p in _images(DATA_ROOT / category / "train" / "good")]
    rows = []
    for w in worker_counts:
        t0 = time.perf_counter(); decode_many([(p, size, False) for p in paths], w)
        dt = time.perf_counter() - t0
        rows.append({"workers": w, "images": len(paths), "seconds": round(dt, 2), "images_per_s": round(len(paths) / dt, 1)})
    return pd.DataFrame(rows)


def load_cache(category, splits, size=IMAGE_SIZE, workers=16):
    """Decode every image (and defect mask) of a category once and keep it in RAM (and on disk)."""
    ensure_dirs()
    path = CACHE_DIR / f"{category}_{size}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["images"].item(), z["masks"].item()
    s = splits[category]
    image_paths = s["fit_good"] + s["val_good"] + s["eval_good"] + [d["path"] for d in s["defect_fit"] + s["defect_eval"]]
    mask_paths = [d["mask"] for d in s["defect_fit"] + s["defect_eval"]]
    images = decode_many([(p, size, False) for p in image_paths], workers)
    masks = decode_many([(p, size, True) for p in mask_paths], workers)
    np.savez(path, images=np.array(images, dtype=object), masks=np.array(masks, dtype=object))
    return images, masks


def load_pil(relpath_or_path):
    p = relpath_or_path if str(relpath_or_path).startswith("/") else DATA_ROOT / relpath_or_path
    return Image.open(p).convert("RGB")
