"""Paths and constants shared by every NanoInspect module."""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("NANOINSPECT_DATA", "/home/hp14/Downloads/mvtec_anomaly_detection"))
PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_DIR / "artifacts"
CACHE_DIR = ARTIFACTS / "cache"
MODELS_DIR = ARTIFACTS / "models"
RESULTS_DIR = ARTIFACTS / "results"

SEED = 42
IMAGE_SIZE = 256            # ResNet input size (MVTec images are 700-1024 px)
VAL_FRACTION = 0.15         # share of train/good held out for checkpoint selection
DEFECT_FIT_FRACTION = 0.5   # share of real test defects that may be used for training (split A)

CATEGORIES = sorted(p.name for p in DATA_ROOT.iterdir() if (p / "train" / "good").is_dir())
TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}

RESNET_VARIANTS = {
    "A_simple_synthetic": "good images + simple synthetic defects (rectangles, lines, noise)",
    "B_realistic_synthetic": "good images + realistic synthetic defects (Perlin-texture and CutPaste scars)",
    "C_real_defects": "B + half of the real defect images (split A) + real defects transplanted onto good parts",
}
PRODUCTION_VARIANT = "C_real_defects"

VLM_ID = os.environ.get("NANOINSPECT_VLM_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
VLM_IMAGE_SIZE = 448        # 448 px -> 256 visual tokens per image
LOCATIONS = ["top-left", "top", "top-right", "left", "center", "right", "bottom-left", "bottom", "bottom-right"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_dirs():
    for d in (ARTIFACTS, CACHE_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
