#!/usr/bin/env bash
# One-shot setup on an HP ZGX Nano (or any NVIDIA GB10 / DGX Spark). Usage:
#   ./setup.sh                      # venv + packages + model weights
#   DATA_URL=... ./setup.sh         # also fetch MVTec AD if not present
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
DATA=${NANOINSPECT_DATA:-$HOME/Downloads/mvtec_anomaly_detection}

echo "== 1/4 Python environment (.venv)"
$PY -m venv .venv
. .venv/bin/activate
pip install --upgrade pip wheel
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

echo "== 2/4 Model weights (downloaded once, then used offline)"
python - <<'PY'
from huggingface_hub import snapshot_download
from torchvision.models import resnet18, ResNet18_Weights
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
resnet18(weights=ResNet18_Weights.DEFAULT)
print("weights cached")
PY

echo "== 3/4 Dataset"
if [ ! -d "$DATA/bottle/train/good" ]; then
  echo "MVTec AD not found at $DATA."
  echo "Download it from https://www.mvtec.com/company/research/datasets/mvtec-ad (CC BY-NC-SA 4.0),"
  echo "extract it there, or set NANOINSPECT_DATA to its location."
else
  echo "found MVTec AD at $DATA"
fi

echo "== 4/4 Trained LoRA adapters (from the GitHub release)"
REL=https://github.com/yuktaablitz/nano-inspect/releases/download/adapters-v1
mkdir -p artifacts/models
for f in nanoinspect-tier1-qwen2.5-vl-7b-lora.zip nanoinspect-tier2-qwen3.8-27b-lora.zip; do
  [ -f "artifacts/models/$f" ] || curl -L -o "artifacts/models/$f" "$REL/$f"
  (cd artifacts/models && unzip -oq "$f")
done
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.8-27B'); s('nvidia/Qwen3.8-27B-NVFP4')"

cat <<'MSG'

Setup done. Next:
  . .venv/bin/activate
  jupyter nbconvert --to notebook --execute --inplace nanoinspect.ipynb    # train + evaluate + stress test (~90 min)
  jupyter nbconvert --to notebook --execute --inplace 02_tier2_finetune_and_capacity.ipynb
  ./start_all.sh       # both model tiers + cloud review tier (:9000) + operator app (:8080), in one command
MSG
