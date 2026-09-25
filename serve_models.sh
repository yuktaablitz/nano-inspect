#!/usr/bin/env bash
# Serve NanoInspect's two vision-language models on the ZGX Nano with vLLM (OpenAI-compatible API).
#   tier 1 (cheap):     Qwen2.5-VL-7B-Instruct + NanoInspect LoRA (fine-tuned on the Nano)  -> :8001
#   tier 2 (expensive): nvidia/Qwen3.8-27B-NVFP4, flags from the vLLM recipe for DGX Spark (GB10) -> :8002
# Usage: ./serve_models.sh [tier1|tier2|both]      Logs: artifacts/logs/vllm_tier{1,2}.log
# ZRT equivalent: replace "vllm serve" with "zrt serve" and keep every flag.
set -euo pipefail
cd "$(dirname "$0")"
VENV=${VENV:-$( [ -x .venv/bin/vllm ] && echo "$PWD/.venv" || echo /home/hp14/jupyterlab/.venv )}
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"      # FlashInfer JIT needs ninja + nvcc
export MAX_JOBS=${MAX_JOBS:-4}   # the first start compiles GB10 kernels; more parallel jobs run out of memory
mkdir -p artifacts/logs
ADAPTER="$PWD/artifacts/models/vlm_lora"

tier1() {
  nohup vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 127.0.0.1 --port 8001 \
    --served-model-name qwen2.5-vl-7b \
    --enable-lora --lora-modules "nanoinspect-7b-lora=$ADAPTER" --max-lora-rank 16 \
    --gpu-memory-utilization ${TIER1_MEM:-0.22} --max-model-len 4096 --max-num-seqs 32 \
    --limit-mm-per-prompt '{"image":1}' --mm-encoder-tp-mode data --enable-prefix-caching \
    > artifacts/logs/vllm_tier1.log 2>&1 &
  echo "tier 1 starting (pid $!)"
}

tier2() {
  # Recipe flags for dgx_spark_gb10, with a smaller context and memory share so both tiers fit together.
  nohup vllm serve nvidia/Qwen3.8-27B-NVFP4 --host 127.0.0.1 --port 8002 \
    --served-model-name qwen3.8-27b \
    --kv-cache-dtype fp8 --gpu-memory-utilization 0.40 --max-model-len 16384 \
    --max-num-seqs 8 --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling \
    --enable-prefix-caching --limit-mm-per-prompt '{"image":2}' --mm-encoder-tp-mode data \
    > artifacts/logs/vllm_tier2.log 2>&1 &
  echo "tier 2 starting (pid $!)"
}

wait_ready() {  # port name
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$1/v1/models" > /dev/null; then echo "$2 ready on :$1"; return 0; fi
    sleep 5
  done
  echo "$2 did not start; see artifacts/logs"; return 1
}

case "${1:-both}" in
  tier1) tier1; wait_ready 8001 "tier 1" ;;
  tier2) tier2; wait_ready 8002 "tier 2" ;;
  both)  tier1; wait_ready 8001 "tier 1"; tier2; wait_ready 8002 "tier 2" ;;
esac
