#!/usr/bin/env bash
# Serve NanoInspect's two model tiers through HP Z Runtime (zrt), HP's vLLM wrapper on the ZGX Nano.
# Same models, same fine-tuned LoRA adapters and same vLLM settings as serve_models.sh; zrt adds its model cache,
# service management, metrics and one OpenAI-compatible HTTPS front door (proxy) for both tiers.
#
#   scripts/zrt_serve.sh setup     once: proxy on 127.0.0.1:8100 (the web app uses 8080), no JWT (localhost only), TLS on
#   scripts/zrt_serve.sh pull      once: download both base models into zrt's cache
#   scripts/zrt_serve.sh start     start tier 1 and tier 2 (stop the direct-vLLM servers on :8001/:8002 first: memory)
#   scripts/zrt_serve.sh status | stop
# Then run the app against the proxy:  SERVE=zrt ./start_all.sh --https
# zrt's proxy routes requests by service label (one name per service), so each service is labelled with its
# fine-tuned adapter's name: requests for nanoinspect-7b-lora / nanoinspect-27b-lora reach the adapters.
# zrt commands need the "zrt" group; until you log out and back in, this script runs them through `sg zrt`.
set -euo pipefail
cd "$(dirname "$0")/.."
Z() { if id -nG | tr ' ' '\n' | grep -qx zrt; then zrt "$@"; else sg zrt -c "zrt $(printf '%q ' "$@")"; fi; }
T1_ADAPTER="$PWD/artifacts/models/vlm_lora"; T2_ADAPTER="$PWD/artifacts/models/vlm_lora_t2"

case "${1:-}" in
  setup)
    Z config set proxy.port 8100; Z config set proxy.host 127.0.0.1; Z config set proxy.auth.type none
    # zrt requires a certificate when TLS is on: a local self-signed one for 127.0.0.1, kept in zrt's own folder
    if [ ! -f /opt/hp/zrt/tls/proxy.pem ]; then
      sg zrt -c "mkdir -p /opt/hp/zrt/tls && openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj /CN=zrt-proxy \
        -addext subjectAltName=DNS:localhost,IP:127.0.0.1 -keyout /opt/hp/zrt/tls/proxy.key -out /opt/hp/zrt/tls/proxy.pem && chmod 640 /opt/hp/zrt/tls/proxy.key"
    fi
    Z config set proxy.tls.cert /opt/hp/zrt/tls/proxy.pem; Z config set proxy.tls.key /opt/hp/zrt/tls/proxy.key; Z config set proxy.tls.enabled true ;;
  pull)
    Z model pull hf:Qwen/Qwen2.5-VL-7B-Instruct
    Z model pull hf:Qwen/Qwen3.8-27B ;;
  start)
    # tier 1: Qwen2.5-VL-7B + NanoInspect LoRA (same flags as serve_models.sh tier1)
    Z service start hf:Qwen/Qwen2.5-VL-7B-Instruct --label nanoinspect-7b-lora --gpu-memory-fraction "${TIER1_MEM:-0.18}" -- \
      --served-model-name qwen2.5-vl-7b --enable-lora --lora-modules "nanoinspect-7b-lora=$T1_ADAPTER" --max-lora-rank 16 \
      --max-model-len 4096 --max-num-seqs 32 --limit-mm-per-prompt '{"image":1}' --mm-encoder-tp-mode data --enable-prefix-caching
    # tier 2: Qwen3.8-27B + NanoInspect LoRA, FP8 weights (same flags as serve_models.sh tier2ft)
    Q=${TIER2_QUANT-fp8}
    Z service start hf:Qwen/Qwen3.8-27B --label nanoinspect-27b-lora --gpu-memory-fraction "${TIER2_MEM:-0.55}" -- \
      --served-model-name qwen3.8-27b --enable-lora --lora-modules "nanoinspect-27b-lora=$T2_ADAPTER" --max-lora-rank 16 \
      ${Q:+--quantization $Q} --kv-cache-dtype fp8 --max-model-len 16384 --max-num-seqs 8 \
      --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling --enable-prefix-caching \
      --limit-mm-per-prompt '{"image":2}' --mm-encoder-tp-mode data
    Z status ;;
  status) Z status ;;
  stop)
    Z service stop hf:Qwen/Qwen3.8-27B@main || true
    Z service stop hf:Qwen/Qwen2.5-VL-7B-Instruct@main || true ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
