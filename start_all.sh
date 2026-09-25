#!/usr/bin/env bash
# One command to run the whole NanoInspect prototype on a ZGX Nano / DGX Spark (GB10) after ./setup.sh.
#   ./start_all.sh              # fine-tuned tiers (7B + LoRA, 27B BF16 + LoRA), cloud tier, edge app
#   T2=nvfp4 ./start_all.sh     # untrained NVFP4 27B instead (faster, less memory, no tier-2 adapter needed)
#   ./start_all.sh stop         # stop everything this script started
# Then open http://localhost:8080 (edge app) and http://localhost:9000 (cloud review console).
# From a laptop: ssh -L 8080:localhost:8080 -L 9000:localhost:9000 <user>@<nano-ip>
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p artifacts/logs
PIDS=artifacts/logs/start_all.pids

if [ "${1:-}" = "stop" ]; then
  [ -f "$PIDS" ] && while read -r p; do kill "$p" 2>/dev/null || true; done < "$PIDS"
  rm -f "$PIDS"; echo "stopped"; exit 0
fi

up() { curl -sf "http://127.0.0.1:$1$2" > /dev/null; }
: > "$PIDS"

# 1. Models (vLLM, OpenAI-compatible API). Skipped if already running.
if ! up 8001 /v1/models; then
  TIER1_MEM=${TIER1_MEM:-0.18} ./serve_models.sh tier1
  pgrep -f "port 8001" >> "$PIDS" || true
fi
if ! up 8002 /v1/models; then
  if [ "${T2:-ft}" = "nvfp4" ] || [ ! -f artifacts/models/vlm_lora_t2/adapter_config.json ]; then
    ./serve_models.sh tier2
  else
    ./serve_models.sh tier2ft
  fi
  pgrep -f "port 8002" >> "$PIDS" || true
fi

# 2. Cloud escalation tier (human review only, no AI).
if ! up 9000 /api/health && ! up 9000 /; then
  nohup ./run_cloud.sh > artifacts/logs/cloud.log 2>&1 & echo $! >> "$PIDS"
fi

# 3. Edge app (operator UI + API).
if ! up 8080 /api/meta; then
  nohup ./run_edge.sh > artifacts/logs/edge.log 2>&1 & echo $! >> "$PIDS"
fi
for i in $(seq 1 90); do up 8080 /api/meta && break; sleep 2; done
up 8080 /api/meta && echo "NanoInspect is running:  edge http://localhost:8080   cloud http://localhost:9000" \
                  || { echo "edge app did not start; see artifacts/logs/edge.log"; exit 1; }
