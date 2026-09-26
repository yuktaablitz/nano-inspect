#!/usr/bin/env bash
# One script to run the whole NanoInspect prototype on an HP ZGX Nano / DGX Spark (GB10), after ./setup.sh.
#
#   ./start_all.sh               start both model tiers (vLLM), the cloud review tier and the edge app; wait until ready
#   ./start_all.sh --https       same, but serve the edge app over HTTPS on the network (phones/laptops can use the camera)
#   ./start_all.sh demo          run the three demo decisions end to end (accept, reject, human review) and check them
#   ./start_all.sh status        health of every component
#   ./start_all.sh stop          stop what this script started
#
#   Tier 2 is the fine-tuned 27B served with FP8 weights (1.8x faster than BF16, same accuracy: artifacts/results/tier2_fp8_vs_bf16.json)
#   TIER2_QUANT= ./start_all.sh  serve it in full BF16 instead;  T2=nvfp4 ./start_all.sh  serve the untrained NVFP4 27B
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p artifacts/logs
PIDS=artifacts/logs/start_all.pids
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo /home/hp14/jupyterlab/.venv/bin/python )}

up()  { curl -skf --max-time 5 "$1" > /dev/null; }
edge_url() { if up "https://127.0.0.1:8080/api/meta"; then echo "https://127.0.0.1:8080"; elif up "http://127.0.0.1:8080/api/meta"; then echo "http://127.0.0.1:8080"; fi; }
lan_ip() { hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' | grep -v '^172\.17\.' | head -1; }

status() {
  local ok=0
  for c in "tier 1 (7B + LoRA)|http://127.0.0.1:8001/v1/models" "tier 2 (27B)|http://127.0.0.1:8002/v1/models" "cloud review tier|http://127.0.0.1:9000/"; do
    if up "${c#*|}"; then printf '  %-22s up\n' "${c%%|*}"; else printf '  %-22s DOWN\n' "${c%%|*}"; ok=1; fi
  done
  local e; e=$(edge_url)
  if [ -n "$e" ]; then printf '  %-22s up (%s)\n' "edge app" "$e"; else printf '  %-22s DOWN\n' "edge app"; ok=1; fi
  return $ok
}

urls() {
  local e ip; e=$(edge_url); ip=$(lan_ip); local scheme=${e%%:*}
  echo
  echo "NanoInspect is running."
  echo "  Operator app : ${scheme}://localhost:8080"
  if [ "$scheme" = "https" ] && [ -n "$ip" ]; then
    echo "  From a phone or laptop on the same network: https://$ip:8080  (accept the self-signed certificate once)"
  else
    echo "  From a laptop: ssh -L 8080:localhost:8080 -L 9000:localhost:9000 $USER@${ip:-<nano-ip>}, then http://localhost:8080"
  fi
  echo "  Pitch        : ${scheme}://localhost:8080/pitch"
  echo "  Cloud review : http://localhost:9000${ip:+   (network: http://$ip:9000)}"
}

demo() {
  local e; e=$(edge_url)
  [ -z "$e" ] && { echo "edge app is not running; start it with ./start_all.sh"; return 1; }
  echo "Running the three demo decisions on the Nano (tier-2 decisions take ~6-10 s each)..."
  E="$e" "$PY" - <<'PYEOF'
import json, os, ssl, sys, time, urllib.request
E = os.environ["E"]; ctx = ssl._create_unverified_context()
def call(path, body=None):
    req = urllib.request.Request(E + path, data=json.dumps(body).encode() if body else None, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, context=ctx, timeout=180))
fails = 0
for s in call("/api/demo_scenarios"):
    t = time.time(); r = call("/api/inspect_path", {"category": s["category"], "path": s["path"]})
    ok = r["decision"] == s["expect"]; fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {s['title']:<18} expected {s['expect']:<13} got {r['decision']:<13} at tier {r['tier']}  {time.time() - t:5.1f} s")
print("All three decisions as expected." if not fails else f"{fails} scenario(s) differed from the expected outcome.")
sys.exit(1 if fails else 0)
PYEOF
}

case "${1:-start}" in
  stop)
    [ -f "$PIDS" ] && while read -r p; do kill "$p" 2>/dev/null || true; done < "$PIDS"
    rm -f "$PIDS"; echo "stopped the components started by this script"; exit 0 ;;
  status) status; exit $? ;;
  demo)   demo; exit $? ;;
  start|--https) ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac

touch "$PIDS"
# 1. Model tiers (vLLM, OpenAI-compatible API on localhost). Skipped if already running.
if ! up http://127.0.0.1:8001/v1/models; then
  TIER1_MEM=${TIER1_MEM:-0.18} ./serve_models.sh tier1 || { echo "tier 1 failed; see artifacts/logs/vllm_tier1.log"; exit 1; }
  pgrep -f -- "--port 8001" >> "$PIDS" || true
fi
if ! up http://127.0.0.1:8002/v1/models; then
  if [ "${T2:-ft}" = "nvfp4" ] || [ ! -f artifacts/models/vlm_lora_t2/adapter_config.json ]; then ./serve_models.sh tier2; else TIER2_QUANT=${TIER2_QUANT-fp8} ./serve_models.sh tier2ft; fi \
    || { echo "tier 2 failed; see artifacts/logs/vllm_tier2.log"; exit 1; }
  pgrep -f -- "--port 8002" >> "$PIDS" || true
fi
# 2. Cloud escalation tier (human review only, no AI).
if ! up http://127.0.0.1:9000/; then
  nohup ./run_cloud.sh > artifacts/logs/cloud.log 2>&1 & echo $! >> "$PIDS"
fi
# 3. Edge app (operator UI + API).
if [ -z "$(edge_url)" ]; then
  if [ "${1:-}" = "--https" ]; then nohup ./run_edge.sh --https > artifacts/logs/edge.log 2>&1 &
  else nohup ./run_edge.sh > artifacts/logs/edge.log 2>&1 & fi
  echo $! >> "$PIDS"
  echo "edge app starting (loads references and the difference map, ~1-2 min)..."
  for i in $(seq 1 90); do [ -n "$(edge_url)" ] && break; sleep 3; done
fi
status || { echo "a component is down; logs are in artifacts/logs/"; exit 1; }
urls
echo
echo "Check it end to end: ./start_all.sh demo"
