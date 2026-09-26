#!/usr/bin/env bash
# Start the NanoInspect edge app on the ZGX Nano.
#   ./run_edge.sh            local only: http://localhost:8080 (from a laptop: ssh -L 8080:localhost:8080 <user>@<nano-ip>)
#   ./run_edge.sh --https    HTTPS on every network interface, so phones and laptops can use the live camera:
#                            https://<nano-ip>:8080 (self-signed certificate: accept the browser warning once)
cd "$(dirname "$0")"
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo /home/hp14/jupyterlab/.venv/bin/python )}
export NANOINSPECT_CLOUD_URL=${NANOINSPECT_CLOUD_URL:-http://127.0.0.1:9000}
export NANOINSPECT_HOST=${NANOINSPECT_HOST:-127.0.0.1}
export NANOINSPECT_DELTA_DEVICE=${NANOINSPECT_DELTA_DEVICE:-cpu}   # both LLM servers hold most GPU memory; the difference map is cheap on CPU
if [ "${1:-}" = "--https" ]; then
  CERTS=artifacts/certs; mkdir -p "$CERTS"
  if [ ! -f "$CERTS/edge.pem" ]; then
    SAN="DNS:localhost,IP:127.0.0.1"; for ip in $(hostname -I); do case "$ip" in *:*) ;; *) SAN="$SAN,IP:$ip";; esac; done
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=nanoinspect-edge" -addext "subjectAltName=$SAN" \
      -keyout "$CERTS/edge.key" -out "$CERTS/edge.pem" 2>/dev/null && echo "created self-signed certificate for $SAN"
  fi
  export NANOINSPECT_HOST=0.0.0.0 NANOINSPECT_SSL_CERT="$CERTS/edge.pem" NANOINSPECT_SSL_KEY="$CERTS/edge.key"
fi
exec "$PY" -m nanoinspect.server
