#!/usr/bin/env bash
# Start the NanoInspect edge app on the ZGX Nano. Open it from a laptop with
#   ssh -L 8080:localhost:8080 -L 9000:localhost:9000 <user>@<nano-ip>   ->  http://localhost:8080
cd "$(dirname "$0")"
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo /home/hp14/jupyterlab/.venv/bin/python )}
export NANOINSPECT_CLOUD_URL=${NANOINSPECT_CLOUD_URL:-http://127.0.0.1:9000}
export NANOINSPECT_HOST=${NANOINSPECT_HOST:-127.0.0.1}
exec "$PY" -m nanoinspect.server
