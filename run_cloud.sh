#!/usr/bin/env bash
# Start the cloud escalation tier (human review console). No AI inference runs here.
cd "$(dirname "$0")"
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo /home/hp14/jupyterlab/.venv/bin/python )}
exec "$PY" -m uvicorn cloud.server:app --host "${CLOUD_HOST:-0.0.0.0}" --port "${CLOUD_PORT:-9000}"
