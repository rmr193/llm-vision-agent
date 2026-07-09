#!/usr/bin/env bash
set -euo pipefail

echo "== Starting virtual display (Xvfb :99) =="
Xvfb :99 -screen 0 1920x1080x24 &
sleep 1

echo "== Starting vLLM server (${LLM_MODEL}) on 127.0.0.1:8000 =="
vllm serve "${LLM_MODEL}" --host 127.0.0.1 --port 8000 --dtype auto > /app/vllm_server.log 2>&1 &

echo "== Starting ScreenSage dashboard on 0.0.0.0:${WEB_PORT} =="
exec python3 -m src.web_ui
