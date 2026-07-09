#!/usr/bin/env bash
# Launches the vLLM server (ROCm) in the background, waits for it to be ready,
# then starts the ScreenSage web dashboard in the foreground.
set -euo pipefail

cd "$(dirname "$0")/.."

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
WEB_PORT="${WEB_PORT:-7860}"

echo "== Starting vLLM server (${LLM_MODEL}) on ${VLLM_HOST}:${VLLM_PORT} =="
vllm serve "${LLM_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --dtype auto \
  > vllm_server.log 2>&1 &
VLLM_PID=$!

echo "== Waiting for vLLM server to become healthy... =="
for i in $(seq 1 60); do
  if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "vLLM server ready after ${i}s."
    break
  fi
  sleep 1
done

cleanup() {
  echo "Shutting down vLLM server (pid ${VLLM_PID})..."
  kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "== Starting ScreenSage dashboard on http://localhost:${WEB_PORT} =="
WEB_PORT="${WEB_PORT}" python3 -m src.web_ui
