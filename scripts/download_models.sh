#!/usr/bin/env bash
# Downloads the YOLOv10 detector weights and pre-warms the HuggingFace cache
# for the LLM, so first inference has no cold-start delay.
set -euo pipefail

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "${MODELS_DIR}"

YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov10n.pt}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

echo "== Downloading YOLOv10 weights (${YOLO_WEIGHTS}) =="
python3 - <<PY
from ultralytics import YOLO
YOLO("${YOLO_WEIGHTS}")  # triggers download to the ultralytics cache
PY
# Copy into our models/ dir for a predictable path referenced by .env
find ~/.cache -iname "${YOLO_WEIGHTS}" -exec cp {} "${MODELS_DIR}/" \; 2>/dev/null || true
find . -maxdepth 1 -iname "${YOLO_WEIGHTS}" -exec mv {} "${MODELS_DIR}/" \; 2>/dev/null || true

echo "== Pre-fetching LLM weights (${LLM_MODEL}) into the HuggingFace cache =="
python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${LLM_MODEL}")
PY

echo "== Done. Start the LLM server with: =="
echo "   vllm serve ${LLM_MODEL} --host 127.0.0.1 --port 8000"
