# syntax=docker/dockerfile:1
# Build explicitly for the judging VM's architecture:
#   docker buildx build --platform linux/amd64 --tag screensage:latest --push .
FROM --platform=linux/amd64 rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_release-2.1.2

LABEL org.opencontainers.image.title="ScreenSage" \
      org.opencontainers.image.description="Local vision-grounded LLM agent on AMD ROCm" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# System deps for screen capture / X11 headless operation inside the container
RUN apt-get update && apt-get install -y --no-install-recommends \
        scrot xvfb x11-utils curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "vllm[rocm]" || true

COPY src ./src
COPY scripts ./scripts
COPY .env.example ./.env

# Bake in model weights at build time so there is no first-request cold start.
ARG YOLO_WEIGHTS=yolov10n.pt
ARG LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
ENV YOLO_WEIGHTS=/app/models/${YOLO_WEIGHTS} \
    LLM_MODEL=${LLM_MODEL} \
    HIP_VISIBLE_DEVICES=0 \
    ROCR_VISIBLE_DEVICES=0 \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=7860 \
    DISPLAY=:99

RUN mkdir -p /app/models \
    && python3 -c "from ultralytics import YOLO; YOLO('${YOLO_WEIGHTS}')" \
    && find / -maxdepth 2 -iname "${YOLO_WEIGHTS}" -exec cp {} /app/models/ \; 2>/dev/null || true

EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=5s --start-period=55s --retries=3 \
  CMD curl -sf http://localhost:7860/healthz || exit 1

# Start a virtual X display for headless screen capture, then run the entrypoint.
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
