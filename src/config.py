"""
Central configuration for ScreenSage.

All values can be overridden via environment variables (or a `.env` file
loaded with python-dotenv), so the same code runs unmodified in a local
dev environment, inside Docker, or in the judging VM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class LLMConfig:
    model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    vllm_host: str = os.getenv("VLLM_HOST", "127.0.0.1")
    vllm_port: int = int(os.getenv("VLLM_PORT", "8000"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))

    @property
    def base_url(self) -> str:
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"


@dataclass(frozen=True)
class VisionConfig:
    weights: str = os.getenv("YOLO_WEIGHTS", str(PROJECT_ROOT / "models" / "yolov10n.pt"))
    conf_threshold: float = float(os.getenv("YOLO_CONF_THRESHOLD", "0.35"))
    iou_threshold: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
    device: str = os.getenv("YOLO_DEVICE", "cuda:0")  # ROCm torch reports GPUs as cuda:N
    max_image_dim: int = int(os.getenv("YOLO_MAX_IMAGE_DIM", "1280"))


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = int(os.getenv("MAX_AGENT_STEPS", "20"))
    step_timeout_seconds: int = int(os.getenv("STEP_TIMEOUT_SECONDS", "30"))
    allow_actions: tuple = field(
        default_factory=lambda: ("click", "move", "type_text", "key", "scroll", "wait", "done")
    )


@dataclass(frozen=True)
class WebConfig:
    host: str = os.getenv("WEB_HOST", "0.0.0.0")
    port: int = int(os.getenv("WEB_PORT", "7860"))
    debug: bool = _env_bool("WEB_DEBUG", False)


@dataclass(frozen=True)
class AMDConfig:
    hip_visible_devices: str = os.getenv("HIP_VISIBLE_DEVICES", "0")
    rocr_visible_devices: str = os.getenv("ROCR_VISIBLE_DEVICES", "0")


LLM = LLMConfig()
VISION = VisionConfig()
AGENT = AgentConfig()
WEB = WebConfig()
AMD = AMDConfig()

# Make sure ROCm picks up the intended GPU(s) as early as possible.
os.environ.setdefault("HIP_VISIBLE_DEVICES", AMD.hip_visible_devices)
os.environ.setdefault("ROCR_VISIBLE_DEVICES", AMD.rocr_visible_devices)
