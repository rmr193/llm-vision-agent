"""
Object / UI-element detector powered by YOLOv10 (via the `ultralytics`
package), running on an AMD GPU through ROCm-enabled PyTorch.

ROCm builds of PyTorch expose the GPU under the same `cuda` device namespace
as NVIDIA builds (this is intentional, upstream, to keep code portable), so
no AMD-specific code branches are needed here beyond picking the device and
logging which backend actually got used.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

from src.config import VISION

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


@dataclass
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple  # (x1, y1, x2, y2) in pixel coords of the input image
    element_id: int  # stable index within this frame, referenced by the LLM

    def center(self) -> tuple:
        x1, y1, x2, y2 = self.box_xyxy
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def to_dict(self) -> dict:
        cx, cy = self.center()
        return {
            "id": self.element_id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "box": [int(v) for v in self.box_xyxy],
            "center": [cx, cy],
        }


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    device: str = "cpu"

    def to_json(self) -> List[dict]:
        return [d.to_dict() for d in self.detections]


def _select_device() -> str:
    """Pick the best available device, preferring the ROCm/CUDA GPU."""
    requested = VISION.device
    if torch is None:
        return "cpu"
    if requested.startswith("cuda") and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        backend = "ROCm/HIP" if getattr(torch.version, "hip", None) else "CUDA"
        logger.info(f"Vision model using GPU 0: {name} (backend: {backend})")
        return requested
    logger.warning("Requested GPU device unavailable, falling back to CPU for detection.")
    return "cpu"


class UIElementDetector:
    """Wraps a YOLOv10 checkpoint to detect UI elements / objects in a frame."""

    def __init__(self, weights_path: str | None = None):
        if YOLO is None:
            raise RuntimeError(
                "The 'ultralytics' package is required. Install with `pip install ultralytics`."
            )
        weights_path = weights_path or VISION.weights
        if not Path(weights_path).exists():
            logger.warning(
                f"Weights not found at {weights_path}. "
                "Run `bash scripts/download_models.sh` first, or ultralytics will "
                "auto-download a default checkpoint on first use."
            )
        self.device = _select_device()
        self.model = YOLO(weights_path)
        logger.info(f"Loaded YOLO model from {weights_path} on device={self.device}")

    def detect(self, frame_rgb: np.ndarray) -> DetectionResult:
        start = time.time()
        results = self.model.predict(
            source=frame_rgb,
            conf=VISION.conf_threshold,
            iou=VISION.iou_threshold,
            device=self.device,
            imgsz=VISION.max_image_dim,
            verbose=False,
        )
        elapsed_ms = (time.time() - start) * 1000

        detections: List[Detection] = []
        if results:
            result = results[0]
            names = result.names
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].tolist()
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    detections.append(
                        Detection(
                            label=label,
                            confidence=conf,
                            box_xyxy=tuple(xyxy),
                            element_id=i,
                        )
                    )

        return DetectionResult(detections=detections, inference_ms=elapsed_ms, device=self.device)
