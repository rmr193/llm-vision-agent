"""
Cross-platform screen / window capture.

Uses `mss` for fast, low-latency full-screen or region grabs. This is the
"eyes" of the agent — every perceive→reason→act loop iteration starts here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import mss
except ImportError:  # pragma: no cover - handled gracefully at runtime
    mss = None

from loguru import logger


@dataclass
class CaptureRegion:
    """A capture region in absolute screen coordinates."""

    left: int
    top: int
    width: int
    height: int

    def as_mss_dict(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


class ScreenCapture:
    """Grabs frames from the display as numpy (H, W, 3) RGB arrays."""

    def __init__(self, monitor_index: int = 1):
        if mss is None:
            raise RuntimeError(
                "The 'mss' package is required for screen capture. Install with `pip install mss`."
            )
        self._sct = mss.mss()
        self.monitor_index = monitor_index
        if monitor_index >= len(self._sct.monitors):
            logger.warning(
                f"Monitor index {monitor_index} not found, falling back to monitor 1 (full virtual screen)."
            )
            self.monitor_index = 1

    @property
    def monitor(self) -> dict:
        return self._sct.monitors[self.monitor_index]

    def grab_full(self) -> np.ndarray:
        """Grab the full configured monitor as an RGB numpy array."""
        raw = self._sct.grab(self.monitor)
        frame = np.array(raw)  # BGRA
        return frame[:, :, [2, 1, 0]]  # -> RGB, drop alpha

    def grab_region(self, region: CaptureRegion) -> np.ndarray:
        raw = self._sct.grab(region.as_mss_dict())
        frame = np.array(raw)
        return frame[:, :, [2, 1, 0]]

    def screen_size(self) -> Tuple[int, int]:
        m = self.monitor
        return m["width"], m["height"]

    def close(self):
        try:
            self._sct.close()
        except Exception:
            pass

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *exc):
        self.close()
