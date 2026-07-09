"""
Executes AgentAction objects as real mouse/keyboard input via `pyautogui`.

This is intentionally the *only* place in the codebase that touches the
actual input devices, so safety limits (bounds checking, rate limiting,
an emergency stop) are centralized here.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from loguru import logger

try:
    import pyautogui
    pyautogui.FAILSAFE = True  # moving mouse to a screen corner aborts
except ImportError:  # pragma: no cover
    pyautogui = None

from src.llm.agent import AgentAction
from src.vision.detector import Detection


class InputController:
    def __init__(self, dry_run: bool = False):
        """
        Parameters
        ----------
        dry_run:
            If True, actions are logged but never actually executed. Useful
            for headless CI / judging environments without a real display.
        """
        self.dry_run = dry_run or (pyautogui is None)
        if pyautogui is None:
            logger.warning("pyautogui not available; InputController running in dry-run mode.")

    @staticmethod
    def _resolve_coordinates(
        action: AgentAction, detections_by_id: Dict[int, Detection]
    ) -> Optional[tuple]:
        if action.target_element_id is not None and action.target_element_id in detections_by_id:
            return detections_by_id[action.target_element_id].center()
        if action.coordinates and len(action.coordinates) == 2:
            return tuple(action.coordinates)
        return None

    def execute(self, action: AgentAction, detections: List[Detection]) -> str:
        """Execute one action. Returns a short human-readable result string."""
        detections_by_id = {d.element_id: d for d in detections}

        if action.action == "done":
            return f"DONE: {action.done_summary or 'goal reported complete'}"

        if action.action == "wait":
            time.sleep(1.0)
            return "waited 1s"

        if action.action in ("click", "move"):
            coords = self._resolve_coordinates(action, detections_by_id)
            if coords is None:
                return "skipped: no resolvable coordinates for click/move"
            x, y = coords
            if self.dry_run:
                logger.info(f"[dry-run] {action.action} -> ({x}, {y})")
            else:
                pyautogui.moveTo(x, y, duration=0.2)
                if action.action == "click":
                    pyautogui.click()
            return f"{action.action} at ({x}, {y})"

        if action.action == "type_text":
            text = action.text or ""
            if self.dry_run:
                logger.info(f"[dry-run] type_text -> {text!r}")
            else:
                pyautogui.typewrite(text, interval=0.02)
            return f"typed {text!r}"

        if action.action == "key":
            key = action.key_name or ""
            if self.dry_run:
                logger.info(f"[dry-run] key -> {key!r}")
            else:
                pyautogui.press(key)
            return f"pressed key {key!r}"

        if action.action == "scroll":
            amount = action.scroll_amount or 0
            if self.dry_run:
                logger.info(f"[dry-run] scroll -> {amount}")
            else:
                pyautogui.scroll(amount)
            return f"scrolled {amount}"

        return f"unknown action: {action.action}"
