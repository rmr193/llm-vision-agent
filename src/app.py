"""
ScreenSage CLI orchestrator.

Runs the perceive -> reason -> act loop:
  1. Capture the screen (src/vision/screen_capture.py)
  2. Detect UI elements with YOLOv10 on ROCm (src/vision/detector.py)
  3. Ask the local LLM for the next single action (src/llm/agent.py)
  4. Execute that action (src/controller/input_controller.py)
  5. Repeat until the LLM reports "done" or MAX_AGENT_STEPS is hit.

Usage:
    python -m src.app "Open the calculator app and compute 42 * 17"
    python -m src.app --dry-run "Click the Settings gear icon"
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, List, Optional

from loguru import logger

from src.config import AGENT
from src.controller.input_controller import InputController
from src.llm.agent import LLMAgent
from src.vision.detector import UIElementDetector
from src.vision.screen_capture import ScreenCapture


def run_agent_loop(
    goal: str,
    dry_run: bool = False,
    on_step: Optional[Callable[[dict], None]] = None,
) -> List[dict]:
    """
    Runs the full loop and returns the step-by-step log (also streamed to
    `on_step` if provided, e.g. from the web dashboard).
    """
    capture = ScreenCapture()
    detector = UIElementDetector()
    agent = LLMAgent()
    controller = InputController(dry_run=dry_run)

    history: List[str] = []
    log: List[dict] = []

    for step in range(1, AGENT.max_steps + 1):
        frame = capture.grab_full()
        detection_result = detector.detect(frame)
        elements = detection_result.to_json()

        action = agent.next_action(goal=goal, elements=elements, history=history)
        result = controller.execute(action, detection_result.detections)

        history.append(f"{action.to_history_line()} -> {result}")

        entry = {
            "step": step,
            "reasoning": action.reasoning,
            "action": action.action,
            "result": result,
            "num_detections": len(elements),
            "inference_ms": round(detection_result.inference_ms, 1),
            "device": detection_result.device,
        }
        log.append(entry)
        logger.info(f"[step {step}] {action.action} -> {result}")

        if on_step:
            on_step(entry)

        if action.action == "done":
            break

        time.sleep(0.3)  # brief pause so UI state can settle before next capture

    capture.close()
    return log


def main():
    parser = argparse.ArgumentParser(description="ScreenSage local vision+LLM agent")
    parser.add_argument("goal", type=str, help="Natural language task for the agent")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without executing input")
    args = parser.parse_args()

    log = run_agent_loop(goal=args.goal, dry_run=args.dry_run)
    print("\n--- Run summary ---")
    for entry in log:
        print(f"step {entry['step']}: {entry['action']} -> {entry['result']}")


if __name__ == "__main__":
    sys.exit(main())
