# Slide Deck Outline — ScreenSage

Use this outline to build the required PDF slide deck (e.g. in Google Slides
/ PowerPoint / Keynote, then export to PDF). Suggested: 8–10 slides.

1. **Title**
   - ScreenSage — Local Vision-Grounded LLM Agent
   - Track 3: Unicorn (Open Innovation)
   - Team name, names, GitHub URL, hosted demo URL

2. **The problem**
   - Cloud "computer-use" agents (browser/desktop automation) require
     sending screenshots to third-party APIs — privacy, latency, and cost
     concerns; nothing works fully offline on your own hardware today.

3. **What we built**
   - One-sentence pitch: a fully local agent that sees your screen, reasons
     about it with an LLM, and acts — all inference on a single AMD GPU.
   - Screenshot of the live dashboard mid-run.

4. **Architecture**
   - Diagram: Screen Capture → YOLOv10 detector (ROCm) → LLM planner (vLLM on
     ROCm) → Input controller → loop.
   - Reuse the ASCII diagram from `docs/architecture.md` as a polished
     graphic.

5. **AMD compute usage**
   - Vision inference on ROCm PyTorch (screenshot: log line showing
     `backend: ROCm/HIP`).
   - LLM inference via vLLM's ROCm build (screenshot: vLLM server startup log
     showing the detected AMD GPU).
   - Per-step latency numbers surfaced live in the dashboard.

6. **Demo walkthrough**
   - 3–4 screenshots of a real run end-to-end (goal entered → detections
     overlay → chosen action → "done" summary).

7. **What makes it general-purpose**
   - Works on any visible window (not a single game or app) because
     reasoning is grounded in detected UI elements + goal text, not a
     game-specific API/SDK.
   - One-line extensibility: adding a new action type touches 3 small,
     isolated files.

8. **Safety & guardrails**
   - Single action per step, re-observation before next step.
   - Hard step cap (`MAX_AGENT_STEPS`), dashboard STOP button,
     `pyautogui.FAILSAFE` corner-abort.

9. **What's next**
   - Fine-tune a small vision-language model directly on UI screenshots to
     drop the separate YOLO stage.
   - Multi-monitor + multi-window awareness.
   - Long-horizon task memory across agent runs.

10. **Thank you / links**
    - GitHub repo URL
    - Hosted demo URL
    - Contact
