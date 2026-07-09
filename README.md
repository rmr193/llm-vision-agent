# ScreenSage — Local Vision-Grounded LLM Agent (AMD ROCm)

**Track 3: Unicorn (Open Innovation) submission.**

ScreenSage is a fully local desktop AI agent that *sees* your screen in real time
(YOLOv10 object/UI-element detection), *reasons* about what to do next with a
local LLM (Qwen2.5 / Llama-3.2 served via vLLM on ROCm), and *acts* by driving
the mouse/keyboard — all on a single AMD GPU, with zero calls to any cloud API.

Think "Operator" or "Claude computer-use", but 100% local and AMD-accelerated.

```
 ┌────────────┐     frames      ┌───────────────┐   detections   ┌──────────────┐
 │ Screen /   │ ───────────────▶│ YOLOv10       │───────────────▶│ Scene Encoder │
 │ Game Window│                 │ (ROCm/PyTorch)│                │ (JSON + crops)│
 └────────────┘                 └───────────────┘                └──────┬───────┘
                                                                         │
        ┌────────────────────────────────────────────────────────────┐ │
        │ Action Controller (pyautogui) ◀── plan JSON ◀── LLM Agent  │◀┘
        │ mouse move / click / key press / wait                     │  (vLLM on ROCm,
        └────────────────────────────────────────────────────────────┘   Qwen2.5-7B-Instruct
                                                                          or Llama-3.2-11B-Vision)
```

## Why this is a good Unicorn entry

* **Real AMD compute usage, not a wrapper**: both the vision model (YOLOv10 via
  `torch` + ROCm) and the LLM (served through `vllm` with the ROCm backend)
  run inference on the GPU — logged and reported (see `docs/amd_usage.md`).
* **General-purpose, not single-game**: the agent operates on *any* visible
  window — a game, a spreadsheet, a browser, a form — because it reasons over
  detected UI elements + a screenshot crop rather than a game-specific API.
* **Fully offline after model download**: no OpenAI/Anthropic/Google API keys,
  no telemetry, works on an airgapped machine.
* **Extensible tool interface**: adding a new action (e.g. `scroll`, `drag`,
  `type_text`) is a one-line addition to `src/controller/input_controller.py`
  and the JSON schema the LLM is prompted with.

## Repository layout

```
llm-vision-agent/
├── src/
│   ├── config.py                # central settings (model names, thresholds, paths)
│   ├── vision/
│   │   ├── screen_capture.py     # cross-platform screen/window grabber (mss)
│   │   └── detector.py           # YOLOv10 wrapper, ROCm device selection
│   ├── llm/
│   │   ├── prompts.py            # system prompt + JSON action schema
│   │   └── agent.py              # vLLM / transformers client, plan parsing
│   ├── controller/
│   │   └── input_controller.py   # pyautogui-based mouse/keyboard executor
│   ├── app.py                    # CLI orchestrator: capture → detect → reason → act loop
│   └── web_ui.py                 # Flask dashboard: live view + logs + manual stop
├── scripts/
│   ├── setup_rocm.sh             # installs ROCm-enabled torch + vllm
│   ├── download_models.sh        # pulls YOLOv10 + LLM weights
│   └── run_demo.sh               # one-command demo launcher
├── tests/
│   ├── test_detector.py
│   └── test_agent.py
├── docs/
│   ├── architecture.md
│   ├── amd_usage.md
│   └── slide_deck_outline.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── LICENSE
```

## Quick start (local machine with an AMD GPU)

```bash
git clone <your-repo-url> && cd llm-vision-agent

# 1. System deps + ROCm PyTorch + vLLM (ROCm build)
bash scripts/setup_rocm.sh

# 2. Pull model weights (YOLOv10n + Qwen2.5-7B-Instruct, ~8GB total)
bash scripts/download_models.sh

# 3. Run the agent against your primary display, with a live web dashboard
bash scripts/run_demo.sh
# -> opens http://localhost:7860
```

Tell the agent what to do from the dashboard, e.g. *"Open the calculator and
compute 42 * 17"* or *"Click the Play button"* — it will screenshot, detect
UI elements, ask the local LLM for the next action, execute it, and loop
until it reports the task is done (or 20 steps pass, whichever first).

## Quick start (Docker, no local GPU driver setup needed on host)

```bash
docker build --platform linux/amd64 -t screensage:latest .
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size 8g \
  -p 7860:7860 screensage:latest
```

The container starts the Flask dashboard on port `7860` and is ready within
60 seconds (health check hits `/healthz`); model weights are baked in at
build time under `/models` so the first request has no cold-start download.

## Configuration

All tunables live in `src/config.py` / `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Local reasoning model served by vLLM |
| `VLLM_HOST` / `VLLM_PORT` | `127.0.0.1:8000` | Where vLLM's OpenAI-compatible server listens |
| `YOLO_WEIGHTS` | `models/yolov10n.pt` | Detector checkpoint |
| `YOLO_CONF_THRESHOLD` | `0.35` | Minimum detection confidence kept |
| `MAX_AGENT_STEPS` | `20` | Safety cap on the perceive→reason→act loop |
| `HIP_VISIBLE_DEVICES` | `0` | Which AMD GPU(s) to expose to ROCm |

## Safety notes

* The agent only ever emits *one* action per step and re-observes the screen
  before the next — no blind multi-step macros.
* A hard `MAX_AGENT_STEPS` cap and a "STOP" button in the dashboard prevent
  runaway loops.
* Input control is confined to the current display session (uses `pyautogui`,
  which respects OS-level accessibility permissions).

## License

MIT — see `LICENSE`.
