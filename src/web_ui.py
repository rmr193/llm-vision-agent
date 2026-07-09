"""
Minimal web dashboard for ScreenSage.

Lets a judge/user type a goal, click "Run", and watch the agent's
perceive -> reason -> act steps stream in live, plus a STOP button and a
health check endpoint for container orchestration.
"""
from __future__ import annotations

import threading
import time

from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from loguru import logger

from src.cli import run_agent_loop
from src.config import WEB

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

_state = {"running": False, "stop_requested": False}

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>ScreenSage</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
  <style>
    body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 760px; margin: 40px auto; background:#0f1115; color:#e8e8ea; }
    h1 { font-weight: 600; }
    input[type=text] { width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #333; background:#1a1d24; color:#fff; font-size: 15px; box-sizing: border-box; }
    button { padding: 10px 18px; border-radius: 8px; border: none; font-weight: 600; cursor: pointer; margin-top: 10px; margin-right: 8px; }
    #run { background: #7c5cff; color: white; }
    #stop { background: #33313a; color: #fff; }
    #log { margin-top: 20px; background: #14161b; border-radius: 10px; padding: 14px; height: 380px; overflow-y: auto; font-family: ui-monospace, monospace; font-size: 13px; white-space: pre-wrap; }
    .step { margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #24262d; }
    .tag { color: #7c5cff; font-weight: 700; }
  </style>
</head>
<body>
  <h1>🦄 ScreenSage</h1>
  <p>Local vision-grounded LLM agent — running on AMD ROCm.</p>
  <input id="goal" type="text" placeholder="e.g. Open the calculator and compute 42 * 17" />
  <div>
    <button id="run">Run agent</button>
    <button id="stop">Stop</button>
  </div>
  <div id="log"></div>

  <script>
    const socket = io();
    const logEl = document.getElementById('log');
    document.getElementById('run').onclick = () => {
      const goal = document.getElementById('goal').value;
      logEl.innerHTML = '';
      socket.emit('start_run', {goal});
    };
    document.getElementById('stop').onclick = () => socket.emit('stop_run');
    socket.on('step', (data) => {
      const div = document.createElement('div');
      div.className = 'step';
      div.innerHTML = `<span class="tag">step ${data.step}</span> — ${data.action} → ${data.result}<br/>`
        + `<small>${data.reasoning || ''} (${data.num_detections} elements, ${data.inference_ms}ms on ${data.device})</small>`;
      logEl.appendChild(div);
      logEl.scrollTop = logEl.scrollHeight;
    });
    socket.on('run_complete', () => {
      const div = document.createElement('div');
      div.innerHTML = '<b>✅ Run complete.</b>';
      logEl.appendChild(div);
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/healthz")
def healthz():
    """Container health check — must respond quickly for the 60s readiness window."""
    return jsonify(status="ok"), 200


def _background_run(goal: str):
    _state["running"] = True
    _state["stop_requested"] = False

    def on_step(entry: dict):
        if _state["stop_requested"]:
            raise StopIteration("stop requested by user")
        socketio.emit("step", entry)
        socketio.sleep(0)

    try:
        run_agent_loop(goal=goal, dry_run=False, on_step=on_step)
    except StopIteration:
        logger.info("Agent run stopped by user request.")
    except Exception as exc:  # keep the dashboard alive even if a run errors
        logger.exception("Agent run failed")
        socketio.emit("step", {"step": -1, "action": "error", "result": str(exc),
                                "reasoning": "", "num_detections": 0, "inference_ms": 0, "device": "n/a"})
    finally:
        _state["running"] = False
        socketio.emit("run_complete", {})


@socketio.on("start_run")
def handle_start_run(data):
    if _state["running"]:
        return
    goal = (data or {}).get("goal", "").strip()
    if not goal:
        return
    threading.Thread(target=_background_run, args=(goal,), daemon=True).start()


@socketio.on("stop_run")
def handle_stop_run():
    _state["stop_requested"] = True


def main():
    logger.info(f"Starting ScreenSage dashboard on {WEB.host}:{WEB.port}")
    socketio.run(app, host=WEB.host, port=WEB.port)


if __name__ == "__main__":
    main()
