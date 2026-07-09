# Architecture

## Perceive → Reason → Act loop

```
┌────────────────────────────────────────────────────────────────────┐
│                            app.py / web_ui.py                      │
│                                                                      │
│   loop (max MAX_AGENT_STEPS times):                                 │
│                                                                      │
│   1. ScreenCapture.grab_full()          -> RGB numpy frame           │
│   2. UIElementDetector.detect(frame)    -> DetectionResult          │
│         (YOLOv10 forward pass on the AMD GPU via ROCm torch)        │
│   3. LLMAgent.next_action(goal, elements, history) -> AgentAction    │
│         (chat completion against local vLLM server, ROCm backend)  │
│   4. InputController.execute(action, detections) -> result string   │
│   5. history.append(...); if action == "done": break                │
└────────────────────────────────────────────────────────────────────┘
```

## Why a JSON action schema instead of free-form tool calls

vLLM's OpenAI-compatible server supports `response_format={"type": "json_object"}`,
which we use to force the LLM's output into a strict, parseable shape without
needing a specific function-calling fine-tune. This keeps the system portable
across different open-weight models (Qwen2.5, Llama-3.2, Mistral, etc.) — you
can swap `LLM_MODEL` in `.env` without touching any parsing code, as long as
the model can follow JSON-mode instructions reasonably well.

## Why element IDs instead of raw coordinates

Raw pixel coordinates from an LLM are notoriously unreliable (models are bad
at precise spatial arithmetic). Instead, the vision stage assigns a stable
integer ID to every detected element in the current frame, and the LLM is
asked to reference `target_element_id`. The `InputController` resolves the ID
back to a pixel center right before executing the action, using the *freshest*
detection results — so even if several seconds pass between capture and
action, coordinates stay grounded in what's actually on screen right now
(re-detected on the very next loop iteration for the following step).

## ROCm-specific notes

* PyTorch's ROCm build intentionally keeps the `torch.cuda.*` API surface
  (device selection, `.to("cuda:0")`, `torch.cuda.is_available()`), so the
  detector code does not need any AMD-specific branches — we just log
  `torch.version.hip` to confirm we're actually on the ROCm backend at
  runtime (see `docs/amd_usage.md`).
* `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` control which physical GPU(s)
  are exposed, mirroring `CUDA_VISIBLE_DEVICES` semantics.
* vLLM's ROCm build serves the LLM with paged-attention KV-cache management,
  giving low per-step latency even though a fresh chat completion is issued
  on every single loop iteration (necessary since each step re-grounds on a
  fresh screenshot).

## Extensibility

Adding a new action type (e.g., `drag`, `double_click`, `right_click`)
requires three small, localized changes:
1. Add the literal to `VALID_ACTIONS` in `src/llm/agent.py`.
2. Document it in the schema/rules inside `src/llm/prompts.py`.
3. Implement its execution branch in `src/controller/input_controller.py`.

No changes are needed to the main loop in `app.py`.
