# AMD Compute Usage

This document explains exactly where and how this project uses AMD GPU
compute, for automated pre-screening and human judges.

## 1. Vision model (YOLOv10) — AMD GPU inference

`src/vision/detector.py` runs every detection forward pass on the GPU:

```python
requested = VISION.device            # "cuda:0" (ROCm torch keeps this namespace)
if requested.startswith("cuda") and torch.cuda.is_available():
    backend = "ROCm/HIP" if getattr(torch.version, "hip", None) else "CUDA"
    logger.info(f"Vision model using GPU 0: {name} (backend: {backend})")
```

At startup and on every run, the log clearly prints `backend: ROCm/HIP` when
running on the provided ROCm-enabled PyTorch base image
(`rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_release-2.1.2`, see
`Dockerfile`). `DetectionResult.device` is also surfaced live in the web
dashboard for every single step (e.g. "12.4ms on cuda:0").

## 2. LLM reasoning — AMD GPU inference via vLLM ROCm

The reasoning model (`Qwen/Qwen2.5-7B-Instruct` by default) is served with:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --dtype auto
```

using vLLM's ROCm build (`pip install "vllm[rocm]"`, see
`scripts/setup_rocm.sh` and the `Dockerfile`). vLLM's paged attention runtime
executes all forward passes for the chat completion API on the AMD GPU
exposed via `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`. Every single agent
step issues a fresh chat completion — this is not a cached/precomputed
response, since the prompt includes live, per-frame detection JSON that
changes every iteration (see the "Do not hardcode or cache answers"
competition rule — the model must literally see fresh screen state each
time and produce a *fresh* completion for it).

## 3. Verifying at runtime

```bash
python3 -c "import torch; print('CUDA/ROCm available:', torch.cuda.is_available()); print('HIP version:', torch.version.hip); print('Device:', torch.cuda.get_device_name(0))"
```

Expected output on an AMD GPU host:

```
CUDA/ROCm available: True
HIP version: 6.1.xxxxx
Device: <e.g. AMD Instinct MI210 / Radeon RX 7900 XTX>
```

Both this check and the vLLM server startup log (which prints the detected
GPU and ROCm version) are captured in `vllm_server.log` inside the running
container for judges to inspect.

## 4. Why AMD (not just "GPU-agnostic")

* PyTorch's ROCm wheels are used explicitly (`--index-url
  https://download.pytorch.org/whl/rocm6.1`), not the default CUDA wheels.
* vLLM is installed with the `[rocm]` extra, which pulls in
  AMD's `hipBLASLt`/`composable_kernel`-backed attention kernels rather than
  the CUDA/FlashAttention path.
* The Docker base image is AMD's official `rocm/pytorch` image, and the
  container is launched with `--device=/dev/kfd --device=/dev/dri` (the ROCm
  kernel driver device nodes), not `--gpus all` (NVIDIA Container Toolkit).
