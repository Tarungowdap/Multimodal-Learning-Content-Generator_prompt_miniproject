"""LOCAL text-to-video: AnimateDiff-Lightning running on this computer. No API key, no account.

Model
  Motion module  ByteDance/AnimateDiff-Lightning (4-step distilled, CreativeML OpenRAIL-M)
  Base model     emilianJR/epiCRealism (Stable Diffusion 1.5 fine-tune; override with LOCAL_T2V_BASE)
Weights (~4.5 GB) are downloaded once from the public Hugging Face Hub into its cache, then the
model runs fully offline.

One short clip (16 frames) is generated per planned scene, each from the scene's own prompt plus the
SAME standardised style block, so a 3-scene lesson gives ~6 s of genuine generated motion that the
video module then fits to 5-10 s, captions and narrates.

Speed: NVIDIA GPU ~10-40 s per scene; CPU-only works but takes several minutes per scene.
Install the extra packages with setup_local_video.bat (or requirements-video.txt).
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .schemas import Scene

REPO = "ByteDance/AnimateDiff-Lightning"
BASE = os.getenv("LOCAL_T2V_BASE", "emilianJR/epiCRealism")
STEPS = int(os.getenv("LOCAL_T2V_STEPS", "4"))            # 1, 2, 4 or 8 (Lightning checkpoints)
FRAMES = 16                                                # what the motion module was trained on
CLIP_FPS = 8                                               # 16 frames -> 2 s per scene
WIDTH, HEIGHT = 640, 384                                   # multiples of 64; cropped to 16:9 later

# Standardised style for every scene (kept short: Stable Diffusion reads at most 77 tokens).
LOCAL_STYLE = ("educational explainer animation, clear subject, soft studio lighting, clean background, "
               "smooth gentle motion, sharp focus, high quality")
LOCAL_NEGATIVE = "text, letters, watermark, logo, blurry, deformed, distorted, low quality, nsfw"

_PIPE = None
_PIPE_LOCK = threading.Lock()


class LocalT2VUnavailable(RuntimeError):
    """Raised when torch/diffusers are not installed or the model cannot be loaded."""


def status() -> dict:
    """What is installed and which device would be used (no model download)."""
    info = {"installed": False, "device": "none", "gpu": "", "vram_gb": 0.0, "reason": ""}
    try:
        import diffusers  # noqa: F401
        import torch
    except Exception as e:  # ImportError or a broken torch install
        info["reason"] = f"torch/diffusers not installed ({type(e).__name__}) - run setup_local_video.bat"
        return info
    info["installed"] = True
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(device="cuda", gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1))
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info.update(device="mps", gpu="Apple GPU")
    else:
        info.update(device="cpu", reason="no CUDA GPU found: generation runs on the CPU and is slow")
    return info


def scene_prompt(scene: Scene, topic: str) -> str:
    subject = " ".join(scene.image_subject.replace("\n", " ").split()[:30]).rstrip(".")
    t = topic.strip()[:60]
    return f"{subject}, {LOCAL_STYLE}" if t.lower() in subject.lower() else f"{subject}, {t}, {LOCAL_STYLE}"


def load_pipeline(log: Callable[[str], None] = lambda m: None):
    """Build (once per process) the AnimateDiff-Lightning pipeline on the best available device."""
    global _PIPE
    with _PIPE_LOCK:
        if _PIPE is not None:
            return _PIPE
        st = status()
        if not st["installed"]:
            raise LocalT2VUnavailable(st["reason"])
        import torch
        from diffusers import AnimateDiffPipeline, EulerDiscreteScheduler, MotionAdapter
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        device = st["device"]
        if device == "cuda":
            try:                                     # driver / GPU too old for this PyTorch build -> use the CPU
                (torch.zeros(1, device="cuda") + 1).item()
            except Exception as e:
                log(f"    local video model: CUDA not usable ({type(e).__name__}), using the CPU instead")
                device, st["vram_gb"] = "cpu", 0.0
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        log(f"    local video model: loading on {device}"
            + (f" ({st['gpu']}, {st['vram_gb']} GB)" if st["gpu"] else "")
            + " - first run downloads ~4.5 GB, later runs are offline")
        try:
            ckpt = hf_hub_download(REPO, f"animatediff_lightning_{STEPS}step_diffusers.safetensors")
            adapter = MotionAdapter()
            adapter.load_state_dict(load_file(ckpt, device="cpu"))
            try:
                pipe = AnimateDiffPipeline.from_pretrained(BASE, motion_adapter=adapter.to(dtype), dtype=dtype)
            except TypeError:                        # diffusers < 0.43 uses the older argument name
                pipe = AnimateDiffPipeline.from_pretrained(BASE, motion_adapter=adapter.to(dtype), torch_dtype=dtype)
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
        except Exception as e:
            raise LocalT2VUnavailable(f"could not load the local video model: {e}") from e
        pipe.set_progress_bar_config(disable=True)
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        if device == "cuda" and st["vram_gb"] < 7.5:
            pipe.enable_model_cpu_offload()          # fits 4-6 GB laptop GPUs
        else:
            pipe.to(device)
        _PIPE = pipe
        return pipe


def generate_scene_frames(pipe, prompt: str, seed: int) -> list[np.ndarray]:
    import torch

    gen = torch.Generator("cpu").manual_seed(seed)
    out = pipe(prompt=prompt, negative_prompt=LOCAL_NEGATIVE, num_frames=FRAMES, width=WIDTH, height=HEIGHT,
               guidance_scale=1.0, num_inference_steps=STEPS, generator=gen, output_type="np")
    frames = out.frames[0]                           # (frames, H, W, 3) floats in [0, 1]
    return [(np.clip(f, 0, 1) * 255).round().astype(np.uint8) for f in frames]


def generate_clip(scenes: list[Scene], topic: str, out_path: Path, seed: int = 42,
                  log: Callable[[str], None] = lambda m: None,
                  progress: Callable[[int, int], None] = lambda i, n: None) -> tuple[Path, list[str]]:
    """Generate one clip per scene and write them back-to-back as an MP4 (8 fps). Returns (path, prompts)."""
    import imageio

    pipe = load_pipeline(log)
    prompts, all_frames = [], []
    for i, sc in enumerate(scenes):
        p = scene_prompt(sc, topic)
        prompts.append(p)
        t0 = time.perf_counter()
        frames = generate_scene_frames(pipe, p, seed + i)
        log(f"    local video: scene {i + 1}/{len(scenes)} generated in {time.perf_counter() - t0:.0f}s")
        all_frames += frames
        progress(i + 1, len(scenes))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out_path), fps=CLIP_FPS, codec="libx264", quality=8,
                            macro_block_size=1, pixelformat="yuv420p") as w:
        for f in all_frames:
            w.append_data(f)
    return out_path, prompts


def prefetch(log: Callable[[str], None] = print) -> None:
    """Download the weights now (used by setup_local_video.bat) and report the device."""
    st = status()
    log(f"device: {st['device']} {st['gpu']} {st['vram_gb'] or ''} {st['reason']}".strip())
    if not st["installed"]:
        raise SystemExit(st["reason"])
    from huggingface_hub import hf_hub_download

    log("downloading motion module…")
    hf_hub_download(REPO, f"animatediff_lightning_{STEPS}step_diffusers.safetensors")
    log(f"downloading base model {BASE} (a few GB, one time only)…")
    log("loading once to check it works…")
    load_pipeline(log)
    log("local text-to-video model is ready.")


if __name__ == "__main__":
    prefetch()
