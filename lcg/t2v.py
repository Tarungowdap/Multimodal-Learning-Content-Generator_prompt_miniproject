"""Text-to-video backends: one prompt -> a raw MP4 clip from a generative video model.

Backends
  gemini       Google Gemini Omni Flash (`gemini-omni-1.1-flash`) via the Interactions API; uses GEMINI_API_KEY.
               (Replaces Veo 3.1 Preview, which Google shut down on 22 Oct 2026.)
  huggingface  Hugging Face Inference Providers (default model Wan-AI/Wan2.2-TI2V-5B); uses HF_TOKEN.
               The live provider for the model is looked up at runtime, so a model moving between
               providers does not break the app.
  novai        NovAI (aiapi-pro.com) `cogvideox-flash`: FREE tier, no credit card (about 5 videos/day,
               5 requests/min, resets 00:00 UTC+8); uses NOVAI_API_KEY. Lower quality than the others.

The raw clip is NOT the final video: `video.make_t2v_video` fits it to 5-10 s, burns in captions
and replaces the audio with the lesson narration.
"""
from __future__ import annotations

import base64
import os
import time
from typing import Callable
from urllib.parse import urlparse

import requests

from .generators import clean_topic
from .schemas import Scene

GEMINI_VIDEO_MODEL = os.getenv("GEMINI_VIDEO_MODEL", "gemini-omni-1.1-flash")
HF_VIDEO_MODEL = os.getenv("HF_VIDEO_MODEL", "Wan-AI/Wan2.2-TI2V-5B")
NOVAI_VIDEO_MODEL = os.getenv("NOVAI_VIDEO_MODEL", "cogvideox-flash")
NOVAI_BASE = "https://aiapi-pro.com/v1"
# The free tier can hold the submit request open for minutes under load. One long submit (never
# auto-retried, because a retry could spend another of the ~5 free daily videos).
NOVAI_SUBMIT_TIMEOUT = float(os.getenv("NOVAI_SUBMIT_TIMEOUT", "300"))
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
HF_ROUTER = "https://router.huggingface.co"
HF_HUB = "https://huggingface.co"

# ---- standardised video style prompt (mirrors images.STYLE_PRESET) -----------------------------
VIDEO_STYLE = (
    "animated flat vector educational explainer, clean minimal shapes, soft pastel palette "
    "(teal, warm yellow, coral, navy), plain light background, smooth slow camera motion, "
    "friendly textbook style, 16:9"
)
VIDEO_NEGATIVE = ("no on-screen text, no letters, no subtitles, no captions, no logos, no watermark, "
                  "no talking people, no dialogue, not photo-realistic")


def build_video_prompt(topic: str, scenes: list[Scene], seconds: int = 8) -> str:
    """Every clip = the lesson's planned shots + the SAME style block => consistent visual identity."""
    shots = " ".join(f"Shot {i + 1}: {s.image_subject.strip().rstrip('.')}." for i, s in enumerate(scenes))
    return (f"A {seconds}-second educational explainer video about {clean_topic(topic)}. {shots} "
            f"Continuous, smooth transitions between shots. Style: {VIDEO_STYLE}. "
            f"Audio: soft ambient background only. Avoid: {VIDEO_NEGATIVE}.")


def compact_prompt(prompt: str, limit: int = 500) -> str:
    """CogVideoX-class models accept ~512 characters: keep the shots, use a shorter style block."""
    if len(prompt) <= limit:
        return prompt
    head = prompt.split(" Continuous, smooth transitions")[0]
    short = f"{head} Flat vector pastel educational animation, smooth slow camera, no text, no people talking."
    return short if len(short) <= limit else short[: limit - 1].rsplit(" ", 1)[0] + "."


class T2VError(RuntimeError):
    pass


# ---- Gemini Omni Flash -------------------------------------------------------------------------
def _find_video(node) -> dict | None:
    """Locate the first {"type": "video", ...data|uri} item anywhere in an Interactions response."""
    if isinstance(node, dict):
        if node.get("type") == "video" and (node.get("data") or node.get("uri")):
            return node
        for v in node.values():
            hit = _find_video(v)
            if hit:
                return hit
    elif isinstance(node, list):
        for v in node:
            hit = _find_video(v)
            if hit:
                return hit
    return None


def gemini_video(prompt: str, key: str, *, model: str = GEMINI_VIDEO_MODEL, timeout: int = 360,
                 poll_every: float = 5.0, log: Callable[[str], None] = lambda m: None,
                 session=requests) -> bytes:
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    body = {"model": model, "input": prompt,
            "response_format": {"type": "video", "aspect_ratio": "16:9", "resolution": "720p"}}
    r = session.post(f"{GEMINI_BASE}/interactions", headers=headers, json=body, timeout=timeout)
    if r.status_code >= 400:
        raise T2VError(f"gemini HTTP {r.status_code}: {r.text[:200]}")
    item = _find_video(r.json())
    if not item:
        raise T2VError("gemini response contained no video")
    if item.get("data"):
        return base64.b64decode(item["data"])
    # URI delivery: poll the file until ACTIVE, then download it.
    file_id = item["uri"].rstrip("/").split("/")[-1].split(":")[0]
    deadline = time.time() + timeout
    while True:
        st = session.get(f"{GEMINI_BASE}/files/{file_id}", headers=headers, timeout=30)
        if st.status_code >= 400:
            raise T2VError(f"gemini file status HTTP {st.status_code}")
        state = st.json().get("state", "")
        if state == "ACTIVE":
            break
        if state == "FAILED" or time.time() > deadline:
            raise T2VError(f"gemini video file {state or 'timed out'}")
        log("    waiting for Gemini video file…")
        time.sleep(poll_every)
    d = session.get(f"{GEMINI_BASE}/files/{file_id}:download", params={"alt": "media"},
                    headers=headers, timeout=120)
    if d.status_code >= 400 or not d.content:
        raise T2VError(f"gemini download HTTP {d.status_code}")
    return d.content


# ---- Hugging Face Inference Providers -------------------------------------------------------
def _hf_provider(model: str, token: str, session) -> tuple[str, str]:
    """Return (provider, provider_model_id) for a live text-to-video mapping of `model`."""
    r = session.get(f"{HF_HUB}/api/models/{model}", params={"expand[]": "inferenceProviderMapping"},
                    headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code >= 400:
        raise T2VError(f"huggingface model lookup HTTP {r.status_code}")
    mapping = r.json().get("inferenceProviderMapping") or {}
    if isinstance(mapping, list):  # newer hub responses use a list of {provider, providerId, ...}
        mapping = {m.get("provider"): m for m in mapping if isinstance(m, dict)}
    for provider in ("fal-ai", "hf-inference"):
        m = mapping.get(provider)
        if isinstance(m, dict) and m.get("status", "live") == "live" and m.get("task", "text-to-video") == "text-to-video":
            return provider, m.get("providerId") or model
    raise T2VError(f"no supported live text-to-video provider for {model} (found: {sorted(mapping)})")


def hf_video(prompt: str, token: str, *, model: str = HF_VIDEO_MODEL, timeout: int = 600,
             poll_every: float = 4.0, seed: int = 42, log: Callable[[str], None] = lambda m: None,
             session=requests) -> bytes:
    auth = {"Authorization": f"Bearer {token}"}
    provider, pid = _hf_provider(model, token, session)
    log(f"    huggingface: {model} via {provider}")
    if provider == "hf-inference":
        r = session.post(f"{HF_ROUTER}/hf-inference/models/{pid}", headers=auth,
                         json={"inputs": prompt, "parameters": {"seed": seed}}, timeout=timeout)
        if r.status_code >= 400 or not r.content:
            raise T2VError(f"hf-inference HTTP {r.status_code}")
        return r.content
    # fal-ai queue protocol, routed (and billed) through Hugging Face.
    r = session.post(f"{HF_ROUTER}/fal-ai/{pid}", params={"_subdomain": "queue"}, headers=auth,
                     json={"prompt": prompt, "seed": seed}, timeout=60)
    if r.status_code >= 400:
        raise T2VError(f"fal-ai submit HTTP {r.status_code}: {r.text[:200]}")
    job = r.json()
    if not job.get("request_id") or not job.get("response_url"):
        raise T2VError("fal-ai returned no request id")
    path = urlparse(job["response_url"]).path
    status, deadline = job.get("status"), time.time() + timeout
    while status != "COMPLETED":
        if status in ("FAILED", "ERROR") or time.time() > deadline:
            raise T2VError(f"fal-ai job {status or 'timed out'}")
        time.sleep(poll_every)
        s = session.get(f"{HF_ROUTER}/fal-ai{path}/status", params={"_subdomain": "queue"},
                        headers=auth, timeout=30)
        if s.status_code >= 400:
            raise T2VError(f"fal-ai status HTTP {s.status_code}")
        status = s.json().get("status")
        log(f"    huggingface job: {status}")
    res = session.get(f"{HF_ROUTER}/fal-ai{path}", params={"_subdomain": "queue"}, headers=auth, timeout=60)
    if res.status_code >= 400:
        raise T2VError(f"fal-ai result HTTP {res.status_code}")
    url = (res.json().get("video") or {}).get("url")
    if not url:
        raise T2VError("fal-ai result contained no video url")
    v = session.get(url, timeout=120)
    if v.status_code >= 400 or not v.content:
        raise T2VError(f"video download HTTP {v.status_code}")
    return v.content


# ---- NovAI (free cogvideox-flash) -------------------------------------------------------------
def novai_video(prompt: str, key: str, *, model: str = NOVAI_VIDEO_MODEL, timeout: int = 600,
                poll_every: float = 5.0, log: Callable[[str], None] = lambda m: None,
                session=requests) -> bytes:
    auth = {"Authorization": f"Bearer {key}"}
    log("    novai: submitting job (the free tier can take a few minutes to accept it)…")
    try:
        r = session.post(f"{NOVAI_BASE}/video/generations", headers=auth,
                         json={"model": model, "prompt": compact_prompt(prompt), "ratio": "16:9"},
                         timeout=(15, NOVAI_SUBMIT_TIMEOUT))
    except requests.Timeout as e:
        raise T2VError(f"novai did not answer within {NOVAI_SUBMIT_TIMEOUT:.0f}s (free tier busy); "
                       "try again later or raise NOVAI_SUBMIT_TIMEOUT") from e
    if r.status_code == 429:
        raise T2VError("novai free daily video limit reached (resets 00:00 UTC+8 = 21:30 IST)")
    if r.status_code >= 400:
        raise T2VError(f"novai submit HTTP {r.status_code}: {r.text[:200]}")
    task = r.json()
    tid = task.get("id")
    if not tid:
        raise T2VError("novai returned no task id")
    status, deadline = task.get("status", "queued"), time.time() + timeout
    while status in ("queued", "processing", "pending", "running"):
        if time.time() > deadline:
            raise T2VError("novai job timed out")
        time.sleep(poll_every)
        try:
            s = session.get(f"{NOVAI_BASE}/video/generations/{tid}", params={"model": model},
                            headers=auth, timeout=(15, 60))
        except requests.RequestException:
            log("    novai: status check timed out, retrying…")   # polling is free: keep waiting
            continue
        if s.status_code >= 400:
            raise T2VError(f"novai status HTTP {s.status_code}")
        task = s.json()
        status = task.get("status", "")
        log(f"    novai job: {status}")
    if status != "succeeded":
        raise T2VError(f"novai job {status or 'returned no status'}")
    url = task.get("video_url") or next((v.get("url") for v in task.get("video_result") or []
                                         if isinstance(v, dict) and v.get("url")), None)
    if not url:
        raise T2VError("novai result contained no video url")
    v = session.get(url, timeout=120)
    if v.status_code >= 400 or not v.content:
        raise T2VError(f"novai video download HTTP {v.status_code}")
    return v.content


def generate_clip(prompt: str, backend: str, key: str, log=lambda m: None, seed: int = 42) -> bytes:
    if backend == "novai":
        return novai_video(prompt, key, log=log)
    if backend == "gemini":
        return gemini_video(prompt, key, log=log)
    if backend == "huggingface":
        return hf_video(prompt, key, seed=seed, log=log)
    raise T2VError(f"unknown text-to-video backend {backend}")
