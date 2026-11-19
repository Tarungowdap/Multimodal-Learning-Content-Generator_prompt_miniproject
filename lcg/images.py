"""Image module: text-to-image with a standardised style prompt so every lesson looks consistent.

Backends (auto order): OpenAI gpt-image-1 -> HuggingFace FLUX.1-schnell -> Pollinations -> infographic card.
The last step is NOT a random placeholder: it renders a readable lesson slide (step badge, heading,
explanation, progress chips) so the video is still useful with no image model at all.
"""
from __future__ import annotations

import base64
import io
import os
import textwrap
import time
import urllib.parse
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from .config import Settings

SIZE = (768, 432)  # 16:9

# ---- standardised style prompt ------------------------------------------------------------
STYLE_PRESET = (
    "flat vector educational illustration, clean minimal shapes, soft pastel palette "
    "(teal, warm yellow, coral, navy), white background, centered composition, "
    "high contrast, friendly textbook style, no text, no watermark"
)
NEGATIVE = "text, letters, words, watermark, logo, photo-realistic, blurry, cluttered, deformed, nsfw"


def build_prompt(subject: str, topic: str) -> str:
    """Every image = subject + the SAME style block => consistent visual identity."""
    return f"{subject.strip()} (concept: {topic.strip()}), {STYLE_PRESET}"


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "segoeuib.ttf", "calibrib.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _regular(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return _font(size)


PALETTES = [("#0f766e", "#ccfbf1", "#fde68a"), ("#1e3a8a", "#dbeafe", "#fca5a5"),
            ("#9a3412", "#ffedd5", "#99f6e4"), ("#4c1d95", "#ede9fe", "#fcd34d")]


def info_card(heading: str, detail: str, topic: str, idx: int, total: int, path: Path) -> Path:
    """Readable infographic slide, used when no text-to-image model is reachable."""
    dark, light, accent = PALETTES[idx % len(PALETTES)]
    img = Image.new("RGB", SIZE, light)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, SIZE[0], 70], fill=dark)                       # top bar with topic
    d.text((28, 18), topic.title()[:44], fill="white", font=_font(30))
    d.ellipse([28, 96, 100, 168], fill=dark)                          # step badge
    num = str(idx + 1)
    d.text((64 - d.textlength(num, font=_font(40)) / 2, 112), num, fill="white", font=_font(40))
    y = 98
    for line in textwrap.wrap(heading, 30)[:2]:                       # heading
        d.text((122, y), line, fill=dark, font=_font(34))
        y += 42
    y = max(y, 190) + 6
    d.rounded_rectangle([28, y, SIZE[0] - 28, y + 120], radius=18, fill="white")   # explanation card
    d.rectangle([28, y + 18, 38, y + 110], fill=accent)
    ty = y + 14
    for line in textwrap.wrap(detail, 52)[:4]:
        d.text((54, ty), line, fill="#1f2937", font=_regular(24))
        ty += 28
    cx = SIZE[0] // 2 - total * 20 + 10                               # progress chips (step flow)
    for k in range(total):
        d.ellipse([cx + k * 40, 338, cx + k * 40 + 22, 360], fill=dark if k == idx else "#cbd5e1")
    img.save(path)
    return path


# ---- backends -----------------------------------------------------------------------------
_last_poll = [0.0]
POLL_GAP = 16.0  # anonymous Pollinations tier ~ 1 request / 15 s


def _pollinations(prompt: str, seed: int, key: str | None, log=lambda m: None) -> bytes:
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
           + f"?width={SIZE[0]}&height={SIZE[1]}&seed={seed}&nologo=true&model=flux"
           + f"&negative={urllib.parse.quote(NEGATIVE)}")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    last = "unknown"
    for attempt in range(4):
        wait = POLL_GAP - (time.time() - _last_poll[0]) if not key else 0
        if wait > 0:
            log(f"    waiting {wait:.0f}s (free-tier rate limit)…")
            time.sleep(wait)
        _last_poll[0] = time.time()
        try:
            r = requests.get(url, headers=headers, timeout=120)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image") and len(r.content) > 5000:
                return r.content
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(2 + attempt * 3)
    raise RuntimeError(f"Pollinations failed ({last})")


def _openai(prompt: str, key: str) -> bytes:
    r = requests.post("https://api.openai.com/v1/images/generations",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": "gpt-image-1", "prompt": prompt + f". Avoid: {NEGATIVE}",
                            "size": "1536x1024", "n": 1}, timeout=120)
    r.raise_for_status()
    return base64.b64decode(r.json()["data"][0]["b64_json"])


def _huggingface(prompt: str, key: str, seed: int) -> bytes:
    r = requests.post("https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"inputs": prompt, "parameters": {"seed": seed, "width": SIZE[0], "height": SIZE[1]}},
                      timeout=120)
    r.raise_for_status()
    return r.content


def generate_image(subject: str, topic: str, idx: int, out_dir: Path, s: Settings, log=lambda m: None,
                   heading: str = "", detail: str = "", total: int = 3) -> tuple[Path, str]:
    """Returns (path, source) where source says which backend produced it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scene_{idx + 1}.png"
    prompt = build_prompt(subject, topic)
    order = ["openai", "huggingface", "pollinations"] if s.image_backend == "auto" else [s.image_backend]
    for b in order:
        if b == "placeholder":
            break
        try:
            if b == "openai":
                if not s.key("openai"):
                    continue
                data = _openai(prompt, s.key("openai"))
            elif b == "huggingface":
                if not s.key("huggingface"):
                    continue
                data = _huggingface(prompt, s.key("huggingface"), s.seed + idx)
            elif b == "pollinations":
                data = _pollinations(prompt, s.seed + idx, s.key("pollinations"), log)
            else:
                continue
            Image.open(io.BytesIO(data)).convert("RGB").resize(SIZE).save(path)
            log(f"  image {idx + 1}: generated via {b}")
            return path, b
        except Exception as e:
            log(f"  image {idx + 1}: {b} failed ({e}); trying next option")
    log(f"  image {idx + 1}: no image model reachable -> infographic slide")
    return info_card(heading or subject, detail or subject, topic, idx, total, path), "infographic"
