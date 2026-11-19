"""Video module: image sequence (Ken-Burns pan/zoom + captions) + TTS narration -> 5-10 s MP4."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from .images import _font

W, H, FPS = 960, 540, 20
MIN_S, MAX_S = 5.0, 10.0
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def audio_duration(path: Path) -> float:
    out = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", out)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def synthesize(text: str, out: Path, log=lambda m: None, *, allow_online: bool = True) -> Path | None:
    """gTTS (online) -> espeak-ng (offline) -> None (silent video with on-screen captions)."""
    if allow_online:
        try:
            from gtts import gTTS

            gTTS(text=text, lang="en", timeout=(5, 15)).save(str(out))
            return out
        except Exception as e:
            log(f"  narration: gTTS unavailable ({type(e).__name__})")
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            wav = out.with_suffix(".wav")
            try:
                result = subprocess.run([exe, "-w", str(wav), text], capture_output=True, timeout=30)
                if result.returncode == 0 and wav.exists():
                    return wav
            except (OSError, subprocess.TimeoutExpired):
                continue
    log("  narration: no TTS engine, producing silent video with captions")
    return None


def _frame(img: Image.Image, t: float, caption: str) -> np.ndarray:
    """t in [0,1]: slow zoom-in (1.0 -> 1.12) with a caption bar."""
    z = 1.0 + 0.12 * t
    iw, ih = img.size
    cw, ch = iw / z, ih / z
    x0, y0 = (iw - cw) / 2, (ih - ch) / 2
    f = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(f, "RGBA")
    d.rectangle([0, H - 84, W, H], fill=(15, 23, 42, 205))
    font = _font(30)
    w = d.textlength(caption, font=font)
    d.text(((W - w) / 2, H - 62), caption, fill="white", font=font)
    return np.asarray(f)


def make_video(images: list[Path], captions: list[str], narration: str, out_path: Path,
               log=lambda m: None, *, allow_online: bool = True) -> tuple[Path, float]:
    if not images or len(images) != len(captions):
        raise ValueError("Video needs at least one image and one caption per image.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        audio = synthesize(narration, td / "narration.mp3", log, allow_online=allow_online)
        dur = audio_duration(audio) if audio else 0.0
        total = min(MAX_S, max(MIN_S, dur + 0.6)) if dur else 8.0   # always within 5-10 s
        per = total / len(images)
        silent = td / "silent.mp4"
        imgs = [Image.open(p).convert("RGB") for p in images]
        n_scene = int(round(per * FPS))
        with imageio.get_writer(str(silent), fps=FPS, codec="libx264", quality=7,
                                macro_block_size=1, pixelformat="yuv420p") as w:
            for img, cap in zip(imgs, captions):
                for k in range(n_scene):
                    w.append_data(_frame(img, k / max(1, n_scene - 1), cap))
        cmd = [FFMPEG, "-y", "-i", str(silent)]
        if audio:
            filters = []
            # Fit all narration into the allowed duration instead of cutting off the ending.
            speed = max(1.0, dur / (total - 0.6))
            while speed > 2.0:
                filters.append("atempo=2.0")
                speed /= 2.0
            if speed > 1.0:
                filters.append(f"atempo={speed:.6f}")
                log("  narration: adjusted speech speed to fit the video")
            filters.append("apad")
            cmd += ["-i", str(audio), "-af", ",".join(filters)]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono"]
        cmd += ["-c:v", "copy", "-c:a", "aac", "-t", f"{total:.2f}", "-movflags", "+faststart", str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError("ffmpeg mux failed: " + r.stderr[-400:])
    return out_path, total
