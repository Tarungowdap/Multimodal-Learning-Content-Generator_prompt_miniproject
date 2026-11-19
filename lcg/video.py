"""Video module -> 5-10 s MP4 with captions + TTS narration.

Two renderers share the same captioning, timing and audio muxing:
  make_t2v_video  clip from a text-to-video model (lcg/t2v.py), fitted to 5-10 s
  make_video      image sequence with Ken-Burns pan/zoom (fallback, works offline)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
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


NARRATION_WPM = int(os.getenv("NARRATION_WPM", "140"))   # calm speaking rate (Windows default is ~200)

_PYTTSX3 = r"""
import sys, pyttsx3
text, rate, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
e = pyttsx3.init()
e.setProperty("rate", rate)
e.save_to_file(text, out)
e.runAndWait()
"""


def word_budget(wpm: int = NARRATION_WPM, seconds: float = MAX_S - 0.6) -> int:
    """How many words fit in the clip when spoken at `wpm` (never speed speech up to make it fit)."""
    return max(6, int(seconds * wpm / 60))


_WEAK_END = set("""a an the of by to in on at for from with and or but which that who whose whom as into onto
is are was were be been its their this these those than then so such very some each any""".split())


def shorten(text: str, max_words: int) -> str:
    """Whole words only, preferring a clause break, never ending on a function word."""
    words = text.replace("\n", " ").split()
    if len(words) > max_words:
        cut = words[:max_words]
        for i in range(len(cut) - 1, max(2, max_words // 2) - 1, -1):     # natural break inside allowance
            if cut[i].endswith((",", ";", ":", ")")):
                cut = cut[: i + 1]
                break
        while len(cut) > 2 and cut[-1].strip(",;:").lower() in _WEAK_END:
            cut = cut[:-1]
        words = cut
    txt = " ".join(words).rstrip(",;:- ")
    return txt if txt.endswith((".", "!", "?")) else txt + "."


def fit_narration(lines: list[str], wpm: int = NARRATION_WPM) -> list[str]:
    """Shorten per-scene narration so the whole script fits the 10 s clip at a natural pace."""
    per = max(3, word_budget(wpm) // max(1, len(lines)))
    return [shorten(ln, per) for ln in lines]


def synthesize(text: str, out: Path, log=lambda m: None, *, allow_online: bool = True,
               wpm: int = NARRATION_WPM) -> Path | None:
    """Offline voices first, at a calm rate: pyttsx3 (Windows SAPI5 / macOS NSSpeech / Linux eSpeak)
    -> espeak-ng -> gTTS (only if installed and online) -> None (silent video with captions)."""
    wav = out.with_suffix(".wav")
    try:
        r = subprocess.run([sys.executable, "-c", _PYTTSX3, text, str(int(wpm)), str(wav)],
                           capture_output=True, timeout=90)
        if r.returncode == 0 and wav.exists() and wav.stat().st_size > 1000:
            return wav
        if r.returncode != 0:
            log(f"  narration: system voice unavailable ({r.stderr.decode(errors='replace').strip().splitlines()[-1:]})")
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"  narration: system voice failed ({type(e).__name__})")
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            try:
                result = subprocess.run([exe, "-s", str(int(wpm)), "-w", str(wav), text], capture_output=True, timeout=30)
                if result.returncode == 0 and wav.exists():
                    return wav
            except (OSError, subprocess.TimeoutExpired):
                continue
    if allow_online:
        try:
            from gtts import gTTS  # optional, not installed by default

            gTTS(text=text, lang="en", timeout=(5, 15)).save(str(out))
            return out
        except Exception as e:
            log(f"  narration: gTTS unavailable ({type(e).__name__})")
    log("  narration: no TTS engine, producing silent video with captions")
    return None


def narrate(lines: list[str], out: Path, wpm: int = NARRATION_WPM, log=lambda m: None, *,
            allow_online: bool = True, limit_s: float = MAX_S - 0.6, tries: int = 4) -> tuple[Path | None, list[str]]:
    """Record the narration and MEASURE it: if the real voice runs longer than the clip allows, shorten
    the script proportionally and record again. Speech is never sped up or cut off.
    Returns (audio file or None, the lines actually spoken)."""
    lines = fit_narration(lines, wpm)
    audio = None
    for attempt in range(tries):
        audio = synthesize(" ".join(lines), out, log, allow_online=allow_online, wpm=wpm)
        if audio is None:
            return None, lines
        dur = audio_duration(audio)
        if dur <= limit_s:
            return audio, lines
        words = [len(ln.split()) for ln in lines]
        ratio = limit_s / dur * 0.95
        new = [shorten(ln, max(2, int(n * ratio))) for ln, n in zip(lines, words)]
        if new == lines:
            break
        log(f"  narration: {dur:.1f}s is longer than the clip, shortening the script (attempt {attempt + 2})")
        lines = new
    return audio, lines


def _frame(img: Image.Image, t: float, caption: str) -> np.ndarray:
    """t in [0,1]: slow zoom-in (1.0 -> 1.12) with a caption bar."""
    z = 1.0 + 0.12 * t
    iw, ih = img.size
    cw, ch = iw / z, ih / z
    x0, y0 = (iw - cw) / 2, (ih - ch) / 2
    f = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.LANCZOS)
    return _caption(f, caption)


def _caption(f: Image.Image, caption: str) -> np.ndarray:
    d = ImageDraw.Draw(f, "RGBA")
    d.rectangle([0, H - 84, W, H], fill=(15, 23, 42, 205))
    font = _font(30)
    w = d.textlength(caption, font=font)
    d.text(((W - w) / 2, H - 62), caption, fill="white", font=font)
    return np.asarray(f)


def _target_duration(audio: Path | None) -> tuple[float, float]:
    """(narration seconds, clip seconds). The clip is always within 5-10 s."""
    dur = audio_duration(audio) if audio else 0.0
    return dur, (min(MAX_S, max(MIN_S, dur + 0.6)) if dur else 8.0)


def _mux(silent: Path, audio: Path | None, dur: float, total: float, out_path: Path, log) -> None:
    """Attach narration (sped up if needed, never cut off) or a silent track; force exact length."""
    cmd = [FFMPEG, "-y", "-i", str(silent)]
    if audio:
        filters = []
        # Narration is budgeted to fit (fit_narration), so it plays at its natural pace. Only a tiny,
        # inaudible tempo change (<= 8 %) is ever applied; anything longer fades out at the end.
        speed = dur / (total - 0.3) if total > 0.3 else 1.0
        if 1.0 < speed <= 1.08:
            filters.append(f"atempo={speed:.4f}")
        elif speed > 1.08:
            filters.append(f"afade=t=out:st={max(0.0, total - 0.8):.2f}:d=0.7")
            log("  narration: longer than the clip, faded out at the end")
        filters.append("apad")
        cmd += ["-i", str(audio), "-af", ",".join(filters)]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono"]
    cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-t", f"{total:.2f}",
            "-movflags", "+faststart", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("ffmpeg mux failed: " + r.stderr[-400:])


def _cover(img: Image.Image) -> Image.Image:
    """Resize to W x H, cropping (not stretching) clips of another aspect ratio."""
    iw, ih = img.size
    s = max(W / iw, H / ih)
    img = img.resize((max(W, round(iw * s)), max(H, round(ih * s))), Image.LANCZOS)
    x0, y0 = (img.width - W) // 2, (img.height - H) // 2
    return img.crop((x0, y0, x0 + W, y0 + H))


def read_native(path: Path, max_s: float = MAX_S + 1.0) -> tuple[list[np.ndarray], float]:
    """Decode a model clip at its own frame rate (W x H, cropped to 16:9). Returns (frames, fps)."""
    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        fps = float(reader.get_meta_data().get("fps") or 8.0)
        frames = []
        for i, fr in enumerate(reader):
            if i / fps > max_s:
                break
            frames.append(np.asarray(_cover(Image.fromarray(fr).convert("RGB"))))
    finally:
        reader.close()
    if len(frames) < 2:
        raise ValueError("text-to-video clip has no decodable frames")
    return frames, fps


def read_clip(path: Path, max_s: float = MAX_S) -> list[np.ndarray]:
    """Clip resampled to FPS (kept for callers that want plain frames)."""
    frames, fps = read_native(path, max_s)
    n_out = max(2, int(round(len(frames) / fps * FPS)))
    return [frames[min(len(frames) - 1, int(k * fps / FPS))] for k in range(n_out)]


MAX_STRETCH = 2.5      # play a short clip up to 2.5x slower (smooth, blended) before looping


def _source_positions(n_src: int, src_fps: float, total: float) -> list[float]:
    """Fractional source-frame position for every output frame: slow-motion stretch first,
    then (only for very short clips) a forward/backward loop. Never a jump cut."""
    n_out = int(round(total * FPS))
    span = n_src - 1
    src_span_s, out_span_s = span / src_fps, (n_out - 1) / FPS     # first-to-last frame times
    factor = 1.0 if src_span_s >= out_span_s else min(out_span_s / src_span_s, MAX_STRETCH)
    out = []
    for k in range(n_out):
        p = (k / FPS) / factor * src_fps                  # position in source frames
        if p > span:                                      # still too short: bounce back and forth
            cyc = p % (2 * span)
            p = cyc if cyc <= span else 2 * span - cyc
        out.append(min(p, span))
    return out


def _blend(frames: list[np.ndarray], p: float) -> np.ndarray:
    i = int(p)
    a = p - i
    if a < 1e-3 or i + 1 >= len(frames):
        return frames[i]
    return (frames[i].astype(np.float32) * (1 - a) + frames[i + 1].astype(np.float32) * a).astype(np.uint8)


def make_t2v_video(clip: Path, captions: list[str], narration: str, out_path: Path,
                   log=lambda m: None, *, allow_online: bool = True, wpm: int = NARRATION_WPM,
                   audio: Path | None | bool = False) -> tuple[Path, float]:
    """Fit a text-to-video clip to 5-10 s (smooth slow-motion, blended frames), burn in captions,
    and replace the audio with the narration at a natural pace."""
    if not captions:
        raise ValueError("Video needs at least one caption.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames, src_fps = read_native(clip)
    src_s = len(frames) / src_fps
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if audio is False:                       # not pre-recorded by the pipeline
            audio = synthesize(narration, td / "narration.mp3", log, allow_online=allow_online, wpm=wpm)
        dur, total = _target_duration(audio)
        pos = _source_positions(len(frames), src_fps, total)
        if src_s + 1e-6 < total:
            how = "slowed down smoothly" if total / src_s <= MAX_STRETCH else "slowed down and looped"
            log(f"  video: model clip {src_s:.1f}s -> {total:.1f}s ({how})")
        elif src_s > total + 1e-6:
            log(f"  video: model clip trimmed to {total:.1f}s")
        silent = td / "silent.mp4"
        n_out = len(pos)
        with imageio.get_writer(str(silent), fps=FPS, codec="libx264", quality=7,
                                macro_block_size=1, pixelformat="yuv420p") as w:
            for k, p in enumerate(pos):
                cap = captions[min(len(captions) - 1, k * len(captions) // n_out)]
                w.append_data(_caption(Image.fromarray(_blend(frames, p)), cap))
        _mux(silent, audio, dur, total, out_path, log)
    return out_path, total


def make_video(images: list[Path], captions: list[str], narration: str, out_path: Path,
               log=lambda m: None, *, allow_online: bool = True, wpm: int = NARRATION_WPM,
               audio: Path | None | bool = False) -> tuple[Path, float]:
    if not images or len(images) != len(captions):
        raise ValueError("Video needs at least one image and one caption per image.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if audio is False:
            audio = synthesize(narration, td / "narration.mp3", log, allow_online=allow_online, wpm=wpm)
        dur, total = _target_duration(audio)
        per = total / len(images)
        silent = td / "silent.mp4"
        imgs = [Image.open(p).convert("RGB") for p in images]
        n_scene = int(round(per * FPS))
        with imageio.get_writer(str(silent), fps=FPS, codec="libx264", quality=7,
                                macro_block_size=1, pixelformat="yuv420p") as w:
            for img, cap in zip(imgs, captions):
                for k in range(n_scene):
                    w.append_data(_frame(img, k / max(1, n_scene - 1), cap))
        _mux(silent, audio, dur, total, out_path, log)
    return out_path, total
