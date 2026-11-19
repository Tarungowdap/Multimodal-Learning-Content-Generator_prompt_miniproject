"""Runtime settings. Everything can come from environment variables or the UI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # optional .env support
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

OUTPUT_DIR = Path(os.getenv("LCG_OUTPUT_DIR", "outputs"))

DEFAULT_MODELS = {
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    "ollama": os.getenv("OLLAMA_MODEL", ""),      # empty = first model installed in Ollama
}
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST


def ollama_models(timeout: float = 1.5) -> list[str]:
    """Models installed in a local Ollama server ([] if Ollama is not running)."""
    try:
        import requests

        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
        return [m["name"] for m in r.json().get("models", [])] if r.ok else []
    except Exception:
        return []


@lru_cache(maxsize=1)
def local_video_installed() -> bool:
    """True when torch + diffusers are importable (the local text-to-video model can run)."""
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers", "transformers"))


@dataclass
class Settings:
    llm_provider: str = "auto"      # auto | anthropic | openai | gemini | offline
    image_backend: str = "auto"     # auto | pollinations | openai | huggingface | placeholder
    video_backend: str = "auto"     # auto | local | novai | gemini | huggingface | slideshow
    narration_wpm: int = int(os.getenv("NARRATION_WPM", "140"))   # speaking rate (lower = slower)
    ollama_model: str = ""          # empty = OLLAMA_MODEL or the first installed model
    api_keys: dict = field(default_factory=dict)  # provider -> key (UI overrides env)
    num_questions: int = 5
    top_k: int = 6
    max_repair_rounds: int = 2
    seed: int = 42

    def key(self, name: str) -> str | None:
        env_names = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "huggingface": "HF_TOKEN",
            "pollinations": "POLLINATIONS_API_KEY",
            "novai": "NOVAI_API_KEY",
        }
        # Strip spaces/newlines picked up when copying keys from a web page.
        key = (self.api_keys.get(name) or "").strip() or os.getenv(env_names.get(name, ""), "").strip()
        return key or None

    def resolve_video(self, allow_online: bool = True) -> list[str]:
        """Text-to-video backends to try, in order. Empty list => image-sequence slideshow."""
        if self.video_backend == "slideshow":
            return []
        if self.video_backend == "local":
            return ["local"]                     # explicit choice: the pipeline reports why if it is unavailable
        if self.video_backend != "auto":
            return [self.video_backend] if allow_online and self.key(self.video_backend) else []
        order = ["local"] if local_video_installed() else []          # no key, runs on this computer
        if allow_online:
            order += [b for b in ("novai", "gemini", "huggingface") if self.key(b)]
        return order

    def resolve_llm(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        for p in ("anthropic", "openai", "gemini"):
            if self.key(p):
                return p
        if ollama_models():                 # local model, no key needed
            return "ollama"
        return "offline"
