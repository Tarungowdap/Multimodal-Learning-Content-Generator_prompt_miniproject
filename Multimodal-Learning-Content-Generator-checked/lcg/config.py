"""Runtime settings. Everything can come from environment variables or the UI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
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
}


@dataclass
class Settings:
    llm_provider: str = "auto"      # auto | anthropic | openai | gemini | offline
    image_backend: str = "auto"     # auto | pollinations | openai | huggingface | placeholder
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
        }
        return self.api_keys.get(name) or os.getenv(env_names.get(name, ""), "") or None

    def resolve_llm(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        for p in ("anthropic", "openai", "gemini"):
            if self.key(p):
                return p
        return "offline"
