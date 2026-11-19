"""Thin REST clients for Anthropic / OpenAI / Gemini + JSON-validated generation."""
from __future__ import annotations

import json
import re
from typing import Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from .config import DEFAULT_MODELS

T = TypeVar("T", bound=BaseModel)


GEMINI_FALLBACKS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str | None = None, timeout: int = 90):
        self.provider, self.api_key, self.timeout = provider, api_key, timeout
        self.model = model or DEFAULT_MODELS[provider]

    def complete(self, system: str, prompt: str, max_tokens: int = 2500, temperature: float = 0.3) -> str:
        p = self.provider
        try:
            if p == "anthropic":
                r = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                    json={"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                          "system": system, "messages": [{"role": "user", "content": prompt}]},
                    timeout=self.timeout)
                r.raise_for_status()
                return self._require_text("".join(b.get("text", "") for b in r.json()["content"]))
            if p == "openai":
                r = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": prompt}]},
                    timeout=self.timeout)
                r.raise_for_status()
                return self._require_text(r.json()["choices"][0]["message"]["content"])
            if p == "gemini":
                last = None
                for m in [self.model] + [x for x in GEMINI_FALLBACKS if x != self.model]:
                    r = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                        headers={"x-goog-api-key": self.api_key},
                        json={"systemInstruction": {"parts": [{"text": system}]},
                              "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                              "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}},
                        timeout=self.timeout)
                    if r.status_code in (404, 400) and m != GEMINI_FALLBACKS[-1]:
                        last = r  # model retired / not available for this key -> try the next one
                        continue
                    r.raise_for_status()
                    self.model = m
                    parts = r.json()["candidates"][0]["content"]["parts"]
                    return self._require_text("".join(x.get("text", "") for x in parts if not x.get("thought")))
                last.raise_for_status()
        except requests.RequestException as e:
            detail = ""
            try:
                detail = " - " + e.response.json()["error"]["message"][:200]
            except Exception:
                pass
            raise LLMError(f"{p} request failed: {e}{detail}") from e
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError(f"{p} returned an empty or malformed response") from e
        raise LLMError(f"Unknown provider {p}")

    @staticmethod
    def _require_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("provider returned no text")
        return text


def extract_json(text: str) -> object:
    """Pull the first balanced JSON object/array out of a model reply."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in reply")
    # The standard decoder handles escaped quotes, backslashes and mixed nesting.
    value, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return value


def generate_json(client: LLMClient, system: str, prompt: str, model: Type[T], retries: int = 2, **kw) -> T:
    """Call the LLM and validate against a pydantic schema; feed errors back to self-repair."""
    last_err = ""
    for attempt in range(retries + 1):
        p = prompt if not last_err else (
            f"{prompt}\n\nYour previous reply was invalid: {last_err}\n"
            "Return ONLY corrected JSON that satisfies the schema.")
        raw = client.complete(system, p, **kw)
        try:
            return model.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as e:
            last_err = str(e)[:600]
    raise LLMError(f"Could not obtain valid JSON after {retries + 1} attempts: {last_err}")
