"""Thin REST clients for Anthropic / OpenAI / Gemini + JSON-validated generation."""
from __future__ import annotations

import json
import os
import re
from typing import Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from .config import DEFAULT_MODELS, OLLAMA_HOST, ollama_models

T = TypeVar("T", bound=BaseModel)


GEMINI_FALLBACKS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str | None = None, timeout: int | None = None):
        # Free-tier "thinking" models can take well over a minute on long course notes.
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", "180"))
        self.provider, self.api_key, self.timeout = provider, api_key, timeout
        self.model = model or DEFAULT_MODELS[provider]
        if provider == "ollama" and not self.model:
            installed = ollama_models()
            if not installed:
                raise LLMError("Ollama is not running or has no models - run: ollama pull qwen2.5:7b")
            self.model = installed[0]

    def complete(self, system: str, prompt: str, max_tokens: int = 2500, temperature: float = 0.3) -> str:
        p = self.provider
        try:
            if p == "ollama":                      # local model: no key, no internet
                r = requests.post(
                    f"{OLLAMA_HOST}/api/chat",
                    json={"model": self.model, "stream": False, "format": "json",
                          "options": {"temperature": temperature, "num_predict": max_tokens},
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": prompt}]},
                    timeout=(10, max(self.timeout, 300)))
                if r.status_code == 404:
                    raise LLMError(f"Ollama has no model '{self.model}' - run: ollama pull {self.model}")
                r.raise_for_status()
                text = re.sub(r"<think>.*?</think>", "", r.json()["message"]["content"], flags=re.S)
                return self._require_text(text)
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
                models = [self.model] + [x for x in GEMINI_FALLBACKS if x != self.model]
                problems = []
                for i, m in enumerate(models):
                    is_last = i == len(models) - 1
                    try:
                        r = requests.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                            headers={"x-goog-api-key": self.api_key},
                            json={"systemInstruction": {"parts": [{"text": system}]},
                                  "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                  "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}},
                            timeout=(15, self.timeout))
                    except (requests.Timeout, requests.ConnectionError) as e:
                        # Slow or overloaded model -> try the next (lighter) one instead of going offline.
                        problems.append(f"{m}: {type(e).__name__}")
                        if is_last:
                            raise
                        continue
                    if r.status_code == 400 and "API key" in r.text:
                        r.raise_for_status()          # a bad key fails the same on every model
                    if r.status_code in (400, 404, 429, 500, 503) and not is_last:
                        # retired model, no free-tier quota for it, or overloaded -> next model
                        problems.append(f"{m}: HTTP {r.status_code}")
                        continue
                    r.raise_for_status()
                    self.model = m                    # later calls in this run start with the model that worked
                    parts = r.json()["candidates"][0]["content"]["parts"]
                    return self._require_text("".join(x.get("text", "") for x in parts if not x.get("thought")))
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


def generate_json_items(client: LLMClient, system: str, prompt: str, item_model: Type[T], key: str,
                        retries: int = 2, **kw) -> tuple[list[T], list[str]]:
    """Like `generate_json`, but validates each element of `reply[key]` on its own.

    One malformed question no longer discards the whole batch: valid items are kept and
    invalid ones are reported. The model is only re-asked when *no* item is valid.
    Returns (valid_items, error_messages_for_rejected_items).
    """
    last_err = ""
    for attempt in range(retries + 1):
        p = prompt if not last_err else (
            f"{prompt}\n\nYour previous reply was invalid: {last_err}\n"
            "Return ONLY corrected JSON that satisfies the schema.")
        raw = client.complete(system, p, **kw)
        try:
            data = extract_json(raw)
        except ValueError as e:
            last_err = str(e)[:600]
            continue
        items = data.get(key) if isinstance(data, dict) else data
        if not isinstance(items, list) or not items:
            last_err = f'expected a non-empty "{key}" list'
            continue
        good, errors = [], []
        for i, item in enumerate(items):
            try:
                good.append(item_model.model_validate(item))
            except ValidationError as e:
                errors.append(f"{key}[{i}]: " + "; ".join(
                    f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())[:300])
        if good:
            return good, errors
        last_err = " | ".join(errors)[:600]
    raise LLMError(f"Could not obtain any valid {key} after {retries + 1} attempts: {last_err}")
