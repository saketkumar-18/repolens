"""OpenAI-compatible chat client with retries, fallback models, and JSON mode.

Works with OpenRouter (default), OpenAI, Groq, Ollama, vLLM, LM Studio —
anything exposing /chat/completions. Pure httpx, no SDK lock-in.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = [
    "minimax/minimax-m3:free",
    "z-ai/glm-5.2:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]


class LLMError(RuntimeError):
    pass


@dataclass
class LLMClient:
    base_url: str = field(default_factory=lambda: os.getenv("REPOLENS_BASE_URL", DEFAULT_BASE_URL))
    api_key: str = field(
        default_factory=lambda: (
            os.getenv("REPOLENS_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
    )
    models: list[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in os.getenv("REPOLENS_MODELS", ",".join(DEFAULT_MODELS)).split(",")
            if m.strip()
        ]
    )
    timeout: float = 180.0
    max_retries: int = 3

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h["HTTP-Referer"] = "https://github.com/saketkumar-18/repolens"
        h["X-Title"] = "RepoLens"
        return h

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
        model: str | None = None,
    ) -> tuple[str, str, int]:
        """Run a chat completion. Returns (content, model_used, approx_tokens).

        Tries each configured model in order; retries transient errors
        (429/5xx/timeouts) with exponential backoff before falling through
        to the next model.
        """
        candidates = [model] if model else list(self.models)
        last_err: Exception | None = None
        for m in candidates:
            payload: dict = {
                "model": m,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            for attempt in range(self.max_retries):
                try:
                    with httpx.Client(timeout=self.timeout) as client:
                        resp = client.post(
                            f"{self.base_url.rstrip('/')}/chat/completions",
                            headers=self._headers(),
                            json=payload,
                        )
                    if resp.status_code in (408, 429, 500, 502, 503, 504):
                        raise LLMError(f"transient HTTP {resp.status_code}: {resp.text[:200]}")
                    if resp.status_code >= 400:
                        # permanent error for this model (bad id, auth) -> try next model
                        last_err = LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                        break
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"] or ""
                    usage = data.get("usage", {})
                    tokens = usage.get("total_tokens", 0)
                    return content, data.get("model", m), tokens
                except (httpx.TimeoutException, httpx.TransportError, LLMError) as e:
                    last_err = e
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** (attempt + 1))
        raise LLMError(f"all models failed; last error: {last_err}")

    def chat_json(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> tuple[dict, str, int]:
        """Chat completion that must return JSON. Extracts + parses robustly."""
        content, used, tokens = self.chat(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=True, model=model
        )
        parsed = extract_json(content)
        if parsed is None:
            # one retry without json_mode (some providers mishandle it)
            content, used, tokens = self.chat(
                messages, temperature=temperature, max_tokens=max_tokens, json_mode=False, model=used
            )
            parsed = extract_json(content)
        if parsed is None:
            raise LLMError(f"model returned no parseable JSON. Raw head: {content[:300]!r}")
        return parsed, used, tokens


def extract_json(text: str) -> dict | None:
    """Best-effort JSON object extraction from an LLM reply."""
    text = text.strip()
    # strip code fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # find the outermost {...}
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
