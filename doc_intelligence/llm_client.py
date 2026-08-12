"""LLM client abstraction.

Two real backends (Anthropic, OpenAI) plus a deterministic ``StubLLM`` used in
tests and for offline demos. All backends return a JSON string when given a
JSON-shaped prompt.

Production callers should set ``ANTHROPIC_API_KEY`` (default) or
``OPENAI_API_KEY`` and pass ``provider="openai"``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Protocol


class LLMClient(Protocol):
    """Minimal protocol — one call returns a string (typically JSON)."""

    def complete(self, *, system: str, user: str, max_tokens: int = 2048) -> str: ...


class AnthropicClient:
    """Wraps the Anthropic Messages API. Lazy-imports the SDK."""

    def __init__(self, model: str = "claude-opus-4-7", api_key: str | None = None) -> None:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - import-time path
            raise RuntimeError(
                "anthropic SDK not installed. `pip install doc-intelligence[anthropic]`"
            ) from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model

    def complete(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate text blocks
        parts = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


class OpenAIClient:
    """Wraps the OpenAI Responses-style chat completion. Lazy-imports the SDK."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai SDK not installed. `pip install doc-intelligence[openai]`"
            ) from exc
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def complete(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class StubLLM:
    """Deterministic LLM for tests / offline demos.

    Supports two modes:

    * Pass a ``responder`` callable that receives ``(system, user)`` and returns a string.
    * Pass a ``mapping`` of substrings → JSON strings; the first key found in the
      ``user`` prompt wins.
    """

    def __init__(
        self,
        *,
        responder: Callable[[str, str], str] | None = None,
        mapping: dict[str, str] | None = None,
        default: str | None = None,
    ) -> None:
        self._responder = responder
        self._mapping = mapping or {}
        self._default = default

    def complete(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        if self._responder is not None:
            return self._responder(system, user)
        for needle, response in self._mapping.items():
            if needle.lower() in user.lower():
                return response
        if self._default is not None:
            return self._default
        raise RuntimeError("StubLLM has no matching response and no default.")


def get_default_client(provider: str = "anthropic", **kwargs: Any) -> LLMClient:
    """Factory used by :class:`Extractor` when the caller doesn't pass one."""
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    if provider == "openai":
        return OpenAIClient(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def parse_json_response(text: str) -> dict:
    """Extract a JSON object from an LLM response that may include code fences or prose."""
    text = text.strip()
    # First try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip code fence
    match = _JSON_BLOCK.search(text)
    if match:
        return json.loads(match.group(1))
    # Last resort: find the first balanced brace block
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Unterminated JSON in LLM response: {text[:200]!r}")
