"""Anthropic structured-output helper.

Phase 10.3.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from mofjson.exceptions import LLMError
from mofjson.llm.openai import LLMResponse
from mofjson.llm.profiles import GetProfileSchema


def anthropic_tool(
    profile: str,
    *,
    name: str | None = None,
    dialect: str = "anthropic",
) -> dict[str, Any]:
    """Return a single forced-tool payload wrapping a profile schema."""
    schema = GetProfileSchema(profile, dialect=dialect)
    tool_name = name or f"mofjson_{profile}"
    return {
        "name": tool_name,
        "description": f"Emit a MOFJSON {profile} profile document.",
        "input_schema": schema,
    }


def _cache_key(model: str, prompt: str, schema: dict[str, Any]) -> str:
    blob = json.dumps(
        {"model": model, "prompt": prompt, "schema": schema},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class AnthropicClient:
    """Anthropic Messages API client using forced tool use for structured output."""

    def __init__(
        self,
        model: str,
        *,
        thinking_budget: int | None = 4096,
        max_tokens: int | None = 16384,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicClient requires the 'llm' extra: pip install mofjson[llm]"
            ) from exc

        self.model = model
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens if max_tokens is not None else 16384
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache: dict[str, LLMResponse] = {}
        self.max_retries = max_retries
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_cache(self, key: str) -> LLMResponse | None:
        if key in self._memory_cache:
            return self._memory_cache[key]
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        resp = LLMResponse(**data)
        self._memory_cache[key] = resp
        return resp

    def _store_cache(self, key: str, resp: LLMResponse) -> None:
        self._memory_cache[key] = resp
        if self._cache_dir is None:
            return
        path = self._cache_dir / f"{key}.json"
        path.write_text(
            json.dumps(
                {
                    "parsed": resp.parsed,
                    "raw_text": resp.raw_text,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "reasoning_tokens": resp.reasoning_tokens,
                    "latency_s": resp.latency_s,
                    "finish_reason": resp.finish_reason,
                    "model": resp.model,
                    "truncated": resp.truncated,
                    "refused": resp.refused,
                    "empty": resp.empty,
                    "metadata": resp.metadata,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def complete(
        self,
        prompt: str,
        *,
        profile: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if schema is None:
            if profile is None:
                raise ValueError("complete() requires profile= or schema=")
            schema = GetProfileSchema(profile, dialect="anthropic")
        tool = {
            "name": f"mofjson_{profile or 'custom'}",
            "description": f"Emit a MOFJSON {profile or 'custom'} document.",
            "input_schema": schema,
        }
        key = _cache_key(self.model, prompt, schema)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                    "tools": [tool],
                    "tool_choice": {"type": "tool", "name": tool["name"]},
                }
                # Thinking budget interacts with max_tokens; record both in metadata.
                if self.thinking_budget is not None and self.thinking_budget > 0:
                    kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget,
                    }
                response = self._client.messages.create(**kwargs)
                latency = time.perf_counter() - t0
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", None) if usage else None
                output_tokens = getattr(usage, "output_tokens", None) if usage else None
                # Map Anthropic thinking tokens onto reasoning_tokens for cross-provider
                # comparison; caveats: not all models expose a separate thinking count.
                reasoning_tokens = None
                if usage is not None:
                    reasoning_tokens = getattr(usage, "thinking_tokens", None)
                    if reasoning_tokens is None:
                        details = getattr(usage, "output_tokens_details", None)
                        if details is not None:
                            reasoning_tokens = getattr(details, "reasoning_tokens", None)

                parsed: dict[str, Any] | None = None
                raw_parts: list[str] = []
                for block in getattr(response, "content", None) or []:
                    btype = getattr(block, "type", None)
                    if btype == "tool_use":
                        inp = getattr(block, "input", None)
                        if isinstance(inp, dict):
                            parsed = inp
                            raw_parts.append(json.dumps(inp, sort_keys=True))
                    elif btype == "text":
                        text = getattr(block, "text", None)
                        if isinstance(text, str):
                            raw_parts.append(text)
                raw = "\n".join(raw_parts)
                stop = getattr(response, "stop_reason", None)
                finish = str(stop) if stop is not None else None
                truncated = finish in ("max_tokens", "model_context_window_exceeded")
                refused = finish == "refusal"
                empty = parsed is None and not raw.strip()
                model_version = getattr(response, "model", None) or self.model
                resp = LLMResponse(
                    parsed=parsed,
                    raw_text=raw,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    latency_s=latency,
                    finish_reason=finish,
                    model=model_version,
                    truncated=truncated,
                    refused=refused,
                    empty=empty,
                    metadata={
                        "thinking_budget": self.thinking_budget,
                        "max_tokens": self.max_tokens,
                    },
                )
                self._store_cache(key, resp)
                return resp
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                name = type(exc).__name__.lower()
                msg = str(exc).lower()
                if "rate" in name or "rate" in msg or "429" in msg:
                    time.sleep(2**attempt)
                    continue
                raise LLMError(f"Anthropic complete failed: {exc}") from exc
        raise LLMError(f"Anthropic complete failed after retries: {last_err}")


__all__ = ["AnthropicClient", "anthropic_tool"]
