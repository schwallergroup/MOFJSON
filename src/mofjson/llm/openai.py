"""OpenAI structured-output helpers (Responses API + Chat Completions / EPFL).

Phase 10.3 / 13.2.1.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mofjson.exceptions import LLMError
from mofjson.llm.profiles import GetProfileSchema

EPFL_DEFAULT_BASE_URL = "https://inference.rcp.epfl.ch/v1"


@dataclass(slots=True)
class LLMResponse:
    """Unified provider response for Phase 13 cross-provider comparison."""

    parsed: dict[str, Any] | None
    raw_text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    finish_reason: str | None = None
    model: str | None = None
    truncated: bool = False
    refused: bool = False
    empty: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class Completer(Protocol):
    """Minimal protocol shared by OpenAI / EPFL / Anthropic clients."""

    model: str

    def complete(
        self,
        prompt: str,
        *,
        profile: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


def openai_response_format(
    profile: str,
    *,
    name: str | None = None,
    dialect: str = "openai",
) -> dict[str, Any]:
    """Return the Responses API ``text.format`` payload for a profile schema."""
    schema = GetProfileSchema(profile, dialect=dialect)
    return {
        "type": "json_schema",
        "name": name or f"mofjson_{profile}",
        "strict": True,
        "schema": schema,
    }


def _cache_key(model: str, prompt: str, schema: dict[str, Any]) -> str:
    blob = json.dumps(
        {"model": model, "prompt": prompt, "schema": schema},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _response_to_dict(resp: LLMResponse) -> dict[str, Any]:
    return {
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
    }


class _CacheMixin:
    _cache_dir: Path | None
    _memory_cache: dict[str, LLMResponse]

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
            json.dumps(_response_to_dict(resp), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _should_retry(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "rate" in name
        or "rate" in msg
        or "429" in msg
        or "timeout" in name
        or "timeout" in msg
        or "503" in msg
        or "502" in msg
        or "connection" in msg
    )


class OpenAIClient(_CacheMixin):
    """Thin OpenAI Responses API client with caching and token accounting."""

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str | None = "low",
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIClient requires the 'llm' extra: pip install mofjson[llm]"
            ) from exc

        self.model = model
        self.reasoning_effort = reasoning_effort
        kwargs: dict[str, Any] = {
            "api_key": api_key or os.environ.get("OPENAI_API_KEY"),
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache: dict[str, LLMResponse] = {}
        self.max_retries = max_retries
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

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
            schema = GetProfileSchema(profile, dialect="openai")
        fmt = {
            "type": "json_schema",
            "name": f"mofjson_{profile or 'custom'}",
            "strict": True,
            "schema": schema,
        }
        key = _cache_key(self.model, prompt, schema)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                call_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "input": prompt,
                    "text": {"verbosity": "low", "format": fmt},
                    "store": False,
                }
                if self.reasoning_effort is not None:
                    call_kwargs["reasoning"] = {"effort": self.reasoning_effort}
                response = self._client.responses.create(**call_kwargs)
                latency = time.perf_counter() - t0
                raw = getattr(response, "output_text", None) or ""
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", None) if usage else None
                output_tokens = getattr(usage, "output_tokens", None) if usage else None
                details = getattr(usage, "output_tokens_details", None) if usage else None
                reasoning_tokens = (
                    getattr(details, "reasoning_tokens", None) if details else None
                )
                status = getattr(response, "status", None)
                finish = str(status) if status is not None else None
                truncated = finish in ("incomplete", "length") or "max_tokens" in (
                    finish or ""
                )
                refused = bool(getattr(response, "refusal", None))
                empty = not raw.strip()
                parsed: dict[str, Any] | None = None
                if raw.strip() and not refused:
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        parsed = None
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
                )
                self._store_cache(key, resp)
                return resp
            except Exception as exc:
                last_err = exc
                if _should_retry(exc):
                    time.sleep(2**attempt)
                    continue
                raise LLMError(f"OpenAI complete failed: {exc}") from exc
        raise LLMError(f"OpenAI complete failed after retries: {last_err}")


class ChatCompletionsClient(_CacheMixin):
    """OpenAI-compatible Chat Completions client (EPFL RCP and similar).

    Uses ``chat.completions.create`` with JSON schema ``response_format`` when
    possible; falls back to ``json_object`` mode if the cluster rejects schemas.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path | None = None,
        max_retries: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        use_json_schema: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "ChatCompletionsClient requires the 'llm' extra: pip install mofjson[llm]"
            ) from exc

        self.model = model
        self.base_url = base_url or os.environ.get("EPFL_BASE_URL") or EPFL_DEFAULT_BASE_URL
        key = (
            api_key
            or os.environ.get("EPFL_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )
        # Hard timeout so a stuck RCP worker cannot hang the whole suite.
        self._client = OpenAI(api_key=key, base_url=self.base_url, timeout=180.0)
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache: dict[str, LLMResponse] = {}
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_json_schema = use_json_schema
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

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
            schema = GetProfileSchema(profile, dialect="openai")

        key = _cache_key(self.model, prompt, schema)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        schema_name = f"mofjson_{profile or 'custom'}"
        last_err: Exception | None = None
        use_schema = self.use_json_schema

        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                if use_schema:
                    response_format: dict[str, Any] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        },
                    }
                else:
                    response_format = {"type": "json_object"}

                create_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": response_format,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                response = self._client.chat.completions.create(**create_kwargs)
                latency = time.perf_counter() - t0
                choice = response.choices[0] if response.choices else None
                raw = ""
                finish = None
                if choice is not None:
                    raw = (choice.message.content or "") if choice.message else ""
                    finish = choice.finish_reason
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                output_tokens = getattr(usage, "completion_tokens", None) if usage else None
                details = getattr(usage, "completion_tokens_details", None) if usage else None
                reasoning_tokens = None
                if details is not None:
                    reasoning_tokens = getattr(details, "reasoning_tokens", None)

                truncated = finish in ("length", "incomplete")
                empty = not raw.strip()
                parsed: dict[str, Any] | None = None
                if raw.strip():
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        stripped = raw.strip()
                        if stripped.startswith("```"):
                            stripped = stripped.strip("`")
                            if stripped.lower().startswith("json"):
                                stripped = stripped[4:].lstrip()
                            try:
                                parsed = json.loads(stripped)
                            except json.JSONDecodeError:
                                parsed = None
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
                    refused=False,
                    empty=empty,
                    metadata={"base_url": self.base_url, "provider": "chat_completions"},
                )
                self._store_cache(key, resp)
                return resp
            except Exception as exc:
                last_err = exc
                msg = str(exc).lower()
                if use_schema and (
                    "response_format" in msg
                    or "json_schema" in msg
                    or "invalid_request" in msg
                    or "400" in msg
                ):
                    use_schema = False
                    continue
                if _should_retry(exc):
                    time.sleep(2**attempt)
                    continue
                raise LLMError(f"ChatCompletions complete failed: {exc}") from exc
        raise LLMError(f"ChatCompletions complete failed after retries: {last_err}")


class EpflClient(ChatCompletionsClient):
    """EPFL RCP inference endpoint (OpenAI-compatible Chat Completions)."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path | None = None,
        max_retries: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        use_json_schema: bool = True,
    ) -> None:
        resolved_model = (
            model
            or os.environ.get("EPFL_MODEL")
            or "mistralai/Mistral-Small-3.2-24B-Instruct-2506-bfloat16"
        )
        super().__init__(
            resolved_model,
            api_key=api_key or os.environ.get("EPFL_API_KEY"),
            base_url=base_url or os.environ.get("EPFL_BASE_URL") or EPFL_DEFAULT_BASE_URL,
            cache_dir=cache_dir,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            use_json_schema=use_json_schema,
        )


def LoadDotEnv(path: str | Path | None = None) -> None:
    """Load a local ``.env`` if python-dotenv is installed (no-op otherwise)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=path)


def MakeClient(
    provider: str,
    model: str | None = None,
    *,
    cache_dir: str | Path | None = None,
    reasoning_effort: str | None = "low",
    thinking_budget: int | None = 4096,
) -> Completer:
    """Factory for ``epfl`` | ``openai`` | ``anthropic`` clients."""
    LoadDotEnv()
    name = provider.lower().strip()
    if name in ("epfl", "chat", "chat_completions", "rcp"):
        return EpflClient(model=model, cache_dir=cache_dir)
    if name in ("openai", "oai"):
        if not model:
            raise ValueError("MakeClient(provider='openai') requires model=")
        return OpenAIClient(
            model, reasoning_effort=reasoning_effort, cache_dir=cache_dir
        )
    if name in ("anthropic", "claude"):
        from mofjson.llm.anthropic import AnthropicClient

        if not model:
            raise ValueError("MakeClient(provider='anthropic') requires model=")
        return AnthropicClient(
            model, thinking_budget=thinking_budget, cache_dir=cache_dir
        )
    raise ValueError(f"Unknown provider {provider!r}; expected epfl|openai|anthropic")


__all__ = [
    "EPFL_DEFAULT_BASE_URL",
    "ChatCompletionsClient",
    "Completer",
    "EpflClient",
    "LLMResponse",
    "LoadDotEnv",
    "MakeClient",
    "OpenAIClient",
    "openai_response_format",
]
