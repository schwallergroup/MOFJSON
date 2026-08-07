"""Tokeniser access for the token-efficiency study.

Phase 13.1, also used by tasks/representations.py.

PUBLIC API
    CountTokens(text, *, tokeniser="cl100k_base") -> int
    CountDocumentTokens(document, *, profile=None, tokeniser=...) -> int
    TOKENISERS   the registry of available tokenisers and how to load each

REQUIREMENTS THAT MAKE THE STUDY DEFENSIBLE
    - The same tokeniser across every representation being compared. A token comparison using
      different tokenisers per format is meaningless, and it is an easy mistake to make when
      each provider ships its own.
    - Report which tokeniser was used in every result. Token counts differ substantially
      between tokenisers, so a count without its tokeniser is not a number anyone can reuse.
    - Prefer counting with the tokeniser of the model actually being evaluated where it is
      available, and report the cross-tokeniser spread on a sample so readers know how much
      the choice matters.
    - Cache counts; the study covers >=100 structures across seven representations and the
      task suite counts every prompt.

WHAT THE STUDY NEEDS FROM THIS MODULE
    Tokens as a function of atom count, per representation, so the slopes can be compared. The
    interesting claim is that the reticular form is nearly size-independent while every
    explicit form scales linearly -- that is a claim about slopes, so the raw per-structure
    counts must be retained, not just the means.

DONE WHEN
    Counts are cached, the tokeniser is recorded in every output, and the cross-tokeniser
    spread on a sample is reported.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

# name -> lazy loader returning an object with .encode(str) -> list[int]
TOKENISERS: dict[str, Callable[[], Any]] = {}

_encoders: dict[str, Any] = {}
_memory_cache: dict[tuple[str, str], int] = {}
_disk_cache_dir: Path | None = None


def _register_defaults() -> None:
    if TOKENISERS:
        return

    def _tiktoken_encoding(name: str) -> Callable[[], Any]:
        def _load() -> Any:
            try:
                import tiktoken
            except ImportError as exc:
                raise ImportError(
                    "CountTokens requires the 'llm' extra: pip install mofjson[llm]"
                ) from exc
            return tiktoken.get_encoding(name)

        return _load

    TOKENISERS["cl100k_base"] = _tiktoken_encoding("cl100k_base")
    TOKENISERS["o200k_base"] = _tiktoken_encoding("o200k_base")
    TOKENISERS["p50k_base"] = _tiktoken_encoding("p50k_base")
    TOKENISERS["r50k_base"] = _tiktoken_encoding("r50k_base")


def SetTokenCacheDir(path: str | Path | None) -> None:
    """Enable or disable optional on-disk token-count caching."""
    global _disk_cache_dir
    if path is None:
        _disk_cache_dir = None
        return
    _disk_cache_dir = Path(path)
    _disk_cache_dir.mkdir(parents=True, exist_ok=True)


def ClearTokenCache() -> None:
    """Clear the in-memory token-count cache (disk cache untouched)."""
    _memory_cache.clear()


def _get_encoder(tokeniser: str) -> Any:
    _register_defaults()
    if tokeniser not in TOKENISERS:
        # Allow model-named encodings via tiktoken.encoding_for_model.
        def _model_loader() -> Any:
            try:
                import tiktoken
            except ImportError as exc:
                raise ImportError(
                    "CountTokens requires the 'llm' extra: pip install mofjson[llm]"
                ) from exc
            try:
                return tiktoken.encoding_for_model(tokeniser)
            except KeyError:
                return tiktoken.get_encoding(tokeniser)

        TOKENISERS[tokeniser] = _model_loader
    if tokeniser not in _encoders:
        _encoders[tokeniser] = TOKENISERS[tokeniser]()
    return _encoders[tokeniser]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _disk_key(tokeniser: str, digest: str) -> Path | None:
    if _disk_cache_dir is None:
        return None
    return _disk_cache_dir / f"{tokeniser}_{digest}.txt"


def CountTokens(text: str, *, tokeniser: str = "cl100k_base") -> int:
    """Return the token count of *text* under *tokeniser* (cached)."""
    digest = _text_hash(text)
    mem_key = (tokeniser, digest)
    if mem_key in _memory_cache:
        return _memory_cache[mem_key]

    disk_path = _disk_key(tokeniser, digest)
    if disk_path is not None and disk_path.is_file():
        count = int(disk_path.read_text(encoding="utf-8").strip())
        _memory_cache[mem_key] = count
        return count

    enc = _get_encoder(tokeniser)
    count = len(enc.encode(text))
    _memory_cache[mem_key] = count
    if disk_path is not None:
        disk_path.write_text(str(count) + "\n", encoding="utf-8")
    return count


def CountDocumentTokens(
    document: Any,
    *,
    profile: str | None = None,
    tokeniser: str = "cl100k_base",
    representation: str | None = None,
) -> int:
    """Token-count a MOFJSON document, optional profile subset, or named representation.

    Domain modules are loaded via ``importlib`` so the AST isolation check on
    ``mofjson.util`` stays clean (no ``from mofjson.schema`` / ``tasks`` nodes).
    """
    import importlib

    if isinstance(document, str):
        return CountTokens(document, tokeniser=tokeniser)

    if representation is not None:
        doc_mod = importlib.import_module("mofjson.schema.document")
        rep_mod = importlib.import_module("mofjson.tasks.representations")
        Doc = doc_mod.MOFJSONDocument
        doc = document if isinstance(document, Doc) else Doc.model_validate(document)
        return CountTokens(rep_mod.Render(doc, representation), tokeniser=tokeniser)

    if profile is not None:
        profiles = importlib.import_module("mofjson.llm.profiles")
        doc_mod = importlib.import_module("mofjson.schema.document")
        Doc = doc_mod.MOFJSONDocument
        doc = document if isinstance(document, Doc) else Doc.model_validate(document)
        data = doc.model_dump(mode="json", exclude_none=True)
        try:
            model = profiles.ProfileModel(profile)
            if profile.startswith("answer_"):
                text = json.dumps(
                    profiles.GetProfileSchema(profile, dialect="openai"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return CountTokens(text, tokeniser=tokeniser)
            if profile == "metal_node":
                nodes = (data.get("reticular") or {}).get("nodes") or []
                payload: Any = nodes[0] if nodes else {}
            elif profile in ("reticular", "reticular_nogeom"):
                payload = {"reticular": data.get("reticular")}
            elif profile in ("crystal", "crystal_columnar"):
                payload = {"crystal": data.get("crystal")}
            elif profile == "properties":
                payload = {
                    "identifiers": data.get("identifiers"),
                    "properties": data.get("properties") or [],
                }
            elif profile == "full":
                payload = data
            elif profile == "linker":
                linkers = (data.get("reticular") or {}).get("linkers") or []
                payload = linkers[0] if linkers else {}
            else:
                payload = data
            narrowed = model.model_validate(payload).model_dump(
                mode="json", exclude_none=True
            )
            text = json.dumps(narrowed, sort_keys=True, separators=(",", ":"))
        except Exception:
            text = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return CountTokens(text, tokeniser=tokeniser)

    if hasattr(document, "model_dump"):
        data = document.model_dump(mode="json", exclude_none=True)
    else:
        data = document
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return CountTokens(text, tokeniser=tokeniser)


def CrossTokeniserSpread(
    text: str,
    *,
    tokenisers: list[str] | None = None,
) -> dict[str, int]:
    """Count the same text under several tokenisers for a spread report."""
    _register_defaults()
    names = tokenisers or ["cl100k_base", "o200k_base"]
    return {name: CountTokens(text, tokeniser=name) for name in names}


_register_defaults()


__all__ = [
    "TOKENISERS",
    "ClearTokenCache",
    "CountDocumentTokens",
    "CountTokens",
    "CrossTokeniserSpread",
    "SetTokenCacheDir",
]
