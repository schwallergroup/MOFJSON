"""Content hashing and mofjson_id construction.

Phase 06.1.

PUBLIC API
    content_hash(obj) / ContentHash(obj) -> str   # "sha256:..." over canonical serialisation
    make_mofjson_id(...) / MakeMofjsonId(...)
    short_id(...) / ShortId(...)
    parse_mofid_line(raw) -> MofidParts
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Util must not import schema models. Duplicate the float canonicalisation used by
# dump_canonical so hashing works on plain dicts without a schema dependency.
_MOFID_SENTINEL_TOKENS = frozenset(
    {
        "ERROR",
        "TIMEOUT",
        "UNKNOWN",
        "NO_REF",
        "unstable",
        "error_unknown",
        "unknown_sql",
        "sql_unknown",
    }
)


def _canonical_default(obj: Any) -> Any:
    if isinstance(obj, float):
        return float(f"{obj:.12g}")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _to_jsonable(obj: Any) -> Any:
    """Convert models / nested structures to plain JSON-able data."""
    if hasattr(obj, "model_dump_sparse"):
        return obj.model_dump_sparse()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _to_canonical_json(obj: Any) -> str:
    """Canonical JSON string for hashing: sorted keys, compact, stable floats."""
    data = _to_jsonable(obj)
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_canonical_default,
    )


def content_hash(obj: Any) -> str:
    """Return ``sha256:<hex>`` over the canonical serialisation of *obj*."""
    payload = _to_canonical_json(obj)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def make_mofjson_id(
    *,
    crystal: Any | None = None,
    reticular: Any | None = None,
    structural: Mapping[str, Any] | None = None,
) -> str:
    """Hash structural layers only (crystal + reticular).

    Never include provenance, timestamps, pipeline versions or warnings.
    """
    if structural is not None:
        payload: Mapping[str, Any] | dict[str, Any] = structural
    else:
        built: dict[str, Any] = {}
        if crystal is not None:
            built["crystal"] = crystal
        if reticular is not None:
            built["reticular"] = reticular
        if not built:
            built = {"crystal": None, "reticular": None}
        payload = built
    return content_hash(payload)


def short_id(hash_str: str, *, length: int = 12) -> str:
    """Human-readable short form of a content hash (hex portion only)."""
    hex_part = hash_str.split(":", 1)[-1]
    if length < 1:
        raise ValueError("length must be >= 1")
    return hex_part[:length]


@dataclass(frozen=True, slots=True)
class MofidParts:
    """Parsed pieces of a full MOFid line (Phase 00.4)."""

    raw: str
    smiles: str | None
    topology_field: str | None
    structure_stem: str | None
    has_sentinel: bool
    sentinel_tokens: tuple[str, ...]


_MOFID_LINE_RE = re.compile(
    r"^(?P<head>.*?)\s+MOFid-v1\.(?P<topo>[^;]*);(?P<stem>.*)$",
    re.DOTALL,
)


def parse_mofid_line(raw: str | None) -> MofidParts:
    """Parse a full MOFid line; flag sentinel corruption in the topology field."""
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return MofidParts(
            raw=raw or "",
            smiles=None,
            topology_field=None,
            structure_stem=None,
            has_sentinel=True,
            sentinel_tokens=("empty",),
        )
    text = str(raw).strip()
    m = _MOFID_LINE_RE.match(text)
    if not m:
        # Bare stem or unparseable — keep as stem candidate, mark unknown.
        upper = text.upper()
        found = [t for t in _MOFID_SENTINEL_TOKENS if t in upper]
        return MofidParts(
            raw=text,
            smiles=None,
            topology_field=None,
            structure_stem=text if ";" not in text else text.rsplit(";", 1)[-1],
            has_sentinel=bool(found),
            sentinel_tokens=tuple(found),
        )

    smiles = m.group("head").strip() or None
    topo = m.group("topo").strip() or None
    stem = m.group("stem").strip() or None
    found_tokens: list[str] = []
    if topo:
        # Tokens may be comma-joined: ERROR,UNKNOWN.cat0 / TIMEOUT.cat0.NO_REF
        chunks = re.split(r"[.,]", topo)
        for chunk in chunks:
            c = chunk.strip()
            if not c:
                continue
            upper = c.upper()
            if upper in _MOFID_SENTINEL_TOKENS or c in _MOFID_SENTINEL_TOKENS:
                found_tokens.append(c if c in _MOFID_SENTINEL_TOKENS else upper)
            elif upper.startswith("NO_REF") or c == "NO_REF":
                found_tokens.append("NO_REF")
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for t in found_tokens:
        key = t.upper()
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return MofidParts(
        raw=text,
        smiles=smiles,
        topology_field=topo,
        structure_stem=stem,
        has_sentinel=bool(ordered),
        sentinel_tokens=tuple(ordered),
    )


# PascalCase aliases matching the stub public API.
ContentHash = content_hash
MakeMofjsonId = make_mofjson_id
ShortId = short_id
ParseMofid = parse_mofid_line

__all__ = [
    "ContentHash",
    "MakeMofjsonId",
    "MofidParts",
    "ParseMofid",
    "ShortId",
    "content_hash",
    "make_mofjson_id",
    "parse_mofid_line",
    "short_id",
]
