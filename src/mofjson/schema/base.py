"""Shared Pydantic configuration, identifier types, and cross-object reference types.

Phase 02.0 -- build this before any other schema module.

WHAT THIS FILE CONTAINS
    MOFJSONModel            the base class every model inherits. Its model_config sets:
                              extra="forbid"      unknown keys are an error, which is what
                                                  makes LLM output verifiable rather than
                                                  plausibly-shaped
                              populate_by_name    so aliases work both directions
                              ser_json_exclude_none / exclude_unset on dump
                                                  so absent fields cost zero tokens
                              frozen=False        documents are built incrementally by the
                                                  pipeline; freeze only if that changes
                              validate_assignment=True
                                                  the pipeline mutates objects in stages
                                                  and silent invalid intermediates are how
                                                  bad data escapes

    LocalId                 a short within-document identifier type ("M1", "L1", "N1").
                            Constrained by pattern. Short because these repeat many times
                            in a large document and every character is a token.
    Ref                     a typed reference to a LocalId elsewhere in the document, so
                            that document.py's validators can check every reference
                            resolves rather than trusting string equality.
    IndexRef                an index into a columnar array (Phase 04's site lists), with
                            bounds validated against the referenced array's length.

    ContentHash             the sha256-based `mofjson_id` type (Phase 06.1)
    dump_canonical(model)   the single canonical serialisation used for hashing, golden
                            files and byte-for-byte comparisons: sorted keys, fixed float
                            format, no insignificant whitespace. Everything that needs
                            determinism calls this, so there is exactly one definition of
                            "the same document".

WHY THIS FILE EXISTS AT ALL
    Every one of these decisions is one that must be identical across all six schema
    modules. Duplicating `model_config` per module guarantees drift, and a single module
    quietly allowing extra keys would undermine the validation guarantee the whole format
    rests on.

DONE WHEN
    Every model in the package inherits MOFJSONModel, and a test asserts that no model
    overrides model_config.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

LocalId = Annotated[
    str,
    Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$"),
]


Ref = Annotated[
    str,
    Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$"),
]


IndexRef = Annotated[int, Field(ge=0)]


ContentHash = Annotated[
    str,
    Field(pattern=r"^sha256:[0-9a-f]{64}$"),
]


def _canonical_default(obj: Any) -> Any:
    if isinstance(obj, float):
        # Fixed, round-trippable float formatting for golden files / hashing.
        return float(f"{obj:.12g}")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


class MOFJSONModel(BaseModel):
    """Base class for every MOFJSON Pydantic model."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        # Documents are built incrementally; do not freeze.
        frozen=False,
    )

    def model_dump_sparse(self, **kwargs: Any) -> dict[str, Any]:
        """Dump excluding unset and None fields."""
        kwargs.setdefault("exclude_unset", True)
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("mode", "json")
        return self.model_dump(**kwargs)


def dump_canonical(model: BaseModel | dict[str, Any]) -> str:
    """Canonical JSON: sorted keys, compact separators, stable floats, no None/unset noise."""
    if isinstance(model, BaseModel):
        if isinstance(model, MOFJSONModel):
            data = model.model_dump_sparse()
        else:
            data = model.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    else:
        data = model
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_canonical_default,
    )


__all__ = [
    "ContentHash",
    "IndexRef",
    "LocalId",
    "MOFJSONModel",
    "Ref",
    "dump_canonical",
]
