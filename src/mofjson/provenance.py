"""
The point of this module is narrow: give every derived/cached block
(building_blocks, net_edges, topology, a SLICES string, ...) a
machine-checkable record of *what it was computed from* and *what computed
it*, so "this cache still matches the raw structure" is a fact you can
verify in roughly constant time rather than an assumption baked into
whoever last ran the pipeline.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


def canonical_json(obj: Any) -> str:
    """Stable, whitespace-free, key-sorted JSON string, used only for
    hashing -- not a general serialization format."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def content_hash(*parts: Any) -> str:
    """SHA-256 over the canonical JSON of one or more raw-structure parts.

    Call as content_hash(atoms, bonds, node_linker_roles). Order matters --
    it's part of what gets hashed -- so always hash the same parts in the
    same order everywhere in the codebase (BuildDocument below is the one
    place that should ever construct this).
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(canonical_json(part).encode("utf-8"))
        h.update(b"\x00")  # separator so hashing ([a], [b]) != ([a, b])
    return h.hexdigest()


class Provenance(BaseModel):
    """Attached to every derived/cached block in a MOFJSON document."""

    method: str = Field(
        description="Name of the function that computed this block, e.g. 'CollapseToBuildingBlocks'."
    )
    version: str = Field(
        default="v1", description="Version tag for the method; bump on any behavior change."
    )
    computed_from: list[str] = Field(
        description="Which raw fields this block is a function of, e.g. ['atoms', 'bonds', 'node_linker_roles']."
    )
    computed_from_hash: str = Field(
        description=(
            "content_hash(...) of the exact raw values this block was derived from, "
            "in the order listed in computed_from."
        )
    )
