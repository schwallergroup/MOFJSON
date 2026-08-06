"""Deterministic serialization for comparing generated MOFJSON designs."""

from __future__ import annotations

import copy
import json
from typing import Any


def _sort_list(container: dict[str, Any], key: str, sort_key: Any) -> None:
    value = container.get(key)
    if isinstance(value, list):
        value.sort(key=sort_key)


def canonicalize(document: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied MOFJSON document with order-insensitive arrays sorted.

    Array order is not part of the identity of the design graph. The canonical
    form is intended for exact-match evaluation and hashing, not as a chemical
    canonicalization algorithm.
    """

    result = copy.deepcopy(document)
    _sort_list(result, "building_blocks", lambda item: item.get("id", ""))

    for block in result.get("building_blocks", []):
        graph = block.get("atom_graph", {})
        _sort_list(graph, "atoms", lambda item: item.get("id", ""))
        _sort_list(
            graph,
            "bonds",
            lambda item: (
                min(item.get("atoms", ["", ""])),
                max(item.get("atoms", ["", ""])),
                str(item.get("order", "")),
            ),
        )
        _sort_list(block, "connection_sites", lambda item: item.get("id", ""))
        for site in block.get("connection_sites", []):
            _sort_list(site, "anchor_atoms", lambda item: item)

    net = result.get("net", {})
    _sort_list(net, "vertices", lambda item: item.get("id", ""))
    _sort_list(net, "edges", lambda item: item.get("id", ""))

    realization = result.get("realization", {})
    _sort_list(realization, "atom_sites", lambda item: item.get("id", ""))
    _sort_list(realization, "bonds", lambda item: item.get("id", ""))
    return result


def canonical_dumps(document: dict[str, Any], *, indent: int | None = 2) -> str:
    """Serialize a document in the draft canonical order."""

    return json.dumps(
        canonicalize(document),
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
    )
