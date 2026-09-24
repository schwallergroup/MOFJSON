"""
AssignRoles -- automatic node/linker/guest classification from geometry
alone. 

Method:
  1. Cut every bond between a metal atom and a non-metal atom (using the
     same _METALS convention conversion.py already uses for its own
     zero-order coordination-bond default).
  2. Take connected components of what's left (metal-metal bonds and
     nonmetal-nonmetal bonds both kept -- only metal/nonmetal edges cut).
  3. A component containing at least one metal atom -> "node".
  4. A metal-free component with at least one cut bond reaching a node
     -> "linker".
  5. A metal-free component with NO cut bond to any node at all -- nothing
     tying it to the framework -- -> "guest" (e.g. solvent sitting in a
     pore with no covalent bond to the SBU).
"""
from __future__ import annotations

from typing import Any

from src.mofjson.conversion import _METALS


def AssignRoles(atoms: list[dict[str, Any]], bonds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = [a["id"] for a in atoms]
    element_of = {a["id"]: a["element"] for a in atoms}

    parent: dict[str, str] = {aid: aid for aid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    cut_bonds: list[dict[str, Any]] = []
    for b in bonds:
        s, t = b["source"], b["target"]
        es, et = element_of.get(s), element_of.get(t)
        if es is None or et is None:
            continue  # dummy '*' attachment atoms etc. -- not real atoms to classify
        s_metal, t_metal = es in _METALS, et in _METALS
        if s_metal != t_metal:
            cut_bonds.append(b)  # metal <-> nonmetal: this is the boundary, cut it
        else:
            union(s, t)  # metal-metal or nonmetal-nonmetal: same component

    components: dict[str, list[str]] = {}
    for aid in ids:
        components.setdefault(find(aid), []).append(aid)

    has_metal = {
        root: any(element_of[a] in _METALS for a in members)
        for root, members in components.items()
    }

    linked_to_node: set[str] = set()
    for b in cut_bonds:
        rs, rt = find(b["source"]), find(b["target"])
        # exactly one side is metal-containing by construction (we only cut
        # bonds where the metal/nonmetal split disagreed).
        if has_metal[rs] and not has_metal[rt]:
            linked_to_node.add(rt)
        elif has_metal[rt] and not has_metal[rs]:
            linked_to_node.add(rs)

    role_of_root: dict[str, str] = {}
    for root in components:
        if has_metal[root]:
            role_of_root[root] = "node"
        elif root in linked_to_node:
            role_of_root[root] = "linker"
        else:
            role_of_root[root] = "guest"

    return [{"atom_id": aid, "role": role_of_root[find(aid)]} for aid in ids]


__all__ = ["AssignRoles"]
