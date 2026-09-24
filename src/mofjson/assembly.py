"""
Assembly -- the reticular-chemistry-proper net: nodes as vertices, AND any
polytopic (>=3-connected) linker also as a vertex, with only ditopic
(exactly 2-connected) linkers contracted down to a plain edge. Everything
else in this codebase (building_blocks, net_edges) is the atomistic/
building-block layer; this is one level of abstraction further up, and is
where RCSR-style invariants (vertex count, edge count, genus, a rough
"kinds of vertex" count) actually live.


Reference: O'Keeffe, M.; Peskov, M. A.; Ramsden, S. J.; Yaghi, O. M. "The
Reticular Chemistry Structure Resource (RCSR) Database of, and Symbols
for, Crystal Nets." Acc. Chem. Res. 2008, 41, 1782-1789.
  - genus g = 1 + e - v (a measure of vertices-per-repeat-unit; used as a
    quick complexity fingerprint, not a unique identifier).
"""
from __future__ import annotations

from typing import Any

from src.mofjson.topology import BuildTopologicalNet


def BuildAssembly(building_blocks: list[dict[str, Any]], net_edges: list[dict[str, Any]]) -> dict[str, Any]:
    blocks_by_id = {b["id"]: b for b in building_blocks}
    vertices, edges, excluded = BuildTopologicalNet(building_blocks, net_edges)

    v_count = len(vertices)
    e_count = len(edges)  # already deduplicated/pruned by BuildTopologicalNet
    genus = 1 + e_count - v_count if v_count else 0

    def _identity(block: dict[str, Any]) -> str:
        return block.get("formula") or block.get("id", "").split("_")[0]

    vertex_kinds = {_identity(blocks_by_id[vid]) for vid in vertices if vid in blocks_by_id}

    return {
        "v": v_count,
        "e": e_count,
        "genus": genus,
        "n_vertex_kinds_proxy": len(vertex_kinds),
        "vertex_block_ids": sorted(vertices),
        "excluded_building_blocks": excluded,
    }


__all__ = ["BuildAssembly"]
