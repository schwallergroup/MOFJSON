"""Stage 6: determine the underlying net.

Phase 08.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mofjson.deconstruct.partition import PartitionResult
from mofjson.schema.metal_node import MetalNode
from mofjson.schema.reticular import Topology, parse_mofid_topology_field
from mofjson.vocab.topology import UNAMBIGUOUS_SIGNATURES


@dataclass
class QuotientGraph:
    """Nodes and linkers as vertices; cut bonds as edges."""

    node_ids: list[str]
    linker_ids: list[str]
    edges: list[tuple[str, str]]  # (node_id, linker_id)
    node_connectivity: dict[str, int] = field(default_factory=dict)
    linker_connectivity: dict[str, int] = field(default_factory=dict)
    n_framework_components: int = 1


def BuildQuotientGraph(partition: PartitionResult) -> QuotientGraph:
    """Build the node–linker quotient graph from a partition."""
    node_ids = [c.id for c in partition.metal_components]
    linkers = [c for c in partition.organic_components if c.role == "linker"]
    linker_ids = [c.id for c in linkers]
    metal_of = {
        idx: c.id for c in partition.metal_components for idx in c.metal_indices
    }
    linker_of = {idx: c.id for c in linkers for idx in c.site_indices}
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for cb in partition.cut_bonds:
        nid = metal_of.get(cb.metal_index)
        lid = linker_of.get(cb.donor_index)
        if nid is None or lid is None:
            continue
        key = (nid, lid)
        if key not in seen:
            seen.add(key)
            edges.append(key)
    node_conn: dict[str, int] = {nid: 0 for nid in node_ids}
    linker_conn: dict[str, int] = {lid: 0 for lid in linker_ids}
    for nid, lid in edges:
        node_conn[nid] = node_conn.get(nid, 0) + 1
        linker_conn[lid] = linker_conn.get(lid, 0) + 1
    return QuotientGraph(
        node_ids=node_ids,
        linker_ids=linker_ids,
        edges=edges,
        node_connectivity=node_conn,
        linker_connectivity=linker_conn,
        n_framework_components=partition.n_framework_components,
    )


def InferTopology(
    quotient_graph: QuotientGraph,
    nodes: list[MetalNode],
    linkers: list[Any],
) -> Topology:
    """Infer topology from connectivity signature; never invent ambiguous symbols."""
    # Representative connectivities (mode of node / linker connectivities).
    from collections import Counter

    n_vals = [n.connectivity for n in nodes if n.connectivity]
    # Prefer MetalNode.connectivity (attachment-point count) over quotient degree.
    if not n_vals:
        n_vals = [c for c in quotient_graph.node_connectivity.values() if c]
    l_vals = []
    for linker in linkers:
        if getattr(linker, "role", None) not in (None, "linker"):
            continue
        if getattr(linker, "connectivity", None):
            l_vals.append(linker.connectivity)
    if not l_vals:
        l_vals = [c for c in quotient_graph.linker_connectivity.values() if c]

    node_c = Counter(n_vals).most_common(1)[0][0] if n_vals else None
    linker_c = Counter(l_vals).most_common(1)[0][0] if l_vals else None
    vertex_connectivity = None
    if node_c is not None and linker_c is not None:
        vertex_connectivity = [int(node_c), int(linker_c)]

    # Dimensionality from node kinds.
    if any(n.kind == "rod" for n in nodes):
        dimensionality = 1
    elif any(n.kind == "layer" for n in nodes):
        dimensionality = 2
    else:
        dimensionality = 3

    interpenetrating = quotient_graph.n_framework_components > 1
    n_nets = quotient_graph.n_framework_components if interpenetrating else 1

    rcsr = None
    confidence = "none"
    source = "manual"
    if node_c is not None and linker_c is not None:
        sig = (int(node_c), int(linker_c))
        if sig in UNAMBIGUOUS_SIGNATURES:
            rcsr = UNAMBIGUOUS_SIGNATURES[sig]
            confidence = "medium"
            source = "manual"
            n_nets = 1
            interpenetrating = False if n_nets == 1 else interpenetrating
        else:
            # Chemistry hint: (4,2) + imidazolate → sod (low confidence).
            donor_groups = {
                d.donor_group
                for n in nodes
                for s in (n.sites or [])
                for d in (s.donors or [])
            }
            if sig == (4, 2) and ("imidazolate_N" in donor_groups or "azolate_N" in donor_groups):
                rcsr = "sod"
                confidence = "low"
                source = "manual"
                n_nets = 1

    if rcsr is None:
        # Schema: null symbol ⇒ n_nets=0 and confidence=none; interpenetrating
        # cannot be True without n_nets>1, so record multiplicity only as a warning
        # via dimensionality / quotient diagnostics.
        return Topology(
            rcsr_symbol=None,
            n_nets=0,
            dimensionality=dimensionality,
            interpenetrating=None,
            vertex_connectivity=vertex_connectivity,
            source=source,
            confidence="none",
        )

    # Signature lookup yields a single net symbol; do not claim interpenetration
    # without a multi-net determination.
    return Topology(
        rcsr_symbol=rcsr,
        n_nets=1,
        dimensionality=dimensionality,
        interpenetrating=False,
        vertex_connectivity=vertex_connectivity,
        source=source,
        confidence=confidence,
    )


def ReconcileTopology(inferred: Topology, asserted: Topology | None) -> Topology:
    """Compare inferred vs asserted (KG/MOFid); keep disagreements as warnings via alt symbols."""
    if asserted is None or asserted.rcsr_symbol is None:
        return inferred
    if inferred.rcsr_symbol is None:
        # Prefer asserted when we have nothing.
        return asserted.model_copy(
            update={
                "alternative_symbols": None,
            }
        )
    inf = inferred.rcsr_symbol
    ass = asserted.rcsr_symbol
    inf_set = {inf} if isinstance(inf, str) else set(inf)
    ass_set = {ass} if isinstance(ass, str) else set(ass)
    if inf_set == ass_set:
        return inferred.model_copy(update={"confidence": "high", "source": asserted.source})
    # Disagreement: keep inferred, record asserted as alternatives.
    alts = sorted(ass_set - inf_set)
    return inferred.model_copy(
        update={
            "alternative_symbols": alts or None,
            "confidence": "low",
        }
    )


def topology_from_mofid(mofid: str | None) -> Topology | None:
    if not mofid:
        return None
    # MOFid line: smiles;topology.catN;...
    parts = mofid.split(";")
    if len(parts) < 2:
        return parse_mofid_topology_field(None)
    return parse_mofid_topology_field(parts[1])


__all__ = [
    "BuildQuotientGraph",
    "InferTopology",
    "QuotientGraph",
    "ReconcileTopology",
    "topology_from_mofid",
]
