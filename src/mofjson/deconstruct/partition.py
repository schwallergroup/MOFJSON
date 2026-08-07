"""Stage 1: split the periodic bond graph into metal nodes and organic components.

Phase 08.1. Everything downstream depends on this being right.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Literal

from mofjson.exceptions import PartitionError
from mofjson.schema.crystal import PeriodicBond
from mofjson.schema.document import MOFJSONDocument
from mofjson.vocab.elements import METALS

# Non-metal neighbours allowed on a node-bridging atom (μ-O / μ-OH).
_NODE_BRIDGE_ALLOWED = frozenset({"H"})
# Merge metals that share at least this many syn-syn carboxylates (paddlewheels).
_CARBOXYLATE_BRIDGE_MIN = 2

ComponentRole = Literal["node", "linker", "ligand", "guest", "counterion"]


@dataclass
class CutBond:
    """A metal–donor bond removed during partition."""

    metal_index: int
    donor_index: int
    metal_id: str
    donor_id: str
    image: tuple[int, int, int]
    order: float


@dataclass
class Component:
    """One connected component after cutting metal–ligand bonds."""

    id: str
    role: ComponentRole
    site_indices: frozenset[int]
    site_ids: tuple[str, ...]
    elements: tuple[str, ...]
    metal_indices: frozenset[int]
    nuclearity: int
    periodicity: int  # 0 finite, 1 rod, 2 layer, 3 framework-of-nodes
    period_axes: tuple[int, ...]
    cut_bonds: tuple[CutBond, ...] = ()


@dataclass
class PartitionResult:
    """Output of :func:`Partition`."""

    metal_components: list[Component]
    organic_components: list[Component]
    cut_bonds: list[CutBond]
    site_ids: list[str]
    elements: list[str]
    n_framework_components: int
    warnings: list[str] = field(default_factory=list)
    # Full undirected adjacency with images for downstream stages.
    adjacency: dict[int, list[tuple[int, tuple[int, int, int]]]] = field(default_factory=dict)
    # Kept (non-cut) edges for intra-component walks.
    kept_adjacency: dict[int, list[tuple[int, tuple[int, int, int]]]] = field(
        default_factory=dict
    )
    document: Any = None


def _require_networkx() -> Any:
    try:
        import networkx as nx
    except ImportError as exc:
        raise PartitionError(
            "networkx is required for partition; install with `pip install mofjson[crystal]`",
        ) from exc
    return nx


def _build_adjacency(
    bonds: list[PeriodicBond],
    id_to_i: dict[str, int],
) -> tuple[
    list[tuple[int, int, tuple[int, int, int], float]],
    dict[int, list[tuple[int, tuple[int, int, int]]]],
]:
    edges: list[tuple[int, int, tuple[int, int, int], float]] = []
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]] = defaultdict(list)
    for bond in bonds:
        i = id_to_i[bond.source]
        j = id_to_i[bond.target]
        img = bond.image_tuple()
        edges.append((i, j, img, float(bond.order)))
        adj[i].append((j, img))
        adj[j].append((i, (-img[0], -img[1], -img[2])))
    return edges, dict(adj)


def _is_node_bridging(
    idx: int,
    elements: list[str],
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]],
    metals: frozenset[str],
) -> bool:
    """True if a non-metal is bonded only to metals (+ optional H) — μ-O / μ-OH."""
    if elements[idx] in metals:
        return False
    metal_nbs = 0
    for j, _ in adj.get(idx, []):
        if elements[j] in metals:
            metal_nbs += 1
        elif elements[j] not in _NODE_BRIDGE_ALLOWED:
            return False
    return metal_nbs >= 1


def _carboxylate_bridge_pairs(
    elements: list[str],
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]],
    metals: frozenset[str],
) -> Counter[tuple[int, int]]:
    """Count syn-syn carboxylate bridges between metal pairs."""
    pairs: Counter[tuple[int, int]] = Counter()
    for c_idx, el in enumerate(elements):
        if el != "C":
            continue
        o_nbs = [j for j, _ in adj.get(c_idx, []) if elements[j] == "O"]
        if len(o_nbs) != 2:
            continue
        metal_touch: set[int] = set()
        for o in o_nbs:
            for j, _ in adj.get(o, []):
                if elements[j] in metals:
                    metal_touch.add(j)
        if len(metal_touch) == 2:
            a, b = sorted(metal_touch)
            pairs[(a, b)] += 1
    return pairs


def _component_periodicity(
    comp: set[int],
    kept_adj: dict[int, list[tuple[int, tuple[int, int, int]]]],
) -> tuple[int, tuple[int, ...]]:
    """Return (n_periodic_axes, axis_indices) for a component."""
    if not comp:
        return 0, ()
    start = next(iter(comp))
    pos: dict[int, tuple[int, int, int]] = {start: (0, 0, 0)}
    q: deque[int] = deque([start])
    axes: set[int] = set()
    while q:
        u = q.popleft()
        pu = pos[u]
        for v, img in kept_adj.get(u, []):
            if v not in comp:
                continue
            pv = (pu[0] + img[0], pu[1] + img[1], pu[2] + img[2])
            if v in pos:
                diff = (pv[0] - pos[v][0], pv[1] - pos[v][1], pv[2] - pos[v][2])
                for ax, d in enumerate(diff):
                    if d != 0:
                        axes.add(ax)
            else:
                pos[v] = pv
                q.append(v)
    return len(axes), tuple(sorted(axes))


def _classify_organic_role(
    site_indices: set[int],
    elements: list[str],
    n_cuts: int,
    *,
    keep_solvent: bool,
) -> ComponentRole:
    if n_cuts == 0:
        # Counterion heuristic: charged-looking small species without C.
        if not any(elements[i] == "C" for i in site_indices):
            return "counterion"
        return "guest"
    # Coordinated solvent / capping: small, few cuts, often aqua/N-donor only.
    n = len(site_indices)
    has_c = any(elements[i] == "C" for i in site_indices)
    if not keep_solvent and n_cuts <= 2 and n <= 8 and not has_c:
        return "ligand"
    if not keep_solvent and n_cuts == 1 and n <= 12:
        # Terminal DMF-like / monodentate solvent with carbon still ligand.
        # Linkers are typically ≥2 connections when bridging; allow exception for
        # imidazolate (2 N cuts) which has C. Monodentate capping with C stays ligand
        # only when single cut and small.
        return "ligand"
    return "linker"


def Partition(
    document: MOFJSONDocument,
    *,
    metals: frozenset[str] | None = None,
    keep_solvent: bool = False,
) -> PartitionResult:
    """Split the crystal bond graph into metal-node and organic components."""
    nx = _require_networkx()
    crystal = document.crystal
    if crystal is None:
        raise PartitionError("document has no crystal layer", field_path="crystal")
    if not crystal.bonds:
        raise PartitionError(
            "crystal layer has no bonds; run DeriveBonds / MOFFromCif(derive_bonds=True)",
            field_path="crystal.bonds",
        )

    metal_set = frozenset(metals) if metals is not None else METALS
    site_ids = [s.id for s in crystal.sites]
    elements = [s.element for s in crystal.sites]
    id_to_i = {sid: i for i, sid in enumerate(site_ids)}
    n = len(elements)
    edges, adj = _build_adjacency(crystal.bonds, id_to_i)

    metal_indices = frozenset(i for i in range(n) if elements[i] in metal_set)
    if not metal_indices:
        raise PartitionError("no metal sites found in structure", field_path="crystal.sites")

    bridging = frozenset(
        i for i in range(n) if _is_node_bridging(i, elements, adj, metal_set)
    )

    cut_bonds: list[CutBond] = []
    kept_edges: list[tuple[int, int, tuple[int, int, int], float]] = []
    for i, j, img, order in edges:
        mi = elements[i] in metal_set
        mj = elements[j] in metal_set
        if mi != mj:
            donor = j if mi else i
            metal_i = i if mi else j
            if donor not in bridging:
                # Orient image as metal → donor.
                if mi:
                    cut_img = img
                else:
                    cut_img = (-img[0], -img[1], -img[2])
                cut_bonds.append(
                    CutBond(
                        metal_index=metal_i,
                        donor_index=donor,
                        metal_id=site_ids[metal_i],
                        donor_id=site_ids[donor],
                        image=cut_img,
                        order=order,
                    )
                )
                continue
        kept_edges.append((i, j, img, order))

    kept_adj: dict[int, list[tuple[int, tuple[int, int, int]]]] = defaultdict(list)
    for i, j, img, _order in kept_edges:
        kept_adj[i].append((j, img))
        kept_adj[j].append((i, (-img[0], -img[1], -img[2])))

    # Metal membership via μ-bridges + paddlewheel carboxylate bridges (≥2).
    metal_graph = nx.Graph()
    metal_graph.add_nodes_from(metal_indices)
    for i, j, _img, _order in kept_edges:
        if i in metal_indices and j in metal_indices:
            metal_graph.add_edge(i, j)
        if i in metal_indices and j in bridging:
            for k, _ in adj.get(j, []):
                if k in metal_indices and k != i:
                    metal_graph.add_edge(i, k)
        if j in metal_indices and i in bridging:
            for k, _ in adj.get(i, []):
                if k in metal_indices and k != j:
                    metal_graph.add_edge(j, k)
    for (a, b), count in _carboxylate_bridge_pairs(elements, adj, metal_set).items():
        if count >= _CARBOXYLATE_BRIDGE_MIN:
            metal_graph.add_edge(a, b)

    metal_groups = [set(c) for c in nx.connected_components(metal_graph)]

    # Attach bridging atoms (and their H) to the metal group they bond to.
    metal_components_sites: list[set[int]] = []
    for group in metal_groups:
        sites = set(group)
        for b in bridging:
            if any(nb in group for nb, _ in adj.get(b, [])):
                sites.add(b)
                for nb, _ in adj.get(b, []):
                    if elements[nb] == "H":
                        sites.add(nb)
        metal_components_sites.append(sites)

    # Organic graph: all non-metal-component atoms with kept edges between them.
    node_atom_set = set().union(*metal_components_sites) if metal_components_sites else set()
    organic_nodes = [i for i in range(n) if i not in node_atom_set]
    org_graph = nx.Graph()
    org_graph.add_nodes_from(organic_nodes)
    for i, j, _img, _order in kept_edges:
        if i in node_atom_set or j in node_atom_set:
            continue
        org_graph.add_edge(i, j)
    organic_comps = [set(c) for c in nx.connected_components(org_graph) if c]

    # Map cut bonds onto organic components.
    cuts_by_org: dict[int, list[CutBond]] = defaultdict(list)
    donor_to_org: dict[int, int] = {}
    for oi, comp in enumerate(organic_comps):
        for idx in comp:
            donor_to_org[idx] = oi
    for cb in cut_bonds:
        oi = donor_to_org.get(cb.donor_index)
        if oi is not None:
            cuts_by_org[oi].append(cb)

    warnings: list[str] = []
    metal_out: list[Component] = []
    for mi, sites in enumerate(metal_components_sites):
        metals_here = frozenset(sites & metal_indices)
        period, axes = _component_periodicity(sites, dict(kept_adj))
        # Cuts touching this metal group.
        group_cuts = tuple(cb for cb in cut_bonds if cb.metal_index in metals_here)
        metal_out.append(
            Component(
                id=f"N{mi + 1}",
                role="node",
                site_indices=frozenset(sites),
                site_ids=tuple(site_ids[i] for i in sorted(sites)),
                elements=tuple(elements[i] for i in sorted(sites)),
                metal_indices=metals_here,
                nuclearity=len(metals_here),
                periodicity=period,
                period_axes=axes,
                cut_bonds=group_cuts,
            )
        )

    organic_out: list[Component] = []
    for oi, sites in enumerate(organic_comps):
        cuts = tuple(cuts_by_org.get(oi, ()))
        role = _classify_organic_role(sites, elements, len(cuts), keep_solvent=keep_solvent)
        organic_out.append(
            Component(
                id=f"O{oi + 1}",
                role=role,
                site_indices=frozenset(sites),
                site_ids=tuple(site_ids[i] for i in sorted(sites)),
                elements=tuple(elements[i] for i in sorted(sites)),
                metal_indices=frozenset(),
                nuclearity=0,
                periodicity=0,
                period_axes=(),
                cut_bonds=cuts,
            )
        )

    # Framework components for interpenetration: reconnect nodes+linkers via cut bonds.
    fw = nx.Graph()
    linkers = [c for c in organic_out if c.role == "linker"]
    for c in metal_out:
        fw.add_node(("n", c.id))
    for c in linkers:
        fw.add_node(("l", c.id))
    metal_comp_of = {}
    for c in metal_out:
        for idx in c.metal_indices:
            metal_comp_of[idx] = c.id
    org_comp_of = {idx: c.id for c in linkers for idx in c.site_indices}
    for cb in cut_bonds:
        nid = metal_comp_of.get(cb.metal_index)
        lid = org_comp_of.get(cb.donor_index)
        if nid and lid:
            fw.add_edge(("n", nid), ("l", lid))
    n_fw = nx.number_connected_components(fw) if fw.number_of_nodes() else 0
    if n_fw > 1:
        warnings.append(f"interpenetrated_or_disconnected_frameworks:{n_fw}")

    return PartitionResult(
        metal_components=metal_out,
        organic_components=organic_out,
        cut_bonds=cut_bonds,
        site_ids=site_ids,
        elements=elements,
        n_framework_components=n_fw,
        warnings=warnings,
        adjacency=adj,
        kept_adjacency=dict(kept_adj),
        document=document,
    )


__all__ = [
    "Component",
    "CutBond",
    "Partition",
    "PartitionResult",
]
