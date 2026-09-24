"""
topology.py -- coordination sequences over the periodic net, matched
against a table of verified RCSR nets.

Narrower than the general problem: assigning an RCSR symbol properly means
canonical-form isomorphism (Systre, ToposPro), which we don't reimplement.

  1. _build_topological_net -- reduces (building_blocks, net_edges) to an
     honest net under Glasby et al. (Chem. Mater. 2024, Sec. 2): drop
     loose ends and chelation, collapse parallel bridges, edges
     undirected. Skipping these inflates degrees with noise.
  2. ComputeCoordinationSequences -- BFS shells, accumulating net_edges'
     shifts under Systre's voltage-graph convention.
  3. IdentifyTopology -- exact signature match or `net` is None, with the
     computed signatures, `excluded_building_blocks` (and why), the
     deconstruction convention used, and a confidence field, since CS
     matching is a fingerprint, not proof of isomorphism.

REFERENCE_NETS stays small until rcsr_import.py is run over RCSR's bulk
.cgd; each entry's `source` says how it got there. Computing the invariant
ourselves makes an unresolved net a table-coverage problem rather than a
dependency on MOFid's Systre lookup.
"""
from __future__ import annotations

from typing import Any

Shift = tuple[int, int, int]
State = tuple[str, Shift]


def _add(a: Shift, b: Shift) -> Shift:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Shift, b: Shift) -> Shift:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _neg(a: Shift) -> Shift:
    return (-a[0], -a[1], -a[2])


def _incidence(
    building_blocks: list[dict[str, Any]], net_edges: list[dict[str, Any]]
) -> dict[str, list[tuple[str, Shift]]]:
    """block_id -> [(neighbor_id, shift), ...], one entry per net_edge
    touching that block (so a block with two connection points bonded to
    the same neighbor id, e.g. a chelating or double-bridging ligand,
    correctly gets two separate entries here, not one) -- `shift` is
    always the neighbor's cell offset relative to THIS block's own cell."""
    inc: dict[str, list[tuple[str, Shift]]] = {b["id"]: [] for b in building_blocks}
    for e in net_edges:
        s = tuple(e.get("shift", [0, 0, 0]))
        inc.setdefault(e["from_block"], []).append((e["to_block"], s))
        inc.setdefault(e["to_block"], []).append((e["from_block"], _neg(s)))
    return inc


def _build_topological_net(
    building_blocks: list[dict[str, Any]], net_edges: list[dict[str, Any]]
) -> tuple[set[str], list[tuple[str, str, Shift]], dict[str, str]]:
    """
    Returns (vertices, edges, excluded):
      vertices -- surviving vertex ids (each has final degree >= 2, or the
        set is empty for a structure with no periodic connectivity at all).
      edges -- deduplicated (a, b, shift) triples, shift meaning b sits at
        `shift` relative to a's own cell; a self-loop has a == b.
      excluded -- {block_id: reason} for every guest_*, every non-vertex
        block that contributed no net edge, and every vertex removed by
        step 5, so nothing is silently missing from the accounting.
    """
    guests = {b["id"] for b in building_blocks if b["id"].startswith("guest_")}
    inc = _incidence(building_blocks, net_edges)

    excluded: dict[str, str] = {bid: "guest_species" for bid in guests}

    passthrough = {
        bid for bid, conns in inc.items()
        if bid not in guests and not bid.startswith("node_") and len(conns) == 2
    }
    for bid, conns in inc.items():
        if bid in guests or bid.startswith("node_") or len(conns) > 1:
            continue
        excluded[bid] = "monodentate_terminal" if len(conns) == 1 else "unbonded"

    vertex_candidates = set(inc) - guests - passthrough - set(excluded)

    raw_edges: list[tuple[str, str, Shift]] = []
    for p in passthrough:
        (n1, s1), (n2, s2) = inc[p]
        if n1 in passthrough or n2 in passthrough:
            excluded[p] = "chained_passthrough_not_supported"
            continue
        combined = _sub(s2, s1)
        if n1 == n2 and combined == (0, 0, 0):
            excluded[p] = "pure_chelation_no_net_displacement"
            continue
        raw_edges.append((n1, n2, combined))

    # Direct vertex-to-vertex net_edges -- no ditopic linker in between at
    # all (e.g. a tritopic/polytopic linker like BTC, or a node bonded
    # straight to another node). Contraction above only produces an edge
    # when there IS a passthrough block to contract through; without this
    # loop, a structure with no ditopic linkers at all (Cu-BTC: every
    # linker is tritopic) would end up with zero edges and every vertex
    # wrongly pruned as isolated.
    for e in net_edges:
        a, b = e["from_block"], e["to_block"]
        if a in vertex_candidates and b in vertex_candidates:
            raw_edges.append((a, b, tuple(e.get("shift", [0, 0, 0]))))

    def _canonical(a: str, b: str, shift: Shift) -> tuple[str, str, Shift]:
        opt_a = (a, b, shift)
        opt_b = (b, a, _neg(shift))
        return min(opt_a, opt_b)

    deduped_edges: set[tuple[str, str, Shift]] = {
        _canonical(a, b, s) for a, b, s in raw_edges
    }

    vertices = set(vertex_candidates)
    edges = set(deduped_edges)
    changed = True
    while changed:
        changed = False
        degree: dict[str, int] = {v: 0 for v in vertices}
        for a, b, s in edges:
            if a not in vertices or b not in vertices:
                continue
            if a == b:
                degree[a] += 2
            else:
                degree[a] += 1
                degree[b] += 1
        to_remove = {v for v in vertices if degree.get(v, 0) < 2}
        if to_remove:
            for v in to_remove:
                excluded.setdefault(v, f"isolated_after_pruning (final_degree={degree.get(v, 0)})")
            vertices -= to_remove
            edges = {(a, b, s) for a, b, s in edges if a in vertices and b in vertices}
            changed = True

    return vertices, sorted(edges), excluded


def BuildTopologicalNet(
    building_blocks: list[dict[str, Any]], net_edges: list[dict[str, Any]]
) -> tuple[set[str], list[tuple[str, str, Shift]], dict[str, str]]:
    """Public entry point for `_build_topological_net` (see that function's
    own docstring for the full algorithm/rules). Exported specifically so
    assembly.py's v/e/genus counting and topology.py's own coordination-
    sequence matching share exactly ONE implementation of "what counts as a
    vertex, what counts as an edge" -- assembly.py used to reimplement a
    weaker, separate version of these same rules (no chelation/double-bridge
    handling, no iterative pruning), which could silently disagree with
    IdentifyTopology's own vertex count for the same structure. There is now
    exactly one place these four net-validity rules live."""
    return _build_topological_net(building_blocks, net_edges)


def _adjacency(edges: list[tuple[str, str, Shift]]) -> dict[str, list[tuple[str, Shift]]]:
    adj: dict[str, list[tuple[str, Shift]]] = {}
    for a, b, s in edges:
        adj.setdefault(a, []).append((b, s))
        adj.setdefault(b, []).append((a, _neg(s)))
    return adj


def _neighbor_states(state: State, adj: dict[str, list[tuple[str, Shift]]]) -> list[State]:
    block_id, cell = state
    return [(nb, _add(cell, s)) for nb, s in adj.get(block_id, [])]


def _shells(start: str, neighbor_fn, max_shell: int) -> list[int]:
    start_state: State = (start, (0, 0, 0))
    visited: set[State] = {start_state}
    frontier: set[State] = {start_state}
    cs: list[int] = []
    for _ in range(max_shell):
        nxt: set[State] = set()
        for state in frontier:
            for nb in neighbor_fn(state):
                if nb not in visited:
                    visited.add(nb)
                    nxt.add(nb)
        cs.append(len(nxt))
        frontier = nxt
        if not frontier:
            break
    while len(cs) < max_shell:
        cs.append(0)
    return cs


def ComputeCoordinationSequences(
    building_blocks: list[dict[str, Any]],
    net_edges: list[dict[str, Any]],
    *,
    max_shell: int = 10,
) -> dict[str, list[int]]:
    """{vertex_block_id: [CS1, ..., CS_max_shell]} for OUR OWN building-block
    graph, after reducing it to an honest topological net via
    `_build_topological_net` (ditopic linkers contracted through, loose
    ends and pure chelation discarded, double-bridges deduplicated -- see
    that function's docstring). This is the function BuildDocument/
    IdentifyTopology actually use. A block excluded by
    `_build_topological_net` (see its `excluded` return value) simply has
    no entry here -- it was never part of the net, not silently zeroed."""
    vertices, edges, _excluded = _build_topological_net(building_blocks, net_edges)
    adj = _adjacency(edges)
    return {v: _shells(v, lambda s: _neighbor_states(s, adj), max_shell) for v in vertices}


def ComputeCoordinationSequencesFromGraph(
    vertex_ids: set[str],
    edges: list[dict[str, Any]],
    *,
    max_shell: int = 6,
) -> dict[str, list[int]]:
    """Same BFS engine as ComputeCoordinationSequences, but with NO
    contraction/pruning/dedup step -- every id in `vertex_ids` is treated
    as a real, countable vertex and every edge is used exactly as given.
    Use this for an ALREADY-REDUCED abstract net graph (e.g. one parsed
    straight out of an RCSR .cgd file by rcsr_import.py), which is already
    a valid simple net by construction -- RCSR's own quotient graphs don't
    need loose-end pruning or double-bridge dedup applied a second time."""
    edges_from: dict[str, list[dict[str, Any]]] = {}
    edges_to: dict[str, list[dict[str, Any]]] = {}
    for e in edges:
        edges_from.setdefault(e["from_block"], []).append(e)
        edges_to.setdefault(e["to_block"], []).append(e)

    def raw_neighbor_states(state: State) -> list[State]:
        block_id, cell = state
        out: list[State] = []
        for e in edges_from.get(block_id, []):
            s = tuple(e.get("shift", [0, 0, 0]))
            out.append((e["to_block"], _add(cell, s)))
        for e in edges_to.get(block_id, []):
            s = tuple(e.get("shift", [0, 0, 0]))
            out.append((e["from_block"], _sub(cell, s)))
        return out

    return {v: _shells(v, raw_neighbor_states, max_shell) for v in vertex_ids}


# Deliberately small -- see module docstring for why (and how to grow this:
# either one net at a time, by hand-transcribing a real RCSR net page the
# way `tbo` was added below, or in bulk via rcsr_import.py once you have
# RCSR's own .cgd export from an environment that can reach rcsr.net/
# github.com). `vertex_kinds` is a LIST because a net can be multinodal
# (tbo has two: one degree-3, one degree-4) -- IdentifyTopology's
# _signatures_match already handles either case, no special-casing needed
# to add a multinodal net here.
#
# Every entry carries a `source` string so a reader can tell a textbook
# closed-form result apart from a hand-transcribed RCSR page apart from a
# (future) bulk verified import -- see IdentifyTopology's
# `matched_reference_source` output field, which surfaces this directly
# without needing to dig into `matched_reference`.
REFERENCE_NETS: dict[str, dict[str, Any]] = {
    "pcu": {
        "description": "simple cubic, 6-connected uninodal net (alpha-Po)",
        "source": "textbook closed-form (N_k = 4k^2 + 2), independently reproducible by hand",
        "vertex_kinds": [{"degree": 6, "coordination_sequence": [6, 18, 38, 66, 102, 146, 198, 258, 326, 402]}],
    },
    "dia": {
        "description": "diamond net, 4-connected uninodal",
        "source": "textbook closed-form, independently reproducible by hand",
        "vertex_kinds": [{"degree": 4, "coordination_sequence": [4, 12, 24, 42, 64, 92, 124, 162, 204, 252]}],
    },
    "sql": {
        "description": "square lattice, 4-connected uninodal, 2-periodic",
        "source": "textbook closed-form (N_k = 4k), independently reproducible by hand",
        "vertex_kinds": [{"degree": 4, "coordination_sequence": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]}],
    },
    "tbo": {
        "description": (
            "twisted boracite net (\"3S net\"); the augmented net of HKUST-1 / "
            "Cu-BTC's paddlewheel node + tritopic (BTC) linker deconstruction"
        ),
        "source": (
            "hand-transcribed from RCSR's own net page, https://rcsr.net/nets/tbo "
            "(space group Fm-3m, a=b=c=4.8990, cell volume 117.5770, genus 11), "
            "cross-checked against this codebase's independently-computed Cu-BTC "
            "signature -- exact match on every shell in common. NOT yet verified "
            "by a bulk Systre .cgd import or by running Systre's canonical-form "
            "algorithm; see IdentifyTopology's confidence field."
        ),
        "vertex_kinds": [
            {
                "degree": 3,
                "coordination_sequence": [3, 9, 15, 33, 45, 82, 90, 153, 150, 241],
                "vertex_symbol": "6.6.6",
                "wyckoff": "32 f",
            },
            {
                "degree": 4,
                "coordination_sequence": [4, 8, 20, 30, 60, 68, 120, 126, 200, 180],
                "vertex_symbol": "6(2).6(2).8(2).8(2).12(2).12(2)",
                "wyckoff": "24 d",
            },
        ],
    },
}

# The deconstruction convention BuildDocument's CollapseToBuildingBlocks
# actually implements, named per Glasby, Cordiner, Cole, Moghadam (Chem.
# Mater. 2024, Sec. 4)'s own taxonomy of MOF deconstruction algorithms.
# Reported verbatim in every IdentifyTopology result -- see that paper's
# own example (CSD JOZWIG matching both **xxv**, all-node, and **ftw**,
# single-node) for why silently implying there is one deconstruction-
# independent net would be dishonest.
DECONSTRUCTION_METHOD: dict[str, str] = {
    "name": "metal_ligand_boundary",
    "closest_named_equivalent": "standard representation / metal-oxo",
    "description": (
        "CollapseToBuildingBlocks (conversion.py) cuts at the metal-linker "
        "bond: every atom with role='node' collapses into one vertex "
        "candidate per contiguous same-role cluster, and every atom with "
        "role='linker' is kept as part of its own building block, "
        "contracted through as a single net edge when ditopic. This is "
        "closest to what Glasby et al. (2024, Sec. 4.2.1/4.2.2) call "
        "'standard representation' or 'metal-oxo' deconstruction -- NOT "
        "'all-node' (which would additionally split a large organic "
        "linker at its own internal branch points into extra vertices) or "
        "'single-node' (which would instead merge each metal cluster and "
        "its directly-bonded carboxylate donor atoms into one combined "
        "vertex). A structure with a large branched linker may therefore "
        "get a different, differently-named net from this codebase than "
        "from a tool using one of those other conventions -- both can be "
        "correct at once; they are answers to a slightly different "
        "question about the same structure."
    ),
}


def _vertex_signature(
    vertex_id: str, cs_by_vertex: dict[str, list[int]], degree_by_vertex: dict[str, int]
) -> tuple[int, tuple[int, ...]]:
    return (degree_by_vertex[vertex_id], tuple(cs_by_vertex[vertex_id]))


def _signatures_match(
    distinct_signatures: set[tuple[int, tuple[int, ...]]], vertex_kinds: list[dict[str, Any]]
) -> bool:
    """True iff there is a 1:1 pairing between the structure's distinct
    vertex signatures and a reference net's vertex_kinds, matching degree
    exactly and coordination_sequence exactly over however many shells the
    structure's own signature has (a reference entry may carry more shells
    than we computed; only the overlapping prefix is compared). Handles
    both uninodal (one signature, one vertex kind) and multinodal nets."""
    if len(distinct_signatures) != len(vertex_kinds):
        return False
    remaining = list(vertex_kinds)
    for degree, cs in distinct_signatures:
        match_idx = None
        for i, kind in enumerate(remaining):
            if kind["degree"] == degree and tuple(kind["coordination_sequence"][: len(cs)]) == cs:
                match_idx = i
                break
        if match_idx is None:
            return False
        remaining.pop(match_idx)
    return True


def IdentifyTopology(
    building_blocks: list[dict[str, Any]],
    net_edges: list[dict[str, Any]],
    *,
    max_shell: int = 10,
    reference_nets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Returns:
        {
          "net": <RCSR symbol or None>,
          "vertex_signatures": {block_id: {"degree": int, "coordination_sequence": [...]}, ...},
          "matched_reference": <reference_nets entry or None>,
          "matched_reference_source": <that entry's "source" string, or None>,
          "excluded_building_blocks": {block_id: reason, ...},
          "deconstruction_method": DECONSTRUCTION_METHOD,
          "determination_method": "coordination_sequence_heuristic",
          "confidence": <see below>,
        }

    `net` is set only when the structure's full set of distinct vertex
    signatures matches one reference net's `vertex_kinds` exactly, in
    either direction -- uninodal (pcu/dia/sql-style, one signature) or
    multinodal (tbo-style, several signatures), see _signatures_match.

    `vertex_signatures`/`net` are computed over the topological net
    from `_build_topological_net`

    `confidence` is deliberately never higher than "heuristic": a
    coordination-sequence match is necessary but not sufficient for true
    graph isomorphism (two different nets can coincidentally share several
    shells of CS) -- the actual proof is Systre's canonical-form algorithm,
    which this module does not reimplement (see module docstring). Values:
    "heuristic_match_unverified_by_isomorphism" when `net` is set,
    "no_match_in_reference_table" when it is None because the structure's
    signature isn't one of the (deliberately few) entries checked, and
    "no_periodic_connectivity" when the structure has no surviving
    vertices at all after pruning (nothing to match against any net).

    Pass `reference_nets` (the dict produced by
    rcsr_import.build_reference_nets, or loaded from its JSON output via
    topology.LoadReferenceNetsFromJSON) to match against a much larger,
    RCSR-derived table instead of the tiny built-in REFERENCE_NETS -- this
    is the versatile, many-MOFs path; see rcsr_import.py."""
    reference_nets = REFERENCE_NETS if reference_nets is None else reference_nets

    vertices, edges, excluded = _build_topological_net(building_blocks, net_edges)
    adj = _adjacency(edges)
    degree_by_vertex = {v: len(adj.get(v, [])) for v in vertices}

    cs_by_vertex = {v: _shells(v, lambda s: _neighbor_states(s, adj), max_shell) for v in vertices}
    signatures = {v: _vertex_signature(v, cs_by_vertex, degree_by_vertex) for v in vertices}
    distinct_signatures = set(signatures.values())

    matched_symbol = None
    matched_ref = None
    for symbol, ref in reference_nets.items():
        if _signatures_match(distinct_signatures, ref["vertex_kinds"]):
            matched_symbol = symbol
            matched_ref = ref
            break

    if matched_symbol is not None:
        confidence = "heuristic_match_unverified_by_isomorphism"
    elif not vertices:
        confidence = "no_periodic_connectivity"
    else:
        confidence = "no_match_in_reference_table"

    return {
        "net": matched_symbol,
        "vertex_signatures": {
            v: {"degree": sig[0], "coordination_sequence": list(sig[1])}
            for v, sig in signatures.items()
        },
        "matched_reference": matched_ref,
        "matched_reference_source": matched_ref["source"] if matched_ref else None,
        "excluded_building_blocks": excluded,
        "deconstruction_method": DECONSTRUCTION_METHOD,
        "determination_method": "coordination_sequence_heuristic",
        "confidence": confidence,
    }


def LoadReferenceNetsFromJSON(path: str) -> dict[str, Any]:
    """Load a big reference-net table previously written by
    rcsr_import.py's `main()` (a JSON dict shaped exactly like
    REFERENCE_NETS above, just much larger). Pass the result as
    IdentifyTopology's `reference_nets=` argument."""
    import json

    with open(path) as f:
        return json.load(f)


__all__ = [
    "BuildTopologicalNet",
    "ComputeCoordinationSequences",
    "ComputeCoordinationSequencesFromGraph",
    "DECONSTRUCTION_METHOD",
    "IdentifyTopology",
    "LoadReferenceNetsFromJSON",
    "REFERENCE_NETS",
]
