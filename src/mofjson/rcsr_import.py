"""
rcsr_import.py -- turn RCSR's bulk Systre export (RCSRnets20190601.cgd)
into topology.REFERENCE_NETS entries, reusing topology.py's
coordination-sequence engine.

The export's CRYSTAL dialect gives one symmetry-unique representative per
vertex/edge orbit, not the expanded graph:

    CRYSTAL
      NAME tbo
      GROUP Fm-3m
      CELL 4.89898 4.89898 4.89898 90.0000 90.0000 90.0000
      NODE 1 3  0.16667 0.16667 0.16667
      NODE 2 4  0.25000 0.25000 0.00000
      EDGE  0.25000 0.25000 0.00000   0.16667 0.16667 0.16667
    END

NODE is (id, coordination number, coords); EDGE is its two endpoints
directly, and an endpoint can fall outside [0,1) -- pcu's single edge runs
(0,0,0) to (0,0,1), and that integer part is the periodic shift. Parsing
alone is therefore not enough: we apply the space group's full symmetry
operations (pymatgen SpaceGroup) to every representative NODE and EDGE to
generate the rest of the cell, as Systre does internally. pcu's one listed
edge is 1 of the 3 its 6-connected vertex needs; Pm-3m generates the other
two.

CELL field count separates periodicities: 2727 blocks with 6 fields are
3-periodic nets and are fully processed; 200 with 3 fields are 2-periodic
layers, parsed but not expanded (pymatgen has no plane-group table). See
`parse_cgd_crystal_blocks`'s `periodicity`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from src.mofjson.topology import ComputeCoordinationSequencesFromGraph

# Space-group symbol aliases: a handful of GROUP strings in the real file
# use an older Hermann-Mauguin spelling pymatgen's own symbol table doesn't
# recognize directly. Each mapped-to symbol was verified to give the SAME
# space group (matching symmetry-operation-count order) as crystallographic
# tables document for that group number.
_GROUP_ALIASES: dict[str, str] = {
    "Cmca": "Cmce",
}


def parse_cgd_crystal_blocks(text: str) -> list[dict[str, Any]]:
    """Parse every CRYSTAL block into
    {name, group, periodicity (2 or 3), cell, nodes: [(id, cn, coords)],
     edges: [(coords1, coords2)]} -- coords are floats, length matching
    periodicity. Comment lines ('#...') are stripped; blank lines and any
    other CRYSTAL sub-keyword this format doesn't need (this file has none
    beyond NAME/GROUP/CELL/NODE/EDGE) are simply ignored."""
    blocks: list[dict[str, Any]] = []
    block: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "CRYSTAL":
            block = {"name": None, "group": None, "cell": None, "nodes": [], "edges": []}
            continue
        if block is None:
            continue
        if line.startswith("NAME"):
            parts = line.split(None, 1)
            block["name"] = parts[1].strip() if len(parts) > 1 else None
        elif line.startswith("GROUP"):
            parts = line.split(None, 1)
            block["group"] = parts[1].strip() if len(parts) > 1 else None
        elif line.startswith("CELL"):
            block["cell"] = [float(x) for x in line.split()[1:]]
        elif line.startswith("NODE"):
            parts = line.split()
            node_id = parts[1]
            cn = int(parts[2])
            coords = tuple(float(x) for x in parts[3:])
            block["nodes"].append((node_id, cn, coords))
        elif line.startswith("EDGE"):
            parts = line.split()[1:]
            dim = len(parts) // 2
            c1 = tuple(float(x) for x in parts[:dim])
            c2 = tuple(float(x) for x in parts[dim:])
            block["edges"].append((c1, c2))
        elif line == "END":
            if block.get("name"):
                dim = len(block["cell"]) if block["cell"] else (3 if block["nodes"] and len(block["nodes"][0][2]) == 3 else 2)
                block["periodicity"] = 3 if dim == 6 else 2
                blocks.append(block)
            block = None
    return blocks


import functools


@functools.lru_cache(maxsize=None)
def _symmetry_ops(group_symbol: str):
    """pymatgen's own (tested, library-provided) symmetry operations for a
    3D space group -- NOT hand-derived. Raises if the symbol (after the
    small, individually-verified alias table above) still isn't
    recognized, so a caller can skip/report that net rather than silently
    treating it as having no symmetry. Cached: the bulk file re-uses the
    same ~170 distinct space-group symbols across 2727 nets, and
    constructing a SpaceGroup from its symbol is the expensive part, not
    applying its already-built operations -- without this cache the same
    space group gets rebuilt from scratch on every single net that uses
    it (verified directly: this was the dominant cost in an initial
    unbounded run of the full file)."""
    from pymatgen.symmetry.groups import SpaceGroup

    resolved = _GROUP_ALIASES.get(group_symbol, group_symbol)
    sg = SpaceGroup(resolved)
    return tuple(sg.symmetry_ops)


def _wrap(coord: tuple[float, ...], tol: float = 1e-4) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Split a (possibly out-of-[0,1)) fractional coordinate into a
    (wrapped_position, integer_shift) pair, snapping anything within `tol`
    of an integer boundary to exactly 0.0 so floating-point noise from the
    symmetry-operation matrix multiply doesn't produce spurious near-1.0/
    near-0.0 mismatches downstream."""
    wrapped = []
    shift = []
    for c in coord:
        s = math.floor(c + 1e-9)
        w = c - s
        if abs(w) < tol:
            w = 0.0
        elif abs(w - 1.0) < tol:
            w = 0.0
            s += 1
        wrapped.append(round(w, 4))
        shift.append(int(s))
    return tuple(wrapped), tuple(shift)


def expand_3d_net(
    nodes: list[tuple[str, int, tuple[float, ...]]],
    edges: list[tuple[tuple[float, ...], tuple[float, ...]]],
    group_symbol: str,
    *,
    tol: float = 1e-4,
) -> tuple[set[str], list[tuple[str, str, tuple[int, int, int]]], dict[str, Any]]:
    """Apply `group_symbol`'s full symmetry-operation set to every
    representative NODE and EDGE, generating the actual periodic graph
    (every symmetry-equivalent vertex and edge in the cell) -- this is the
    step RCSR's raw file leaves to the reader (Systre itself does this
    internally). Returns (vertex_ids, edges[(id1, id2, shift)], diagnostics).

    diagnostics carries {"unmatched_edge_endpoints": int,
    "degree_mismatches": {vertex_id: (stated_cn, actual_degree)}} so a
    caller can tell a clean expansion from one that silently dropped data,
    instead of assuming success."""
    ops = _symmetry_ops(group_symbol)

    # Step 1: every symmetry-equivalent copy of every NODE, deduplicated
    # by wrapped position (special-position vertices legitimately produce
    # FEWER than len(ops) distinct copies -- deduplication is what
    # correctly recovers that multiplicity without any separate
    # Wyckoff-position bookkeeping).
    position_to_id: dict[tuple[float, ...], str] = {}
    cn_by_orbit: dict[str, int] = {}
    for node_id, cn, coords in nodes:
        seen: dict[tuple[float, ...], None] = {}
        for op in ops:
            p = op.operate(list(coords))
            wrapped, _shift = _wrap(tuple(p), tol=tol)
            seen.setdefault(wrapped, None)
        for k, wrapped in enumerate(sorted(seen)):
            vid = f"{node_id}_{k}"
            position_to_id[wrapped] = vid
            cn_by_orbit[vid] = cn

    def find_vertex(coord: tuple[float, ...]) -> tuple[str | None, tuple[int, int, int]]:
        wrapped, shift = _wrap(coord, tol=tol)
        vid = position_to_id.get(wrapped)
        if vid is not None:
            return vid, shift
        # Tolerance fallback: try the 3^3 neighboring roundings in case the
        # matrix multiply landed just across a rounding boundary the plain
        # dict lookup missed.
        step = 10 ** -4
        for dx in (-step, 0, step):
            for dy in (-step, 0, step):
                for dz in (-step, 0, step):
                    alt = (round(wrapped[0] + dx, 4), round(wrapped[1] + dy, 4), round(wrapped[2] + dz, 4))
                    if alt in position_to_id:
                        return position_to_id[alt], shift
        return None, shift

    raw_edges: list[tuple[str, str, tuple[int, int, int]]] = []
    unmatched = 0
    for c1, c2 in edges:
        for op in ops:
            p1 = op.operate(list(c1))
            p2 = op.operate(list(c2))
            v1, s1 = find_vertex(tuple(p1))
            v2, s2 = find_vertex(tuple(p2))
            if v1 is None or v2 is None:
                unmatched += 1
                continue
            shift = (s2[0] - s1[0], s2[1] - s1[1], s2[2] - s1[2])
            raw_edges.append((v1, v2, shift))

    def _canonical(a: str, b: str, s: tuple[int, int, int]):
        opt_a = (a, b, s)
        opt_b = (b, a, (-s[0], -s[1], -s[2]))
        return min(opt_a, opt_b)

    deduped_edges = sorted({_canonical(a, b, s) for a, b, s in raw_edges})

    vertex_ids = set(position_to_id.values())
    degree: dict[str, int] = {v: 0 for v in vertex_ids}
    for a, b, s in deduped_edges:
        if a == b:
            degree[a] += 2
        else:
            degree[a] += 1
            degree[b] += 1
    degree_mismatches = {
        vid: (cn_by_orbit[vid], degree.get(vid, 0))
        for vid in vertex_ids
        if degree.get(vid, 0) != cn_by_orbit[vid]
    }

    diagnostics = {"unmatched_edge_endpoints": unmatched, "degree_mismatches": degree_mismatches}
    return vertex_ids, deduped_edges, diagnostics


def build_reference_nets(
    blocks: list[dict[str, Any]], *, max_shell: int = 10, only_clean: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expand every 3-periodic CRYSTAL block into a real periodic graph
    (via expand_3d_net) and compute its coordination-sequence signature
    with topology.py's own BFS engine (ComputeCoordinationSequencesFromGraph
    -- no second implementation of that algorithm). Returns
    (reference_nets, report):

      reference_nets -- {symbol: {"description", "source", "vertex_kinds"}},
      shaped exactly like topology.REFERENCE_NETS.

      report -- {"total_3d": int, "imported": int, "skipped": {symbol: reason}}
      so nothing is silently dropped: a net whose GROUP symbol pymatgen
      can't parse, or whose expansion produced an unmatched edge endpoint
      or a degree mismatch against its own stated coordination number, is
      recorded with a reason rather than either crashing the whole import
      or being included with unverified data.

    `only_clean=True` (default) excludes any net whose expansion produced
    unmatched edges or a degree mismatch from `reference_nets` (they still
    appear in `report["skipped"]`) -- matching this codebase's existing
    rule that IdentifyTopology's REFERENCE_NETS entries are only ones
    that have actually been checked, not merely attempted.
    """
    reference: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    total_3d = 0

    for block in blocks:
        if block["periodicity"] != 3:
            continue
        total_3d += 1
        symbol = block["name"]
        try:
            vertex_ids, edges, diag = expand_3d_net(block["nodes"], block["edges"], block["group"])
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            skipped[symbol] = f"symmetry expansion failed: {exc}"
            continue

        if diag["unmatched_edge_endpoints"]:
            skipped[symbol] = f"{diag['unmatched_edge_endpoints']} edge endpoint(s) did not match any generated vertex"
            if only_clean:
                continue
        if diag["degree_mismatches"]:
            skipped[symbol] = f"degree mismatch vs stated cn: {diag['degree_mismatches']}"
            if only_clean:
                continue
        if not vertex_ids:
            skipped[symbol] = "no vertices generated"
            continue

        edge_dicts = [{"from_block": a, "to_block": b, "shift": list(s)} for a, b, s in edges]
        cs_by_vertex = ComputeCoordinationSequencesFromGraph(vertex_ids, edge_dicts, max_shell=max_shell)
        degree_by_vertex: dict[str, int] = {v: 0 for v in vertex_ids}
        for e in edge_dicts:
            degree_by_vertex[e["from_block"]] += 1
            degree_by_vertex[e["to_block"]] += 1
        signatures = {(degree_by_vertex[v], tuple(cs_by_vertex[v])) for v in vertex_ids}

        reference[symbol] = {
            "description": f"RCSR net {symbol} (space group {block['group']}), bulk-imported",
            "source": (
                "bulk-imported from RCSR's own Systre .cgd export "
                "(RCSRnets20190601.cgd) via rcsr_import.py: NODE/EDGE "
                "representatives expanded into the full periodic graph "
                "using pymatgen.symmetry.groups.SpaceGroup's real symmetry "
                "operations for this net's own space group, then run "
                "through topology.py's own coordination-sequence BFS -- "
                "not hand-transcribed, and cross-checked for a degree-vs-"
                "stated-coordination-number match before being included."
            ),
            "vertex_kinds": [
                {"degree": degree, "coordination_sequence": list(cs)} for degree, cs in sorted(signatures)
            ],
        }

    report = {"total_3d": total_3d, "imported": len(reference), "skipped": skipped}
    return reference, report


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m src.mofjson.rcsr_import path/to/rcsr.cgd")
        sys.exit(1)

    cgd_path = Path(sys.argv[1])
    blocks = parse_cgd_crystal_blocks(cgd_path.read_text())
    n3d = sum(1 for b in blocks if b["periodicity"] == 3)
    n2d = sum(1 for b in blocks if b["periodicity"] == 2)
    print(f"parsed {len(blocks)} CRYSTAL block(s): {n3d} 3-periodic, {n2d} 2-periodic (layers, not yet expanded)")
    if not blocks:
        print(
            "0 blocks parsed -- the file's actual layout doesn't match the "
            "CRYSTAL/NAME/GROUP/CELL/NODE/EDGE shape this parser assumes "
            "(see this module's docstring). Paste a few real lines back "
            "and the parser can be adjusted to match."
        )
        sys.exit(1)

    reference, report = build_reference_nets(blocks)
    out_path = cgd_path.with_suffix(".reference_nets.json")
    out_path.write_text(json.dumps(reference, indent=2, sort_keys=True))
    print(f"wrote {out_path} ({report['imported']} / {report['total_3d']} 3-periodic nets imported cleanly)")
    if report["skipped"]:
        report_path = cgd_path.with_suffix(".reference_nets.skipped.json")
        report_path.write_text(json.dumps(report["skipped"], indent=2, sort_keys=True))
        print(f"{len(report['skipped'])} net(s) skipped (with reasons) -- see {report_path}")
    for symbol in ("pcu", "dia", "tbo", "nbo", "srs"):
        if symbol in reference:
            print(f"  spot-check {symbol}: {reference[symbol]['vertex_kinds']}")


if __name__ == "__main__":
    main()


__all__ = [
    "build_reference_nets",
    "expand_3d_net",
    "parse_cgd_crystal_blocks",
]
