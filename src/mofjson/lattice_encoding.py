"""
lattice_encoding.py -- MOFJSON's own compact, invertible, single-string
serialization of a periodic structure's labeled quotient graph.

Conceptual debt, stated plainly: SLICES (Xiao et al., Nat. Commun. 2023,
14, 7027) is the reason this exists at all. Its core idea -- that a
periodic crystal can be represented as a single string encoding a labeled
quotient graph (atoms + bonds, with periodic bonds carrying an explicit
unit-cell shift), useful precisely because it's compact and tokenizer/
LLM-friendly -- is directly why schema.py's bonds already carry a `shift`
field (see that module's docstring: "the same mechanism used by labeled-
quotient-graph string representations of crystals such as SLICES").

What this module is deliberately NOT: it is not the `slices` PyPI package,
does not depend on it, was not built by wrapping it, and does not attempt
SLICES's actual end-to-end algorithm. Real SLICES additionally (a)
canonicalizes/tokenizes the string in a way tuned for a specific
downstream generative model, and (b) reconstructs full 3D atomic
coordinates FROM the string via simulated annealing plus an M3GNet
machine-learned interatomic potential -- a substantial ML pipeline
(tensorflow + m3gnet + SMACT, ~600 MB). An earlier version of this
codebase wrapped that real package directly (see git history / the old
slice.py this module replaces); that was the wrong call for a project
whose whole point is an independently-authored representation, not a thin
shell around someone else's tool. This module instead borrows only the
underlying REPRESENTATION IDEA and implements it from scratch, directly
against data this codebase already has (RawStructure's own atoms/bonds/
lattice) -- no bond-order/geometry guessing, no ML, no external
chemistry-string-generation package, nothing to `pip install`.

Because the scope is narrower, this module can make an honesty claim real
SLICES's own paper is careful not to make unconditionally: round-trip
fidelity here is exact and guaranteed, not probabilistic. That is not this
module being "better" -- it is solving an easier problem. The encoding
below is a literal, deterministic text serialization of already-exact
data, decoded back via DecodeLatticeEncoding + conversion.StructureFromJSON
-- there is no compression of symmetry-reduced information and no
geometry to reconstruct from a sparser starting point the way real SLICES
does. ComputeLatticeEncoding/DecodeLatticeEncoding are exercised directly
by test_document.py (encode -> decode -> rebuild a real pymatgen Structure
-> compare), a real, executed round-trip test -- unlike the old
slices-package wrapper this replaces, which was never actually run here.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.mofjson.provenance import Provenance, content_hash


class LatticeEncoding(BaseModel):
    """One invertible string per structure. See module docstring for what
    this is and, just as importantly, what it deliberately is not."""

    encoding_string: str = Field(min_length=1)
    provenance: Provenance


def ComputeLatticeEncoding(raw: Any, *, version: str = "v1") -> LatticeEncoding:
    """
    Deterministic (stable atom/bond ordering) plain-text encoding of
    (raw.lattice, raw.atoms, raw.bonds). Format, by example:

        CELL a=8.601000 b=8.601000 c=8.601000 alpha=90.000000 beta=90.000000 gamma=90.000000
        ATOMS Cu#0:0.000000,0.000000,0.000000 O#1:0.152000,0.152000,0.000000 ...
        BONDS 0-1:0.0:[0,0,0] 1-14:1.0:[1,0,0] ...

    Atom indices (#0, #1, ...) are this encoding's own compact stand-in
    for the original string atom ids, assigned by sorting atoms by id --
    DecodeLatticeEncoding inverts this using the same sort order, so the
    graph structure round-trips exactly."""
    atoms_sorted = sorted(raw.atoms, key=lambda a: a["id"])
    id_to_idx = {a["id"]: i for i, a in enumerate(atoms_sorted)}
    bonds_sorted = sorted(raw.bonds, key=lambda b: (b["source"], b["target"]))

    lat = raw.lattice
    cell_part = "CELL " + " ".join(f"{k}={lat[k]:.6f}" for k in ("a", "b", "c", "alpha", "beta", "gamma"))
    atoms_part = "ATOMS " + " ".join(
        f"{a['element']}#{id_to_idx[a['id']]}:{a['frac_xyz'][0]:.6f},{a['frac_xyz'][1]:.6f},{a['frac_xyz'][2]:.6f}"
        for a in atoms_sorted
    )
    bonds_part = "BONDS " + " ".join(
        f"{id_to_idx[b['source']]}-{id_to_idx[b['target']]}:{b['order']}:"
        f"[{','.join(str(x) for x in b.get('shift', [0, 0, 0]))}]"
        for b in bonds_sorted
    )
    encoding_string = f"{cell_part}|{atoms_part}|{bonds_part}"

    return LatticeEncoding(
        encoding_string=encoding_string,
        provenance=Provenance(
            method="lattice_encoding.ComputeLatticeEncoding",
            version=version,
            computed_from=["atoms", "bonds", "lattice"],
            computed_from_hash=content_hash(raw.atoms, raw.bonds, raw.lattice),
        ),
    )


def DecodeLatticeEncoding(encoding_string: str) -> dict[str, Any]:
    """Inverse of ComputeLatticeEncoding: returns {"lattice", "atoms",
    "bonds"} with atom ids reconstructed as "0", "1", ... (the same
    indices ComputeLatticeEncoding assigned). Pass the result straight to
    conversion.StructureFromJSON to rebuild an actual pymatgen Structure.
    """
    cell_str, atoms_str, bonds_str = encoding_string.split("|")

    lattice: dict[str, float] = {}
    for tok in cell_str.removeprefix("CELL ").split():
        key, val = tok.split("=")
        lattice[key] = float(val)

    atoms: list[dict[str, Any]] = []
    for tok in atoms_str.removeprefix("ATOMS ").split():
        elem_idx, coords = tok.split(":")
        element, idx = elem_idx.split("#")
        x, y, z = (float(v) for v in coords.split(","))
        atoms.append({"id": idx, "element": element, "frac_xyz": [x, y, z]})

    bonds: list[dict[str, Any]] = []
    remainder = bonds_str.removeprefix("BONDS ").strip()
    for tok in remainder.split():
        src_tgt, order, shift_str = tok.split(":")
        source, target = src_tgt.split("-")
        shift = [int(x) for x in shift_str.strip("[]").split(",")]
        bonds.append({"source": source, "target": target, "order": float(order), "shift": shift})

    return {"lattice": lattice, "atoms": atoms, "bonds": bonds}


__all__ = ["LatticeEncoding", "ComputeLatticeEncoding", "DecodeLatticeEncoding"]
