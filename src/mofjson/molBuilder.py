"""
fragment_to_rwmol -- shared "atoms/bonds dict -> RDKit RWMol" construction.

Previously duplicated near-identically in linker.py (IdentifyLinkers) and
DetectPolymerUnit.py (DetectPolymerUnits): both built their own
Chem.RWMol(), both defined their own copy of the bond-order-to-RDKit-
BondType map, both looped over atoms/bonds the same way, both did the same
aromatic-flagging pass afterward. 
"""
from __future__ import annotations

from typing import Any

from rdkit import Chem

_DUMMY_SYMBOL = "*"

ORDER_TO_BONDTYPE = {
    0.0: Chem.BondType.ZERO,
    1.0: Chem.BondType.SINGLE,
    1.5: Chem.BondType.AROMATIC,
    2.0: Chem.BondType.DOUBLE,
    3.0: Chem.BondType.TRIPLE,
}


def fragment_to_rwmol(
    fragment: dict[str, Any], charge_by_id: dict[str, int] | None = None
) -> tuple[Chem.RWMol, dict[str, int]]:
    """Builds an RDKit RWMol from one building-block fragment's atoms/bonds.

    Dummy '*' attachment atoms become atomic-number-0 placeholder atoms;
    their bonds keep whatever order is actually stored on them (usually
    0.0 / BondType.ZERO) -- never forced to a single bond, which would
    over-value any real atom with more than one attachment point (this was
    a real bug once: a bridging carboxylate oxygen with two attachment
    points, forced to two single bonds, raised an RDKit valence error).

    `charge_by_id` is optional: pass it when the caller wants formal
    charges pre-set before sanitization (linker.py's smiles/formula
    extraction needs this; DetectPolymerUnit.py's bigsmiles construction
    doesn't). Returns (rwmol, id_to_idx) -- the caller still owns calling
    Chem.SanitizeMol(rwmol.GetMol()) and handling its exceptions, since
    what counts as "failed" differs slightly between the two callers.
    """
    charge_by_id = charge_by_id or {}
    rw = Chem.RWMol()
    id_to_idx: dict[str, int] = {}
    for a in fragment["atoms"]:
        atom = Chem.Atom(0) if a["element"] == _DUMMY_SYMBOL else Chem.Atom(a["element"])
        if a["id"] in charge_by_id:
            atom.SetFormalCharge(charge_by_id[a["id"]])
        id_to_idx[a["id"]] = rw.AddAtom(atom)

    for b in fragment["bonds"]:
        rw.AddBond(id_to_idx[b["source"]], id_to_idx[b["target"]], ORDER_TO_BONDTYPE[b["order"]])

    aromatic_atom_idxs = {
        idx for b in rw.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC
        for idx in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    }
    for idx in aromatic_atom_idxs:
        rw.GetAtomWithIdx(idx).SetIsAromatic(True)

    return rw, id_to_idx


__all__ = ["fragment_to_rwmol", "ORDER_TO_BONDTYPE"]
