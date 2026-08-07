"""Stage 5: turn organic components into MolFragments with attachment points.

Phase 08.5.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from mofjson.deconstruct.partition import Component, PartitionResult
from mofjson.deconstruct.site_geometry import _classify_donor_group
from mofjson.exceptions import DeconstructionError
from mofjson.schema.molecular import (
    Atom,
    AttachmentPoint,
    Bond,
    Charge,
    HCount,
    MolFragment,
)
from mofjson.schema.reticular import FunctionalGroup, Linker
from mofjson.vocab.elements import METALS


def _require_rdkit() -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DeconstructionError(
            "rdkit is required for linker extraction; install with `pip install mofjson[chem]`",
        ) from exc
    return Chem


def _component_bonds(
    comp: Component,
    partition: PartitionResult,
) -> list[tuple[int, int, float]]:
    """Intra-component bonds (no image) from kept adjacency / original bonds."""
    crystal = partition.document.crystal
    assert crystal is not None
    id_to_i = {sid: i for i, sid in enumerate(partition.site_ids)}
    sites = set(comp.site_indices)
    out: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for bond in crystal.bonds:
        i = id_to_i[bond.source]
        j = id_to_i[bond.target]
        if i in sites and j in sites:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((i, j, float(bond.order)))
    return out


def _guess_bond_order(order: float, el_i: str, el_j: str) -> float:
    """CrystalNN orders are often 0.5 for M–L; organic edges may be 1.0 placeholders."""
    if abs(order - 0.5) < 1e-9:
        return 1.0
    if order in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        return order
    return 1.0


def _build_fragment(
    comp: Component,
    partition: PartitionResult,
    *,
    structure: Any | None = None,
) -> MolFragment:
    """Build an anionic-framework MolFragment for one organic component."""
    indices = sorted(comp.site_indices)
    idx_to_local = {idx: f"{partition.elements[idx]}{k + 1}" for k, idx in enumerate(indices)}
    # Prefer stable crystal site ids when unique elements allow; use local ids for fragment.
    atoms = [
        Atom(id=idx_to_local[idx], element=partition.elements[idx]) for idx in indices
    ]
    bonds: list[Bond] = []
    for i, j, order in _component_bonds(comp, partition):
        oi = _guess_bond_order(order, partition.elements[i], partition.elements[j])
        bonds.append(
            Bond(source=idx_to_local[i], target=idx_to_local[j], order=oi)
        )

    # Attachment points from cut bonds; carboxylate O keeps charge -1, no extra H.
    charges: list[Charge] = []
    attachment_points: list[AttachmentPoint] = []
    seen_ap_atoms: set[int] = set()
    ap_n = 0
    for cb in comp.cut_bonds:
        if cb.donor_index not in comp.site_indices:
            continue
        if cb.donor_index in seen_ap_atoms:
            # Still one attachment per donor atom; carboxylate may have 2 O.
            pass
        group = _classify_donor_group(
            cb.donor_index, partition.elements, partition.adjacency
        )
        # Collapse carboxylate to one AP per carboxylate carbon when possible.
        ap_atom = cb.donor_index
        role = group
        if partition.elements[cb.donor_index] == "O":
            for j, _ in partition.adjacency.get(cb.donor_index, []):
                if partition.elements[j] == "C":
                    o_on_c = [
                        k
                        for k, _ in partition.adjacency.get(j, [])
                        if partition.elements[k] == "O"
                    ]
                    if len(o_on_c) >= 2:
                        # Use first O as representative if not yet used.
                        ap_atom = min(o_on_c)
                        role = "carboxylate_O"
                        break
        if ap_atom in seen_ap_atoms:
            continue
        seen_ap_atoms.add(ap_atom)
        ap_n += 1
        attachment_points.append(
            AttachmentPoint(
                id=f"A{ap_n}",
                atom_id=idx_to_local[ap_atom],
                role=role,
                expected_partner="metal",
            )
        )
        # Anionic framework form: carboxylate O and imidazolate N get -1.
        if role in ("carboxylate_O", "carboxylate_chelating", "imidazolate_N", "azolate_N"):
            charges.append(Charge(atom_id=idx_to_local[ap_atom], formal_charge=-1))

    # Explicit hcounts: hydrogens present in the structure are real atoms already.
    # For heavy atoms, set hcount=0 so RDKit does not invent protons on carboxylates.
    hcounts = [
        HCount(atom_id=idx_to_local[idx], hcount=0)
        for idx in indices
        if partition.elements[idx] != "H"
    ]

    # Coords require attachment_points.vector (MolFragment invariant); omit until
    # connection vectors are computed in a later enrichment pass.
    _ = structure
    fragment = MolFragment(
        atoms=atoms,
        bonds=bonds,
        charges=charges or None,
        hcounts=hcounts or None,
        attachment_points=attachment_points or None,
    )
    return fragment


def _canonical_smiles(fragment: MolFragment) -> str | None:
    Chem = _require_rdkit()
    from mofjson.convert.moljson_interop import FragmentToMol

    try:
        mol = FragmentToMol(fragment, cap_attachment_points=False, sanitize=True)
        return Chem.MolToSmiles(mol)
    except Exception:
        try:
            mol = FragmentToMol(fragment, cap_attachment_points=False, sanitize=False)
            return Chem.MolToSmiles(mol)
        except Exception:
            return None


def _detect_functional_groups(smiles: str | None, elements: tuple[str, ...]) -> list[FunctionalGroup]:
    groups: list[FunctionalGroup] = []
    counts = Counter(elements)
    for el, name in (("Br", "Br"), ("F", "F"), ("Cl", "Cl"), ("I", "I"), ("S", "S")):
        if counts.get(el, 0):
            groups.append(FunctionalGroup(group=name, count=counts[el]))
    if smiles:
        if "N(=O)O" in smiles or "[N+](=O)[O-]" in smiles:
            groups.append(FunctionalGroup(group="NO2", count=smiles.count("N")))
    return groups


def ExtractLinkers(partition: PartitionResult) -> list[Linker]:
    """Convert organic components into :class:`Linker` records."""
    structure = None
    try:
        from mofjson.convert.pymatgen_bridge import MOFToStructure

        if partition.document is not None:
            structure = MOFToStructure(partition.document)
    except Exception:
        structure = None

    linkers: list[Linker] = []
    role_map = {
        "linker": "linker",
        "ligand": "capping_ligand",
        "guest": "guest_solvent",
        "counterion": "counterion",
        "pillar": "pillar",
    }
    # Multiplicity: count identical compositions per role among linker-role comps.
    linker_comps = [c for c in partition.organic_components if c.role == "linker"]
    formula_key = lambda c: tuple(sorted(Counter(c.elements).items()))
    multiplicity = Counter(formula_key(c) for c in linker_comps)

    seen_keys: set[tuple] = set()
    n = 0
    for comp in partition.organic_components:
        if comp.role not in ("linker", "ligand", "guest", "counterion"):
            continue
        # Deduplicate linker instances of the same formula into one Linker with count.
        key = (comp.role, formula_key(comp))
        if comp.role == "linker" and key in seen_keys:
            continue
        if comp.role == "linker":
            seen_keys.add(key)

        n += 1
        fragment = _build_fragment(comp, partition, structure=structure)
        smiles = _canonical_smiles(fragment)
        fragment = fragment.model_copy(update={"smiles": smiles})
        donor_groups = None
        if fragment.attachment_points:
            donor_groups = [ap.role for ap in fragment.attachment_points]
        connectivity = len(fragment.attachment_points or [])
        count = multiplicity[formula_key(comp)] if comp.role == "linker" else 1
        linkers.append(
            Linker(
                id=f"L{n}",
                connectivity=connectivity if connectivity else None,
                role=role_map.get(comp.role, "linker"),
                fragment=fragment,
                smiles=smiles,
                donor_groups=donor_groups,
                denticity=len(donor_groups) if donor_groups else None,
                count_per_cell=count,
                functional_groups=_detect_functional_groups(smiles, comp.elements) or None,
            )
        )
    return linkers


def DeduplicateLinkers(linkers: list[Linker]) -> list[Linker]:
    """Collapse chemically identical linkers by canonical SMILES."""
    by_smiles: dict[str, Linker] = {}
    no_smiles: list[Linker] = []
    for linker in linkers:
        if not linker.smiles:
            no_smiles.append(linker)
            continue
        prev = by_smiles.get(linker.smiles)
        if prev is None:
            by_smiles[linker.smiles] = linker
            continue
        # Merge counts.
        c1 = prev.count_per_cell or 1
        c2 = linker.count_per_cell or 1
        by_smiles[linker.smiles] = prev.model_copy(update={"count_per_cell": c1 + c2})
    # Re-id
    out = list(by_smiles.values()) + no_smiles
    return [
        linker.model_copy(update={"id": f"L{i + 1}"}) for i, linker in enumerate(out)
    ]


__all__ = ["DeduplicateLinkers", "ExtractLinkers"]
