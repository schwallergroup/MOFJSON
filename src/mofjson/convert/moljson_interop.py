"""MolFragment <-> plain MolJSON, and RDKit Mol <-> MolFragment.

Phase 02.8.

PUBLIC API
    FragmentToMolJSON(fragment, *, strict=True) -> dict
    FragmentFromMolJSON(obj) -> MolFragment
    FragmentFromMol(mol) -> MolFragment          # RDKit
    FragmentToMol(fragment) -> rdkit.Chem.Mol
    CheckRoundTrip(smiles) -> RoundTripResult    # the audit helper for Phase 02.1
    NormalizeDeMetalatedFragment(fragment) -> (MolFragment, warnings)
"""

from __future__ import annotations

import importlib
import re
import warnings as py_warnings
from dataclasses import dataclass, field
from typing import Any, Literal

from mofjson.exceptions import ConversionError, LossyConversionError
from mofjson.schema.molecular import (
    AromaticNH,
    Atom,
    AttachmentPoint,
    Bond,
    Charge,
    Coord,
    HCount,
    MolFragment,
    Radical,
)

_DUMMY_SYMBOL = "*"
_METAL_SYMBOLS: frozenset[str] | None = None


def _rdkit() -> Any:
    from rdkit import Chem

    return Chem


def _periodic_table() -> Any:
    Chem = _rdkit()
    return Chem.GetPeriodicTable()


def _metal_symbols() -> frozenset[str]:
    global _METAL_SYMBOLS
    if _METAL_SYMBOLS is None:
        # Metals commonly seen in MOF node SMILES / dative bonds.
        _METAL_SYMBOLS = frozenset(
            {
                "Li",
                "Na",
                "K",
                "Rb",
                "Cs",
                "Be",
                "Mg",
                "Ca",
                "Sr",
                "Ba",
                "Sc",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
                "Y",
                "Zr",
                "Nb",
                "Mo",
                "Tc",
                "Ru",
                "Rh",
                "Pd",
                "Ag",
                "Cd",
                "Hf",
                "Ta",
                "W",
                "Re",
                "Os",
                "Ir",
                "Pt",
                "Au",
                "Hg",
                "Al",
                "Ga",
                "In",
                "Tl",
                "Ge",
                "Sn",
                "Pb",
                "Sb",
                "Bi",
                "La",
                "Ce",
                "Pr",
                "Nd",
                "Pm",
                "Sm",
                "Eu",
                "Gd",
                "Tb",
                "Dy",
                "Ho",
                "Er",
                "Tm",
                "Yb",
                "Lu",
                "Th",
                "U",
            }
        )
    return _METAL_SYMBOLS


def _is_valid_element_symbol(sym: Any) -> bool:
    if not isinstance(sym, str):
        return False
    s = sym.strip()
    if not s:
        return False
    if s == _DUMMY_SYMBOL:
        return True
    try:
        return int(_periodic_table().GetAtomicNumber(s)) > 0
    except Exception:
        return False


def _make_rdkit_atom(sym: str) -> Any:
    Chem = _rdkit()
    if sym == _DUMMY_SYMBOL:
        return Chem.Atom(0)
    return Chem.Atom(sym)


def _order_to_bondtype(order: float) -> Any:
    Chem = _rdkit()
    mapping = {
        0.0: Chem.BondType.ZERO,
        0.5: Chem.BondType.DATIVE,
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
    }
    for key, bt in mapping.items():
        if abs(order - key) < 1e-9:
            return bt
    raise ConversionError(f"Unsupported bond order: {order!r}", field_path="order", value=order)


def _bondtype_to_order(bt: Any) -> float:
    Chem = _rdkit()
    mapping = {
        Chem.BondType.ZERO: 0.0,
        Chem.BondType.DATIVE: 0.5,
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.AROMATIC: 1.5,
        Chem.BondType.DOUBLE: 2.0,
        Chem.BondType.TRIPLE: 3.0,
    }
    if bt not in mapping:
        raise ConversionError(
            f"Unsupported bond type: {bt}",
            field_path="order",
            value=str(bt),
        )
    return mapping[bt]


def _infer_and_mark_aromatic_atoms(mol: Any) -> None:
    Chem = _rdkit()
    aromatic_atom_idxs: set[int] = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            aromatic_atom_idxs.add(bond.GetBeginAtomIdx())
            aromatic_atom_idxs.add(bond.GetEndAtomIdx())
    for idx in aromatic_atom_idxs:
        mol.GetAtomWithIdx(idx).SetIsAromatic(True)


def mol_to_canonical_smiles(mol: Any) -> str:
    Chem = _rdkit()
    m = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return str(Chem.MolToSmiles(m, canonical=True, isomericSmiles=False))


def fragment_formula(mol: Any) -> str:
    from collections import Counter

    counts: Counter[str] = Counter()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "H":
            continue
        counts[sym] += 1
        # Include explicit + total Hs toward formula.
        counts["H"] += int(atom.GetTotalNumHs())
    # Hill system-ish: C, H, then alphabetical.
    parts: list[str] = []
    for el in ("C", "H"):
        if counts.get(el):
            n = counts.pop(el)
            parts.append(el if n == 1 else f"{el}{n}")
    for el in sorted(counts):
        n = counts[el]
        parts.append(el if n == 1 else f"{el}{n}")
    return "".join(parts)


def _assign_atom_ids(mol: Any, *, atom_id_style: str = "element") -> dict[int, str]:
    kept_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"]
    idx_to_id: dict[int, str] = {}
    if atom_id_style == "a":
        for n, idx in enumerate(kept_idxs, start=1):
            idx_to_id[idx] = f"a{n}"
    else:
        element_counts: dict[str, int] = {}
        for idx in kept_idxs:
            sym = mol.GetAtomWithIdx(idx).GetSymbol()
            element_counts[sym] = element_counts.get(sym, 0) + 1
            idx_to_id[idx] = f"{sym}{element_counts[sym]}"
    return idx_to_id


def _infer_h_via_clone(atom: Any) -> int:
    """What RDKit would assign as implicit H if we cleared explicit H and radicals."""
    parent = atom.GetOwningMol()
    Chem = _rdkit()
    m = Chem.Mol(parent)
    a = m.GetAtomWithIdx(atom.GetIdx())
    a.SetNumExplicitHs(0)
    a.SetNumRadicalElectrons(0)
    a.SetNoImplicit(False)
    try:
        m.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        return int(a.GetTotalNumHs())
    except Exception:
        return 0


def FragmentFromMolJSON(obj: dict[str, Any]) -> MolFragment:
    """Always succeed for a valid MolJSON document (strict subset of MolFragment)."""
    if not isinstance(obj, dict):
        raise ConversionError("MolJSON must be an object", value=type(obj).__name__)

    atoms_raw = obj.get("atoms") or []
    bonds_raw = obj.get("bonds") or []
    charges_raw = obj.get("charges") or []
    anh_raw = obj.get("aromatic_n_h") or []

    atoms = [Atom(id=a["id"], element=a["element"]) for a in atoms_raw]
    bonds = [
        Bond(
            source=b["source"],
            target=b["target"],
            order=float(b["order"]),
            dative_to=b.get("dative_to"),
        )
        for b in bonds_raw
    ]
    charges = (
        [
            Charge(atom_id=c["atom_id"], formal_charge=int(c["formal_charge"]))
            for c in charges_raw
            if int(c["formal_charge"]) != 0
        ]
        or None
    )
    aromatic_n_h = (
        [AromaticNH(atom_id=e["atom_id"], hcount=1) for e in anh_raw] or None
    )

    # Pass through MOF extensions if present (superset tolerance).
    hcounts = None
    if obj.get("hcounts"):
        hcounts = [HCount(atom_id=e["atom_id"], hcount=int(e["hcount"])) for e in obj["hcounts"]]
    radicals = None
    if obj.get("radicals"):
        radicals = [
            Radical(
                atom_id=e["atom_id"],
                num_radical_electrons=int(e["num_radical_electrons"]),
            )
            for e in obj["radicals"]
        ]
    attachment_points = None
    if obj.get("attachment_points"):
        attachment_points = [
            AttachmentPoint(
                id=e["id"],
                atom_id=e["atom_id"],
                role=e["role"],
                expected_partner=e.get("expected_partner"),
                vector=e.get("vector"),
                partner_distance=e.get("partner_distance"),
            )
            for e in obj["attachment_points"]
        ]

    coords = None
    if obj.get("coords"):
        coords = [Coord(atom_id=e["atom_id"], xyz=list(e["xyz"])) for e in obj["coords"]]

    return MolFragment(
        atoms=atoms,
        bonds=bonds,
        charges=charges,
        aromatic_n_h=aromatic_n_h,
        hcounts=hcounts,
        radicals=radicals,
        attachment_points=attachment_points,
        coords=coords,
        coords_frame=obj.get("coords_frame"),
        coords_origin=obj.get("coords_origin"),
        coords_source=obj.get("coords_source"),
        coords_precision=obj.get("coords_precision"),
        smiles=obj.get("smiles"),
        formula=obj.get("formula"),
    )


def FragmentToMolJSON(
    fragment: MolFragment,
    *,
    strict: bool = True,
    include_empty_fields: bool = False,
) -> dict[str, Any]:
    """Convert MolFragment to a plain MolJSON dict.

    With strict=True, raise LossyConversionError if MOF-only fields would be dropped.
    With strict=False, drop extensions and emit warnings.
    """
    warnings: list[str] = []
    if fragment.has_mof_extensions():
        msg = (
            "MolFragment uses MOFJSON-only fields "
            "(hcounts/radicals/attachment_points/coords/dative order 0.5)"
        )
        if strict:
            raise LossyConversionError(msg, field_path="fragment")
        warnings.append(msg)
        if fragment.coords is not None:
            warnings.append("dropping coords (MOFJSON-only geometry block)")
        for w in warnings:
            py_warnings.warn(w, stacklevel=2)

    bonds_out: list[dict[str, Any]] = []
    for b in fragment.bonds:
        order = b.order
        if abs(order - 0.5) < 1e-9:
            # Already handled by strict path; in non-strict, degrade to single.
            order = 1.0
        bonds_out.append({"source": b.source, "target": b.target, "order": order})

    out: dict[str, Any] = {
        "atoms": [{"id": a.id, "element": a.element} for a in fragment.atoms],
        "bonds": bonds_out,
    }
    if fragment.charges:
        out["charges"] = [
            {"atom_id": c.atom_id, "formal_charge": c.formal_charge} for c in fragment.charges
        ]
    elif include_empty_fields:
        out["charges"] = None

    if fragment.aromatic_n_h:
        out["aromatic_n_h"] = [
            {"atom_id": e.atom_id, "hcount": e.hcount} for e in fragment.aromatic_n_h
        ]
    elif include_empty_fields:
        out["aromatic_n_h"] = None

    return out


def FragmentFromMol(mol: Any, *, atom_id_style: str = "element") -> MolFragment:
    """Build a MolFragment from an RDKit molecule, preserving H/radicals/dative bonds."""
    Chem = _rdkit()
    if mol is None:
        raise ConversionError("mol is None")

    m = Chem.Mol(mol)
    try:
        m.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    idx_to_id = _assign_atom_ids(m, atom_id_style=atom_id_style)
    kept_set = set(idx_to_id)

    # Preserve RDKit atom order (non-H) for MolJSON-compatible id assignment.
    atoms: list[Atom] = []
    for atom in m.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        atoms.append(Atom(id=idx_to_id[atom.GetIdx()], element=atom.GetSymbol()))

    bonds: list[Bond] = []
    for bond in m.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i not in kept_set or j not in kept_set:
            continue
        order = _bondtype_to_order(bond.GetBondType())
        dative_to = None
        if abs(order - 0.5) < 1e-9:
            si = m.GetAtomWithIdx(i).GetSymbol()
            sj = m.GetAtomWithIdx(j).GetSymbol()
            metals = _metal_symbols()
            if si in metals and sj not in metals:
                dative_to = idx_to_id[i]
            elif sj in metals and si not in metals:
                dative_to = idx_to_id[j]
            else:
                dative_to = idx_to_id[j]
        bonds.append(
            Bond(
                source=idx_to_id[i],
                target=idx_to_id[j],
                order=order,
                dative_to=dative_to,
            )
        )

    charges: list[Charge] = []
    hcounts: list[HCount] = []
    radicals: list[Radical] = []
    aromatic_n_h: list[AromaticNH] = []

    for idx, atom_id in idx_to_id.items():
        atom = m.GetAtomWithIdx(idx)
        fc = int(atom.GetFormalCharge())
        if fc != 0:
            charges.append(Charge(atom_id=atom_id, formal_charge=fc))

        n_rad = int(atom.GetNumRadicalElectrons())
        if n_rad > 0:
            radicals.append(Radical(atom_id=atom_id, num_radical_electrons=n_rad))

        total_h = int(atom.GetTotalNumHs())
        inferred = _infer_h_via_clone(atom)
        no_implicit = bool(atom.GetNoImplicit())

        # MolJSON aromatic_n_h for ordinary aromatic N-H (no MOF extensions needed).
        if (
            atom.GetSymbol() == "N"
            and atom.GetIsAromatic()
            and total_h == 1
            and n_rad == 0
            and not no_implicit
        ):
            aromatic_n_h.append(AromaticNH(atom_id=atom_id, hcount=1))
            continue

        # Sparse hcounts: only where locked/radical H differs from default inference.
        if n_rad > 0 or (no_implicit and total_h != inferred) or (
            not no_implicit and total_h != inferred
        ):
            hcounts.append(HCount(atom_id=atom_id, hcount=total_h))

    return MolFragment(
        atoms=atoms,
        bonds=bonds,
        charges=charges or None,
        aromatic_n_h=aromatic_n_h or None,
        hcounts=hcounts or None,
        radicals=radicals or None,
    )


def FragmentToMol(
    fragment: MolFragment,
    *,
    cap_attachment_points: bool = False,
    sanitize: bool = True,
) -> Any:
    """Build an RDKit molecule from a MolFragment."""
    Chem = _rdkit()
    rw = Chem.RWMol()
    id_to_idx: dict[str, int] = {}

    for atom in fragment.atoms:
        if not _is_valid_element_symbol(atom.element):
            raise ConversionError(
                f"Unsupported/invalid element symbol: {atom.element!r}",
                field_path="atoms",
                value=atom.element,
            )
        idx = rw.AddAtom(_make_rdkit_atom(atom.element))
        id_to_idx[atom.id] = idx

    charge_map = {c.atom_id: c.formal_charge for c in (fragment.charges or [])}
    for atom_id, fc in charge_map.items():
        rw.GetAtomWithIdx(id_to_idx[atom_id]).SetFormalCharge(int(fc))

    hcount_map = {h.atom_id: h.hcount for h in (fragment.hcounts or [])}
    radical_map = {
        r.atom_id: r.num_radical_electrons for r in (fragment.radicals or [])
    }
    anh_map: dict[str, int] = {e.atom_id: int(e.hcount) for e in (fragment.aromatic_n_h or [])}

    for bond in fragment.bonds:
        i = id_to_idx[bond.source]
        j = id_to_idx[bond.target]
        bt = _order_to_bondtype(bond.order)
        # RDKit dative bonds are directional: begin -> end means electrons donated
        # toward end. If dative_to is set, make that atom the end (acceptor).
        if abs(bond.order - 0.5) < 1e-9 and bond.dative_to is not None:
            acceptor = id_to_idx[bond.dative_to]
            donor = j if acceptor == i else i
            rw.AddBond(donor, acceptor, bt)
        else:
            a, b = (i, j) if i < j else (j, i)
            rw.AddBond(a, b, bt)

    mol = rw.GetMol()
    _infer_and_mark_aromatic_atoms(mol)

    for atom_id, hcount in anh_map.items():
        atom = mol.GetAtomWithIdx(id_to_idx[atom_id])
        atom.SetIsAromatic(True)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(int(hcount))

    for atom_id, hcount in hcount_map.items():
        atom = mol.GetAtomWithIdx(id_to_idx[atom_id])
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(int(hcount))

    for atom_id, n_rad in radical_map.items():
        atom = mol.GetAtomWithIdx(id_to_idx[atom_id])
        atom.SetNumRadicalElectrons(int(n_rad))
        # Radical centres in de-metalated SMILES usually need locked H counts.
        if atom_id not in hcount_map:
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(int(atom.GetNumExplicitHs()))

    if cap_attachment_points and fragment.attachment_points:
        for ap in fragment.attachment_points:
            atom = mol.GetAtomWithIdx(id_to_idx[ap.atom_id])
            # Cap dangling valence with one H if the atom looks under-valent.
            atom.SetNoImplicit(False)
            try:
                mol.UpdatePropertyCache(strict=False)
            except Exception:
                pass
            # Force +1 explicit H for carboxylate O attachment points etc.
            current = int(atom.GetTotalNumHs())
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(current + 1)
            if int(atom.GetNumRadicalElectrons()) > 0:
                atom.SetNumRadicalElectrons(0)

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # MOF fragments often fail full sanitization; leave unsanitized.
            pass
    return mol


def NormalizeDeMetalatedFragment(
    fragment: MolFragment,
) -> tuple[MolFragment, list[str]]:
    """Rewrite radical aromatic carbons toward closed-shell + attachment points.

    Canonical MOFJSON form prefers attachment_points over radicals for de-metalated
    donor atoms. This normaliser is best-effort for carboxylate-bearing aromatics
    (e.g. BTC radical SMILES).
    """
    warns: list[str] = []
    if not fragment.radicals:
        return fragment, warns

    # Identify carboxylate oxygens (O with formal charge -1 bonded to a C that is
    # double-bonded to another O) and radical carbons on the aromatic ring.
    charge_map = {c.atom_id: c.formal_charge for c in (fragment.charges or [])}
    element = {a.id: a.element for a in fragment.atoms}

    # Build adjacency
    adj: dict[str, list[tuple[str, float]]] = {a.id: [] for a in fragment.atoms}
    for b in fragment.bonds:
        adj[b.source].append((b.target, b.order))
        adj[b.target].append((b.source, b.order))

    carboxylate_o: list[str] = []
    for oid, el in element.items():
        if el != "O" or charge_map.get(oid, 0) != -1:
            continue
        for nbr, _order in adj[oid]:
            if element.get(nbr) != "C":
                continue
            # C should also bond to a carbonyl O (order 2)
            for nbr2, order2 in adj[nbr]:
                if nbr2 != oid and element.get(nbr2) == "O" and abs(order2 - 2.0) < 1e-9:
                    carboxylate_o.append(oid)
                    break

    if not carboxylate_o:
        return fragment, warns

    # Clear C radicals when carboxylates are present (BTC-style de-metalated rings).
    cleared = [r.atom_id for r in fragment.radicals if element[r.atom_id] in {"C", "c"}]
    new_radicals = [r for r in fragment.radicals if r.atom_id not in cleared]

    hcount_map = {h.atom_id: h.hcount for h in (fragment.hcounts or [])}
    for cid in cleared:
        # Carboxylate-bearing or carboxylate-adjacent ring C -> 0 H; else aromatic CH.
        bonded_to_carbox = any(n in carboxylate_o for n, _ in adj[cid])
        if not bonded_to_carbox:
            for nbr, _ in adj[cid]:
                if element.get(nbr) == "C" and any(n in carboxylate_o for n, _ in adj[nbr]):
                    bonded_to_carbox = True
                    break
        hcount_map[cid] = 0 if bonded_to_carbox else 1

    # Attachment points on carboxylate O atoms (canonical form).
    existing_ap_atoms = {ap.atom_id for ap in (fragment.attachment_points or [])}
    new_aps = list(fragment.attachment_points or [])
    next_id = len(new_aps) + 1
    for oid in carboxylate_o:
        if oid in existing_ap_atoms:
            continue
        new_aps.append(
            AttachmentPoint(
                id=f"X{next_id}",
                atom_id=oid,
                role="carboxylate_O",
                expected_partner="metal",
            )
        )
        next_id += 1
        existing_ap_atoms.add(oid)

    warns.append(
        "Normalized de-metalated radical carboxylate fragment to closed-shell "
        "aromatic with attachment_points (preferred MOFJSON form)."
    )

    return (
        MolFragment(
            atoms=fragment.atoms,
            bonds=fragment.bonds,
            charges=fragment.charges,
            aromatic_n_h=fragment.aromatic_n_h,
            hcounts=[HCount(atom_id=k, hcount=v) for k, v in hcount_map.items()] or None,
            radicals=new_radicals or None,
            attachment_points=new_aps or None,
            smiles=None,
            formula=None,
        ),
        warns,
    )


@dataclass
class RoundTripResult:
    smiles: str
    parses: bool
    roundtrips: bool
    error: str | None = None
    has_radical_atom: bool = False
    has_dative: bool = False
    input_smiles: str | None = None
    output_smiles: str | None = None
    representation: str = "moljson"
    warnings: list[str] = field(default_factory=list)


def _smiles_has_radical_atom(smiles: str) -> bool:
    # Heuristic matching Phase 00: de-metalated radicals like [C], [CH], [N], [O].
    # Do not treat charged species such as [O-] / [n-] as radicals.
    return bool(re.search(r"\[(?:CH\d*|NH\d*|[CNOS])\]", smiles))


def _smiles_has_dative(smiles: str) -> bool:
    # RDKit uses -> or <- for dative in SMILES.
    return "->" in smiles or "<-" in smiles


def _parse_smiles(
    smiles: str,
    *,
    allow_unsanitized: bool,
) -> tuple[Any | None, bool, str | None]:
    """Return (mol, sanitized_ok, error).

    For the MolJSON baseline, molecules that only parse with sanitize=False count as
    parse failures (Phase 00.6: 369 / 5,149). The MolFragment path may still recover them.
    """
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol, True, None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, False, "rdkit_parse_failed"
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    if not allow_unsanitized:
        return None, False, "rdkit_parse_failed"
    return mol, False, None


def CheckRoundTrip(
    smiles: str,
    *,
    representation: Literal["moljson", "mofjson-fragment"] = "moljson",
) -> RoundTripResult:
    """Round-trip a SMILES string via MolJSON or MolFragment and classify the outcome."""
    has_rad = _smiles_has_radical_atom(smiles)
    has_dat = _smiles_has_dative(smiles)

    allow_unsanitized = representation == "mofjson-fragment"
    mol, _sanitized_ok, parse_err = _parse_smiles(
        smiles, allow_unsanitized=allow_unsanitized
    )
    if mol is None:
        return RoundTripResult(
            smiles=smiles,
            parses=False,
            roundtrips=False,
            error=parse_err or "rdkit_parse_failed",
            has_radical_atom=has_rad,
            has_dative=has_dat,
            representation=representation,
        )

    try:
        input_smi = mol_to_canonical_smiles(mol)
    except Exception as exc:
        return RoundTripResult(
            smiles=smiles,
            parses=False,
            roundtrips=False,
            error=f"canonical_smiles_failed: {exc}",
            has_radical_atom=has_rad,
            has_dative=has_dat,
            representation=representation,
        )

    try:
        if representation == "moljson":
            from moljson import CheckRoundTrip as MolJSONCheckRoundTrip
            from moljson import MolFromJSON, MolToJSON

            # Detect dative bonds before calling MolJSON (which raises).
            Chem = _rdkit()
            for bond in mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.DATIVE:
                    has_dat = True
                    return RoundTripResult(
                        smiles=smiles,
                        parses=True,
                        roundtrips=False,
                        error="Unsupported bond type: DATIVE",
                        has_radical_atom=has_rad,
                        has_dative=True,
                        input_smiles=input_smi,
                        representation=representation,
                    )
            ok, in_s, out_s, _ = MolJSONCheckRoundTrip(mol)
            # Also try MolToJSON path for molecules that sanitize oddly.
            if not ok:
                try:
                    mj = MolToJSON(mol)
                    mol2 = MolFromJSON(mj)
                    out_s = mol_to_canonical_smiles(mol2)
                    ok = in_s == out_s
                except Exception as exc:
                    return RoundTripResult(
                        smiles=smiles,
                        parses=True,
                        roundtrips=False,
                        error=str(exc),
                        has_radical_atom=has_rad,
                        has_dative=has_dat,
                        input_smiles=in_s,
                        representation=representation,
                    )
            return RoundTripResult(
                smiles=smiles,
                parses=True,
                roundtrips=bool(ok),
                error=None if ok else "molecule_changed",
                has_radical_atom=has_rad,
                has_dative=has_dat,
                input_smiles=in_s,
                output_smiles=out_s,
                representation=representation,
            )

        # mofjson-fragment path
        Chem = _rdkit()
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DATIVE:
                has_dat = True

        # Prefer a sanitized working copy, but never let a failed SanitizeMol
        # partially mutate the molecule we round-trip (it clears aromatic flags).
        work = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(work)
        except Exception:
            work = Chem.Mol(mol)
            try:
                work.UpdatePropertyCache(strict=False)
            except Exception:
                pass
            for atom in work.GetAtoms():
                if atom.GetNumRadicalElectrons() > 0:
                    atom.SetNoImplicit(True)

        frag = FragmentFromMol(work)
        mol2 = FragmentToMol(frag, sanitize=False)
        # Graph equality first — canonical SMILES is unreliable for unsanitized
        # aromatic / radical fragments (SanitizeMol inside MolToSmiles mutates).
        ok = _graph_equivalent(work, mol2)
        out_s = None
        if not ok:
            out_s = mol_to_canonical_smiles(mol2)
            ok = input_smi == out_s
        else:
            try:
                out_s = mol_to_canonical_smiles(Chem.Mol(mol2))
            except Exception:
                out_s = input_smi
        return RoundTripResult(
            smiles=smiles,
            parses=True,
            roundtrips=bool(ok),
            error=None if ok else "molecule_changed",
            has_radical_atom=has_rad,
            has_dative=has_dat,
            input_smiles=input_smi,
            output_smiles=out_s,
            representation=representation,
        )
    except Exception as exc:
        err = str(exc)
        if "DATIVE" in err.upper():
            has_dat = True
        return RoundTripResult(
            smiles=smiles,
            parses=True,
            roundtrips=False,
            error=err,
            has_radical_atom=has_rad,
            has_dative=has_dat,
            input_smiles=input_smi,
            representation=representation,
        )


def SuperposeRMSD(
    mobile: list[list[float]],
    target: list[list[float]],
) -> float:
    """Kabsch superposition RMSD (Å) between two equal-length point sets."""
    if len(mobile) != len(target) or not mobile:
        raise ConversionError(
            "SuperposeRMSD requires non-empty equal-length point sets",
            value={"mobile": len(mobile), "target": len(target)},
        )
    try:
        # importlib keeps mypy from parsing numpy's py.typed stubs (py3.12 syntax).
        np = importlib.import_module("numpy")
    except ImportError as exc:  # pragma: no cover
        raise ConversionError(
            "SuperposeRMSD requires numpy",
            value=str(exc),
        ) from exc

    p = np.asarray(mobile, dtype=float)
    q = np.asarray(target, dtype=float)
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    h = p.T @ q
    u, _s, vt = np.linalg.svd(h)
    v = vt.T
    d = float(np.linalg.det(v @ u.T))
    sign = np.diag([1.0, 1.0, 1.0 if d >= 0.0 else -1.0])
    r = v @ sign @ u.T
    aligned = p @ r.T
    diff = aligned - q
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _graph_equivalent(mol_a: Any, mol_b: Any) -> bool:
    """Structural equality when canonical SMILES disagrees due to sanitization."""

    def _atom_key(a: Any) -> tuple[Any, ...]:
        try:
            total_h = int(a.GetTotalNumHs())
        except Exception:
            total_h = int(a.GetNumExplicitHs())
        return (
            a.GetSymbol(),
            int(a.GetFormalCharge()),
            int(a.GetNumRadicalElectrons()),
            total_h,
            bool(a.GetIsAromatic()),
        )

    def _sig(m: Any) -> tuple[Any, ...]:
        atoms = [_atom_key(a) for a in m.GetAtoms() if a.GetSymbol() != "H"]
        bonds = []
        for b in m.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            si = m.GetAtomWithIdx(i).GetSymbol()
            sj = m.GetAtomWithIdx(j).GetSymbol()
            if si == "H" or sj == "H":
                continue
            bonds.append((tuple(sorted((si, sj))), str(b.GetBondType()), bool(b.GetIsAromatic())))
        return (tuple(sorted(atoms)), tuple(sorted(bonds)))

    try:
        return _sig(mol_a) == _sig(mol_b)
    except Exception:
        return False


__all__ = [
    "CheckRoundTrip",
    "FragmentFromMol",
    "FragmentFromMolJSON",
    "FragmentToMol",
    "FragmentToMolJSON",
    "NormalizeDeMetalatedFragment",
    "RoundTripResult",
    "SuperposeRMSD",
    "fragment_formula",
    "mol_to_canonical_smiles",
]
