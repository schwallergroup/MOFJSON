"""MolFragment: MolJSON extended so that real MOF linkers survive a round trip.

Phase 02.

Schema shape follows Phase 02.7: sparse MolJSON-style lists (atoms/bonds plus optional
charges, aromatic_n_h, hcounts, radicals, attachment_points) and optional local 3D
coords from 02.6. Bond order 0.5 encodes a dative/coordination bond with optional
dative_to; attachment points may carry vector / partner_distance when coords exist.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from mofjson.exceptions import SchemaError
from mofjson.schema.base import MOFJSONModel

ALLOWED_BOND_ORDERS: frozenset[float] = frozenset({0.0, 0.5, 1.0, 1.5, 2.0, 3.0})

CoordsFrame = Literal["local_cartesian"]
CoordsOrigin = Literal["centroid", "first_atom", "metal_centroid"]
CoordsSource = Literal["deconstruction", "forcefield", "template", "llm", "manual"]

# Covalent radii (Å) for geometry sanity checks against the bond graph.
_COVALENT_RADII: dict[str, float] = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "*": 0.70,
}

_UNIT_VECTOR_TOL = 1e-3
_BOND_LENGTH_LO = 0.55  # fraction of sum of covalent radii
_BOND_LENGTH_HI = 1.55


class Atom(MOFJSONModel):
    id: str = Field(min_length=1)
    element: str = Field(min_length=1)


class Bond(MOFJSONModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    order: Annotated[float, Field(description="Bond order; 0.5 = dative/coordination.")]
    dative_to: str | None = None

    @field_validator("order")
    @classmethod
    def _order_allowed(cls, v: float) -> float:
        fv = float(v)
        for allowed in ALLOWED_BOND_ORDERS:
            if abs(fv - allowed) < 1e-9:
                return allowed
        raise SchemaError(
            f"Unsupported bond order: {v!r} (allowed: 0, 0.5, 1, 1.5, 2, 3)",
            field_path="order",
            value=v,
        )

    @model_validator(mode="after")
    def _dative_rules(self) -> Bond:
        if self.dative_to is not None:
            if abs(self.order - 0.5) > 1e-9:
                raise SchemaError(
                    "dative_to is only allowed when order == 0.5",
                    field_path="dative_to",
                    value=self.dative_to,
                )
            if self.dative_to not in (self.source, self.target):
                raise SchemaError(
                    "dative_to must equal source or target",
                    field_path="dative_to",
                    value=self.dative_to,
                )
        return self


class Charge(MOFJSONModel):
    atom_id: str = Field(min_length=1)
    formal_charge: int


class AromaticNH(MOFJSONModel):
    atom_id: str = Field(min_length=1)
    hcount: Literal[1] = 1


class HCount(MOFJSONModel):
    atom_id: str = Field(min_length=1)
    hcount: Annotated[int, Field(ge=0)]


class Radical(MOFJSONModel):
    atom_id: str = Field(min_length=1)
    num_radical_electrons: Annotated[int, Field(ge=0)]


class AttachmentPoint(MOFJSONModel):
    id: str = Field(min_length=1)
    atom_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    expected_partner: str | None = None
    # Outward unit direction of the severed bond, in the fragment local frame (02.5/02.6).
    vector: list[float] | None = None
    # Bond length (Å) before the cut.
    partner_distance: float | None = None

    @field_validator("vector")
    @classmethod
    def _vector_shape(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        if len(v) != 3 or not all(isinstance(x, (int, float)) for x in v):
            raise SchemaError(
                "attachment_points.vector must be a length-3 numeric list",
                field_path="vector",
                value=v,
            )
        return [float(x) for x in v]


class Coord(MOFJSONModel):
    atom_id: str = Field(min_length=1)
    xyz: list[float]

    @field_validator("xyz")
    @classmethod
    def _xyz_shape(cls, v: list[float]) -> list[float]:
        if len(v) != 3 or not all(isinstance(x, (int, float)) for x in v):
            raise SchemaError(
                "coords.xyz must be a length-3 numeric list",
                field_path="xyz",
                value=v,
            )
        return [float(x) for x in v]


class MolFragment(MOFJSONModel):
    atoms: list[Atom]
    bonds: list[Bond]
    charges: list[Charge] | None = None
    aromatic_n_h: list[AromaticNH] | None = None
    hcounts: list[HCount] | None = None
    radicals: list[Radical] | None = None
    attachment_points: list[AttachmentPoint] | None = None
    coords: list[Coord] | None = None
    coords_frame: CoordsFrame | None = None
    coords_origin: CoordsOrigin | None = None
    coords_source: CoordsSource | None = None
    coords_precision: Annotated[int, Field(ge=0, le=8)] | None = None
    smiles: str | None = None
    formula: str | None = None

    @model_validator(mode="after")
    def _invariants(self) -> MolFragment:
        atom_ids = [a.id for a in self.atoms]
        if any(not i for i in atom_ids):
            raise SchemaError("atoms[].id must be non-empty", field_path="atoms")
        if len(atom_ids) != len(set(atom_ids)):
            dupes = sorted({i for i in atom_ids if atom_ids.count(i) > 1})
            raise SchemaError(
                f"atoms[].id must be unique; duplicates: {dupes}",
                field_path="atoms",
                value=dupes,
            )
        id_set = set(atom_ids)
        element_of = {a.id: a.element for a in self.atoms}

        def _require(atom_id: str, field_path: str) -> None:
            if atom_id not in id_set:
                raise SchemaError(
                    f"{field_path} references unknown atom_id: {atom_id}",
                    field_path=field_path,
                    value=atom_id,
                )

        seen_edges: dict[tuple[str, str], float] = {}
        for bi, bond in enumerate(self.bonds):
            _require(bond.source, f"bonds[{bi}].source")
            _require(bond.target, f"bonds[{bi}].target")
            if bond.source == bond.target:
                raise SchemaError(
                    f"Self-bonds are not allowed: {bond.source}",
                    field_path=f"bonds[{bi}]",
                    value=bond.source,
                )
            a, b = (
                (bond.source, bond.target)
                if bond.source <= bond.target
                else (bond.target, bond.source)
            )
            key = (a, b)
            prev = seen_edges.get(key)
            if prev is not None and abs(prev - bond.order) > 1e-9:
                raise SchemaError(
                    f"Conflicting duplicate edge between {bond.source} and {bond.target}",
                    field_path=f"bonds[{bi}]",
                    value={"previous": prev, "new": bond.order},
                )
            seen_edges[key] = bond.order
            if bond.dative_to is not None:
                _require(bond.dative_to, f"bonds[{bi}].dative_to")

        for field_name in ("charges", "aromatic_n_h", "hcounts", "radicals"):
            entries = getattr(self, field_name)
            if not entries:
                continue
            seen: set[str] = set()
            for ei, entry in enumerate(entries):
                aid = entry.atom_id
                _require(aid, f"{field_name}[{ei}].atom_id")
                if aid in seen:
                    raise SchemaError(
                        f"Duplicate atom_id in {field_name}: {aid}",
                        field_path=f"{field_name}[{ei}].atom_id",
                        value=aid,
                    )
                seen.add(aid)

        if self.attachment_points:
            ap_ids: set[str] = set()
            for ei, ap in enumerate(self.attachment_points):
                _require(ap.atom_id, f"attachment_points[{ei}].atom_id")
                if not ap.id:
                    raise SchemaError(
                        "attachment_points[].id must be non-empty",
                        field_path=f"attachment_points[{ei}].id",
                    )
                if ap.id in ap_ids:
                    raise SchemaError(
                        f"Duplicate attachment_points id: {ap.id}",
                        field_path=f"attachment_points[{ei}].id",
                        value=ap.id,
                    )
                ap_ids.add(ap.id)

        if self.coords is not None:
            _validate_coords_block(self, id_set, element_of)

        if self.smiles is not None or self.formula is not None:
            _check_smiles_formula_agree(self)

        return self

    def has_mof_extensions(self) -> bool:
        """True if this fragment uses fields MolJSON cannot represent."""
        if self.hcounts:
            return True
        if self.radicals:
            return True
        if self.attachment_points:
            return True
        if self.coords is not None:
            return True
        for bond in self.bonds:
            if abs(bond.order - 0.5) < 1e-9 or bond.dative_to is not None:
                return True
        return False


def _vec_norm(v: list[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vec_dot(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _covalent_radius(element: str) -> float:
    return _COVALENT_RADII.get(element, 1.0)


def _validate_coords_block(
    frag: MolFragment,
    id_set: set[str],
    element_of: dict[str, str],
) -> None:
    assert frag.coords is not None

    # Invariant 7: cover every atom exactly once; metadata present.
    if frag.coords_frame is None or frag.coords_origin is None or frag.coords_source is None:
        raise SchemaError(
            "coords_frame, coords_origin and coords_source are required when coords is present",
            field_path="coords",
        )

    coord_ids = [c.atom_id for c in frag.coords]
    if len(coord_ids) != len(set(coord_ids)):
        raise SchemaError(
            "coords must reference each atom_id at most once",
            field_path="coords",
        )
    if set(coord_ids) != id_set:
        missing = sorted(id_set - set(coord_ids))
        extra = sorted(set(coord_ids) - id_set)
        raise SchemaError(
            "coords must cover every atom exactly once",
            field_path="coords",
            value={"missing": missing, "extra": extra},
        )

    xyz_of = {c.atom_id: c.xyz for c in frag.coords}
    n = len(frag.coords)
    centroid = [
        sum(c.xyz[0] for c in frag.coords) / n,
        sum(c.xyz[1] for c in frag.coords) / n,
        sum(c.xyz[2] for c in frag.coords) / n,
    ]

    # Invariant 8: every AP has a unit outward vector when coords are present.
    if frag.attachment_points:
        for ei, ap in enumerate(frag.attachment_points):
            if ap.vector is None:
                raise SchemaError(
                    "attachment_points.vector is required when coords is present",
                    field_path=f"attachment_points[{ei}].vector",
                )
            norm = _vec_norm(ap.vector)
            if abs(norm - 1.0) > _UNIT_VECTOR_TOL:
                raise SchemaError(
                    "attachment_points.vector must be unit length (tolerance 1e-3)",
                    field_path=f"attachment_points[{ei}].vector",
                    value={"norm": norm, "vector": ap.vector},
                )
            pos = xyz_of[ap.atom_id]
            outward = _vec_sub(pos, centroid)
            outward_norm = _vec_norm(outward)
            if outward_norm > 1e-8 and _vec_dot(ap.vector, outward) < 0:
                raise SchemaError(
                    "attachment_points.vector must point away from the fragment",
                    field_path=f"attachment_points[{ei}].vector",
                    value=ap.vector,
                )

    # Invariant 9: bonded distances within element-dependent bounds.
    for bi, bond in enumerate(frag.bonds):
        p = xyz_of[bond.source]
        q = xyz_of[bond.target]
        dist = _vec_norm(_vec_sub(p, q))
        r_sum = _covalent_radius(element_of[bond.source]) + _covalent_radius(
            element_of[bond.target]
        )
        lo = r_sum * _BOND_LENGTH_LO
        hi = r_sum * _BOND_LENGTH_HI
        # Zero-order / very long contacts get a looser upper bound.
        if abs(bond.order) < 1e-9:
            hi = max(hi, 3.5)
        if dist < lo or dist > hi:
            raise SchemaError(
                "coords disagree with the bond graph: bonded distance out of bounds",
                field_path=f"bonds[{bi}]",
                value={
                    "source": bond.source,
                    "target": bond.target,
                    "distance": dist,
                    "allowed": [lo, hi],
                },
            )


def _check_smiles_formula_agree(frag: MolFragment) -> None:
    """If smiles/formula are present, they must agree with the graph (lazy RDKit)."""
    try:
        from mofjson.convert import moljson_interop as interop
    except Exception as exc:  # pragma: no cover
        raise SchemaError(
            "smiles/formula validation requires the chem extra (rdkit)",
            field_path="smiles",
            value=str(exc),
        ) from exc

    try:
        mol = interop.FragmentToMol(frag, cap_attachment_points=False)
        graph_smiles = interop.mol_to_canonical_smiles(mol)
        graph_formula = interop.fragment_formula(mol)
    except Exception as exc:
        raise SchemaError(
            f"Could not build RDKit mol from fragment graph: {exc}",
            field_path="atoms",
            value=str(exc),
        ) from exc

    if frag.smiles is not None and frag.smiles != graph_smiles:
        raise SchemaError(
            "smiles does not agree with the molecular graph",
            field_path="smiles",
            value={"provided": frag.smiles, "graph": graph_smiles},
        )
    if frag.formula is not None and frag.formula != graph_formula:
        raise SchemaError(
            "formula does not agree with the molecular graph",
            field_path="formula",
            value={"provided": frag.formula, "graph": graph_formula},
        )


__all__ = [
    "ALLOWED_BOND_ORDERS",
    "AromaticNH",
    "Atom",
    "AttachmentPoint",
    "Bond",
    "Charge",
    "Coord",
    "CoordsFrame",
    "CoordsOrigin",
    "CoordsSource",
    "HCount",
    "MolFragment",
    "Radical",
]
