"""
MOFJSON schema definition.

MOFJSON extends MolJSON with the minimum needed to describe a periodic
reticular structure:

- Periodicity: bonds may carry an integer `shift` vector naming which
  neighbouring cell the target atom sits in, the mechanism
  labelled-quotient-graph encodings use (e.g. SLICES, Xiao et al., Nat.
  Commun. 2023, 14, 7027). We don't depend on the `slices` package; see
  lattice_encoding.py for our own encoding built on this field.
- Node/linker/guest classification is sparse and optional, never required
  for validity, as MolJSON already does for `charges`/`aromatic_n_h`.
- `topology` is nullable metadata, not a closed lookup into an external
  registry (contrast MOFid, where an unresolved net becomes ERROR/UNKNOWN).

Atoms, bonds, charges and aromatic_n_h are unchanged, so a fragment with
no `lattice` key degrades back into a plain MolJSON object.

A building_blocks[] entry also carries, optionally: smiles / formula /
identity_provenance for linker_* fragments (linker.IdentifyLinkers), and
polymer, a PolymerUnit set automatically for any fragment bonding to its
own next periodic copy (DetectPolymerUnit.DetectPolymerUnits). PolymerUnit
appears here as a generic object; the authoritative shape is the pydantic
model -- call PolymerUnit.model_json_schema() if you need it strictly.
"""


from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

_DUMMY_SYMBOL = "*"

# Same 118-element + dummy-atom enum as MolJSON's schema, kept as a literal
# list here (rather than generated from a chemistry library) so that
# schema.py has zero external dependencies -- pymatgen is only needed by
# conversion.py, where real chemistry/geometry work actually happens.
_ELEMENT_ENUM = [
    _DUMMY_SYMBOL, "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
    "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

_NODE_LINKER_ROLES = ["node", "linker", "guest"]


def _build_charge_schema(*, use_enum: bool) -> Dict[str, Any]:
    if use_enum:
        return {"type": "integer", "enum": list(range(-5, 6))}
    return {"type": "integer", "minimum": -5, "maximum": 5}


def _build_hcount_schema(*, mode: str) -> Dict[str, Any]:
    if mode == "default":
        return {"type": "integer", "enum": [1]}
    if mode == "paper":
        return {"type": "integer", "minimum": 1, "maximum": 2}
    raise ValueError(f"Unsupported hcount schema mode: {mode}")


def _build_shift_schema() -> Dict[str, Any]:
    return {
        "type": "array",
        "description": (
            "Periodic image shift [a, b, c], in whole unit cells, that the "
            "target atom's coordinates must be translated by to realize this "
            "bond. Omit (or use [0, 0, 0]) for an ordinary intra-cell bond. "
            "This is the direct analogue of SLICES's labeled-quotient-graph "
            "edge shift vectors."
        ),
        "items": {"type": "integer"},
        "minItems": 3,
        "maxItems": 3,
    }


def _build_lattice_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "description": (
            "Unit cell parameters defining the periodic repeat. Required: a "
            "MOFJSON object with no lattice is not a periodic structure."
        ),
        "properties": {
            "a": {"type": "number", "exclusiveMinimum": 0},
            "b": {"type": "number", "exclusiveMinimum": 0},
            "c": {"type": "number", "exclusiveMinimum": 0},
            "alpha": {"type": "number", "exclusiveMinimum": 0, "maximum": 180},
            "beta": {"type": "number", "exclusiveMinimum": 0, "maximum": 180},
            "gamma": {"type": "number", "exclusiveMinimum": 0, "maximum": 180},
        },
        "required": ["a", "b", "c", "alpha", "beta", "gamma"],
    }


def _build_topology_schema() -> Dict[str, Any]:
    return {
        "type": ["object", "null"],
        "description": (
            "Optional, non-load-bearing net annotation. Null means no "
            "topology has been assigned -- this never blocks validity the "
            "way an unresolved RCSR lookup does for MOFid/MOFkey."
        ),
        "additionalProperties": False,
        "properties": {
            "net": {
                "type": ["string", "null"],
                "description": "e.g. an RCSR three-letter net code, if known.",
            },
            "source": {
                "type": "string",
                "description": "Where the net label came from, e.g. 'RCSR' or 'user'.",
            },
            "independent_nets": {
                "type": "integer",
                "minimum": 1,
                "description": "Degree of catenation/interpenetration.",
            },
        },
    }


def _build_node_linker_role_item() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "atom_id": {"type": "string"},
            "role": {"type": "string", "enum": _NODE_LINKER_ROLES},
        },
        "required": ["atom_id", "role"],
    }


def _build_polymer_unit_schema() -> Dict[str, Any]:
    """Loose mirror of polymer.PolymerUnit -- see that module for the
    authoritative shape and its cross-field validation (is_infinite implies
    periodic_directions required/non-empty/duplicate-free and
    degree_of_polymerization/end_groups both absent; finite implies
    degree_of_polymerization required and periodic_directions absent).
    That validation is NOT re-expressed here as JSON Schema conditionals --
    call PolymerUnit.model_json_schema() directly if you need it enforced
    at the JSON Schema level rather than at the pydantic layer."""
    return {
        "type": ["object", "null"],
        "additionalProperties": False,
        "properties": {
            "repeat_unit_smiles": {"type": "string", "minLength": 1},
            "is_infinite": {"type": "boolean"},
            "periodic_directions": {
                "type": ["array", "null"],
                "items": {"type": "string", "enum": ["a", "b", "c"]},
            },
            "degree_of_polymerization": {"type": ["integer", "null"], "minimum": 1},
            "end_groups": {"type": ["array", "null"], "items": {"type": "string"}},
        },
        "required": ["repeat_unit_smiles", "is_infinite"],
    }


def _build_schema(*, charge_use_enum: bool, hcount_mode: str) -> Dict[str, Any]:
    element_schema: Dict[str, Any] = {
        "type": "string",
        "enum": _ELEMENT_ENUM,
        "description": "Element symbol like 'C' or 'Cl', or '*' dummy atom.",
    }

    atom_item: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "description": "Unique atom id."},
            "element": element_schema,
            "frac_xyz": {
                "type": "array",
                "description": "Fractional coordinates [x, y, z] within the unit cell.",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
            "occupancy": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Site occupancy fraction. Omit for fully occupied (1.0) sites.",
            },
            "connection_id": {
                "type": ["integer", "null"],
                "description": (
                    "Only meaningful when element is '*'. Marks this dummy atom "
                    "as a point of extension on a building-block fragment, wired "
                    "up to another fragment's point of extension via net_edges -- "
                    "the MOFJSON analogue of a BigSMILES bonding descriptor."
                ),
            },
        },
        "required": ["id", "element", "frac_xyz"],
    }

    # A "fragment" is the shape of one building block: the same atoms/bonds/
    # charges/aromatic_n_h core every top-level MOFJSON object has, minus
    # lattice and net-level fields, plus its own id (referenced by net_edges
    # and by no one else -- the same reference pattern bonds already use to
    # point at atom ids, just one nesting level up), plus optional chemical
    # identity (smiles/formula/identity_provenance) and an optional polymer
    # (BigSMILES PolymerUnit) attribute -- see module docstring.
    fragment_schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "atoms", "bonds", "charges", "aromatic_n_h"],
        "properties": {
            "id": {"type": "string", "description": "Unique building-block id."},
            "atoms": {"type": "array", "items": atom_item},
            "bonds": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "order": {
                            "type": "number",
                            "enum": [0, 1, 1.5, 2, 3],
                        },
                    },
                    "required": ["source", "target", "order"],
                },
                "description": "Internal bonds only; a fragment's own bonds never carry a shift.",
            },
            "charges": {"type": ["array", "null"]},
            "aromatic_n_h": {"type": ["array", "null"]},
            "smiles": {
                "type": ["string", "null"],
                "description": (
                    "Chemical identity for linker_* fragments, from RDKit "
                    "(linker.IdentifyLinkers). Null for node_*/"
                    "guest_* fragments and for any linker where extraction "
                    "failed -- see identity_provenance for why."
                ),
            },
            "formula": {
                "type": ["string", "null"],
                "description": "Molecular formula paired with smiles above. Null under the same conditions.",
            },
            "identity_provenance": {
                "type": ["object", "null"],
                "description": (
                    "Diagnostic from the smiles/formula extraction step, e.g. "
                    "{'ok': True}, {'ok': False, 'reason': ...} (extraction "
                    "attempted and failed), or {'ok': None, 'reason': ...} "
                    "(not attempted -- fragment contains a metal atom)."
                ),
            },
            "polymer": _build_polymer_unit_schema(),
        },
    }

    bond_order_schema: Dict[str, Any] = {
        "type": "number",
        "enum": [0, 1, 1.5, 2, 3],
        "description": (
            "Bond order. Aromatic bonds are 1.5. ZERO bonds are 0 and are the "
            "default choice for metal coordination bonds."
        ),
    }

    bond_item: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source": {"type": "string"},
            "target": {"type": "string"},
            "order": bond_order_schema,
            "shift": _build_shift_schema(),
        },
        "required": ["source", "target", "order"],
    }

    charges_item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "atom_id": {"type": "string"},
            "formal_charge": _build_charge_schema(use_enum=charge_use_enum),
        },
        "required": ["atom_id", "formal_charge"],
    }

    aromatic_n_h_item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "atom_id": {"type": "string"},
            "hcount": _build_hcount_schema(mode=hcount_mode),
        },
        "required": ["atom_id", "hcount"],
    }

    net_edge_item: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "from_block": {"type": "string"},
            "from_point": {"type": "integer"},
            "to_block": {"type": "string"},
            "to_point": {"type": "integer"},
            "order": bond_order_schema,
            "shift": _build_shift_schema(),
        },
        "required": ["from_block", "from_point", "to_block", "to_point", "order"],
        "description": (
            "Wires one building block's point of extension (connection_id) to "
            "another's, with an explicit periodic shift -- the SBU-level "
            "analogue of a SLICES quotient-graph edge. from_block == to_block "
            "with a nonzero shift is valid and represents a self-repeating "
            "fragment (e.g. an infinite rod-shaped node bonding to its own "
            "next periodic copy) -- this is exactly the signal "
            "DetectPolymerUnit.DetectPolymerUnits reads to auto-populate a "
            "fragment's `polymer` field."
        ),
    }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "lattice",
            "atoms",
            "bonds",
            "charges",
            "aromatic_n_h",
            "node_linker_roles",
            "topology",
            "building_blocks",
            "net_edges",
        ],
        "properties": {
            "lattice": _build_lattice_schema(),
            "atoms": {"type": "array", "items": atom_item},
            "bonds": {"type": "array", "items": bond_item},
            "charges": {
                "type": ["array", "null"],
                "description": "Sparse list of NON-ZERO formal charges. Null means none.",
                "items": charges_item,
            },
            "aromatic_n_h": {
                "type": ["array", "null"],
                "description": (
                    "Sparse list of aromatic nitrogens with explicit hydrogen "
                    "count. Null means none."
                ),
                "items": aromatic_n_h_item,
            },
            "node_linker_roles": {
                "type": ["array", "null"],
                "description": (
                    "Sparse, optional reticular-chemistry classification of "
                    "atoms into node/linker/guest. Null means no "
                    "classification has been assigned; this is metadata, "
                    "never required for structural validity."
                ),
                "items": _build_node_linker_role_item(),
            },
            "topology": {
                **_build_topology_schema(),
                "x-derived-from": ["atoms", "bonds", "node_linker_roles"],
            },
            "building_blocks": {
                "type": ["array", "null"],
                "description": (
                    "Optional compact repeating-unit layer: one fragment per "
                    "SBU instance, each carrying its own free-text id -- the "
                    "same array-of-named-things shape 'atoms' already uses, "
                    "one nesting level up. When present alongside net_edges, "
                    "ExpandNet() can mechanically derive the full atoms/bonds "
                    "layer from it. Null means this MOFJSON only carries the "
                    "flat atomistic representation. This field is a CACHE, "
                    "not independently-authored data -- see x-derived-from. "
                    "A consumer that also has document.py's MOFJSONDocument "
                    "should prefer that over hand-checking this field, since "
                    "the plain dict shape here carries no provenance/hash."
                ),
                "items": fragment_schema,
                "x-derived-from": ["atoms", "bonds", "node_linker_roles"],
            },
            "net_edges": {
                "type": ["array", "null"],
                "description": (
                    "Sparse list wiring building_blocks together. Null unless "
                    "building_blocks is also non-null. Same caching caveat as "
                    "building_blocks: see x-derived-from."
                ),
                "items": net_edge_item,
                "x-derived-from": ["atoms", "bonds", "node_linker_roles"],
            },
        },
    }


def GetSchema() -> Dict[str, Any]:
    """Return a deep copy of the MOFJSON schema (portable, enum-based)."""
    return deepcopy(_build_schema(charge_use_enum=True, hcount_mode="default"))


def GetPaperSchema() -> Dict[str, Any]:
    """Return the MOFJSON schema variant with min/max numeric ranges,
    mirroring MolJSON's GetPaperSchema for cross-provider structured-output
    compatibility."""
    return deepcopy(_build_schema(charge_use_enum=False, hcount_mode="paper"))


def GetDocumentSchema() -> Dict[str, Any]:
    """Return the JSON Schema for an actual document.MOFJSONDocument --
    the nested {raw, roles, derived, slices, classification} object
    BuildDocument() really constructs -- generated directly from the
    pydantic model rather than hand-maintained in parallel with it.

    This is not the same schema as GetSchema()/GetPaperSchema()
    above (see module docstring): those describe the flat conversion.py
    dict. Calling MOFJSONDocument.model_json_schema() here.
    """
    from src.mofjson.document import MOFJSONDocument

    return MOFJSONDocument.model_json_schema()
