"""Element sets, expected coordination numbers, and (later) bond-length bounds.

Phase 03.1 and Phase 11.7.

METALS is the predicate that splits the bond graph in Phase 08.1. Include alkali,
alkaline earth, transition, post-transition, lanthanide and actinide. Exclude B, Si,
Ge, As, Sb, Te from METALS (they are metalloids / main-group framework atoms when
present); keep them in METALLOIDS so the choice is visible.
"""

from __future__ import annotations

from types import MappingProxyType

LANTHANIDES: frozenset[str] = frozenset(
    {
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
    }
)

ACTINIDES: frozenset[str] = frozenset(
    {
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
    }
)

# Ambiguous main-group / metalloid set — not treated as framework metals by default.
METALLOIDS: frozenset[str] = frozenset({"B", "Si", "Ge", "As", "Sb", "Te"})

_ALKALI = frozenset({"Li", "Na", "K", "Rb", "Cs", "Fr"})
_ALKALINE_EARTH = frozenset({"Be", "Mg", "Ca", "Sr", "Ba", "Ra"})
_TRANSITION = frozenset(
    {
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
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
    }
)
_POST_TRANSITION = frozenset({"Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi", "Po", "Nh", "Fl", "Mc", "Lv"})

METALS: frozenset[str] = frozenset(
    _ALKALI
    | _ALKALINE_EARTH
    | _TRANSITION
    | _POST_TRANSITION
    | LANTHANIDES
    | ACTINIDES
)

DONOR_ELEMENTS: frozenset[str] = frozenset({"O", "N", "S", "P", "C", "F", "Cl", "Br", "I"})

ORGANIC_ELEMENTS: frozenset[str] = frozenset(
    {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Se"}
)

# Typical CN ranges for OMS / chemistry sanity (Phase 03.4, Phase 11.7).
# Unusual coordination is a warning, not a hard rejection.
_LN_CN = (6, 12)
EXPECTED_CN: MappingProxyType[str, tuple[int, int]] = MappingProxyType(
    {
        "Li": (4, 6),
        "Na": (4, 8),
        "K": (6, 8),
        "Mg": (4, 6),
        "Ca": (6, 8),
        "Al": (4, 6),
        "Sc": (6, 8),
        "Ti": (4, 6),
        "V": (4, 6),
        "Cr": (4, 6),
        "Mn": (4, 6),
        "Fe": (4, 6),
        "Co": (4, 6),
        "Ni": (4, 6),
        "Cu": (4, 6),
        "Zn": (4, 6),
        "Y": (6, 9),
        "Zr": (6, 8),
        "Nb": (4, 6),
        "Mo": (4, 6),
        "Ru": (4, 6),
        "Rh": (4, 6),
        "Pd": (4, 6),
        "Ag": (2, 4),
        "Cd": (4, 6),
        "Hf": (6, 8),
        "W": (4, 6),
        "Re": (4, 6),
        "Os": (4, 6),
        "Ir": (4, 6),
        "Pt": (4, 6),
        "Au": (2, 4),
        "In": (4, 6),
        "Sn": (4, 6),
        "Pb": (4, 8),
        "Bi": (4, 8),
        **{sym: _LN_CN for sym in LANTHANIDES},
        "Th": (6, 12),
        "U": (6, 12),
    }
)

CONNECTIVITY_SOURCES: tuple[str, ...] = (
    "computed",
    "mofid",
    "literature",
    "llm",
    "unknown",
)

PARTIAL_CHARGE_METHODS: tuple[str, ...] = (
    "pbe_ddec",
    "pbe_cm5",
    "pbe_bader",
    "other",
    "unknown",
)

# Covalent radii (Å). Organic values match schema/molecular.py and convert/bonds.py;
# metal values are Pyykkö–Atsumi-style single-bond radii for sanity checks.
COVALENT_RADII: MappingProxyType[str, float] = MappingProxyType(
    {
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
        "Li": 1.28,
        "Na": 1.66,
        "K": 2.03,
        "Mg": 1.41,
        "Ca": 1.76,
        "Al": 1.21,
        "Sc": 1.48,
        "Ti": 1.36,
        "V": 1.34,
        "Cr": 1.22,
        "Mn": 1.19,
        "Fe": 1.16,
        "Co": 1.11,
        "Ni": 1.10,
        "Cu": 1.12,
        "Zn": 1.18,
        "Y": 1.63,
        "Zr": 1.54,
        "Nb": 1.47,
        "Mo": 1.38,
        "Ru": 1.25,
        "Rh": 1.25,
        "Pd": 1.20,
        "Ag": 1.28,
        "Cd": 1.36,
        "In": 1.42,
        "Sn": 1.39,
        "Hf": 1.52,
        "W": 1.37,
        "Re": 1.31,
        "Os": 1.28,
        "Ir": 1.22,
        "Pt": 1.23,
        "Au": 1.24,
        "Pb": 1.46,
        "Bi": 1.48,
        "La": 1.80,
        "Ce": 1.63,
        "Pr": 1.76,
        "Nd": 1.74,
        "Sm": 1.72,
        "Eu": 1.68,
        "Gd": 1.69,
        "Tb": 1.68,
        "Dy": 1.67,
        "Ho": 1.66,
        "Er": 1.65,
        "Tm": 1.64,
        "Yb": 1.70,
        "Lu": 1.62,
        "Th": 1.75,
        "U": 1.70,
        "*": 0.70,
    }
)

# Default metal–ligand distance window (Å). Phase 11.7: flag bonds outside this
# unless a per-pair override says otherwise.
DEFAULT_METAL_LIGAND_BOUNDS: tuple[float, float] = (1.5, 3.2)

# Optional per-pair (metal, donor) overrides; keys are sorted alphabetically only
# when both are metals — here always (metal, donor_element).
_BOND_LENGTH_PAIR_OVERRIDES: dict[tuple[str, str], tuple[float, float]] = {
    ("Cu", "O"): (1.7, 2.5),
    ("Cu", "N"): (1.7, 2.5),
    ("Zn", "O"): (1.7, 2.5),
    ("Zn", "N"): (1.7, 2.5),
    ("Zr", "O"): (1.8, 2.6),
    ("Al", "O"): (1.6, 2.3),
    ("Fe", "O"): (1.7, 2.5),
    ("Fe", "N"): (1.7, 2.5),
    ("Sm", "O"): (2.0, 3.0),
    ("Sm", "N"): (2.0, 3.0),
}

BOND_LENGTH_BOUNDS: MappingProxyType[tuple[str, str], tuple[float, float]] = MappingProxyType(
    dict(_BOND_LENGTH_PAIR_OVERRIDES)
)


def bond_length_bounds(metal: str, donor: str) -> tuple[float, float]:
    """Return (lo, hi) Å for a metal–donor contact."""
    return BOND_LENGTH_BOUNDS.get((metal, donor), DEFAULT_METAL_LIGAND_BOUNDS)


def covalent_radius(element: str) -> float:
    """Covalent radius in Å; unknown elements default to 1.0."""
    return COVALENT_RADII.get(element, 1.0)


__all__ = [
    "ACTINIDES",
    "BOND_LENGTH_BOUNDS",
    "CONNECTIVITY_SOURCES",
    "COVALENT_RADII",
    "DEFAULT_METAL_LIGAND_BOUNDS",
    "DONOR_ELEMENTS",
    "EXPECTED_CN",
    "LANTHANIDES",
    "METALLOIDS",
    "METALS",
    "ORGANIC_ELEMENTS",
    "PARTIAL_CHARGE_METHODS",
    "bond_length_bounds",
    "covalent_radius",
]
