"""Coordination polyhedron names and their ideal coordination numbers.

Phase 03.2.

WHAT THIS FILE CONTAINS
    POLYHEDRA               ordered tuple of allowed MetalSite.geometry values, keyed to
                            ChemEnv symbols so the mapping is mechanical (Phase 03.1).
    POLYHEDRON_CN           polyhedron -> ideal CN (other/unknown omitted).
    CN_TO_POLYHEDRA         inverse map for constraining classifier candidates by CN.
    CHEMENV_TO_POLYHEDRON   ChemEnv IUPAC symbol -> POLYHEDRA name.
    CSM_THRESHOLDS          continuous-symmetry-measure cutoffs for naming vs other.
    GEOMETRY_SOURCES        provenance enum for how geometry was obtained.

WHY BOTH `other` AND `unknown`
    `other` means the geometry was computed and matches nothing named; `unknown` means
    it was never computed or classification failed. Never emit ERROR/TIMEOUT.
"""

from __future__ import annotations

from types import MappingProxyType

# Ordered for stable JSON Schema enums and never-shrink tests.
POLYHEDRA: tuple[str, ...] = (
    "single",
    "linear",
    "angular",
    "trigonal_planar",
    "trigonal_pyramidal",
    "tetrahedral",
    "square_planar",
    "seesaw",
    "trigonal_bipyramidal",
    "square_pyramidal",
    "octahedral",
    "trigonal_prismatic",
    "pentagonal_bipyramidal",
    "capped_octahedral",
    "capped_trigonal_prismatic",
    "cubic",
    "square_antiprismatic",
    "dodecahedral",
    "bicapped_trigonal_prismatic",
    "tricapped_trigonal_prismatic",
    "capped_square_antiprismatic",
    "bicapped_square_antiprismatic",
    "icosahedral",
    "cuboctahedral",
    "other",
    "unknown",
)

POLYHEDRON_CN: MappingProxyType[str, int] = MappingProxyType(
    {
        "single": 1,
        "linear": 2,
        "angular": 2,
        "trigonal_planar": 3,
        "trigonal_pyramidal": 3,
        "tetrahedral": 4,
        "square_planar": 4,
        "seesaw": 4,
        "trigonal_bipyramidal": 5,
        "square_pyramidal": 5,
        "octahedral": 6,
        "trigonal_prismatic": 6,
        "pentagonal_bipyramidal": 7,
        "capped_octahedral": 7,
        "capped_trigonal_prismatic": 7,
        "cubic": 8,
        "square_antiprismatic": 8,
        "dodecahedral": 8,
        "bicapped_trigonal_prismatic": 8,
        "tricapped_trigonal_prismatic": 9,
        "capped_square_antiprismatic": 9,
        "bicapped_square_antiprismatic": 10,
        "icosahedral": 12,
        "cuboctahedral": 12,
    }
)

CN_TO_POLYHEDRA: MappingProxyType[int, tuple[str, ...]] = MappingProxyType(
    {
        cn: tuple(name for name, ideal in POLYHEDRON_CN.items() if ideal == cn)
        for cn in sorted({ideal for ideal in POLYHEDRON_CN.values()})
    }
)

CHEMENV_TO_POLYHEDRON: MappingProxyType[str, str] = MappingProxyType(
    {
        "S:1": "single",
        "L:2": "linear",
        "A:2": "angular",
        "TL:3": "trigonal_planar",
        "TY:3": "trigonal_pyramidal",
        "T:4": "tetrahedral",
        "S:4": "square_planar",
        "SS:4": "seesaw",
        "TB:5": "trigonal_bipyramidal",
        "SP:5": "square_pyramidal",
        "O:6": "octahedral",
        "PP:6": "trigonal_prismatic",
        "PB:7": "pentagonal_bipyramidal",
        "CO:7": "capped_octahedral",
        "CTP:7": "capped_trigonal_prismatic",
        "C:8": "cubic",
        "SA:8": "square_antiprismatic",
        "DDPN:8": "dodecahedral",
        # Square-face bicapped trigonal prism — common ChemEnv label for Zr6 sites.
        "SBT:8": "bicapped_trigonal_prismatic",
        "BTPR:8": "bicapped_trigonal_prismatic",
        "TCTPR:9": "tricapped_trigonal_prismatic",
        "CSA:9": "capped_square_antiprismatic",
        "BSA:10": "bicapped_square_antiprismatic",
        "IC:12": "icosahedral",
        "CU:12": "cuboctahedral",
    }
)

# CSM cutoff above which a site is labelled `other` rather than the nearest named
# polyhedron. Distorted MOF nodes are the norm; keep the threshold generous and
# auditable (Phase 08.2 may tune per-CN).
# Chosen from a stratified QMOF ChemEnv sample (n=8300 metal sites, 2029 structures):
# CSM P50≈1.40, P90≈6.76, P95≈8.59; ~6% of sites exceed 8.0. See
# project_notes/phases/08_csm_sample.json and 08_deconstruction_pipeline.md §08.2.
CSM_THRESHOLDS: MappingProxyType[str, float] = MappingProxyType(
    {
        "default": 8.0,
        "named_max": 8.0,
    }
)

GEOMETRY_SOURCES: tuple[str, ...] = (
    "chemenv",
    "order_parameters",
    "cn_heuristic",
    "literature",
    "llm",
)


def map_chemenv_symbol(symbol: str | None) -> str:
    """Map a ChemEnv symbol to a POLYHEDRA name; unmapped -> other; None/empty -> unknown."""
    if not symbol:
        return "unknown"
    return CHEMENV_TO_POLYHEDRON.get(symbol, "other")


__all__ = [
    "CHEMENV_TO_POLYHEDRON",
    "CN_TO_POLYHEDRA",
    "CSM_THRESHOLDS",
    "GEOMETRY_SOURCES",
    "POLYHEDRA",
    "POLYHEDRON_CN",
    "map_chemenv_symbol",
]
