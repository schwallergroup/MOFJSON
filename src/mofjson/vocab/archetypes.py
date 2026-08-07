"""Named secondary-building-unit archetypes.

Phase 03.3 and Phase 08.3.

WHAT THIS FILE CONTAINS
    ARCHETYPES              allowed MetalNode.archetype values (Phase 03.1).
    NODE_KINDS              periodicity class: mononuclear / polynuclear / rod / layer /
                            unknown — whether a finite MolFragment can represent the node.
    ARCHETYPE_EXPECTED_CONNECTIVITY
                            typical linker attachment counts for cross-checks.
    ARCHETYPE_FINGERPRINTS  deferred to Phase 08.3 (classifier rules as data).
"""

from __future__ import annotations

from types import MappingProxyType

ARCHETYPES: tuple[str, ...] = (
    "paddlewheel",
    "hexanuclear_M6O4OH4",
    "basic_metal_carboxylate_M4O",
    "trinuclear_mu3_oxo_M3O",
    "binuclear_mu2_OH",
    "mononuclear_tetrahedral",
    "mononuclear_octahedral",
    "infinite_rod",
    "infinite_layer",
    "dinuclear_M2",
    "polyoxometalate",
    "mixed_metal_cluster",
    "other",
    "unknown",
)

NODE_KINDS: tuple[str, ...] = (
    "mononuclear",
    "polynuclear",
    "rod",
    "layer",
    "unknown",
)

ARCHETYPE_EXPECTED_CONNECTIVITY: MappingProxyType[str, tuple[int, ...]] = MappingProxyType(
    {
        "paddlewheel": (4,),
        "hexanuclear_M6O4OH4": (12, 8, 6),
        "basic_metal_carboxylate_M4O": (6,),
        "trinuclear_mu3_oxo_M3O": (6,),
        "mononuclear_tetrahedral": (4,),
        "mononuclear_octahedral": (2, 4, 6),
        "infinite_rod": (2, 4),
        "dinuclear_M2": (4,),
    }
)

# Declarative classifier rules for Phase 08.3. Order matters: first high-scoring
# match wins in ClassifyNode (scored walk). Rods matched on kind/periodicity.
ARCHETYPE_FINGERPRINTS: tuple[dict, ...] = (
    {
        "archetype": "infinite_rod",
        "periodicity_kinds": ("rod",),
        "min_score": 1.0,
    },
    {
        "archetype": "infinite_layer",
        "periodicity_kinds": ("layer",),
        "min_score": 1.0,
    },
    {
        "archetype": "hexanuclear_M6O4OH4",
        "nuclearity": 6,
        "exact_metal_elements": 1,
        "cn": 8,
        "require_mu3_O": True,
        "require_mu3_OH": True,
        "connectivity_any": (12, 8, 6),
        "min_score": 0.8,
    },
    {
        "archetype": "basic_metal_carboxylate_M4O",
        "nuclearity": 4,
        "exact_metal_elements": 1,
        "cn": 4,
        "require_mu4_O": True,
        "geometry_any": ("tetrahedral",),
        "connectivity_any": (6,),
        "min_score": 0.8,
    },
    {
        "archetype": "trinuclear_mu3_oxo_M3O",
        "nuclearity": 3,
        "exact_metal_elements": 1,
        "cn": 6,
        "require_mu3_O": True,
        "geometry_any": ("octahedral",),
        "connectivity_any": (6,),
        "min_score": 0.8,
    },
    {
        "archetype": "paddlewheel",
        "nuclearity": 2,
        "exact_metal_elements": 1,
        "cn_any": (4, 5),
        "geometry_any": ("square_planar", "square_pyramidal", "other"),
        "donor_group_any": ("carboxylate_O", "carboxylate_chelating"),
        "connectivity_any": (4,),
        "min_score": 0.75,
    },
    {
        "archetype": "mononuclear_tetrahedral",
        "nuclearity": 1,
        "exact_metal_elements": 1,
        "cn": 4,
        "geometry_any": ("tetrahedral",),
        "donor_group_any": ("imidazolate_N", "azolate_N", "pyridyl_N"),
        "min_score": 0.75,
    },
    {
        "archetype": "mononuclear_octahedral",
        "nuclearity": 1,
        "exact_metal_elements": 1,
        "cn": 6,
        "geometry_any": ("octahedral",),
        "min_score": 0.85,
    },
    {
        "archetype": "binuclear_mu2_OH",
        "nuclearity": 2,
        "exact_metal_elements": 1,
        "require_mu2_OH": True,
        "min_score": 0.85,
    },
    {
        "archetype": "dinuclear_M2",
        "nuclearity": 2,
        "exact_metal_elements": 1,
        "min_score": 0.9,
    },
)


__all__ = [
    "ARCHETYPES",
    "ARCHETYPE_EXPECTED_CONNECTIVITY",
    "ARCHETYPE_FINGERPRINTS",
    "NODE_KINDS",
]
