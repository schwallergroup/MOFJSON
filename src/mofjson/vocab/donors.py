"""Donor groups, bridging modes, and removable solvent donors.

Phase 03.4 (schema vocab; classifier uses these in Phase 08).
"""

from __future__ import annotations

from types import MappingProxyType

DONOR_GROUPS: tuple[str, ...] = (
    "carboxylate_O",
    "carboxylate_chelating",
    "phosphonate_O",
    "sulfonate_O",
    "hydroxide_O",
    "oxide_O",
    "aqua_O",
    "alkoxide_O",
    "imidazolate_N",
    "pyridyl_N",
    "amine_N",
    "azolate_N",
    "nitrile_N",
    "thiolate_S",
    "halide",
    "cyanide_C",
    "other",
    "unknown",
)

# Element(s) that actually bond to the metal for each donor group.
DONOR_GROUP_ATOMS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "carboxylate_O": ("O",),
        "carboxylate_chelating": ("O",),
        "phosphonate_O": ("O",),
        "sulfonate_O": ("O",),
        "hydroxide_O": ("O",),
        "oxide_O": ("O",),
        "aqua_O": ("O",),
        "alkoxide_O": ("O",),
        "imidazolate_N": ("N",),
        "pyridyl_N": ("N",),
        "amine_N": ("N",),
        "azolate_N": ("N",),
        "nitrile_N": ("N",),
        "thiolate_S": ("S",),
        "halide": ("F", "Cl", "Br", "I"),
        "cyanide_C": ("C",),
        "other": (),
        "unknown": (),
    }
)

BRIDGING_MODES: tuple[str, ...] = (
    "terminal",
    "mu2",
    "mu3",
    "mu4",
    "mu5",
    "mu6",
)

# Removable / activation ligands used for open-metal-site evidence (Phase 08.4).
# Justified: aqua and common synthesis solvents that leave a vacant site when stripped;
# framework carboxylates / azolates are deliberately excluded.
SOLVENT_DONORS: frozenset[str] = frozenset(
    {
        "aqua_O",
        "alkoxide_O",  # MeOH/EtOH-bound alkoxides when terminal
    }
)

OMS_METHODS: tuple[str, ...] = (
    "coordination_deficit",
    "solvent_removal",
    "literature",
    "unknown",
)


__all__ = [
    "BRIDGING_MODES",
    "DONOR_GROUPS",
    "DONOR_GROUP_ATOMS",
    "OMS_METHODS",
    "SOLVENT_DONORS",
]
