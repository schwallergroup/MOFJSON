
"""
PolymerUnit -- repeat-unit attribute for building blocks that aren't one
finite molecule: rod/sheet/net SBUs (e.g. the -M-O-M-O- chain in
MOF-74/CPO-27) and variable-length linkers. SMILES only describes a
finite molecule, so MolFragment can't express a periodic rod.

Not BigSMILES (Lin, Olsen & Jensen, ACS Cent. Sci. 2019, 5, 1523-1531),
whose stochastic-object notation {[$]...[$]} represents an ensemble of
repeat arrangements and end groups. We take only the idea that a
repeating fragment needs its own representation: `repeat_unit_smiles` is
plain RDKit SMILES with '*' attachment points (as elsewhere, see
molBuilder.py), and whether it repeats lives in the typed `is_infinite` /
`periodic_directions` fields rather than in the string.

`periodic_directions` is plural because Batten et al. (IUPAC, Pure Appl.
Chem. 2013, 85, 1715-1724) classify coordination polymers by degree of
extension. A rod repeating along one lattice vector and a sheet repeating
along two are both infinite but not the same object.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PolymerUnit(BaseModel):
    repeat_unit_smiles: str = Field(
        min_length=1,
        description=(
            "Plain RDKit SMILES of the repeating fragment, with '*' dummy "
            "atoms at attachment points (the same convention every other "
            "fragment in this schema uses). NOT BigSMILES stochastic-object "
            "notation -- see module docstring for why."
        ),
    )
    is_infinite: bool = Field(
        description=(
            "True for a genuine periodic repeat (rod/sheet/net SBU, infinite "
            "chain). False for a finite but variable-length/disordered oligomer."
        )
    )
    periodic_directions: list[Literal["a", "b", "c"]] | None = Field(
        default=None,
        description=(
            "Lattice vector(s) the repeat propagates along. Required, "
            "non-empty, and duplicate-free when is_infinite is True. One "
            "axis = a rod (1D), two = a sheet (2D), three = a 3D net."
        ),
    )
    degree_of_polymerization: int | None = Field(
        default=None,
        ge=1,
        description="Number of repeat units, when finite and known. Must be omitted when is_infinite is True.",
    )
    end_groups: list[str] | None = Field(
        default=None,
        description="End-group SMILES fragments. Must be omitted when is_infinite is True (no ends on an infinite repeat).",
    )

    @property
    def dimensionality(self) -> int | None:
        """1, 2, or 3 for an infinite repeat; None for a finite oligomer
        (degree_of_polymerization measures size there, not dimensionality)."""
        return len(self.periodic_directions) if self.periodic_directions else None

    @model_validator(mode="after")
    def _infinite_is_consistent(self) -> "PolymerUnit":
        if self.is_infinite:
            if not self.periodic_directions:
                raise ValueError("periodic_directions is required (non-empty) when is_infinite is True")
            if len(set(self.periodic_directions)) != len(self.periodic_directions):
                raise ValueError(f"periodic_directions must not repeat an axis: {self.periodic_directions!r}")
            if self.degree_of_polymerization is not None:
                raise ValueError("degree_of_polymerization must be omitted when is_infinite is True")
            if self.end_groups is not None:
                raise ValueError("end_groups must be omitted when is_infinite is True (no ends on an infinite repeat)")
        else:
            if self.periodic_directions is not None:
                raise ValueError("periodic_directions must be omitted when is_infinite is False")
            if self.degree_of_polymerization is None:
                raise ValueError("degree_of_polymerization is required when is_infinite is False")
        return self


__all__ = ["PolymerUnit"]
