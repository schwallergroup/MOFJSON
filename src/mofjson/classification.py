"""
IUPAC classification tag (Batten et al., "Terminology of metal-organic
frameworks and coordination polymers", IUPAC Recommendations 2013, Pure
Appl. Chem. 85, 1715-1724).

The paper defines a strict hierarchy, each term a subset of the last:

  coordination polymer  -- most general: a coordination compound with
                           repeating coordination entities extending in
                           1, 2, or 3 dimensions.
  coordination network  -- SUBSET: either (a) 1D but with cross-links
                           between chains, loops, or spiro-links, or
                           (b) extends in 2 or 3 dimensions.
  metal-organic framework (mof) -- FURTHER SUBSET of coordination network:
                           organic ligands, AND contains potential voids.
                           Crystallinity is NOT required, and porosity does
                           NOT need to be measured -- "potential" voids is
                           the IUPAC-defining structural criterion, not an
                           experimental one.

"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ClassificationTag(BaseModel):
    term: Literal["coordination_polymer", "coordination_network", "mof"]
    dimensionality: Literal[1, 2, 3] = Field(
        description="Degree of extension of the coordination polymer (Batten et al. 2013)."
    )
    has_potential_voids: bool | None = Field(
        default=None,
        description="IUPAC's defining structural criterion for 'mof' specifically. Required True when term == 'mof'.",
    )
    source: Literal["manual", "heuristic"] = "manual"

    @model_validator(mode="after")
    def _mof_requires_voids(self) -> "ClassificationTag":
        if self.term == "mof" and self.has_potential_voids is not True:
            raise ValueError(
                "term == 'mof' requires has_potential_voids == True -- IUPAC's own "
                "defining criterion for the term, not optional metadata."
            )
        return self


__all__ = ["ClassificationTag"]
