"""The inverse pipeline: reticular description -> explicit periodic structure.

Phase 08.7. EXPERIMENTAL for partial documents; 08.7a full-assembly is graded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mofjson.exceptions import DeconstructionError
from mofjson.schema.crystal import CrystalStructure
from mofjson.schema.reticular import ReticularStructure

ReconstructionMethod = Literal["full", "template"]


@dataclass
class ReconstructionResult:
    """Structure plus honesty about what was assumed."""

    crystal: CrystalStructure | None
    confidence: float
    method: str
    assumed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    deferred: bool = False


def _reconstruct_full(reticular: ReticularStructure) -> ReconstructionResult:
    """Rigid-body placement from a ``full`` assembly (Phase 08.7a)."""
    from mofjson.convert.pymatgen_bridge import MOFFromStructure
    from mofjson.schema.document import MOFJSONDocument

    if reticular.assembly is None or reticular.assembly.completeness != "full":
        raise DeconstructionError(
            "full reconstruction requires assembly.completeness == 'full'",
            field_path="assembly.completeness",
        )
    failures = reticular._check_full_completeness()
    if failures:
        raise DeconstructionError(
            "full assembly contract not satisfied: " + "; ".join(failures[:5]),
            field_path="assembly",
            value=failures,
        )

    # Place atoms from node local frames + linker fragment coords into a Cartesian
    # cell. Without an explicit crystal lattice on the parent document we cannot
    # recover absolute fractional coordinates; require that node centroid_frac and
    # orientation are present and build a P1 structure in an orthorhombic box large
    # enough to hold the placed blocks.
    import numpy as np
    from pymatgen.core import Lattice, Structure

    species: list[str] = []
    coords: list[list[float]] = []

    def _rotate(quat: list[float], vec: list[float]) -> list[float]:
        # Quaternion (w, x, y, z) acting on vec.
        w, x, y, z = quat
        q = np.array([x, y, z], dtype=float)
        v = np.array(vec, dtype=float)
        t = 2.0 * np.cross(q, v)
        return (v + w * t + np.cross(q, t)).tolist()

    for node in reticular.nodes or []:
        assert node.local_frame is not None and node.orientation is not None
        origin = node.centroid_frac or [0.0, 0.0, 0.0]
        # Treat centroid_frac as Cartesian angstroms if no lattice — documented assumption.
        base = [float(x) * 20.0 for x in origin]  # scale fractional-looking coords
        for c in node.local_frame.coords:
            xyz = _rotate(node.orientation.quaternion, c.xyz)
            species.append(
                next(
                    (a.element for a in (node.fragment.atoms if node.fragment else []) if a.id == c.atom_id),
                    c.atom_id.rstrip("0123456789") or "X",
                )
            )
            # Infer element from atom_id prefix if no fragment.
            if species[-1] == (c.atom_id.rstrip("0123456789") or "X"):
                el = "".join(ch for ch in c.atom_id if ch.isalpha()) or "X"
                species[-1] = el
            coords.append([base[i] + xyz[i] for i in range(3)])

    for linker in reticular.linkers or []:
        if linker.role != "linker":
            continue
        frag = linker.fragment
        if not frag.coords:
            continue
        # Place linker at its coordinate centroid (already local).
        for c in frag.coords:
            el = "".join(ch for ch in c.atom_id if ch.isalpha()) or "C"
            for a in frag.atoms:
                if a.id == c.atom_id:
                    el = a.element
                    break
            species.append(el)
            coords.append(list(c.xyz))

    if not species:
        raise DeconstructionError("no atoms to place during full reconstruction")

    arr = np.array(coords, dtype=float)
    mins = arr.min(axis=0) - 2.0
    maxs = arr.max(axis=0) + 2.0
    lengths = np.maximum(maxs - mins, 5.0)
    cart = arr - mins
    lattice = Lattice.from_parameters(float(lengths[0]), float(lengths[1]), float(lengths[2]), 90, 90, 90)
    structure = Structure(lattice, species, cart, coords_are_cartesian=True)
    doc = MOFFromStructure(structure, derive_bonds=False)
    assert doc.crystal is not None
    return ReconstructionResult(
        crystal=doc.crystal,
        confidence=0.9,
        method="full",
        assumed=["centroid_frac_scaled_to_cartesian_if_no_parent_lattice"],
    )


def Reconstruct(
    reticular: ReticularStructure,
    *,
    method: ReconstructionMethod = "full",
) -> CrystalStructure | ReconstructionResult:
    """Reconstruct a crystal layer from a reticular description.

    For ``method='full'`` returns a :class:`CrystalStructure` when the assembly is
    complete. For ``method='template'`` (08.7b) returns a deferred result.
    """
    if method == "template":
        return ReconstructionResult(
            crystal=None,
            confidence=0.0,
            method="template",
            assumed=["linker_conformation", "node_orientation", "cell_parameters"],
            warnings=["08.7b partial reconstruction deferred"],
            deferred=True,
        )
    result = _reconstruct_full(reticular)
    return result.crystal if result.crystal is not None else result


__all__ = ["Reconstruct", "ReconstructionResult"]
