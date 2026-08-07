"""Cell and symmetry normalisation: primitive, conventional, Niggli, P1 expansion.

Phase 07.5.
"""

from __future__ import annotations

from typing import Any, Literal

from mofjson.convert.pymatgen_bridge import (
    MOFFromStructure,
    MOFToStructure,
    SymmetryFromStructure,
)
from mofjson.exceptions import ConversionError
from mofjson.schema.crystal import Symmetry
from mofjson.schema.document import MOFJSONDocument
from mofjson.schema.metadata import PipelineStep

NormalizeTarget = Literal["primitive", "conventional", "niggli", "none", "p1"]


def _require_pymatgen() -> None:
    try:
        import pymatgen  # noqa: F401
    except ImportError as exc:
        raise ConversionError(
            "pymatgen is required for normalisation; install with `pip install mofjson[crystal]`",
        ) from exc


def DetectSymmetry(document: MOFJSONDocument, *, symprec: float = 0.01) -> Symmetry:
    """Detect symmetry of the crystal layer via spglib."""
    structure = MOFToStructure(document)
    return SymmetryFromStructure(structure, symprec=symprec)


def _from_transformed(
    document: MOFJSONDocument,
    structure: Any,
    *,
    step: str,
    symprec: float,
    derive_bonds: bool = True,
) -> MOFJSONDocument:
    """Rebuild document from a transformed Structure, carrying identifiers forward."""
    # Prefer re-deriving bonds so indices stay consistent with new site order.
    new_doc = MOFFromStructure(
        structure,
        derive_bonds=derive_bonds and bool(document.crystal and document.crystal.bonds),
        charge_method=document.crystal.charge_method if document.crystal else None,
        symprec=symprec,
        source_path=None,
        warnings=list(document.warnings),
    )
    new_doc.identifiers = document.identifiers
    # Carry charges by element+approx frac match if counts match and species match
    pipe = list(document.provenance.pipeline) + [
        PipelineStep(step=step, tool="pymatgen", params={"symprec": symprec})
    ]
    new_doc.provenance = new_doc.provenance.model_copy(
        update={
            "pipeline": pipe,
            "warnings": list(document.provenance.warnings) + list(new_doc.provenance.warnings),
            "sources": list(document.provenance.sources),
        }
    )
    new_doc.warnings = list(dict.fromkeys(list(document.warnings) + list(new_doc.warnings)))
    return new_doc


def ToP1(document: MOFJSONDocument, *, symprec: float = 0.01) -> MOFJSONDocument:
    """Expand to a P1 representation (all sites explicit, identity symop)."""
    _require_pymatgen()
    if document.crystal is None:
        raise ConversionError("document has no crystal layer", field_path="crystal")
    structure = MOFToStructure(document)
    # Sites are already Cartesian/fractional positions; mark symmetry as P1 expanded.
    # If not already expanded, use SpacegroupAnalyzer to get conventional and keep all sites.
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        # get_symmetrized_structure then expand — or simply keep structure and rewrite symmetry
        structure = sga.get_refined_structure()
    except Exception:
        pass
    new_doc = _from_transformed(document, structure, step="ToP1", symprec=symprec)
    if new_doc.crystal is not None:
        sym = new_doc.crystal.symmetry.model_copy(
            update={
                "is_p1_expanded": True,
                "symmetry_operations": None,
                "space_group_number": 1,
                "space_group_symbol": "P1",
                "crystal_system": "triclinic",
            }
        )
        # Keep detected number in declared if we wiped it — actually DetectSymmetry separately.
        # Better: keep detected group in space_group_* but mark is_p1_expanded.
        detected = DetectSymmetry(document, symprec=symprec)
        sym = new_doc.crystal.symmetry.model_copy(
            update={
                "space_group_number": detected.space_group_number,
                "space_group_symbol": detected.space_group_symbol,
                "crystal_system": detected.crystal_system,
                "is_p1_expanded": True,
                "symmetry_operations": None,
                "symprec": symprec,
            }
        )
        new_doc.crystal = new_doc.crystal.model_copy(update={"symmetry": sym})
    return new_doc


def ToPrimitive(document: MOFJSONDocument, *, symprec: float = 0.01) -> MOFJSONDocument:
    """Reduce to a primitive cell."""
    _require_pymatgen()
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure = MOFToStructure(document)
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        prim = sga.get_primitive_standard_structure()
    except Exception as exc:
        raise ConversionError(f"ToPrimitive failed: {exc}") from exc
    return _from_transformed(document, prim, step="ToPrimitive", symprec=symprec)


def ToConventional(document: MOFJSONDocument, *, symprec: float = 0.01) -> MOFJSONDocument:
    """Convert to the conventional standard setting."""
    _require_pymatgen()
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure = MOFToStructure(document)
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        conv = sga.get_conventional_standard_structure()
    except Exception as exc:
        raise ConversionError(f"ToConventional failed: {exc}") from exc
    return _from_transformed(document, conv, step="ToConventional", symprec=symprec)


def ToNiggli(document: MOFJSONDocument, *, symprec: float = 0.01) -> MOFJSONDocument:
    """Niggli-reduce the lattice."""
    _require_pymatgen()
    structure = MOFToStructure(document)
    try:
        reduced = structure.get_reduced_structure(reduction_algo="niggli")
    except Exception as exc:
        raise ConversionError(f"ToNiggli failed: {exc}") from exc
    return _from_transformed(document, reduced, step="ToNiggli", symprec=symprec)


def Normalize(
    document: MOFJSONDocument,
    *,
    to: NormalizeTarget = "primitive",
    symprec: float = 0.01,
) -> MOFJSONDocument:
    """Dispatcher matching the phase-note ``normalize(doc, to=...)`` API."""
    if to == "none":
        return document
    if to == "primitive":
        return ToPrimitive(document, symprec=symprec)
    if to == "conventional":
        return ToConventional(document, symprec=symprec)
    if to == "niggli":
        return ToNiggli(document, symprec=symprec)
    if to == "p1":
        return ToP1(document, symprec=symprec)
    raise ConversionError(f"Unknown normalisation target: {to!r}", field_path="to", value=to)


__all__ = [
    "DetectSymmetry",
    "Normalize",
    "NormalizeTarget",
    "ToConventional",
    "ToNiggli",
    "ToP1",
    "ToPrimitive",
]
