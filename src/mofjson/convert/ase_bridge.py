"""ASE Atoms <-> MOFJSON crystal layer.

Phase 07.3.
"""

from __future__ import annotations

from typing import Any

from mofjson.convert.pymatgen_bridge import MOFFromStructure, MOFToStructure
from mofjson.exceptions import ConversionError
from mofjson.schema.document import MOFJSONDocument
from mofjson.schema.metadata import Identifiers, PipelineStep


def _require_ase() -> Any:
    try:
        import ase
        from ase import Atoms
    except ImportError as exc:
        raise ConversionError(
            "ase is required; install with `pip install mofjson[crystal]`",
        ) from exc
    return Atoms


def _atoms_to_structure(atoms: Any) -> Any:
    """Convert ASE Atoms to pymatgen Structure via ASE→pymatgen bridge."""
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise ConversionError(
            "pymatgen is required for ASE bridge; install with `pip install mofjson[crystal]`",
        ) from exc
    return AseAtomsAdaptor.get_structure(atoms)


def _structure_to_atoms(structure: Any) -> Any:
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise ConversionError(
            "pymatgen is required for ASE bridge; install with `pip install mofjson[crystal]`",
        ) from exc
    return AseAtomsAdaptor.get_atoms(structure)


def _identifiers_from_atoms_info(info: dict[str, Any]) -> Identifiers:
    """Map common ODAC / ASE info keys into Identifiers."""
    kwargs: dict[str, Any] = {}
    if "qmof_id" in info and info["qmof_id"] is not None:
        kwargs["qmof_id"] = str(info["qmof_id"])
    if "mofid" in info and info["mofid"] is not None:
        kwargs["mofid"] = str(info["mofid"])
    if "odac_name" in info and info["odac_name"] is not None:
        kwargs["odac_name"] = str(info["odac_name"])
    elif "name" in info and info["name"] is not None:
        kwargs["odac_name"] = str(info["name"])
    if "refcode" in info and info["refcode"] is not None:
        kwargs["csd_refcode"] = str(info["refcode"])
    if "csd_refcode" in info and info["csd_refcode"] is not None:
        kwargs["csd_refcode"] = str(info["csd_refcode"])
    if "structure_stem" in info and info["structure_stem"] is not None:
        kwargs["structure_stem"] = str(info["structure_stem"])
    return Identifiers(**kwargs)


def MOFFromAtoms(
    atoms: Any,
    *,
    derive_bonds: bool = True,
    symprec: float = 0.01,
    source_path: str | None = None,
) -> MOFJSONDocument:
    """Convert ASE Atoms to MOFJSONDocument.

    ASE has no symmetry concept: structures are P1 by construction.
    """
    _require_ase()
    structure = _atoms_to_structure(atoms)
    info = dict(getattr(atoms, "info", {}) or {})
    doc = MOFFromStructure(
        structure,
        derive_bonds=derive_bonds,
        symprec=symprec,
        declared_space_group_symbol="P1",
        is_p1_expanded=True,
        source_path=source_path,
        warnings=["ASE Atoms has no symmetry; recorded as P1 by construction"],
    )
    # Identifiers from atoms.info
    try:
        ids = _identifiers_from_atoms_info(info)
        doc.identifiers = ids
    except Exception:
        pass
    step = PipelineStep(
        step="MOFFromAtoms",
        tool="ase",
        params={"derive_bonds": derive_bonds, "symprec": symprec, "info_keys": sorted(info)},
    )
    doc.provenance = doc.provenance.model_copy(
        update={"pipeline": list(doc.provenance.pipeline) + [step]}
    )
    # Force symmetry number/symbol consistency for ASE path
    if doc.crystal is not None:
        sym = doc.crystal.symmetry.model_copy(
            update={
                "declared_space_group_symbol": "P1",
                "is_p1_expanded": True,
            }
        )
        doc.crystal = doc.crystal.model_copy(update={"symmetry": sym})
    return doc


def MOFToAtoms(document: MOFJSONDocument) -> Any:
    """Convert MOFJSONDocument crystal layer to ASE Atoms."""
    _require_ase()
    structure = MOFToStructure(document)
    atoms = _structure_to_atoms(structure)
    # Preserve identifiers into info
    info = dict(atoms.info)
    ids = document.identifiers
    if ids.qmof_id is not None:
        info["qmof_id"] = ids.qmof_id
    if ids.odac_name is not None:
        info["odac_name"] = ids.odac_name
        info["name"] = ids.odac_name
    if ids.csd_refcode is not None:
        info["csd_refcode"] = ids.csd_refcode
        info["refcode"] = ids.csd_refcode
    if ids.mofid is not None:
        info["mofid"] = ids.mofid
    if ids.structure_stem is not None:
        info["structure_stem"] = ids.structure_stem
    atoms.info = info
    return atoms


__all__ = ["MOFFromAtoms", "MOFToAtoms"]
