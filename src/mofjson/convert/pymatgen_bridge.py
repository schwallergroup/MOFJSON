"""pymatgen Structure <-> MOFJSON crystal layer.

Phase 07.3.

Convention: PeriodicBond.image == StructureGraph to_jimage (lattice translation of target).
"""

from __future__ import annotations

from typing import Any

from mofjson.exceptions import ConversionError
from mofjson.schema.crystal import (
    CrystalStructure,
    Lattice,
    PeriodicBond,
    Site,
    Symmetry,
    crystal_system_from_number,
    generate_site_ids,
    normalize_space_group_symbol,
)
from mofjson.schema.document import MOFJSONDocument
from mofjson.schema.metadata import DocumentProvenance, PipelineStep, SourceRef


def _require_pymatgen() -> None:
    try:
        import pymatgen  # noqa: F401
    except ImportError as exc:
        raise ConversionError(
            "pymatgen is required; install with `pip install mofjson[crystal]`",
        ) from exc


def _spglib_symmetry(
    structure: Any,
    *,
    symprec: float = 0.01,
) -> tuple[int, str, str]:
    """Return (number, symbol, crystal_system) from spglib via pymatgen."""
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        number = int(sga.get_space_group_number())
        symbol = normalize_space_group_symbol(sga.get_space_group_symbol())
        system = crystal_system_from_number(number)
        return number, symbol, system
    except Exception:
        return 1, "P1", "triclinic"


def LatticeFromStructure(structure: Any) -> Lattice:
    """Build Lattice from a pymatgen Structure."""
    _require_pymatgen()
    lat = structure.lattice
    return Lattice(
        a=lat.a,
        b=lat.b,
        c=lat.c,
        alpha=lat.alpha,
        beta=lat.beta,
        gamma=lat.gamma,
        volume=lat.volume,
        matrix=[[float(x) for x in row] for row in lat.matrix.tolist()],
    )


def SitesFromStructure(structure: Any) -> list[Site]:
    """Build Site list with stable MolJSON-style ids."""
    elements = [str(site.specie.symbol) for site in structure]
    ids = generate_site_ids(elements)
    sites: list[Site] = []
    for i, site in enumerate(structure):
        occ = 1.0
        try:
            # Ordered: single specie with occupancy 1; disordered: Composition weights.
            if hasattr(site, "species") and hasattr(site.species, "values"):
                vals = list(site.species.values())
                if len(vals) == 1:
                    occ = float(vals[0])
                else:
                    occ = float(sum(vals))
        except Exception:
            occ = 1.0
        if occ <= 0 or occ > 1:
            occ = 1.0
        props = getattr(site, "properties", {}) or {}
        charge = props.get("charge")
        magmom = props.get("magmom")
        label = props.get("label")
        sites.append(
            Site(
                id=ids[i],
                element=elements[i],
                frac=[float(x) for x in site.frac_coords],
                occupancy=occ,
                label=str(label) if label else None,
                charge=float(charge) if charge is not None else None,
                magmom=float(magmom) if magmom is not None else None,
            )
        )
    return sites


def SymmetryFromStructure(
    structure: Any,
    *,
    symprec: float = 0.01,
    declared_space_group_symbol: str | None = None,
    is_p1_expanded: bool | None = None,
) -> Symmetry:
    """Detect symmetry with spglib; record declared CIF symbol separately."""
    number, symbol, system = _spglib_symmetry(structure, symprec=symprec)
    # Heuristic: if declared is P1 but spglib finds higher, sites are P1-expanded.
    declared = (
        normalize_space_group_symbol(declared_space_group_symbol)
        if declared_space_group_symbol
        else None
    )
    if is_p1_expanded is None:
        if (declared in ("P1", "P-1") and number > 1) or number == 1:
            is_p1_expanded = True
        else:
            is_p1_expanded = False
    ops: list[str] | None = None
    if not is_p1_expanded:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            sga = SpacegroupAnalyzer(structure, symprec=symprec)
            ops = [op.as_xyz_str() for op in sga.get_symmetry_operations(cartesian=False)]
        except Exception:
            ops = ["x,y,z"]
    return Symmetry(
        space_group_number=number,
        space_group_symbol=symbol,
        crystal_system=system,
        is_p1_expanded=bool(is_p1_expanded),
        symmetry_operations=ops,
        symprec=symprec,
        declared_space_group_symbol=declared,
    )


def MOFToStructure(document: MOFJSONDocument | CrystalStructure) -> Any:
    """Convert MOFJSON crystal layer to a pymatgen Structure."""
    _require_pymatgen()
    from pymatgen.core import Lattice as PmgLattice
    from pymatgen.core import Structure

    crystal = document if isinstance(document, CrystalStructure) else document.crystal
    if crystal is None:
        raise ConversionError("document has no crystal layer", field_path="crystal")
    matrix = crystal.lattice.to_matrix()
    pmg_lat = PmgLattice(matrix)
    species = [s.element for s in crystal.sites]
    coords = [list(s.frac) for s in crystal.sites]
    props: dict[str, list[Any]] = {"mofjson_id": [s.id for s in crystal.sites]}
    charges = [s.charge for s in crystal.sites]
    if any(c is not None for c in charges):
        props["charge"] = [0.0 if c is None else c for c in charges]
    magmoms = [s.magmom for s in crystal.sites]
    if any(m is not None for m in magmoms):
        props["magmom"] = [0.0 if m is None else m for m in magmoms]
    # Site properties as lists
    structure = Structure(
        pmg_lat,
        species,
        coords,
        coords_are_cartesian=False,
        site_properties={k: v for k, v in props.items()},
    )
    # Apply occupancies for partial sites via replace if needed
    for i, site in enumerate(crystal.sites):
        if abs(site.occupancy - 1.0) > 1e-9:
            structure.replace(
                i,
                {site.element: site.occupancy},
                properties=structure[i].properties,
            )
    return structure


def MOFFromStructure(
    structure: Any,
    *,
    derive_bonds: bool = True,
    charge_method: str | None = None,
    symprec: float = 0.01,
    declared_space_group_symbol: str | None = None,
    is_p1_expanded: bool | None = None,
    site_charges: list[float | None] | None = None,
    site_magmoms: list[float | None] | None = None,
    source_path: str | None = None,
    warnings: list[str] | None = None,
) -> MOFJSONDocument:
    """Build a MOFJSONDocument from a pymatgen Structure."""
    _require_pymatgen()
    warn = list(warnings or [])
    lattice = LatticeFromStructure(structure)
    symmetry = SymmetryFromStructure(
        structure,
        symprec=symprec,
        declared_space_group_symbol=declared_space_group_symbol,
        is_p1_expanded=is_p1_expanded,
    )
    sites = SitesFromStructure(structure)
    if site_charges is not None:
        if len(site_charges) != len(sites):
            raise ConversionError(
                "site_charges length mismatch",
                field_path="site_charges",
                value={"n": len(site_charges), "n_sites": len(sites)},
            )
        sites = [
            s.model_copy(update={"charge": site_charges[i]})
            for i, s in enumerate(sites)
        ]
    if site_magmoms is not None:
        if len(site_magmoms) != len(sites):
            raise ConversionError("site_magmoms length mismatch", field_path="site_magmoms")
        sites = [
            s.model_copy(update={"magmom": site_magmoms[i]})
            for i, s in enumerate(sites)
        ]

    bonds: list[PeriodicBond] = []
    bond_graph_source = None
    if derive_bonds:
        from mofjson.convert.bonds import DeriveBonds

        bonds, bond_warns = DeriveBonds(
            structure,
            method="crystalnn",
            site_ids=[s.id for s in sites],
        )
        warn.extend(bond_warns)
        bond_graph_source = "crystalnn"

    # No hydrogens?
    if not any(s.element == "H" for s in sites):
        msg = "no hydrogens present"
        if msg not in warn:
            warn.append(msg)

    crystal = CrystalStructure(
        lattice=lattice,
        symmetry=symmetry,
        sites=sites,
        bonds=bonds,
        charge_method=charge_method,
        bond_graph_source=bond_graph_source,
        coordinate_precision=4,
    )

    provenance = DocumentProvenance(
        sources=[SourceRef(name=source_path, kind="structure", file=source_path)]
        if source_path
        else [],
        pipeline=[
            PipelineStep(
                step="MOFFromStructure",
                tool="pymatgen",
                params={"derive_bonds": derive_bonds, "symprec": symprec},
            )
        ],
        warnings=list(warn),
    )
    return MOFJSONDocument(crystal=crystal, provenance=provenance, warnings=list(warn))


def MOFToStructureGraph(document: MOFJSONDocument | CrystalStructure) -> Any:
    """Build a StructureGraph from crystal bonds.

    ``PeriodicBond.image`` maps directly to StructureGraph ``to_jimage``.
    """
    _require_pymatgen()
    from pymatgen.analysis.graphs import StructureGraph

    structure = MOFToStructure(document)
    crystal = document if isinstance(document, CrystalStructure) else document.crystal
    assert crystal is not None
    # Start empty graph then add edges from our bonds for exact image control
    sg = StructureGraph.from_empty_graph(structure, name="bonds")
    id_to_idx = {s.id: i for i, s in enumerate(crystal.sites)}
    for bond in crystal.bonds:
        i = id_to_idx[bond.source]
        j = id_to_idx[bond.target]
        to_jimage = bond.image_tuple()
        sg.add_edge(i, j, to_jimage=to_jimage, weight=bond.order)
    return sg


def MOFFromStructureGraph(
    graph: Any,
    *,
    charge_method: str | None = None,
    symprec: float = 0.01,
    source_path: str | None = None,
) -> MOFJSONDocument:
    """Build MOFJSON from a StructureGraph (bonds already known)."""
    _require_pymatgen()
    structure = graph.structure
    doc = MOFFromStructure(
        structure,
        derive_bonds=False,
        charge_method=charge_method,
        symprec=symprec,
        source_path=source_path,
    )
    assert doc.crystal is not None
    ids = [s.id for s in doc.crystal.sites]
    bonds: list[PeriodicBond] = []
    # Iterate unique edges
    for edge in graph.graph.edges(data=True):
        # networkx: (u, v, data) — StructureGraph stores to_jimage in data
        u, v, data = edge
        to_jimage = data.get("to_jimage") or data.get("to_jimage ", (0, 0, 0))
        if to_jimage is None:
            to_jimage = (0, 0, 0)
        img = tuple(int(x) for x in to_jimage)
        # Prefer the orientation matching edge direction u→v
        weight = data.get("weight", 1.0)
        try:
            order = float(weight) if weight is not None else 1.0
        except (TypeError, ValueError):
            order = 1.0
        if order not in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            order = 1.0
        img_list = None if img == (0, 0, 0) else [img[0], img[1], img[2]]
        bonds.append(
            PeriodicBond(
                source=ids[int(u)],
                target=ids[int(v)],
                order=order,
                image=img_list,
            )
        )
    doc.crystal = doc.crystal.model_copy(
        update={"bonds": bonds, "bond_graph_source": "manual"}
    )
    return doc


__all__ = [
    "LatticeFromStructure",
    "MOFFromStructure",
    "MOFFromStructureGraph",
    "MOFToStructure",
    "MOFToStructureGraph",
    "SitesFromStructure",
    "SymmetryFromStructure",
]
