"""
Paste a CIF path in (CIF_PATH below, or as argv[1]) and this gives you two
things straight back:

  1. The full MOFJSON schema for that structure -- building_blocks,
     net_edges, topology, lattice_encoding -- written to <stem>.mofjson.json.
  2. A reconstructed CIF, rebuilt FROM the MOFJSON (not copied from the
     original), written to <stem>_reconstructed.cif -- plus a pass/fail on
     whether it actually matches the original structure.

This is a trimmed-down version of test_document.py's step 0/1 (role
assignment + BuildDocument) with the internal-consistency demo steps
(2-8: tamper detection, PolymerUnit/Assembly/ClassificationTag invariants,
lattice_encoding round trip, schema-shape comparison, topology deep dive)
removed -- those test that the package's OWN validators work, which is a
different question from "show me the schema and the reconstructed CIF for
this one structure." Same `from src.mofjson import (...)` public-API import
style as test_document.py, same CifParser(..., primitive=False) loading
convention -- nothing about how the structure is read changed, only what
gets printed/written at the end.

Two IMPORTANT things worth understanding about what "reconstructed" means
here, carried over from the same distinction test_document.py's step 6
makes for lattice_encoding specifically:

  - The reconstructed CIF is rebuilt from the FLAT layer only (lattice +
    atoms + bonds, via StructureToJSON/StructureFromJSON) -- this is the
    one genuinely invertible slice of the schema. It is NOT rebuilt from
    building_blocks/net_edges/topology, and it structurally can't be: an
    SBU partition isn't unique, so "here are the building blocks" cannot
    be losslessly inverted back into "here are the exact atom
    coordinates." That information only flows raw -> derived, one way.
  - CheckRoundTrip uses pymatgen's StructureMatcher (lattice + site
    comparison under symmetry operations and numerical tolerance), not a
    string/coordinate equality check -- MOFJSON has no single canonical
    serialization the way MolJSON can compare canonical SMILES strings, so
    "did it round-trip" has to be answered geometrically.

Run against a different structure with:
    python test_document_roundtrip.py path/to/other.cif
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher

from src.mofjson import (
    AssignRoles,
    BuildDocument,
    CheckRoundTrip,
    RawStructure,
    StructureFromJSON,
    StructureToJSON,
)

CIF_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/mofjson_test/Cu-BTC.cif"


def load_structure_and_roles():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = CifParser(CIF_PATH)
        structure = parser.parse_structures(primitive=False)[0]
    mofjson_flat = StructureToJSON(structure, include_empty_fields=True)
    raw = RawStructure(
        lattice=mofjson_flat["lattice"],
        atoms=mofjson_flat["atoms"],
        bonds=mofjson_flat["bonds"],
        charges=mofjson_flat["charges"] or None,
        aromatic_n_h=mofjson_flat["aromatic_n_h"] or None,
    )
    roles = AssignRoles(mofjson_flat["atoms"], mofjson_flat["bonds"])
    return structure, mofjson_flat, raw, roles


def main() -> None:
    cif_path = Path(CIF_PATH)
    stem = cif_path.stem
    print(f"loading {cif_path} ...")
    structure, mofjson_flat, raw, roles = load_structure_and_roles()

    role_counts: dict[str, int] = {}
    for r in roles:
        role_counts[r["role"]] = role_counts.get(r["role"], 0) + 1
    print(f"role assignment: {role_counts}")

    # --- 1. the full schema -------------------------------------------------
    doc = BuildDocument(raw, roles)
    doc.verify_structurally()  # cheap, and catches a bad derivation before we hand you a wrong schema
    schema_path = cif_path.with_name(f"{stem}.mofjson.json")
    schema_path.write_text(doc.model_dump_json(indent=2))

    topology_net = doc.derived.topology.get("net") if doc.derived.topology else None
    print(
        f"wrote full schema -> {schema_path}  "
        f"({len(doc.derived.building_blocks)} building_blocks, "
        f"{len(doc.derived.net_edges)} net_edges, topology.net={topology_net!r})"
    )

    # --- 2. the reconstructed CIF (from the FLAT layer, see module docstring) ---
    ok, _ = CheckRoundTrip(structure)
    reconstructed = StructureFromJSON(mofjson_flat)
    cif_out_path = cif_path.with_name(f"{stem}_reconstructed.cif")
    reconstructed.to(filename=str(cif_out_path), fmt="cif")
    print(f"wrote reconstructed CIF -> {cif_out_path}")
    print(f"round trip (StructureMatcher.fit, original vs. reconstructed): {'PASS' if ok else 'FAIL'}")

    if not ok:
        matcher = StructureMatcher()
        rms = matcher.get_rms_dist(structure, reconstructed)
        print(f"  get_rms_dist: {rms} (None = matcher couldn't align them at all, e.g. different composition)")
        print(f"  original:      formula={structure.composition.reduced_formula!r}, sites={len(structure)}, "
              f"lattice=({structure.lattice.a:.4f}, {structure.lattice.b:.4f}, {structure.lattice.c:.4f})")
        print(f"  reconstructed: formula={reconstructed.composition.reduced_formula!r}, sites={len(reconstructed)}, "
              f"lattice=({reconstructed.lattice.a:.4f}, {reconstructed.lattice.b:.4f}, {reconstructed.lattice.c:.4f})")


if __name__ == "__main__":
    main()