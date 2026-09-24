"""
MOFJSONDocument -- ties the raw structure, the role-assignment input, and
the derived/cached views together, with the "computed from" relationship
enforced as a validator rather than assumed.

Three tiers, deliberately kept separate:

  raw (CIF files. lattice + atoms + bonds (+ charges, aromatic_n_h))

  roles (node_linker_roles) 

  derived (building_blocks, net_edges, topology, optionally slices. Strictly a function of (raw, roles).)

  lattice_encoding 

  classification 

Two representations exist in this package: this nested, hash-enforced
MOFJSONDocument, and the flatter dict schema.py's GetSchema()/
GetPaperSchema() describe (the shape conversion.py's own StructureToJSON/
CollapseToBuildingBlocks/ExpandNet natively produce/consume). They are NOT
the same schema for the same object -- call schema.GetDocumentSchema() if
you specifically need MOFJSONDocument's own JSON Schema.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

import src.mofjson.conversion as _conversion
from src.mofjson.classification import ClassificationTag
# NOTE: unlike conversion.py (which keeps its own `from rdkit import Chem`
# lazy/local to InferOrganicBondOrders so a broken RDKit install never
# blocks StructureToJSON/StructureFromJSON/CheckNetConsistency), linker.py
# and DetectPolymerUnit.py import RDKit at module level (via
# molBuilder.py), and BuildDocument below always calls both -- so
# importing document.py now requires a working RDKit install. This is an
# intentional tradeoff: SMILES identity and polymer detection are part of
# the default pipeline, not optional add-ons. If you need conversion.py's
# core round-trip machinery in an environment without RDKit, import
# conversion.py directly instead of going through document.py.
from src.mofjson.DetectPolymerUnit import DetectPolymerUnits
from src.mofjson.lattice_encoding import ComputeLatticeEncoding, LatticeEncoding
from src.mofjson.linker import IdentifyLinkers
from src.mofjson.polymer import PolymerUnit
from src.mofjson.provenance import Provenance, content_hash
from src.mofjson.topology import IdentifyTopology, LoadReferenceNetsFromJSON

import functools
import pathlib

# rcsr_import.py's bulk output, generated once from RCSR's own published
# Systre export (RCSRnets20190601.cgd -- 2727 3-periodic nets, of which 2645
# import cleanly; see rcsr_import.py's docstring for the 82 excluded and
# why) and shipped alongside this module so BuildDocument gets real RCSR
# coverage out of the box instead of topology.py's tiny 4-entry built-in
# table (pcu/dia/sql/tbo). Loaded lazily and cached -- it's ~2.7MB of JSON,
# no reason to pay that cost at import time or more than once per process.
_BUNDLED_REFERENCE_NETS_PATH = pathlib.Path(__file__).parent / "data" / "rcsr_reference_nets.json"


@functools.lru_cache(maxsize=1)
def _bundled_reference_nets() -> dict[str, Any] | None:
    """The ~2645-net RCSR reference table bundled with this package, or
    None if it isn't present (e.g. a stripped-down install) -- in which
    case BuildDocument falls back to topology.py's tiny built-in table."""
    if not _BUNDLED_REFERENCE_NETS_PATH.exists():
        return None
    return LoadReferenceNetsFromJSON(str(_BUNDLED_REFERENCE_NETS_PATH))


class RawStructure(BaseModel):
    lattice: dict[str, float]
    atoms: list[dict[str, Any]]
    bonds: list[dict[str, Any]]
    charges: list[dict[str, Any]] | None = None
    aromatic_n_h: list[dict[str, Any]] | None = None


class RoleAssignment(BaseModel):
    node_linker_roles: list[dict[str, Any]] = Field(
        description=(
            "Required input to the derived tier. Not ground truth, not a cache -- "
            "see module docstring. An empty list means 'not yet assigned', which "
            "is valid on its own, but `derived` must then also be absent."
        )
    )


class DerivedFragment(BaseModel):
    """One building_blocks[] entry, extended with:
      - polymer: set automatically by DetectPolymerUnit.DetectPolymerUnits
        for any block that bonds to its own next periodic copy (previously
        this was a type-only slot nothing ever populated).
      - smiles / formula / identity_provenance: set automatically for every
        linker_* block by linker.IdentifyLinkers. Null for node_* and
        guest_* blocks -- node chemical identity is a separate, still-open
        gap (see linker.py's module docstring).
    """

    id: str
    atoms: list[dict[str, Any]]
    bonds: list[dict[str, Any]]
    charges: list[dict[str, Any]] | None = None
    aromatic_n_h: list[dict[str, Any]] | None = None
    polymer: PolymerUnit | None = None
    smiles: str | None = None
    formula: str | None = None
    identity_provenance: dict[str, Any] | None = None


class DerivedViews(BaseModel):
    building_blocks: list[DerivedFragment]
    net_edges: list[dict[str, Any]]
    topology: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Set automatically by BuildDocument via topology.IdentifyTopology: "
            "genuine coordination-sequence invariants computed by BFS over an "
            "honest topological net (loose ends, pure chelation, and duplicate "
            "double-bridges pruned/deduplicated per the four net-validity rules "
            "in topology.py's module docstring -- NOT the raw building_blocks/ "
            "net_edges graph unmodified), plus an RCSR `net` symbol IF AND ONLY "
            "IF that computed signature exactly matches one of the (deliberately "
            "few, individually-verified) entries in topology.REFERENCE_NETS "
            "(pcu/dia/sql from textbook closed-form counting, tbo hand-"
            "transcribed from RCSR's own net page and cross-checked against "
            "this codebase's own Cu-BTC computation -- see each entry's "
            "`source` field, and `matched_reference_source` on the result). "
            "`net` is null -- honestly, not as a bug -- for any structure whose "
            "signature isn't in that table yet (see topology.py's module "
            "docstring for exactly how to add more nets, either one at a time "
            "by hand or in bulk via rcsr_import.py). The vertex_signatures are "
            "still real, computed data even when net is null. Also carries "
            "`excluded_building_blocks` (which blocks were pruned from the net "
            "and why -- nothing silently dropped), `deconstruction_method` "
            "(which of Glasby et al. 2024's named MOF-deconstruction "
            "conventions this document's building_blocks/net_edges follows -- "
            "a different, equally valid convention can assign a different net "
            "symbol to the same structure), and `determination_method`/"
            "`confidence` (a coordination-sequence match is a necessary-but-not-"
            "sufficient fingerprint, never a proof of graph isomorphism -- these "
            "two fields say plainly which kind of result `net` is, rather than "
            "one string pretending to be both a fingerprint and a proof)."
        ),
    )
    provenance: Provenance


class MOFJSONDocument(BaseModel):
    raw: RawStructure
    roles: RoleAssignment | None = None
    derived: DerivedViews | None = None
    lattice_encoding: LatticeEncoding | None = None
    classification: ClassificationTag | None = None

    @model_validator(mode="after")
    def _derived_requires_roles(self) -> "MOFJSONDocument":
        if self.derived is not None and self.roles is None:
            raise ValueError(
                "derived is present but roles is None -- derived cannot exist "
                "without the roles it's a function of"
            )
        return self

    @model_validator(mode="after")
    def _derived_matches_raw(self) -> "MOFJSONDocument":
        """The actual enforced edge: derived's provenance hash must match a
        fresh hash of the raw+roles it claims to be computed from. This is
        the same class of check MolFragment already runs for smiles/coords,
        applied here to the building_blocks/net_edges cache instead."""
        if self.derived is None:
            return self
        expected = content_hash(
            self.raw.atoms,
            self.raw.bonds,
            self.roles.node_linker_roles if self.roles else [],
        )
        if self.derived.provenance.computed_from_hash != expected:
            raise ValueError(
                "derived.provenance.computed_from_hash does not match a fresh hash of "
                "(raw.atoms, raw.bonds, roles.node_linker_roles) -- this document's "
                "building_blocks/net_edges is stale relative to its own raw structure."
            )
        return self

    @model_validator(mode="after")
    def _lattice_encoding_matches_raw(self) -> "MOFJSONDocument":
        """Same class of check as _derived_matches_raw, applied to the
        lattice_encoding tier. It is a function of (raw.atoms, raw.bonds,
        raw.lattice) alone -- it does NOT depend on roles, unlike derived --
        so it gets its own, independently-scoped hash rather than reusing
        derived's. Before this validator existed, a stale/hand-edited
        `lattice_encoding` block had nothing checking it at all."""
        if self.lattice_encoding is None:
            return self
        expected = content_hash(self.raw.atoms, self.raw.bonds, self.raw.lattice)
        if self.lattice_encoding.provenance.computed_from_hash != expected:
            raise ValueError(
                "lattice_encoding.provenance.computed_from_hash does not match a fresh "
                "hash of (raw.atoms, raw.bonds, raw.lattice) -- this document's lattice "
                "encoding is stale relative to its own raw structure."
            )
        return self

    def verify_structurally(self) -> None:
        """Stronger, opt-in check: actually expand building_blocks/net_edges
        and compare against raw.atoms/raw.bonds via CheckNetConsistency,
        instead of only the cheap hash comparison above. Call this before
        writing a document out as benchmark ground truth; the hash check
        alone is enough to reject a document at parse time, but it can't
        catch a hash that was (incorrectly) computed by hand to match
        tampered building_blocks -- only re-deriving and comparing can."""
        if self.derived is None:
            return
        mofjson_dict = {
            "atoms": self.raw.atoms,
            "bonds": self.raw.bonds,
            "building_blocks": [
                bb.model_dump(exclude_none=True) for bb in self.derived.building_blocks
            ],
            "net_edges": self.derived.net_edges,
        }
        node_linker_roles = self.roles.node_linker_roles if self.roles else None
        ok, _ = _conversion.CheckNetConsistency(mofjson_dict, node_linker_roles=node_linker_roles)
        if not ok:
            raise ValueError(
                "CheckNetConsistency failed: derived.building_blocks/net_edges do not "
                "expand back to raw.atoms/raw.bonds"
            )


def BuildDocument(
    raw: RawStructure,
    node_linker_roles: list[dict[str, Any]],
    *,
    method: str = "CollapseToBuildingBlocks",
    version: str = "v1",
    detect_polymers: bool = True,
    include_lattice_encoding: bool = True,
    topology_reference_nets: dict[str, Any] | None = None,
) -> MOFJSONDocument:
    """The one sanctioned way to produce a MOFJSONDocument with a `derived`
    block: run the real derivation, hash exactly what it consumed, and hand
    back a document whose own validator has already checked itself on
    construction.
    """
    mofjson_dict = {
        "atoms": raw.atoms,
        "bonds": raw.bonds,
        "charges": raw.charges or [],
        "aromatic_n_h": raw.aromatic_n_h or [],
        "node_linker_roles": node_linker_roles,
    }
    collapsed = _conversion.CollapseToBuildingBlocks(mofjson_dict)
    identified_blocks = IdentifyLinkers(collapsed["building_blocks"], raw.lattice)

    if detect_polymers:
        polymers_by_block = DetectPolymerUnits(collapsed["building_blocks"], collapsed["net_edges"])
        for block in identified_blocks:
            if block["id"] in polymers_by_block:
                block["polymer"] = polymers_by_block[block["id"]]

    effective_reference_nets = (
        topology_reference_nets if topology_reference_nets is not None else _bundled_reference_nets()
    )
    topology = IdentifyTopology(
        collapsed["building_blocks"], collapsed["net_edges"], reference_nets=effective_reference_nets
    )

    provenance = Provenance(
        method=method,
        version=version,
        computed_from=["atoms", "bonds", "node_linker_roles"],
        computed_from_hash=content_hash(raw.atoms, raw.bonds, node_linker_roles),
    )
    derived = DerivedViews(
        building_blocks=identified_blocks,
        net_edges=collapsed["net_edges"],
        topology=topology,
        provenance=provenance,
    )
    return MOFJSONDocument(
        raw=raw,
        roles=RoleAssignment(node_linker_roles=node_linker_roles),
        derived=derived,
        lattice_encoding=ComputeLatticeEncoding(raw, version=version) if include_lattice_encoding else None,
    )


__all__ = [
    "RawStructure",
    "RoleAssignment",
    "DerivedFragment",
    "DerivedViews",
    "MOFJSONDocument",
    "BuildDocument",
]
