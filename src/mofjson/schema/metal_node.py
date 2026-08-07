"""MetalNode and MetalSite: the coordination geometry of the inorganic building unit.

Phase 03. This is the central schema contribution of the project.

Field names follow project_notes/phases/03_schema_metal_node_geometry.md:
    Donor, MetalSite (per-centre geometry), MetalNode (SBU), LocalFrame, Orientation.
Phase 08 computes values; this module defines what "computed correctly" means.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from mofjson.exceptions import SchemaError
from mofjson.schema.base import MOFJSONModel
from mofjson.schema.molecular import Coord, CoordsOrigin, CoordsSource, MolFragment
from mofjson.vocab.archetypes import ARCHETYPES, NODE_KINDS
from mofjson.vocab.donors import BRIDGING_MODES, DONOR_GROUPS, OMS_METHODS
from mofjson.vocab.elements import CONNECTIVITY_SOURCES, PARTIAL_CHARGE_METHODS
from mofjson.vocab.geometry import GEOMETRY_SOURCES, POLYHEDRA, POLYHEDRON_CN


def _require_vocab(value: str, allowed: tuple[str, ...] | frozenset[str], field_path: str) -> str:
    if value not in allowed:
        raise SchemaError(
            f"{field_path} must be one of {sorted(allowed)}; got {value!r}",
            field_path=field_path,
            value=value,
        )
    return value


def _vec3(v: list[float], field_path: str) -> list[float]:
    if len(v) != 3 or not all(isinstance(x, (int, float)) for x in v):
        raise SchemaError(
            f"{field_path} must be a length-3 numeric list",
            field_path=field_path,
            value=v,
        )
    return [float(x) for x in v]


def _quat(v: list[float], field_path: str) -> list[float]:
    if len(v) != 4 or not all(isinstance(x, (int, float)) for x in v):
        raise SchemaError(
            f"{field_path} must be a length-4 numeric list",
            field_path=field_path,
            value=v,
        )
    return [float(x) for x in v]


class Donor(MOFJSONModel):
    """One coordinating atom as seen from the metal centre."""

    atom_id: str = Field(min_length=1)
    element: str = Field(min_length=1)
    donor_group: str
    bridging_mode: str
    distance: Annotated[float, Field(ge=0.0)]

    @field_validator("donor_group")
    @classmethod
    def _donor_group(cls, v: str) -> str:
        return _require_vocab(v, DONOR_GROUPS, "donor_group")

    @field_validator("bridging_mode")
    @classmethod
    def _bridging_mode(cls, v: str) -> str:
        return _require_vocab(v, BRIDGING_MODES, "bridging_mode")


class BondLengthSummary(MOFJSONModel):
    min: float
    max: float
    mean: float
    std: Annotated[float, Field(ge=0.0)]

    @model_validator(mode="after")
    def _ordered(self) -> BondLengthSummary:
        if self.min > self.max:
            raise SchemaError(
                "bond_lengths.min must be <= max",
                field_path="bond_lengths",
                value={"min": self.min, "max": self.max},
            )
        return self


class BondAngleSummary(MOFJSONModel):
    min: float
    max: float
    mean: float

    @model_validator(mode="after")
    def _ordered(self) -> BondAngleSummary:
        if self.min > self.max:
            raise SchemaError(
                "bond_angles.min must be <= max",
                field_path="bond_angles",
                value={"min": self.min, "max": self.max},
            )
        return self


class OpenMetalSiteEvidence(MOFJSONModel):
    observed_cn: Annotated[int, Field(ge=0)]
    expected_cn: Annotated[int, Field(ge=0)]
    removable_ligand: str | None = None
    solvent_removed: bool | None = None


class MetalCount(MOFJSONModel):
    element: str = Field(min_length=1)
    count: Annotated[int, Field(ge=1)]


class LigandCount(MOFJSONModel):
    formula: str = Field(min_length=1)
    bridging_mode: str | None = None
    count: Annotated[int, Field(ge=0)]

    @field_validator("bridging_mode")
    @classmethod
    def _bridging_mode(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _require_vocab(v, BRIDGING_MODES, "bridging_mode")


class NodeAttachmentPoint(MOFJSONModel):
    id: str = Field(min_length=1)
    atom_id: str = Field(min_length=1)
    donor_group: str
    linker_id: str | None = None

    @field_validator("donor_group")
    @classmethod
    def _donor_group(cls, v: str) -> str:
        return _require_vocab(v, DONOR_GROUPS, "donor_group")


class MetalMetalContact(MOFJSONModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    distance: Annotated[float, Field(ge=0.0)]
    bonded: bool


class ConnectionVector(MOFJSONModel):
    attachment_id: str = Field(min_length=1)
    position: list[float]
    vector: list[float]
    length: Annotated[float, Field(ge=0.0)]

    @field_validator("position")
    @classmethod
    def _position(cls, v: list[float]) -> list[float]:
        return _vec3(v, "position")

    @field_validator("vector")
    @classmethod
    def _vector(cls, v: list[float]) -> list[float]:
        return _vec3(v, "vector")


class LocalFrame(MOFJSONModel):
    """Node geometry in its own Cartesian frame (Phase 03.5)."""

    coords: list[Coord]
    origin: CoordsOrigin
    connection_vectors: list[ConnectionVector] | None = None
    source: CoordsSource
    precision: Annotated[int, Field(ge=0, le=8)]
    # Translation period (Å) along rod_axis for kind rod/layer.
    period: float | None = None

    @field_validator("origin")
    @classmethod
    def _origin(cls, v: str) -> str:
        allowed = ("centroid", "first_atom", "metal_centroid")
        if v not in allowed:
            raise SchemaError(
                f"local_frame.origin must be one of {allowed}",
                field_path="origin",
                value=v,
            )
        return v

    @field_validator("source")
    @classmethod
    def _source(cls, v: str) -> str:
        allowed = ("deconstruction", "forcefield", "template", "llm", "manual")
        if v not in allowed:
            raise SchemaError(
                f"local_frame.source must be one of {allowed}",
                field_path="source",
                value=v,
            )
        return v


class Orientation(MOFJSONModel):
    """Rotation placing local_frame into the parent cell (Phase 03.5)."""

    quaternion: list[float]
    vertex_id: str | None = None

    @field_validator("quaternion")
    @classmethod
    def _quaternion(cls, v: list[float]) -> list[float]:
        return _quat(v, "quaternion")


class MetalSite(MOFJSONModel):
    """Per-centre coordination geometry."""

    site_id: str = Field(min_length=1)
    element: str = Field(min_length=1)
    wyckoff: str | None = None
    oxidation_state: int | None = None
    coordination_number: Annotated[int, Field(ge=0)]
    geometry: str
    geometry_symbol: str | None = None
    geometry_csm: float | None = None
    geometry_source: str | None = None
    geometry_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    donors: list[Donor] | None = None
    donor_composition: dict[str, int] | None = None
    bond_lengths: BondLengthSummary | None = None
    bond_angles: BondAngleSummary | None = None
    open_metal_site: bool | None = None
    open_metal_site_method: str | None = None
    open_metal_site_evidence: OpenMetalSiteEvidence | None = None
    magnetic_moment: float | None = None
    partial_charge: float | None = None
    partial_charge_method: str | None = None

    @field_validator("geometry")
    @classmethod
    def _geometry(cls, v: str) -> str:
        return _require_vocab(v, POLYHEDRA, "geometry")

    @field_validator("geometry_source")
    @classmethod
    def _geometry_source(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _require_vocab(v, GEOMETRY_SOURCES, "geometry_source")

    @field_validator("open_metal_site_method")
    @classmethod
    def _oms_method(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _require_vocab(v, OMS_METHODS, "open_metal_site_method")

    @field_validator("partial_charge_method")
    @classmethod
    def _pcm(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _require_vocab(v, PARTIAL_CHARGE_METHODS, "partial_charge_method")

    @field_validator("donor_composition")
    @classmethod
    def _donor_comp_counts(cls, v: dict[str, int] | None) -> dict[str, int] | None:
        if v is None:
            return None
        for el, n in v.items():
            if not el or not isinstance(n, int) or n < 0:
                raise SchemaError(
                    "donor_composition values must be non-negative integers",
                    field_path="donor_composition",
                    value=v,
                )
        return v

    @model_validator(mode="after")
    def _invariants(self) -> MetalSite:
        if self.geometry not in ("unknown", "other"):
            ideal = POLYHEDRON_CN.get(self.geometry)
            if ideal is not None and self.coordination_number != ideal:
                raise SchemaError(
                    f"geometry {self.geometry!r} requires coordination_number "
                    f"{ideal}, got {self.coordination_number}",
                    field_path="coordination_number",
                    value=self.coordination_number,
                )

        if self.geometry != "unknown":
            if self.geometry_csm is None:
                raise SchemaError(
                    "geometry_csm is required when geometry is not 'unknown'",
                    field_path="geometry_csm",
                )
            if self.geometry_source is None:
                raise SchemaError(
                    "geometry_source is required when geometry is not 'unknown'",
                    field_path="geometry_source",
                )

        if self.donors is not None and self.donor_composition is not None:
            agg = Counter(d.element for d in self.donors)
            expected = {k: v for k, v in self.donor_composition.items() if v}
            actual = dict(agg)
            if actual != expected:
                raise SchemaError(
                    "donor_composition must equal aggregation of donors[].element",
                    field_path="donor_composition",
                    value={"expected": expected, "from_donors": actual},
                )

        if self.donors is not None and self.coordination_number != len(self.donors):
            # Allow mismatch only when composition alone is used; if donors listed, CN
            # should match donor count.
            raise SchemaError(
                "coordination_number must equal len(donors) when donors are present",
                field_path="coordination_number",
                value={
                    "coordination_number": self.coordination_number,
                    "n_donors": len(self.donors),
                },
            )

        return self


class MetalNode(MOFJSONModel):
    """Secondary building unit (SBU) record."""

    id: str = Field(min_length=1)
    kind: str
    archetype: str
    formula: str | None = None
    nuclearity: Annotated[int, Field(ge=0)]
    metals: list[MetalCount]
    bridging_ligands: list[LigandCount] | None = None
    terminal_ligands: list[LigandCount] | None = None
    # Required key; nullable when unknown (OpenAI-strict: never omit).
    connectivity: int | None
    connectivity_source: str | None = None
    attachment_points: list[NodeAttachmentPoint] | None = None
    sites: list[MetalSite] | None = None
    metal_metal_contacts: list[MetalMetalContact] | None = None
    point_group: str | None = None
    point_group_tolerance: float | None = None
    rod_axis: list[float] | None = None
    fragment: MolFragment | None = None
    node_smiles: str | None = None
    centroid_frac: list[float] | None = None
    local_frame: LocalFrame | None = None
    orientation: Orientation | None = None

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        return _require_vocab(v, NODE_KINDS, "kind")

    @field_validator("archetype")
    @classmethod
    def _archetype(cls, v: str) -> str:
        return _require_vocab(v, ARCHETYPES, "archetype")

    @field_validator("connectivity_source")
    @classmethod
    def _conn_src(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _require_vocab(v, CONNECTIVITY_SOURCES, "connectivity_source")

    @field_validator("rod_axis")
    @classmethod
    def _rod_axis(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        return _vec3(v, "rod_axis")

    @field_validator("centroid_frac")
    @classmethod
    def _centroid(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        return _vec3(v, "centroid_frac")

    @model_validator(mode="after")
    def _invariants(self) -> MetalNode:
        metal_sum = sum(m.count for m in self.metals)
        if self.nuclearity != metal_sum:
            raise SchemaError(
                "nuclearity must equal sum(metals[].count)",
                field_path="nuclearity",
                value={"nuclearity": self.nuclearity, "sum_counts": metal_sum},
            )

        if self.kind in ("rod", "layer"):
            if self.rod_axis is None:
                raise SchemaError(
                    f"rod_axis is required when kind is {self.kind!r}",
                    field_path="rod_axis",
                )
        elif self.rod_axis is not None:
            raise SchemaError(
                "rod_axis is only allowed when kind is 'rod' or 'layer'",
                field_path="rod_axis",
                value=self.rod_axis,
            )

        site_ids: set[str] = set()
        if self.sites is not None:
            for s in self.sites:
                if s.site_id in site_ids:
                    raise SchemaError(
                        f"duplicate site_id: {s.site_id}",
                        field_path="sites",
                        value=s.site_id,
                    )
                site_ids.add(s.site_id)

        if self.metal_metal_contacts is not None and site_ids:
            for i, c in enumerate(self.metal_metal_contacts):
                for attr in ("source", "target"):
                    ref = getattr(c, attr)
                    if ref not in site_ids:
                        raise SchemaError(
                            f"metal_metal_contacts[{i}].{attr} references unknown site_id",
                            field_path=f"metal_metal_contacts[{i}].{attr}",
                            value=ref,
                        )

        if self.connectivity is not None and self.attachment_points is not None:
            if self.connectivity != len(self.attachment_points):
                raise SchemaError(
                    "connectivity must equal len(attachment_points) when both are set",
                    field_path="connectivity",
                    value={
                        "connectivity": self.connectivity,
                        "n_attachment_points": len(self.attachment_points),
                    },
                )

        if self.attachment_points is not None:
            ap_ids = [ap.id for ap in self.attachment_points]
            if len(ap_ids) != len(set(ap_ids)):
                raise SchemaError(
                    "attachment_points[].id must be unique",
                    field_path="attachment_points",
                    value=ap_ids,
                )

        # When a fragment is present, donor atom_ids must resolve into it. Rod/layer
        # fragments are still full coordination shells for the recorded sites.
        if self.fragment is not None and self.sites is not None:
            frag_ids = {a.id for a in self.fragment.atoms}
            for si, site in enumerate(self.sites):
                if not site.donors:
                    continue
                for di, donor in enumerate(site.donors):
                    if donor.atom_id not in frag_ids:
                        raise SchemaError(
                            f"donors[{di}].atom_id references unknown atom_id: "
                            f"{donor.atom_id}",
                            field_path=f"sites[{si}].donors[{di}].atom_id",
                            value=donor.atom_id,
                        )

        if self.local_frame is not None and self.local_frame.connection_vectors is not None:
            if self.attachment_points is not None:
                ap_set = {ap.id for ap in self.attachment_points}
                for cv in self.local_frame.connection_vectors:
                    if cv.attachment_id not in ap_set:
                        raise SchemaError(
                            "connection_vectors.attachment_id must reference "
                            "attachment_points[].id",
                            field_path="local_frame.connection_vectors",
                            value=cv.attachment_id,
                        )

        return self


__all__ = [
    "BondAngleSummary",
    "BondLengthSummary",
    "ConnectionVector",
    "Donor",
    "LigandCount",
    "LocalFrame",
    "MetalCount",
    "MetalMetalContact",
    "MetalNode",
    "MetalSite",
    "NodeAttachmentPoint",
    "OpenMetalSiteEvidence",
    "Orientation",
]
