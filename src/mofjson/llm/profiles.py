"""Named field subsets of the schema, and the machinery to derive them.

Phase 10.2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator

from mofjson.exceptions import SchemaError
from mofjson.schema.base import MOFJSONModel
from mofjson.schema.crystal import CrystalStructure
from mofjson.schema.document import MOFJSONDocument
from mofjson.schema.metadata import Identifiers, Property
from mofjson.schema.metal_node import (
    LigandCount,
    LocalFrame,
    MetalCount,
    MetalMetalContact,
    MetalNode,
    NodeAttachmentPoint,
    Orientation,
)
from mofjson.schema.molecular import MolFragment
from mofjson.schema.reticular import (
    Assembly,
    Defect,
    Linker,
    ReticularStructure,
    Topology,
)


@dataclass(frozen=True, slots=True)
class Profile:
    """Named subset of the MOFJSON schema for structured-output prompts."""

    name: str
    purpose: str
    layers: tuple[str, ...]
    model_name: str


# Geometry fields stripped for the Phase 13.2 ablation (reticular_nogeom).
_METAL_SITE_GEOM_FIELDS = frozenset(
    {
        "geometry",
        "geometry_symbol",
        "geometry_csm",
        "geometry_source",
        "geometry_confidence",
        "donors",
        "donor_composition",
        "bond_lengths",
        "bond_angles",
    }
)


class MetalSiteNoGeom(MOFJSONModel):
    """MetalSite without Phase 03 geometry fields (ablation control)."""

    site_id: str = Field(min_length=1)
    element: str = Field(min_length=1)
    wyckoff: str | None = None
    oxidation_state: int | None = None
    coordination_number: Annotated[int, Field(ge=0)]
    open_metal_site: bool | None = None
    open_metal_site_method: str | None = None
    magnetic_moment: float | None = None
    partial_charge: float | None = None
    partial_charge_method: str | None = None


class MetalNodeNoGeom(MOFJSONModel):
    """MetalNode whose sites omit geometry fields."""

    id: str = Field(min_length=1)
    kind: str
    archetype: str
    formula: str | None = None
    nuclearity: Annotated[int, Field(ge=0)]
    metals: list[MetalCount]
    bridging_ligands: list[LigandCount] | None = None
    terminal_ligands: list[LigandCount] | None = None
    connectivity: int | None
    connectivity_source: str | None = None
    attachment_points: list[NodeAttachmentPoint] | None = None
    sites: list[MetalSiteNoGeom] | None = None
    metal_metal_contacts: list[MetalMetalContact] | None = None
    point_group: str | None = None
    point_group_tolerance: float | None = None
    rod_axis: list[float] | None = None
    fragment: MolFragment | None = None
    node_smiles: str | None = None
    centroid_frac: list[float] | None = None
    local_frame: LocalFrame | None = None
    orientation: Orientation | None = None


class ReticularNoGeom(MOFJSONModel):
    """ReticularStructure with geometry-stripped nodes."""

    topology: Topology | None = None
    nodes: list[MetalNodeNoGeom] | None = None
    linkers: list[Linker] | None = None
    assembly: Assembly | None = None
    defects: list[Defect] | None = None
    formula_unit: str | None = None
    z: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_from_full(cls, data: Any) -> Any:
        """Accept a full reticular dict and drop geometry fields from nodes."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        nodes = out.get("nodes")
        if isinstance(nodes, list):
            stripped = []
            for n in nodes:
                if not isinstance(n, dict):
                    stripped.append(n)
                    continue
                nd = {k: v for k, v in n.items() if k != "sites"}
                sites = n.get("sites")
                if isinstance(sites, list):
                    nd["sites"] = [
                        {k: v for k, v in s.items() if k not in _METAL_SITE_GEOM_FIELDS}
                        if isinstance(s, dict)
                        else s
                        for s in sites
                    ]
                stripped.append(nd)
            out["nodes"] = stripped
        return out


class CrystalProfileDoc(MOFJSONModel):
    crystal: CrystalStructure

    @model_validator(mode="after")
    def _layout_verbose(self) -> CrystalProfileDoc:
        if self.crystal.layout != "verbose":
            raise SchemaError(
                "crystal profile requires layout='verbose'",
                field_path="crystal.layout",
                value=self.crystal.layout,
            )
        return self


class CrystalColumnarProfileDoc(MOFJSONModel):
    crystal: CrystalStructure

    @model_validator(mode="after")
    def _layout_columnar(self) -> CrystalColumnarProfileDoc:
        if self.crystal.layout != "columnar":
            raise SchemaError(
                "crystal_columnar profile requires layout='columnar'",
                field_path="crystal.layout",
                value=self.crystal.layout,
            )
        return self


class ReticularProfileDoc(MOFJSONModel):
    reticular: ReticularStructure


class ReticularNoGeomProfileDoc(MOFJSONModel):
    reticular: ReticularNoGeom


class PropertiesProfileDoc(MOFJSONModel):
    identifiers: Identifiers = Field(default_factory=Identifiers)
    properties: list[Property] = Field(default_factory=list)


class MinimalReticular(MOFJSONModel):
    topology: Topology | None = None


class MinimalProfileDoc(MOFJSONModel):
    identifiers: Identifiers = Field(default_factory=Identifiers)
    reticular: MinimalReticular = Field(default_factory=MinimalReticular)


class AnswerInteger(MOFJSONModel):
    answer: int


class AnswerString(MOFJSONModel):
    answer: str


class AnswerBoolean(MOFJSONModel):
    answer: bool


class AnswerCif(MOFJSONModel):
    cif: str


class AnswerMofid(MOFJSONModel):
    mofid: str


class AnswerLatticeCoordsText(MOFJSONModel):
    lattice_coords_text: str


_PROFILE_MODELS: dict[str, type[BaseModel]] = {
    "crystal": CrystalProfileDoc,
    "crystal_columnar": CrystalColumnarProfileDoc,
    "reticular": ReticularProfileDoc,
    "reticular_nogeom": ReticularNoGeomProfileDoc,
    "metal_node": MetalNode,
    "properties": PropertiesProfileDoc,
    "minimal": MinimalProfileDoc,
    "answer_integer": AnswerInteger,
    "answer_string": AnswerString,
    "answer_boolean": AnswerBoolean,
    "answer_cif": AnswerCif,
    "answer_mofid": AnswerMofid,
    "answer_lattice_coords_text": AnswerLatticeCoordsText,
}

PROFILES: dict[str, Profile] = {
    "crystal": Profile(
        name="crystal",
        purpose="Lossless structure for round-trip and translation tasks.",
        layers=("crystal",),
        model_name="CrystalProfileDoc",
    ),
    "crystal_columnar": Profile(
        name="crystal_columnar",
        purpose="Token-efficient columnar crystal for structure generation.",
        layers=("crystal",),
        model_name="CrystalColumnarProfileDoc",
    ),
    "reticular": Profile(
        name="reticular",
        purpose="Topology + nodes + linkers for design and generation.",
        layers=("reticular",),
        model_name="ReticularProfileDoc",
    ),
    "reticular_nogeom": Profile(
        name="reticular_nogeom",
        purpose="Reticular without metal-node geometry (Phase 13.2 ablation).",
        layers=("reticular",),
        model_name="ReticularNoGeomProfileDoc",
    ),
    "metal_node": Profile(
        name="metal_node",
        purpose="Single MetalNode for coordination/connectivity Q&A.",
        layers=("metal_node",),
        model_name="MetalNode",
    ),
    "properties": Profile(
        name="properties",
        purpose="Identifiers + properties for extraction from text.",
        layers=("identifiers", "properties"),
        model_name="PropertiesProfileDoc",
    ),
    "minimal": Profile(
        name="minimal",
        purpose="Identifiers + topology for cheap classification.",
        layers=("identifiers", "topology"),
        model_name="MinimalProfileDoc",
    ),
    "answer_integer": Profile(
        name="answer_integer",
        purpose="Single-key integer answer schema for fair format comparison.",
        layers=(),
        model_name="AnswerInteger",
    ),
    "answer_string": Profile(
        name="answer_string",
        purpose="Single-key string answer schema for fair format comparison.",
        layers=(),
        model_name="AnswerString",
    ),
    "answer_boolean": Profile(
        name="answer_boolean",
        purpose="Single-key boolean answer schema for fair format comparison.",
        layers=(),
        model_name="AnswerBoolean",
    ),
    "answer_cif": Profile(
        name="answer_cif",
        purpose="Single-key CIF text schema for fair format comparison.",
        layers=(),
        model_name="AnswerCif",
    ),
    "answer_mofid": Profile(
        name="answer_mofid",
        purpose="Single-key MOFid schema for fair format comparison.",
        layers=(),
        model_name="AnswerMofid",
    ),
    "answer_lattice_coords_text": Profile(
        name="answer_lattice_coords_text",
        purpose="Single-key lattice+coords text schema for fair format comparison.",
        layers=(),
        model_name="AnswerLatticeCoordsText",
    ),
}


def ProfileModel(name: str) -> type[BaseModel]:
    """Return the Pydantic model used to validate a profile document."""
    try:
        return _PROFILE_MODELS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown profile {name!r}; known: {sorted(PROFILES)}"
        ) from exc


def GetProfileSchema(name: str, *, dialect: str = "jsonschema") -> dict[str, Any]:
    """Emit the JSON Schema for a named profile in the requested dialect."""
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; known: {sorted(PROFILES)}")
    from mofjson.schema.jsonschema import _emit_model_schema

    model = ProfileModel(name)
    title = f"MOFJSONProfile_{name}"
    return _emit_model_schema(model, title, dialect)


def MergeProfileIntoDocument(
    obj: dict[str, Any] | BaseModel,
    profile: str,
    *,
    base: MOFJSONDocument | dict[str, Any] | None = None,
) -> MOFJSONDocument:
    """Merge profile-shaped output into a full document without clobbering structure.

    Properties-profile output updates identifiers/properties only. Other profiles
    update their layer. Existing base fields outside the profile are preserved.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}")

    validated = ProfileModel(profile).model_validate(
        obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    )
    payload = validated.model_dump(mode="json", exclude_none=True)

    if base is None:
        doc_data: dict[str, Any] = {}
    elif isinstance(base, MOFJSONDocument):
        doc_data = base.model_dump(mode="json", exclude_none=True)
    else:
        doc_data = dict(base)

    if profile == "metal_node":
        reticular = dict(doc_data.get("reticular") or {})
        nodes = list(reticular.get("nodes") or [])
        # Replace or append by id.
        node_id = payload.get("id")
        replaced = False
        for i, n in enumerate(nodes):
            if isinstance(n, dict) and n.get("id") == node_id:
                nodes[i] = payload
                replaced = True
                break
        if not replaced:
            nodes.append(payload)
        reticular["nodes"] = nodes
        doc_data["reticular"] = reticular
    elif profile.startswith("answer_"):
        raise ValueError(f"Profile {profile!r} is not a document layer; cannot merge")
    else:
        for key, value in payload.items():
            if key == "reticular" and isinstance(value, dict) and "reticular" in doc_data:
                merged = dict(doc_data["reticular"] or {})
                merged.update(value)
                doc_data["reticular"] = merged
            else:
                doc_data[key] = value

    return MOFJSONDocument.model_validate(doc_data)


def CountProfileSchemaTokens(name: str, *, tokeniser: str = "cl100k_base") -> int:
    """Token count of a profile's openai dialect schema (requires tiktoken / llm extra)."""
    from mofjson.util.tokens import CountTokens

    schema = GetProfileSchema(name, dialect="openai")
    text = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return CountTokens(text, tokeniser=tokeniser)


__all__ = [
    "PROFILES",
    "AnswerBoolean",
    "AnswerCif",
    "AnswerInteger",
    "AnswerLatticeCoordsText",
    "AnswerMofid",
    "AnswerString",
    "CountProfileSchemaTokens",
    "GetProfileSchema",
    "MergeProfileIntoDocument",
    "Profile",
    "ProfileModel",
]
