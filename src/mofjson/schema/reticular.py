"""The reticular layer: a MOF as topology + nodes + linkers. The token-efficient view.

Phase 05.

THE CLAIM THIS LAYER MAKES
    A MOF is not an arbitrary collection of atoms; it is a small number of building blocks
    repeated on a net. So its description should be roughly the size of its building
    blocks, not of its unit cell. Every explicit format (CIF, coordinate text, the crystal
    layer) grows linearly with atom count; this layer should be nearly flat. Phase 13.1
    tests that by plotting tokens against atom count and reporting slopes -- and must also
    report the cases where this layer loses, which will be small dense structures where
    JSON punctuation dominates.

WHAT THIS FILE CONTAINS
    Topology        RCSR net symbol (vocab.topology), optional point symbol and vertex
                    symbol, the dimensionality of the framework (1/2/3), and how the net
                    was determined -- asserted by the source, inferred from connectivity,
                    or computed. Provenance here matters because the KG's topology codes
                    are the most reliable field it has but are still dirty, and an
                    inferred net must never be presentable as a measured one.
    Linker          the organic building block: a MolFragment (Phase 02), its connectivity
                    (denticity, i.e. how many nodes it bridges), its donor groups, an
                    optional common name and canonical SMILES, and a count/multiplicity.
    MetalNode       reused unchanged from metal_node.py. Do not redefine it here -- the
                    whole point of Phase 03 is that there is one definition of a node.
    Assembly        which node connects to which linker, how many times, and optionally at
                    which net vertex. Mark this experimental in v0.1.0 (Phase 13.6): it is
                    the part most likely to change, and over-promising it would be worse
                    than shipping it as provisional.
    Defect          missing linkers, missing nodes, capping ligands, and their
                    concentration if known. Defects are the difference between an idealised
                    net and a real material; a schema with no way to say "this is
                    defective" forces every real structure to be described as if it were
                    perfect.
    ReticularStructure
                    topology, nodes, linkers, assembly, defects, formula unit, and the
                    Z/multiplicity needed to relate it back to a unit cell.

RELATIONSHIP TO THE CRYSTAL LAYER
    Reticular is a lossy projection of crystal. It is derived by Phase 08 and, when both
    layers are present, document.py must validate that they agree on composition and metal
    count. Going the other way (reticular -> crystal) is reconstruction, is hard, and is
    explicitly experimental in Phase 08.7 -- so this file must not imply the inverse is
    available.

    The `assembly.completeness == "full"` contract (05.4) has five clauses. Clauses 1, 2, 3
    and 5 are checkable from this layer alone and are enforced by ReticularStructure's own
    validator. Clause 4 ("crystal.lattice is present, or the net has a standard embedding")
    genuinely needs the sibling crystal layer, which does not exist as a typed model yet
    (Phase 04) -- it is exposed here as `has_standard_net_embedding()` and
    `completeness_clause4_satisfied()` for Phase 06.6's document-level validator to call
    once both layers are typed.

DONE WHEN
    A hand-written reticular description of UiO-66 validates, and the derived reticular
    layer for every fixture structure agrees with its crystal layer on composition.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from mofjson.exceptions import SchemaError
from mofjson.schema.base import MOFJSONModel
from mofjson.schema.metal_node import MetalNode
from mofjson.schema.molecular import MolFragment
from mofjson.util.ids import parse_mofid_line
from mofjson.vocab.donors import DONOR_GROUPS
from mofjson.vocab.topology import TOPOLOGY_ALIASES, TOPOLOGY_CONFIDENCE, TOPOLOGY_SOURCES

# ---------------------------------------------------------------------------
# Local vocab (reticular-schema-only enums; not reused by other phases, so kept
# here rather than in vocab/, mirroring how metal_node.py keeps its own local
# helpers next to the models that use them).
# ---------------------------------------------------------------------------

LINKER_ROLES: tuple[str, ...] = (
    "linker",
    "pillar",
    "capping_ligand",
    "bridging_ligand",
    "counterion",
    "guest_solvent",
)

OCCUPANT_TYPES: tuple[str, ...] = ("node", "linker")

ASSEMBLY_COMPLETENESS: tuple[str, ...] = ("full", "partial", "topology_only")

DEFECT_TYPES: tuple[str, ...] = (
    "missing_linker",
    "missing_node",
    "capping_ligand",
    "interpenetration",
    "other",
)

# Nets with a template embedding available elsewhere in the pipeline (PORMAKE / ToBaCCo net
# libraries) sufficient to place a `full` reticular record without an explicit
# crystal.lattice. Conservative and documented on purpose -- Phase 08.7 owns extending it.
_NETS_WITH_STANDARD_EMBEDDING: frozenset[str] = frozenset({"pcu", "fcu", "sod", "dia", "bcu"})

# Lowercase RCSR-style net symbol: a short alphanumeric body plus the hyphenated single-
# letter suffixes that really occur (tfz-d, bcu-f, npo-a, cem-d, llw-z) and must not be
# stripped (Phase 05.1).
_NET_SYMBOL_RE = re.compile(r"^[a-z][a-z0-9]{1,7}(-[a-z])?$")

_MOFID_SUFFIX_RE = re.compile(r"\.(cat\d+|no_ref)\b", re.IGNORECASE)
_MOFID_VERSION_RE = re.compile(r"MOFid-v(\d+)\.", re.IGNORECASE)
_MOFID_CATENATION_RE = re.compile(r"\.cat(\d+)\b", re.IGNORECASE)


def _require_vocab(value: str, allowed: tuple[str, ...] | frozenset[str], field_path: str) -> str:
    if value not in allowed:
        raise SchemaError(
            f"{field_path} must be one of {sorted(allowed)}; got {value!r}",
            field_path=field_path,
            value=value,
        )
    return value


def _check_not_sentinel(symbol: str, field_path: str) -> str:
    """Reject the literal upstream sentinel strings (Phase 00.4/05.1): they become null,
    never a string value."""
    if symbol.strip().lower() in TOPOLOGY_ALIASES:
        raise SchemaError(
            f"{field_path} must not be a sentinel string (got {symbol!r}); "
            "use rcsr_symbol=null with confidence='none' instead",
            field_path=field_path,
            value=symbol,
        )
    return symbol


def _check_net_symbol_format(symbol: str, field_path: str) -> str:
    _check_not_sentinel(symbol, field_path)
    if not _NET_SYMBOL_RE.match(symbol):
        raise SchemaError(
            f"{field_path} must be a lowercase RCSR-style net symbol, optionally with a "
            f"hyphenated single-letter suffix (e.g. 'fcu', 'tfz-d'); got {symbol!r}",
            field_path=field_path,
            value=symbol,
        )
    return symbol


# ---------------------------------------------------------------------------
# 05.1 Topology
# ---------------------------------------------------------------------------


class Topology(MOFJSONModel):
    """RCSR net assignment and how it was determined."""

    rcsr_symbol: str | list[str] | None = None
    alternative_symbols: list[str] | None = None
    point_symbol: str | None = None
    vertex_symbol: str | None = None
    # 1 = chain/rod, 2 = layer, 3 = ordinary 3-periodic framework.
    dimensionality: int | None = None
    n_nets: Annotated[int, Field(ge=0)] = 0
    interpenetrating: bool | None = None
    vertex_connectivity: list[int] | None = None
    source: str
    confidence: str

    @field_validator("rcsr_symbol")
    @classmethod
    def _rcsr_symbol(cls, v: str | list[str] | None) -> str | list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            return _check_net_symbol_format(v, "rcsr_symbol")
        return [_check_net_symbol_format(s, "rcsr_symbol") for s in v]

    @field_validator("alternative_symbols")
    @classmethod
    def _alternative_symbols(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return [_check_net_symbol_format(s, "alternative_symbols") for s in v]

    @field_validator("dimensionality")
    @classmethod
    def _dimensionality(cls, v: int | None) -> int | None:
        if v is not None and v not in (1, 2, 3):
            raise SchemaError(
                "dimensionality must be 1, 2 or 3", field_path="dimensionality", value=v
            )
        return v

    @field_validator("source")
    @classmethod
    def _source(cls, v: str) -> str:
        return _require_vocab(v, TOPOLOGY_SOURCES, "source")

    @field_validator("confidence")
    @classmethod
    def _confidence(cls, v: str) -> str:
        return _require_vocab(v, TOPOLOGY_CONFIDENCE, "confidence")

    @model_validator(mode="after")
    def _invariants(self) -> Topology:
        if self.rcsr_symbol is None:
            if self.confidence != "none":
                raise SchemaError(
                    "confidence must be 'none' when rcsr_symbol is null",
                    field_path="confidence",
                    value=self.confidence,
                )
            if self.n_nets != 0:
                raise SchemaError(
                    "n_nets must be 0 when rcsr_symbol is null",
                    field_path="n_nets",
                    value=self.n_nets,
                )
        elif isinstance(self.rcsr_symbol, str):
            if self.confidence == "none":
                raise SchemaError(
                    "confidence must not be 'none' when rcsr_symbol is set",
                    field_path="confidence",
                    value=self.confidence,
                )
            if self.n_nets != 1:
                raise SchemaError(
                    "n_nets must be 1 when rcsr_symbol is a single symbol",
                    field_path="n_nets",
                    value=self.n_nets,
                )
        else:
            if self.confidence == "none":
                raise SchemaError(
                    "confidence must not be 'none' when rcsr_symbol is set",
                    field_path="confidence",
                    value=self.confidence,
                )
            if self.n_nets != len(self.rcsr_symbol):
                raise SchemaError(
                    "n_nets must equal len(rcsr_symbol) for a multi-net value",
                    field_path="n_nets",
                    value={"n_nets": self.n_nets, "len_rcsr_symbol": len(self.rcsr_symbol)},
                )
            if self.n_nets < 2:
                raise SchemaError(
                    "rcsr_symbol as a list requires n_nets >= 2",
                    field_path="rcsr_symbol",
                    value=self.rcsr_symbol,
                )

        if self.interpenetrating is True and self.n_nets < 2:
            raise SchemaError(
                "interpenetrating=True requires n_nets > 1",
                field_path="interpenetrating",
                value=self.interpenetrating,
            )
        return self


def parse_mofid_topology_field(raw: str | None) -> Topology:
    """Parse the topology portion of a MOFid line into a `Topology` (Phase 05.1).

    Handles a clean single net (``fcu.cat0``), comma-joined multi-nets
    (``fud,npo-a.cat1``), sentinel-corrupted values (``TIMEOUT.cat0.NO_REF``,
    ``ERROR,UNKNOWN.cat0``), and missing/empty input -- every case maps sentinels to
    ``rcsr_symbol=None, confidence="none"``, never to a sentinel string.
    """
    if raw is None or not str(raw).strip():
        return Topology(rcsr_symbol=None, n_nets=0, source="mofid", confidence="none")

    nets: list[str] = []
    for token in str(raw).split(","):
        stripped = _MOFID_SUFFIX_RE.sub("", token).strip()
        if not stripped:
            continue
        if stripped.lower() in TOPOLOGY_ALIASES:
            continue
        nets.append(stripped.lower())

    if not nets:
        return Topology(rcsr_symbol=None, n_nets=0, source="mofid", confidence="none")
    if len(nets) == 1:
        return Topology(rcsr_symbol=nets[0], n_nets=1, source="mofid", confidence="medium")
    return Topology(rcsr_symbol=nets, n_nets=len(nets), source="mofid", confidence="medium")


# ---------------------------------------------------------------------------
# 05.2 Linker
# ---------------------------------------------------------------------------


class FunctionalGroup(MOFJSONModel):
    """One functionalisation on a linker (e.g. an ODAC/ToBaCCo `_2NO2_` branch tag)."""

    group: str = Field(min_length=1)
    count: Annotated[int, Field(ge=0)]
    position: str | None = None


class Linker(MOFJSONModel):
    """The organic (or otherwise non-metal-node) building block."""

    id: str = Field(min_length=1)
    connectivity: int | None
    role: str
    fragment: MolFragment
    smiles: str | None = None
    # Never populated from the KG's :hasChemicalName (Phase 00.6/05.2: that field is the
    # parent MOF's name, not the linker's). If the KG string is wanted, it goes here.
    name: str | None = None
    provenance_source_name: str | None = None
    common_abbreviation: str | None = None
    donor_groups: list[str] | None = None
    denticity: Annotated[int, Field(ge=0)] | None = None
    count_per_cell: Annotated[int, Field(ge=0)] | None = None
    functional_groups: list[FunctionalGroup] | None = None

    @field_validator("role")
    @classmethod
    def _role(cls, v: str) -> str:
        return _require_vocab(v, LINKER_ROLES, "role")

    @field_validator("donor_groups")
    @classmethod
    def _donor_groups(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return [_require_vocab(g, DONOR_GROUPS, "donor_groups") for g in v]

    @model_validator(mode="after")
    def _invariants(self) -> Linker:
        if self.connectivity is not None and self.fragment.attachment_points is not None:
            n_ap = len(self.fragment.attachment_points)
            if self.connectivity != n_ap:
                raise SchemaError(
                    "connectivity must equal len(fragment.attachment_points) when both "
                    "are set",
                    field_path="connectivity",
                    value={"connectivity": self.connectivity, "n_attachment_points": n_ap},
                )
        if (
            self.denticity is not None
            and self.donor_groups is not None
            and self.denticity != len(self.donor_groups)
        ):
            raise SchemaError(
                "denticity must equal len(donor_groups) when both are set",
                field_path="denticity",
                value={
                    "denticity": self.denticity,
                    "n_donor_groups": len(self.donor_groups),
                },
            )
        return self


# ---------------------------------------------------------------------------
# 05.4 Assembly, 05.5 Defect / SolidSolutionEntry
# ---------------------------------------------------------------------------


class AssemblyVertex(MOFJSONModel):
    vertex_id: str = Field(min_length=1)
    occupant_type: str
    occupant_id: str = Field(min_length=1)
    connectivity: Annotated[int, Field(ge=0)]

    @field_validator("occupant_type")
    @classmethod
    def _occupant_type(cls, v: str) -> str:
        return _require_vocab(v, OCCUPANT_TYPES, "occupant_type")


class AssemblyEdge(MOFJSONModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    via: str | None = None
    count: Annotated[int, Field(ge=0)] | None = None

    @field_validator("via")
    @classmethod
    def _via(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _require_vocab(v, DONOR_GROUPS, "via")


class Assembly(MOFJSONModel):
    """How nodes and linkers sit on the net (Phase 05.4). Experimental in v0.1.0."""

    vertices: list[AssemblyVertex]
    edges: list[AssemblyEdge] | None = None
    stoichiometry: dict[str, int] | None = None
    completeness: str

    @field_validator("completeness")
    @classmethod
    def _completeness(cls, v: str) -> str:
        return _require_vocab(v, ASSEMBLY_COMPLETENESS, "completeness")

    @model_validator(mode="after")
    def _invariants(self) -> Assembly:
        vertex_ids = [v.vertex_id for v in self.vertices]
        if len(vertex_ids) != len(set(vertex_ids)):
            dupes = sorted({v for v in vertex_ids if vertex_ids.count(v) > 1})
            raise SchemaError(
                "assembly.vertices[].vertex_id must be unique",
                field_path="vertices",
                value=dupes,
            )
        vertex_set = set(vertex_ids)
        for ei, edge in enumerate(self.edges or []):
            if edge.source not in vertex_set:
                raise SchemaError(
                    f"edges[{ei}].source references an unknown vertex_id",
                    field_path=f"edges[{ei}].source",
                    value=edge.source,
                )
            if edge.target not in vertex_set:
                raise SchemaError(
                    f"edges[{ei}].target references an unknown vertex_id",
                    field_path=f"edges[{ei}].target",
                    value=edge.target,
                )
        return self


class Defect(MOFJSONModel):
    """Missing linkers/nodes, capping ligands, and their concentration if known."""

    type: str
    linker_id: str | None = None
    node_id: str | None = None
    fraction: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    compensating_ligand: str | None = None
    source: str | None = None
    # The 05.4 `full`-contract clause 5 field: a defect claimed under a `full` assembly
    # must carry a placement, not only a statistical fraction.
    placement_vertex_id: str | None = None

    @field_validator("type")
    @classmethod
    def _type(cls, v: str) -> str:
        return _require_vocab(v, DEFECT_TYPES, "type")


class SolidSolutionEntry(MOFJSONModel):
    """One element's occupancy fraction at a mixed-metal (or mixed-linker) site."""

    site: str = Field(min_length=1)
    element: str = Field(min_length=1)
    fraction: Annotated[float, Field(ge=0.0, le=1.0)]


def has_standard_net_embedding(rcsr_symbol: str | list[str] | None) -> bool:
    """True if every symbol in *rcsr_symbol* has a known standard-embedding template.

    Used by clause 4 of the 05.4 `full` contract as the alternative to an explicit
    crystal lattice.
    """
    if rcsr_symbol is None:
        return False
    symbols = [rcsr_symbol] if isinstance(rcsr_symbol, str) else rcsr_symbol
    return bool(symbols) and all(s in _NETS_WITH_STANDARD_EMBEDDING for s in symbols)


def completeness_clause4_satisfied(
    topology: Topology | None, *, has_crystal_lattice: bool
) -> bool:
    """Phase 05.4 clause 4: `crystal.lattice` is present, or the net has a standard embedding.

    This is genuinely cross-layer (it needs the sibling crystal layer, Phase 04), so it is
    a plain function rather than something `ReticularStructure` enforces on itself.
    Phase 06.6's document-level validator calls this once both layers are typed.
    """
    if has_crystal_lattice:
        return True
    if topology is None:
        return False
    return has_standard_net_embedding(topology.rcsr_symbol)


# ---------------------------------------------------------------------------
# 05.7 MOFid / MOFkey
# ---------------------------------------------------------------------------


class MofidParsed(MOFJSONModel):
    """Structured pieces of a MOFid line, plus the derived reticular-layer bits."""

    version: int | None = None
    topology: str | list[str] | None = None
    catenation: int | None = None
    has_ref: bool = True
    name: str | None = None
    component_smiles: list[str] | None = None


def parse_mofid(raw: str | None) -> tuple[MofidParsed, Topology]:
    """Parse a full MOFid line into `(MofidParsed, Topology)` (Phase 05.7).

    Delegates the line-level split (SMILES head / topology field / structure-stem tail,
    and sentinel detection) to `mofjson.util.ids.parse_mofid_line` so the grammar is
    defined exactly once; this function adds the reticular-layer structuring (version,
    catenation, has_ref, and the parsed `Topology`) that util.ids intentionally leaves
    to the layer that owns net symbols.

    Handles, per the phase's own "done when": a normal line, `ERROR`, `TIMEOUT`,
    `.NO_REF`, comma-joined topology, an empty tail, and a `<REFCODE>_clean` tail.
    """
    parts = parse_mofid_line(raw)
    topology = parse_mofid_topology_field(parts.topology_field)

    version: int | None = None
    catenation: int | None = None
    if raw is not None and str(raw).strip():
        text = str(raw)
        version_match = _MOFID_VERSION_RE.search(text)
        if version_match:
            version = int(version_match.group(1))
        catenation_match = _MOFID_CATENATION_RE.search(text)
        if catenation_match:
            catenation = int(catenation_match.group(1))

    component_smiles = parts.smiles.split(".") if parts.smiles else None

    mofid_parsed = MofidParsed(
        version=version,
        topology=topology.rcsr_symbol,
        catenation=catenation,
        has_ref="NO_REF" not in parts.sentinel_tokens,
        name=parts.structure_stem,
        component_smiles=component_smiles,
    )
    return mofid_parsed, topology


# ---------------------------------------------------------------------------
# 05.3 / 05.6 ReticularStructure
# ---------------------------------------------------------------------------


class ReticularStructure(MOFJSONModel):
    """A MOF as topology + nodes + linkers + assembly (+ defects). The reticular layer."""

    topology: Topology | None = None
    # MetalNode reused unchanged from Phase 03 -- no duplicate node fields here.
    nodes: list[MetalNode] | None = None
    linkers: list[Linker] | None = None
    assembly: Assembly | None = None
    defects: list[Defect] | None = None
    solid_solution: list[SolidSolutionEntry] | None = None
    formula_unit: str | None = None
    z: Annotated[int, Field(ge=0)] | None = None
    mofid: str | None = None
    mofid_parsed: MofidParsed | None = None
    mofkey: str | None = None

    def _check_full_completeness(self) -> list[str]:
        """Clauses 1/2/3/5 of the 05.4 `full` contract -- everything checkable from this
        layer alone. Clause 4 is cross-layer; see `completeness_clause4_satisfied`.

        Clause 3 is scoped to what this layer can see: every vertex declared with a
        non-zero connectivity must actually be touched by at least one assembly edge --
        a vertex with connectivity > 0 and no edge is a connection the document asserts
        exists but never describes, which is exactly what `full` promises not to do.
        `AssemblyEdge.count` is a net-wide multiplicity (how many such contacts occur per
        cell), not a per-vertex-instance load, so it is not compared against
        `vertex.connectivity` here. Occupant-id resolution is already enforced
        unconditionally by `_invariants`, so it is not repeated here. Matching individual
        connection vectors to individual edges is reconstruction-time geometry and is
        explicitly Phase 08.7's job, not this layer's.
        """
        failures: list[str] = []

        for node in self.nodes or []:
            lf = node.local_frame
            if lf is None or lf.coords is None or lf.connection_vectors is None:
                failures.append(f"missing local_frame.coords/connection_vectors on {node.id}")
            if node.orientation is None:
                failures.append(f"missing orientation on {node.id}")

        for linker in self.linkers or []:
            if linker.fragment.coords is None:
                failures.append(f"missing fragment.coords on {linker.id}")

        if self.assembly is not None:
            touched_vertices: set[str] = set()
            for edge in self.assembly.edges or []:
                touched_vertices.add(edge.source)
                touched_vertices.add(edge.target)
            for vertex in self.assembly.vertices:
                if vertex.connectivity > 0 and vertex.vertex_id not in touched_vertices:
                    failures.append(
                        f"vertex {vertex.vertex_id} has connectivity {vertex.connectivity} "
                        "but no assembly edge touches it"
                    )

        for i, defect in enumerate(self.defects or []):
            if defect.placement_vertex_id is None:
                failures.append(f"defects[{i}] ({defect.type}) has no placement_vertex_id")

        return failures

    @model_validator(mode="after")
    def _invariants(self) -> ReticularStructure:
        node_ids = [n.id for n in self.nodes or []]
        linker_ids = [lk.id for lk in self.linkers or []]
        all_ids = node_ids + linker_ids
        if len(all_ids) != len(set(all_ids)):
            dupes = sorted({i for i in all_ids if all_ids.count(i) > 1})
            raise SchemaError(
                "node and linker ids must form one collision-free namespace",
                field_path="nodes/linkers",
                value=dupes,
            )
        id_set = set(all_ids)

        linker_id_set = set(linker_ids)
        for ni, node in enumerate(self.nodes or []):
            if not node.attachment_points:
                continue
            for ai, ap in enumerate(node.attachment_points):
                if ap.linker_id is not None and linker_id_set and ap.linker_id not in linker_id_set:
                    raise SchemaError(
                        f"attachment_points[].linker_id references unknown linker_id: "
                        f"{ap.linker_id}",
                        field_path=f"nodes[{ni}].attachment_points[{ai}].linker_id",
                        value=ap.linker_id,
                    )

        if self.topology is not None and self.topology.vertex_connectivity is not None:
            if self.assembly is not None and self.assembly.vertices:
                declared = [v.connectivity for v in self.assembly.vertices]
            else:
                declared = [c for c in (n.connectivity for n in self.nodes or []) if c is not None]
                declared += [
                    c for c in (lk.connectivity for lk in self.linkers or []) if c is not None
                ]
            if declared and sorted(declared) != sorted(self.topology.vertex_connectivity):
                raise SchemaError(
                    "topology.vertex_connectivity must match the connectivity of the "
                    "assembly vertices (or nodes[]/linkers[] when there is no assembly)",
                    field_path="topology.vertex_connectivity",
                    value={
                        "topology": sorted(self.topology.vertex_connectivity),
                        "observed": sorted(declared),
                    },
                )

        if self.assembly is not None:
            nodes_by_id = {n.id: n for n in self.nodes or []}
            linkers_by_id = {lk.id: lk for lk in self.linkers or []}
            for vi, vertex in enumerate(self.assembly.vertices):
                if id_set and vertex.occupant_id not in id_set:
                    raise SchemaError(
                        f"assembly.vertices[{vi}].occupant_id references an unknown id",
                        field_path=f"assembly.vertices[{vi}].occupant_id",
                        value=vertex.occupant_id,
                    )
                if vertex.occupant_id in node_ids and vertex.occupant_type != "node":
                    raise SchemaError(
                        f"assembly.vertices[{vi}].occupant_type does not match occupant_id",
                        field_path=f"assembly.vertices[{vi}].occupant_type",
                        value={
                            "occupant_id": vertex.occupant_id,
                            "occupant_type": vertex.occupant_type,
                        },
                    )
                if vertex.occupant_id in linker_ids and vertex.occupant_type != "linker":
                    raise SchemaError(
                        f"assembly.vertices[{vi}].occupant_type does not match occupant_id",
                        field_path=f"assembly.vertices[{vi}].occupant_type",
                        value={
                            "occupant_id": vertex.occupant_id,
                            "occupant_type": vertex.occupant_type,
                        },
                    )
                occupant = (
                    nodes_by_id.get(vertex.occupant_id)
                    if vertex.occupant_type == "node"
                    else linkers_by_id.get(vertex.occupant_id)
                )
                if (
                    occupant is not None
                    and occupant.connectivity is not None
                    and occupant.connectivity != vertex.connectivity
                ):
                    raise SchemaError(
                        f"assembly.vertices[{vi}].connectivity disagrees with the "
                        "occupant's own connectivity",
                        field_path=f"assembly.vertices[{vi}].connectivity",
                        value={
                            "vertex": vertex.connectivity,
                            "occupant": occupant.connectivity,
                        },
                    )

            if self.assembly.stoichiometry is not None and id_set:
                for key in self.assembly.stoichiometry:
                    if key not in id_set:
                        raise SchemaError(
                            "assembly.stoichiometry key references an unknown node/linker "
                            "id",
                            field_path="assembly.stoichiometry",
                            value=key,
                        )

        if self.solid_solution:
            by_site: dict[str, float] = {}
            for entry in self.solid_solution:
                by_site[entry.site] = by_site.get(entry.site, 0.0) + entry.fraction
            for site, total in by_site.items():
                if abs(total - 1.0) > 1e-6:
                    raise SchemaError(
                        f"solid_solution fractions for site {site!r} must sum to 1.0",
                        field_path="solid_solution",
                        value={"site": site, "sum": total},
                    )

        if self.assembly is not None and self.assembly.completeness == "full":
            failures = self._check_full_completeness()
            if failures:
                raise SchemaError(
                    "assembly.completeness='full' but the contract is violated: "
                    + "; ".join(failures),
                    field_path="assembly.completeness",
                    value=failures,
                )

        return self


__all__ = [
    "ASSEMBLY_COMPLETENESS",
    "DEFECT_TYPES",
    "LINKER_ROLES",
    "OCCUPANT_TYPES",
    "Assembly",
    "AssemblyEdge",
    "AssemblyVertex",
    "Defect",
    "FunctionalGroup",
    "Linker",
    "MofidParsed",
    "ReticularStructure",
    "SolidSolutionEntry",
    "Topology",
    "completeness_clause4_satisfied",
    "has_standard_net_embedding",
    "parse_mofid",
    "parse_mofid_topology_field",
]
