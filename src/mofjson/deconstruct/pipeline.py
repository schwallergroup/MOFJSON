"""Orchestration: run the deconstruction stages in order and report what happened.

Phase 08.8.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mofjson.deconstruct.archetype import ClassifyNode
from mofjson.deconstruct.connectivity import (
    BridgingModes,
    NodeConnectivity,
    apply_oms_to_sites,
    connectivity_archetype_warnings,
    summarize_bridging_ligands,
    _attachment_key,
)
from mofjson.deconstruct.linkers import DeduplicateLinkers, ExtractLinkers
from mofjson.deconstruct.partition import Partition, PartitionResult
from mofjson.deconstruct.site_geometry import AnalyseSites
from mofjson.deconstruct.topology import (
    BuildQuotientGraph,
    InferTopology,
    ReconcileTopology,
    topology_from_mofid,
)
from mofjson.exceptions import DeconstructionError, PartitionError
from mofjson.schema.document import MOFJSONDocument
from mofjson.schema.metal_node import (
    LigandCount,
    MetalCount,
    MetalMetalContact,
    MetalNode,
    NodeAttachmentPoint,
)
from mofjson.schema.metadata import DocumentProvenance, PipelineStep
from mofjson.schema.reticular import (
    Assembly,
    AssemblyEdge,
    AssemblyVertex,
    ReticularStructure,
    Topology,
)
from mofjson.vocab.donors import DONOR_GROUPS

GeometryMethodName = Literal["crystalnn", "chemenv", "both"]


@dataclass
class PipelineReport:
    """Per-stage success counts, failure categories, timings, distributions."""

    n_total: int = 0
    n_success: int = 0
    stage_success: dict[str, int] = field(default_factory=dict)
    stage_attempts: dict[str, int] = field(default_factory=dict)
    failure_categories: Counter = field(default_factory=Counter)
    archetype_hist: Counter = field(default_factory=Counter)
    connectivity_hist: Counter = field(default_factory=Counter)
    geometry_hist: Counter = field(default_factory=Counter)
    topology_hist: Counter = field(default_factory=Counter)
    topology_null: int = 0
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        attempts = self.stage_attempts or {k: self.n_total for k in self.stage_success}
        rates = {
            k: (self.stage_success.get(k, 0) / attempts.get(k, 1))
            for k in sorted(set(attempts) | set(self.stage_success))
        }
        return {
            "n_total": self.n_total,
            "n_success": self.n_success,
            "throughput_per_s": (self.n_total / self.elapsed_s) if self.elapsed_s > 0 else None,
            "stage_success": dict(self.stage_success),
            "stage_success_rate": rates,
            "failure_categories": dict(self.failure_categories.most_common(20)),
            "archetype_hist": dict(self.archetype_hist.most_common()),
            "connectivity_hist": {str(k): v for k, v in self.connectivity_hist.most_common()},
            "geometry_hist": dict(self.geometry_hist.most_common()),
            "topology_hist": dict(self.topology_hist.most_common()),
            "topology_null": self.topology_null,
            "topology_null_rate": (self.topology_null / self.n_total) if self.n_total else None,
            "elapsed_s": self.elapsed_s,
        }


def _node_kind(comp_periodicity: int, nuclearity: int) -> str:
    if comp_periodicity == 1:
        return "rod"
    if comp_periodicity == 2:
        return "layer"
    if nuclearity == 1:
        return "mononuclear"
    return "polynuclear"


def _donor_group_safe(group: str) -> str:
    return group if group in DONOR_GROUPS else "other"


def _build_metal_nodes(
    partition: PartitionResult,
    structure: Any,
    *,
    geometry: bool,
    method: GeometryMethodName,
) -> tuple[list[MetalNode], list[str]]:
    warnings: list[str] = []
    nodes: list[MetalNode] = []
    geom_method: GeometryMethodName = method if geometry else "crystalnn"

    for comp in partition.metal_components:
        metal_idxs = sorted(comp.metal_indices)
        site_ids = [f"M{k + 1}" for k in range(len(metal_idxs))]
        try:
            sites = AnalyseSites(
                structure,
                metal_idxs,
                method=geom_method if geometry else "crystalnn",
                site_ids=site_ids,
                elements=partition.elements,
                adjacency=partition.adjacency,
                # Reuse Phase 07 bonds — avoid a second CrystalNN pass on bulk runs.
                use_bond_graph=(not geometry) or method == "crystalnn",
            )
            if not geometry:
                sites = [
                    s.model_copy(
                        update={
                            "geometry": "unknown",
                            "geometry_symbol": None,
                            "geometry_csm": None,
                            "geometry_source": None,
                            "geometry_confidence": None,
                        }
                    )
                    for s in sites
                ]
        except Exception as exc:
            warnings.append(f"geometry_failed:{comp.id}:{exc}")
            sites = []

        metals_counter = Counter(partition.elements[i] for i in metal_idxs)
        metals = [MetalCount(element=el, count=n) for el, n in sorted(metals_counter.items())]
        kind = _node_kind(comp.periodicity, comp.nuclearity)
        rod_axis = None
        if kind in ("rod", "layer") and comp.period_axes:
            axis = [0.0, 0.0, 0.0]
            for ax in comp.period_axes:
                axis[ax] = 1.0
            rod_axis = axis

        bridging_assignments = BridgingModes(partition, comp)
        bridging_dicts = summarize_bridging_ligands(bridging_assignments)
        bridging_ligands = [
            LigandCount(
                formula=d["formula"],
                bridging_mode=d["bridging_mode"],
                count=d["count"],
            )
            for d in bridging_dicts
        ] or None

        connectivity = NodeConnectivity(partition, comp)

        aps: list[NodeAttachmentPoint] = []
        seen: set[tuple[str, int]] = set()
        ap_n = 0
        for cb in comp.cut_bonds:
            key = _attachment_key(cb.donor_index, partition)
            if key[1] < 0 or key in seen:
                continue
            seen.add(key)
            ap_n += 1
            donor_id = partition.site_ids[cb.donor_index]
            group = "carboxylate_O" if key[0] == "carboxylate" else "other"
            if sites:
                for s in sites:
                    for d in s.donors or []:
                        if d.element == partition.elements[cb.donor_index]:
                            group = d.donor_group
                            break
            aps.append(
                NodeAttachmentPoint(
                    id=f"X{ap_n}",
                    atom_id=donor_id,
                    donor_group=_donor_group_safe(group),
                    linker_id=None,
                )
            )

        mm: list[MetalMetalContact] = []
        if structure is not None and len(metal_idxs) >= 2:
            for a in range(len(metal_idxs)):
                for b in range(a + 1, len(metal_idxs)):
                    i, j = metal_idxs[a], metal_idxs[b]
                    try:
                        dist = float(structure.get_distance(i, j))
                    except Exception:
                        continue
                    if dist < 4.0:
                        mm.append(
                            MetalMetalContact(
                                source=site_ids[a],
                                target=site_ids[b],
                                distance=round(dist, 4),
                                bonded=dist < 3.0,
                            )
                        )

        centroid = None
        if structure is not None and metal_idxs:
            fracs = [structure[i].frac_coords for i in metal_idxs]
            c = sum(fracs) / len(fracs)
            centroid = [round(float(x), 6) for x in c]

        formula_parts = [f"{el}{n if n > 1 else ''}" for el, n in sorted(metals_counter.items())]
        if bridging_ligands:
            for lig in bridging_ligands:
                suffix = str(lig.count) if lig.count > 1 else ""
                formula_parts.append(f"({lig.formula}){suffix}")
        formula = "".join(formula_parts) if formula_parts else None

        node = MetalNode(
            id=comp.id,
            kind=kind,
            archetype="unknown",
            formula=formula,
            nuclearity=comp.nuclearity,
            metals=metals,
            bridging_ligands=bridging_ligands,
            connectivity=connectivity,
            connectivity_source="computed",
            attachment_points=aps or None,
            sites=sites or None,
            metal_metal_contacts=mm or None,
            rod_axis=rod_axis,
            centroid_frac=centroid,
        )
        match = ClassifyNode(node)
        node = node.model_copy(update={"archetype": match.archetype})
        if match.archetype == "other" and match.near_misses:
            warnings.append(f"archetype_other:{comp.id}:{match.near_misses[0]}")
        node = apply_oms_to_sites(node, require_activation=True)
        warnings.extend(connectivity_archetype_warnings(node))
        nodes.append(node)
    return nodes, warnings


def _dedupe_nodes(nodes: list[MetalNode]) -> tuple[list[MetalNode], dict[str, int]]:
    """Collapse identical node instances to types; return (types, stoichiometry)."""
    groups: dict[tuple, list[MetalNode]] = {}
    for node in nodes:
        key = (
            node.archetype,
            node.kind,
            node.nuclearity,
            node.connectivity,
            tuple((m.element, m.count) for m in node.metals),
        )
        groups.setdefault(key, []).append(node)
    types: list[MetalNode] = []
    stoich: dict[str, int] = {}
    for i, (_key, group) in enumerate(groups.items(), start=1):
        rep = group[0].model_copy(update={"id": f"N{i}"})
        types.append(rep)
        stoich[rep.id] = len(group)
    return types, stoich


def _link_attachment_linker_ids(
    nodes: list[MetalNode],
    linkers: list[Any],
    partition: PartitionResult,
) -> list[MetalNode]:
    formula_to_lid: dict[tuple, str] = {}
    for linker in linkers:
        if linker.role != "linker":
            continue
        for comp in partition.organic_components:
            if comp.role != "linker":
                continue
            key = tuple(sorted(Counter(comp.elements).items()))
            formula_to_lid.setdefault(key, linker.id)

    donor_to_linker: dict[str, str] = {}
    for comp in partition.organic_components:
        if comp.role != "linker":
            continue
        key = tuple(sorted(Counter(comp.elements).items()))
        lid = formula_to_lid.get(key)
        if lid is None:
            continue
        for idx in comp.site_indices:
            donor_to_linker[partition.site_ids[idx]] = lid

    out = []
    for node in nodes:
        if not node.attachment_points:
            out.append(node)
            continue
        new_aps = [
            ap.model_copy(update={"linker_id": donor_to_linker.get(ap.atom_id)})
            for ap in node.attachment_points
        ]
        out.append(node.model_copy(update={"attachment_points": new_aps}))
    return out


def Deconstruct(
    document: MOFJSONDocument,
    *,
    geometry: bool = True,
    method: GeometryMethodName = "chemenv",
    timeout: float | None = None,
) -> MOFJSONDocument:
    """Run partition → geometry → archetype → connectivity → linkers → topology."""
    _ = timeout
    warnings: list[str] = list(document.warnings or [])
    prov = document.provenance or DocumentProvenance()
    pipeline = list(prov.pipeline or [])

    if document.crystal is None:
        raise DeconstructionError("document has no crystal layer", field_path="crystal")
    if not document.crystal.bonds:
        from mofjson.convert.bonds import DeriveBonds
        from mofjson.convert.pymatgen_bridge import MOFToStructure

        structure = MOFToStructure(document)
        bonds, bond_warns = DeriveBonds(
            structure,
            method="crystalnn",
            site_ids=[s.id for s in document.crystal.sites],
        )
        warnings.extend(bond_warns)
        document = document.model_copy(
            update={
                "crystal": document.crystal.model_copy(
                    update={"bonds": bonds, "bond_graph_source": "crystalnn"}
                )
            }
        )

    from mofjson.convert.pymatgen_bridge import MOFToStructure

    structure = MOFToStructure(document)

    try:
        partition = Partition(document)
        pipeline.append(
            PipelineStep(step="partition", tool="mofjson", params={"keep_solvent": False})
        )
        warnings.extend(partition.warnings)
    except PartitionError:
        raise
    except Exception as exc:
        raise PartitionError(f"partition failed: {exc}") from exc

    nodes, node_warns = _build_metal_nodes(
        partition, structure, geometry=geometry, method=method
    )
    warnings.extend(node_warns)
    pipeline.append(
        PipelineStep(
            step="node_geometry",
            tool="chemenv" if geometry and method != "crystalnn" else "crystalnn",
            params={
                "geometry": geometry,
                "method": method,
                "distance_cutoff": 1.4,
                "angle_cutoff": 0.3,
            },
        )
    )
    pipeline.append(PipelineStep(step="archetype", tool="mofjson", params={}))
    pipeline.append(PipelineStep(step="connectivity", tool="mofjson", params={}))

    try:
        linkers = ExtractLinkers(partition)
        linkers = DeduplicateLinkers(linkers)
        pipeline.append(PipelineStep(step="linkers", tool="rdkit", params={"form": "anionic"}))
    except Exception as exc:
        warnings.append(f"linkers_failed:{exc}")
        linkers = []

    nodes = _link_attachment_linker_ids(nodes, linkers, partition)
    node_types, node_stoich = _dedupe_nodes(nodes)

    # Remap linker ids referenced on attachment points after node re-id (ids unchanged for linkers).
    try:
        qg = BuildQuotientGraph(partition)
        inferred = InferTopology(qg, node_types, [lk for lk in linkers if lk.role == "linker"])
        asserted = None
        if document.identifiers and document.identifiers.mofid:
            asserted = topology_from_mofid(document.identifiers.mofid)
        topology = ReconcileTopology(inferred, asserted)
        pipeline.append(
            PipelineStep(
                step="topology",
                tool="mofjson",
                params={"signature": topology.vertex_connectivity},
            )
        )
    except Exception as exc:
        warnings.append(f"topology_failed:{exc}")
        topology = Topology(rcsr_symbol=None, n_nets=0, source="manual", confidence="none")

    linker_types = [lk for lk in linkers if lk.role == "linker"]
    guests = [lk for lk in linkers if lk.role != "linker"]

    vertices = [
        AssemblyVertex(
            vertex_id=f"v_{n.id}",
            occupant_type="node",
            occupant_id=n.id,
            connectivity=n.connectivity or 0,
        )
        for n in node_types
    ]
    for linker in linker_types:
        vertices.append(
            AssemblyVertex(
                vertex_id=f"v_{linker.id}",
                occupant_type="linker",
                occupant_id=linker.id,
                connectivity=linker.connectivity or 0,
            )
        )
    # Build edges between type vertices when any instance edge exists.
    type_by_key = {
        (
            n.archetype,
            n.kind,
            n.nuclearity,
            n.connectivity,
            tuple((m.element, m.count) for m in n.metals),
        ): n.id
        for n in node_types
    }
    orig_key = {
        n.id: (
            n.archetype,
            n.kind,
            n.nuclearity,
            n.connectivity,
            tuple((m.element, m.count) for m in n.metals),
        )
        for n in nodes
    }
    # Original component ids are N1.. before dedupe; nodes still have those ids.
    linker_comp_to_type: dict[str, str] = {}
    for comp in partition.organic_components:
        if comp.role != "linker":
            continue
        for linker in linker_types:
            if linker.fragment and len(linker.fragment.atoms) == len(comp.site_indices):
                linker_comp_to_type[comp.id] = linker.id
                break
        if comp.id not in linker_comp_to_type and len(linker_types) == 1:
            linker_comp_to_type[comp.id] = linker_types[0].id

    edge_set: set[tuple[str, str]] = set()
    for nid, lid in BuildQuotientGraph(partition).edges:
        # nid is component id like N1
        if nid not in orig_key:
            continue
        tid = type_by_key.get(orig_key[nid])
        # Find linker type for component lid (O1..)
        ltid = linker_comp_to_type.get(lid)
        if tid and ltid:
            edge_set.add((f"v_{tid}", f"v_{ltid}"))
    edges = [AssemblyEdge(source=a, target=b) for a, b in sorted(edge_set)]

    stoich = dict(node_stoich)
    for linker in linker_types:
        stoich[linker.id] = linker.count_per_cell or 1

    # Align topology.vertex_connectivity with assembly vertices (schema invariant).
    if vertices:
        topology = topology.model_copy(
            update={"vertex_connectivity": [v.connectivity for v in vertices]}
        )

    assembly = Assembly(
        vertices=vertices,
        edges=edges or None,
        stoichiometry=stoich,
        completeness="partial",
    )

    reticular = ReticularStructure(
        topology=topology,
        nodes=node_types,
        linkers=linker_types + guests,
        assembly=assembly,
    )

    new_prov = DocumentProvenance(
        sources=list(prov.sources or []),
        pipeline=pipeline,
        warnings=list(dict.fromkeys(list(prov.warnings or []) + warnings)),
    )
    return document.model_copy(
        update={
            "reticular": reticular,
            "provenance": new_prov,
            "warnings": list(dict.fromkeys(list(document.warnings or []) + warnings)),
        }
    )


def DeconstructCorpus(
    paths_or_documents: Iterable[str | Path | MOFJSONDocument],
    *,
    workers: int | None = None,
    out: str | Path | None = None,
    geometry: bool = True,
    method: GeometryMethodName = "chemenv",
    timeout: float | None = 120.0,
) -> PipelineReport:
    """Run :func:`Deconstruct` over many structures; stream NDJSON if ``out`` is set."""
    _ = workers
    from mofjson.convert.cif import MOFFromCif
    from mofjson.util.io import ResumableWriter

    report = PipelineReport()
    t0 = time.perf_counter()
    stages = [
        "partition",
        "node_geometry",
        "archetype",
        "connectivity",
        "linkers",
        "topology",
    ]
    for s in stages:
        report.stage_success[s] = 0
        report.stage_attempts[s] = 0

    def _key_fn(obj: dict[str, Any]) -> str:
        ids = obj.get("identifiers") or {}
        return str(
            ids.get("source_filename")
            or ids.get("structure_stem")
            or ids.get("mofjson_id")
            or obj.get("_key")
            or ""
        )

    items = list(paths_or_documents)

    def _run_one(item: str | Path | MOFJSONDocument) -> None:
        nonlocal report
        report.n_total += 1
        try:
            if isinstance(item, MOFJSONDocument):
                doc = item
                key = (
                    (doc.identifiers.source_filename if doc.identifiers else None)
                    or (doc.identifiers.structure_stem if doc.identifiers else None)
                    or f"doc-{report.n_total}"
                )
            else:
                path = Path(item)
                key = path.name
                doc = MOFFromCif(path, derive_bonds=True)
                if doc.identifiers is None:
                    from mofjson.schema.metadata import Identifiers

                    doc = doc.model_copy(
                        update={"identifiers": Identifiers(source_filename=path.name)}
                    )
            for s in stages:
                report.stage_attempts[s] += 1
            result = Deconstruct(doc, geometry=geometry, method=method, timeout=timeout)
            ret = result.reticular
            assert ret is not None
            step_names = {
                s.step for s in ((result.provenance.pipeline if result.provenance else []) or [])
            }
            for s in stages:
                if s in step_names:
                    report.stage_success[s] += 1
            for n in ret.nodes or []:
                report.archetype_hist[n.archetype] += 1
                if n.connectivity is not None:
                    report.connectivity_hist[n.connectivity] += 1
                for site in n.sites or []:
                    report.geometry_hist[site.geometry] += 1
            sym = ret.topology.rcsr_symbol if ret.topology else None
            if sym is None:
                report.topology_null += 1
                report.topology_hist["null"] += 1
            elif isinstance(sym, str):
                report.topology_hist[sym] += 1
            else:
                report.topology_hist[",".join(sym)] += 1
            report.n_success += 1
            if out is not None:
                payload = result.model_dump(mode="json")
                payload["_key"] = key
                return payload
        except Exception as exc:
            report.failure_categories[type(exc).__name__] += 1
            report.warnings.append(f"{item}: {exc}")
        return None

    if out is not None:
        with ResumableWriter(out, key_fn=_key_fn) as writer:
            for item in items:
                if isinstance(item, (str, Path)) and Path(item).name in writer.seen:
                    continue
                payload = _run_one(item)
                if payload is not None:
                    writer.write(payload)
    else:
        for item in items:
            _run_one(item)

    report.elapsed_s = time.perf_counter() - t0
    return report


__all__ = ["Deconstruct", "DeconstructCorpus", "PipelineReport"]
