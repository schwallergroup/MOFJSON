"""Dependency-free structural and referential validation for MOFJSON v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _duplicates(values: Iterable[Any]) -> set[Any]:
    seen: set[Any] = set()
    repeated: set[Any] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated


def _require_keys(
    value: dict[str, Any], keys: set[str], path: str, issues: list[ValidationIssue]
) -> None:
    for key in sorted(keys - value.keys()):
        issues.append(ValidationIssue(path, f"missing required key {key!r}"))


def _validate_vector(
    value: Any,
    length: int,
    path: str,
    issues: list[ValidationIssue],
    *,
    integer: bool = False,
) -> None:
    if not isinstance(value, list) or len(value) != length:
        issues.append(ValidationIssue(path, f"must be an array of length {length}"))
        return
    if integer:
        predicate = lambda item: isinstance(item, int) and not isinstance(item, bool)
    else:
        predicate = _is_number
    if not all(predicate(item) for item in value):
        kind = "integers" if integer else "numbers"
        issues.append(ValidationIssue(path, f"must contain only {kind}"))


def validate(document: Any) -> list[ValidationIssue]:
    """Validate the v0.1 schema plus graph references and connection-site use."""

    issues: list[ValidationIssue] = []
    if not isinstance(document, dict):
        return [ValidationIssue("$", "document must be a JSON object")]

    _require_keys(document, {"schema_version", "building_blocks", "net"}, "$", issues)
    if document.get("schema_version") != "0.1":
        issues.append(ValidationIssue("$.schema_version", "must equal '0.1'"))

    blocks = document.get("building_blocks")
    if not isinstance(blocks, list) or not blocks:
        issues.append(ValidationIssue("$.building_blocks", "must be a non-empty array"))
        blocks = []

    block_ids = [
        block.get("id")
        for block in blocks
        if isinstance(block, dict) and isinstance(block.get("id"), str)
    ]
    for repeated in sorted(_duplicates(block_ids), key=str):
        issues.append(ValidationIssue("$.building_blocks", f"duplicate id {repeated!r}"))

    sites_by_block: dict[str, set[str]] = {}
    for index, block in enumerate(blocks):
        path = f"$.building_blocks[{index}]"
        if not isinstance(block, dict):
            issues.append(ValidationIssue(path, "must be an object"))
            continue
        _require_keys(block, {"id", "role", "atom_graph", "connection_sites"}, path, issues)
        block_id = block.get("id")
        if not isinstance(block_id, str) or not block_id:
            issues.append(ValidationIssue(f"{path}.id", "must be a non-empty string"))
        if block.get("role") not in {"node", "linker"}:
            issues.append(ValidationIssue(f"{path}.role", "must be 'node' or 'linker'"))

        graph = block.get("atom_graph")
        if not isinstance(graph, dict):
            issues.append(ValidationIssue(f"{path}.atom_graph", "must be an object"))
            graph = {}
        _require_keys(graph, {"atoms", "bonds"}, f"{path}.atom_graph", issues)
        atoms = graph.get("atoms", [])
        bonds = graph.get("bonds", [])
        if not isinstance(atoms, list) or not atoms:
            issues.append(ValidationIssue(f"{path}.atom_graph.atoms", "must be non-empty"))
            atoms = []
        atom_ids = {
            atom.get("id")
            for atom in atoms
            if isinstance(atom, dict) and isinstance(atom.get("id"), str)
        }
        if len(atom_ids) != len(atoms):
            issues.append(
                ValidationIssue(
                    f"{path}.atom_graph.atoms", "atom ids must be present and unique"
                )
            )
        for atom_index, atom in enumerate(atoms):
            atom_path = f"{path}.atom_graph.atoms[{atom_index}]"
            if not isinstance(atom, dict):
                issues.append(ValidationIssue(atom_path, "must be an object"))
                continue
            _require_keys(atom, {"id", "element"}, atom_path, issues)
            if not isinstance(atom.get("element"), str) or not atom.get("element"):
                issues.append(ValidationIssue(f"{atom_path}.element", "must be a string"))

        if not isinstance(bonds, list):
            issues.append(ValidationIssue(f"{path}.atom_graph.bonds", "must be an array"))
            bonds = []
        for bond_index, bond in enumerate(bonds):
            bond_path = f"{path}.atom_graph.bonds[{bond_index}]"
            if not isinstance(bond, dict):
                issues.append(ValidationIssue(bond_path, "must be an object"))
                continue
            _require_keys(bond, {"atoms", "order"}, bond_path, issues)
            refs = bond.get("atoms")
            if not isinstance(refs, list) or len(refs) != 2:
                issues.append(ValidationIssue(f"{bond_path}.atoms", "must contain two atom ids"))
            else:
                for ref in refs:
                    if ref not in atom_ids:
                        issues.append(
                            ValidationIssue(f"{bond_path}.atoms", f"unknown atom id {ref!r}")
                        )
            if bond.get("order") not in {0, 1, 1.5, 2, 3, "coordination"}:
                issues.append(
                    ValidationIssue(
                        f"{bond_path}.order",
                        "must be 0, 1, 1.5, 2, 3, or 'coordination'",
                    )
                )

        sites = block.get("connection_sites")
        if not isinstance(sites, list) or not sites:
            issues.append(ValidationIssue(f"{path}.connection_sites", "must be non-empty"))
            sites = []
        site_ids = [
            site.get("id")
            for site in sites
            if isinstance(site, dict) and isinstance(site.get("id"), str)
        ]
        if len(set(site_ids)) != len(sites):
            issues.append(
                ValidationIssue(f"{path}.connection_sites", "site ids must be present and unique")
            )
        if isinstance(block_id, str):
            sites_by_block[block_id] = set(site_ids)
        for site_index, site in enumerate(sites):
            site_path = f"{path}.connection_sites[{site_index}]"
            if not isinstance(site, dict):
                issues.append(ValidationIssue(site_path, "must be an object"))
                continue
            _require_keys(site, {"id", "anchor_atoms"}, site_path, issues)
            anchors = site.get("anchor_atoms")
            if not isinstance(anchors, list) or not anchors:
                issues.append(ValidationIssue(f"{site_path}.anchor_atoms", "must be non-empty"))
            else:
                for ref in anchors:
                    if ref not in atom_ids:
                        issues.append(
                            ValidationIssue(f"{site_path}.anchor_atoms", f"unknown atom id {ref!r}")
                        )
            if "vector" in site:
                _validate_vector(site["vector"], 3, f"{site_path}.vector", issues)

    net = document.get("net")
    if not isinstance(net, dict):
        issues.append(ValidationIssue("$.net", "must be an object"))
        net = {}
    _require_keys(net, {"dimension", "vertices", "edges"}, "$.net", issues)
    if net.get("dimension") not in {1, 2, 3}:
        issues.append(ValidationIssue("$.net.dimension", "must be 1, 2, or 3"))

    vertices = net.get("vertices", [])
    if not isinstance(vertices, list) or not vertices:
        issues.append(ValidationIssue("$.net.vertices", "must be a non-empty array"))
        vertices = []
    vertex_ids = [
        item.get("id")
        for item in vertices
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if len(set(vertex_ids)) != len(vertices):
        issues.append(ValidationIssue("$.net.vertices", "vertex ids must be present and unique"))
    block_by_vertex: dict[str, str] = {}
    for index, vertex in enumerate(vertices):
        path = f"$.net.vertices[{index}]"
        if not isinstance(vertex, dict):
            issues.append(ValidationIssue(path, "must be an object"))
            continue
        _require_keys(vertex, {"id", "building_block"}, path, issues)
        if vertex.get("building_block") not in sites_by_block:
            issues.append(
                ValidationIssue(
                    f"{path}.building_block",
                    f"unknown building block {vertex.get('building_block')!r}",
                )
            )
        if isinstance(vertex.get("id"), str) and isinstance(vertex.get("building_block"), str):
            block_by_vertex[vertex["id"]] = vertex["building_block"]

    edges = net.get("edges", [])
    if not isinstance(edges, list) or not edges:
        issues.append(ValidationIssue("$.net.edges", "must be a non-empty array"))
        edges = []
    edge_ids = [
        item.get("id")
        for item in edges
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if len(set(edge_ids)) != len(edges):
        issues.append(ValidationIssue("$.net.edges", "edge ids must be present and unique"))

    used_sites: set[tuple[str, str]] = set()
    for edge_index, edge in enumerate(edges):
        edge_path = f"$.net.edges[{edge_index}]"
        if not isinstance(edge, dict):
            issues.append(ValidationIssue(edge_path, "must be an object"))
            continue
        _require_keys(edge, {"id", "source", "target"}, edge_path, issues)
        for endpoint_name in ("source", "target"):
            endpoint = edge.get(endpoint_name)
            path = f"{edge_path}.{endpoint_name}"
            if not isinstance(endpoint, dict):
                issues.append(ValidationIssue(path, "must be an object"))
                continue
            _require_keys(endpoint, {"vertex", "connection_site", "image"}, path, issues)
            vertex_id = endpoint.get("vertex")
            site_id = endpoint.get("connection_site")
            if vertex_id not in block_by_vertex:
                issues.append(ValidationIssue(f"{path}.vertex", f"unknown vertex {vertex_id!r}"))
            else:
                block_id = block_by_vertex[vertex_id]
                if site_id not in sites_by_block.get(block_id, set()):
                    issues.append(
                        ValidationIssue(
                            f"{path}.connection_site",
                            f"unknown site {site_id!r} on building block {block_id!r}",
                        )
                    )
                pair = (vertex_id, site_id)
                if pair in used_sites:
                    issues.append(
                        ValidationIssue(path, f"connection site {vertex_id!r}/{site_id!r} is reused")
                    )
                used_sites.add(pair)
            _validate_vector(endpoint.get("image"), 3, f"{path}.image", issues, integer=True)

    realization = document.get("realization")
    if realization is not None:
        if not isinstance(realization, dict):
            issues.append(ValidationIssue("$.realization", "must be an object"))
        else:
            lattice = realization.get("lattice_vectors")
            if not isinstance(lattice, list) or len(lattice) != 3:
                issues.append(
                    ValidationIssue("$.realization.lattice_vectors", "must contain three vectors")
                )
            else:
                for index, vector in enumerate(lattice):
                    _validate_vector(
                        vector, 3, f"$.realization.lattice_vectors[{index}]", issues
                    )
            atom_sites = realization.get("atom_sites", [])
            if not isinstance(atom_sites, list):
                issues.append(ValidationIssue("$.realization.atom_sites", "must be an array"))
            else:
                for index, atom_site in enumerate(atom_sites):
                    path = f"$.realization.atom_sites[{index}]"
                    if not isinstance(atom_site, dict):
                        issues.append(ValidationIssue(path, "must be an object"))
                        continue
                    _require_keys(atom_site, {"id", "element", "fractional"}, path, issues)
                    _validate_vector(atom_site.get("fractional"), 3, f"{path}.fractional", issues)

    return issues
