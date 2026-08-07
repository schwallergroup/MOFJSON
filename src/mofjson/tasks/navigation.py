"""Family B -- navigation: integer-answer questions about structure. Tests reading only.

Phase 12.3.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from mofjson.llm.prompts import RenderPrompt
from mofjson.schema.document import MOFJSONDocument
from mofjson.tasks.item import ItemProvenance, ItemSource, ItemStrata, TaskItem
from mofjson.tasks.representations import Render, representation_can_ask
from mofjson.vocab.elements import METALS
from mofjson.vocab.topology import UNAMBIGUOUS_SIGNATURES

GENERATOR_VERSION = "tasks/navigation.py@v1"


@dataclass(frozen=True, slots=True)
class NavigationTask:
    id: str
    question_template: str
    answer_type: str  # integer | boolean | string
    grader: str
    response_schema: str
    needs_paths: bool = True
    needs_coords: bool = True
    ground_truth: Callable[[MOFJSONDocument], list[dict[str, Any]]] | None = None


def _site_by_id(doc: MOFJSONDocument) -> dict[str, Any]:
    if doc.crystal is None:
        return {}
    return {s.id: s for s in doc.crystal.sites}


def _metal_sites(doc: MOFJSONDocument) -> list[Any]:
    if doc.crystal is None:
        return []
    return [s for s in doc.crystal.sites if s.element in METALS or s.is_metal]


def _ensure_bonds(doc: MOFJSONDocument) -> MOFJSONDocument:
    if doc.crystal is None:
        return doc
    if doc.crystal.bonds:
        return doc
    from mofjson.convert.bonds import DeriveBonds
    from mofjson.convert.pymatgen_bridge import MOFToStructure

    structure = MOFToStructure(doc)
    bonds, _warns = DeriveBonds(
        structure,
        method="crystalnn",
        site_ids=[s.id for s in doc.crystal.sites],
    )
    crystal = doc.crystal.model_copy(update={"bonds": bonds, "bond_graph_source": "crystalnn"})
    return doc.model_copy(update={"crystal": crystal})


def _bond_graph(doc: MOFJSONDocument) -> Any:
    import networkx as nx

    doc = _ensure_bonds(doc)
    g = nx.Graph()
    if doc.crystal is None:
        return g
    for s in doc.crystal.sites:
        g.add_node(s.id, element=s.element)
    for b in doc.crystal.bonds:
        # Collapse periodic images for shortest-path bond counts (MolJSON-style).
        if g.has_edge(b.source, b.target):
            continue
        g.add_edge(b.source, b.target, image=tuple(b.image or [0, 0, 0]))
    return g


def gt_metal_to_metal_path(doc: MOFJSONDocument, *, max_pairs: int = 16) -> list[dict[str, Any]]:
    import networkx as nx

    metals = _metal_sites(doc)
    if len(metals) < 2:
        return []
    g = _bond_graph(doc)
    out: list[dict[str, Any]] = []
    for a, b in itertools.combinations(metals, 2):
        if len(out) >= max_pairs:
            break
        if a.id not in g or b.id not in g:
            continue
        try:
            length = nx.shortest_path_length(g, a.id, b.id)
        except nx.NetworkXNoPath:
            continue
        if not (2 <= length <= 18):
            continue
        out.append(
            {
                "answer": int(length),
                "vars": {
                    "M1": f"{a.element} ({a.id})",
                    "M2": f"{b.element} ({b.id})",
                },
                "question": (
                    f"How many bonds are on the shortest path between metal centres "
                    f"{a.element} ({a.id}) and {b.element} ({b.id})?"
                ),
                "method": "networkx_shortest_path",
                "strata": {"path_length": int(length)},
            }
        )
    return out


def gt_metal_to_functional_group(doc: MOFJSONDocument) -> list[dict[str, Any]]:
    import networkx as nx

    if doc.crystal is None:
        return []
    metals = _metal_sites(doc)
    # NH2 nitrogen heuristic: N bonded to exactly two H (or labelled) — use element N
    # that is not a metal donor in carboxylate sense; keep simple: N sites.
    nitrogens = [s for s in doc.crystal.sites if s.element == "N"]
    if not metals or not nitrogens:
        return []
    g = _bond_graph(doc)
    out: list[dict[str, Any]] = []
    for m in metals:
        best = None
        best_n = None
        for n in nitrogens:
            if m.id not in g or n.id not in g:
                continue
            try:
                length = nx.shortest_path_length(g, m.id, n.id)
            except nx.NetworkXNoPath:
                continue
            if best is None or length < best:
                best = length
                best_n = n
        if best is None or best_n is None or not (1 <= best <= 18):
            continue
        out.append(
            {
                "answer": int(best),
                "vars": {"M1": f"{m.element} ({m.id})", "N": f"{best_n.id}"},
                "question": (
                    f"How many bonds are on the shortest path from metal {m.element} ({m.id}) "
                    f"to the nearest nitrogen atom ({best_n.id})?"
                ),
                "method": "networkx_shortest_path_fg",
                "strata": {"path_length": int(best)},
            }
        )
    return out


def gt_donor_to_donor_path(doc: MOFJSONDocument) -> list[dict[str, Any]]:
    """Bonds between two carboxylate O atoms of the first linker, if reticular present."""
    import networkx as nx

    if doc.reticular is None or doc.crystal is None:
        return []
    ret = doc.reticular
    linkers = ret.linkers if hasattr(ret, "linkers") else (ret.get("linkers") if isinstance(ret, dict) else None)
    if not linkers:
        return []
    linker = linkers[0]
    lid = linker.id if hasattr(linker, "id") else linker.get("id", "L1")
    donors = linker.donor_groups if hasattr(linker, "donor_groups") else linker.get("donor_groups")
    # Fall back: pick two O sites assigned to this linker.
    oxygens = [s for s in doc.crystal.sites if s.element == "O" and (s.linker_id == lid or s.linker_id is None)]
    if len(oxygens) < 2:
        return []
    g = _bond_graph(doc)
    o1, o2 = oxygens[0], oxygens[1]
    if o1.id not in g or o2.id not in g:
        return []
    try:
        length = nx.shortest_path_length(g, o1.id, o2.id)
    except nx.NetworkXNoPath:
        return []
    if not (1 <= length <= 18):
        return []
    _ = donors
    return [
        {
            "answer": int(length),
            "vars": {"L1": lid, "O1": o1.id, "O2": o2.id},
            "question": (
                f"How many bonds separate carboxylate oxygen atoms {o1.id} and {o2.id} "
                f"of linker {lid}?"
            ),
            "method": "networkx_linker_donor_path",
            "strata": {"path_length": int(length)},
        }
    ]


def gt_across_linker_path(doc: MOFJSONDocument) -> list[dict[str, Any]]:
    """Bonds between two nodes bridged by the first linker (metal of each node)."""
    import networkx as nx

    if doc.reticular is None:
        return []
    ret = doc.reticular
    nodes = ret.nodes if hasattr(ret, "nodes") else (ret.get("nodes") if isinstance(ret, dict) else None)
    linkers = ret.linkers if hasattr(ret, "linkers") else (ret.get("linkers") if isinstance(ret, dict) else None)
    if not nodes or len(nodes) < 1 or not linkers:
        return []
    # Use two metal sites from crystal as proxy for node-to-node across linker.
    metals = _metal_sites(doc)
    if len(metals) < 2:
        return []
    g = _bond_graph(doc)
    a, b = metals[0], metals[-1]
    try:
        length = nx.shortest_path_length(g, a.id, b.id)
    except nx.NetworkXNoPath:
        return []
    if not (2 <= length <= 18):
        return []
    lid = linkers[0].id if hasattr(linkers[0], "id") else linkers[0].get("id", "L1")
    return [
        {
            "answer": int(length),
            "vars": {"L1": lid, "M1": a.id, "M2": b.id},
            "question": (
                f"How many bonds are on the shortest path between the two metal centres "
                f"{a.id} and {b.id} bridged by linker {lid}?"
            ),
            "method": "networkx_across_linker",
            "strata": {"path_length": int(length)},
        }
    ]


def gt_coordination_number(doc: MOFJSONDocument) -> list[dict[str, Any]]:
    doc = _ensure_bonds(doc)
    out: list[dict[str, Any]] = []
    # Prefer reticular site CN when present.
    if doc.reticular is not None:
        nodes = doc.reticular.nodes if hasattr(doc.reticular, "nodes") else doc.reticular.get("nodes")
        if nodes:
            for node in nodes:
                sites = node.sites if hasattr(node, "sites") else node.get("sites")
                if not sites:
                    continue
                for site in sites:
                    sid = site.site_id if hasattr(site, "site_id") else site.get("site_id")
                    el = site.element if hasattr(site, "element") else site.get("element")
                    cn = site.coordination_number if hasattr(site, "coordination_number") else site.get("coordination_number")
                    if cn is None:
                        continue
                    out.append(
                        {
                            "answer": int(cn),
                            "vars": {"M1": f"{el} ({sid})"},
                            "question": f"What is the coordination number of metal site {el} ({sid})?",
                            "method": "reticular_metal_site_cn",
                            "strata": {},
                        }
                    )
            if out:
                return out
    # Fallback: degree in bond graph for metal sites.
    g = _bond_graph(doc)
    for s in _metal_sites(doc):
        if s.id not in g:
            continue
        cn = int(g.degree(s.id))
        out.append(
            {
                "answer": cn,
                "vars": {"M1": f"{s.element} ({s.id})"},
                "question": f"What is the coordination number of metal site {s.element} ({s.id})?",
                "method": "bond_graph_degree",
                "strata": {},
            }
        )
    return out


def gt_node_nuclearity(doc: MOFJSONDocument) -> list[dict[str, Any]]:
    if doc.reticular is None:
        return []
    nodes = doc.reticular.nodes if hasattr(doc.reticular, "nodes") else doc.reticular.get("nodes")
    if not nodes:
        return []
    out: list[dict[str, Any]] = []
    for node in nodes:
        nuc = node.nuclearity if hasattr(node, "nuclearity") else node.get("nuclearity")
        nid = node.id if hasattr(node, "id") else node.get("id")
        sites = node.sites if hasattr(node, "sites") else node.get("sites")
        m1 = None
        if sites:
            s0 = sites[0]
            m1 = s0.site_id if hasattr(s0, "site_id") else s0.get("site_id")
        if nuc is None:
            continue
        label = m1 or nid
        out.append(
            {
                "answer": int(nuc),
                "vars": {"M1": label, "N": nid},
                "question": f"How many metal atoms are in the node containing site {label}?",
                "method": "reticular_nuclearity",
                "strata": {},
            }
        )
    return out


def gt_node_connectivity(doc: MOFJSONDocument) -> list[dict[str, Any]]:
    if doc.reticular is None:
        return []
    nodes = doc.reticular.nodes if hasattr(doc.reticular, "nodes") else doc.reticular.get("nodes")
    if not nodes:
        return []
    out: list[dict[str, Any]] = []
    for node in nodes:
        conn = node.connectivity if hasattr(node, "connectivity") else node.get("connectivity")
        nid = node.id if hasattr(node, "id") else node.get("id")
        sites = node.sites if hasattr(node, "sites") else node.get("sites")
        m1 = None
        if sites:
            s0 = sites[0]
            m1 = s0.site_id if hasattr(s0, "site_id") else s0.get("site_id")
        if conn is None:
            continue
        label = m1 or nid
        out.append(
            {
                "answer": int(conn),
                "vars": {"M1": label},
                "question": f"How many linkers attach to the node containing site {label}?",
                "method": "reticular_connectivity",
                "strata": {"node_connectivity": int(conn)},
            }
        )
    return out


def gt_distinct_linkers(doc: MOFJSONDocument) -> list[dict[str, Any]]:
    if doc.reticular is None:
        return []
    linkers = doc.reticular.linkers if hasattr(doc.reticular, "linkers") else doc.reticular.get("linkers")
    if not linkers:
        return []
    smiles_set: set[str] = set()
    for linker in linkers:
        smi = linker.smiles if hasattr(linker, "smiles") else linker.get("smiles")
        if smi:
            smiles_set.add(smi)
        else:
            fid = linker.id if hasattr(linker, "id") else linker.get("id")
            smiles_set.add(str(fid))
    n = len(smiles_set)
    return [
        {
            "answer": n,
            "vars": {},
            "question": "How many chemically distinct linkers are in the framework?",
            "method": "canonical_smiles_set",
            "strata": {"n_distinct_linkers": n},
        }
    ]


def gt_metals_per_cell(doc: MOFJSONDocument) -> list[dict[str, Any]]:
    metals = _metal_sites(doc)
    n = len(metals)
    if n == 0:
        return []
    return [
        {
            "answer": n,
            "vars": {},
            "question": "How many metal atoms are in the unit cell?",
            "method": "site_count",
            "strata": {"n_metals": n},
        }
    ]


def gt_periodic_bond(doc: MOFJSONDocument) -> list[dict[str, Any]]:
    doc = _ensure_bonds(doc)
    if doc.crystal is None:
        return []
    out: list[dict[str, Any]] = []
    periodic = [b for b in doc.crystal.bonds if b.image and list(b.image) != [0, 0, 0]]
    nonperiodic = [b for b in doc.crystal.bonds if not b.image or list(b.image) == [0, 0, 0]]
    # One yes and one no when available.
    for b, ans in ((periodic[0], True) if periodic else (None, None),):
        if b is None:
            break
        out.append(
            {
                "answer": True,
                "vars": {"X": b.source, "Y": b.target},
                "question": (
                    f"Are atoms {b.source} and {b.target} bonded across a cell boundary?"
                ),
                "method": "bond_image",
                "strata": {},
            }
        )
    if nonperiodic:
        b = nonperiodic[0]
        out.append(
            {
                "answer": False,
                "vars": {"X": b.source, "Y": b.target},
                "question": (
                    f"Are atoms {b.source} and {b.target} bonded across a cell boundary?"
                ),
                "method": "bond_image",
                "strata": {},
            }
        )
    return out


def gt_topology_from_connectivity(doc: MOFJSONDocument) -> list[dict[str, Any]]:
    if doc.reticular is None:
        return []
    ret = doc.reticular
    nodes = ret.nodes if hasattr(ret, "nodes") else ret.get("nodes")
    linkers = ret.linkers if hasattr(ret, "linkers") else ret.get("linkers")
    if not nodes or not linkers:
        return []
    node_conn = nodes[0].connectivity if hasattr(nodes[0], "connectivity") else nodes[0].get("connectivity")
    linker_conn = linkers[0].connectivity if hasattr(linkers[0], "connectivity") else linkers[0].get("connectivity")
    if node_conn is None or linker_conn is None:
        return []
    sig = (int(node_conn), int(linker_conn))
    if sig not in UNAMBIGUOUS_SIGNATURES:
        return []
    net = UNAMBIGUOUS_SIGNATURES[sig]
    return [
        {
            "answer": net,
            "vars": {"node_c": node_conn, "linker_c": linker_conn},
            "question": (
                f"Given node connectivity {node_conn} and linker connectivity {linker_conn}, "
                f"what is the RCSR net symbol?"
            ),
            "method": "unambiguous_signature",
            "strata": {"topology": net, "node_connectivity": int(node_conn)},
        }
    ]


def _make_tasks() -> dict[str, NavigationTask]:
    return {
        "metal_to_metal_path": NavigationTask(
            id="metal_to_metal_path",
            question_template="",
            answer_type="integer",
            grader="exact_int",
            response_schema="profiles/answer_integer.schema.json",
            ground_truth=gt_metal_to_metal_path,
        ),
        "metal_to_functional_group": NavigationTask(
            id="metal_to_functional_group",
            question_template="",
            answer_type="integer",
            grader="exact_int",
            response_schema="profiles/answer_integer.schema.json",
            ground_truth=gt_metal_to_functional_group,
        ),
        "donor_to_donor_path": NavigationTask(
            id="donor_to_donor_path",
            question_template="",
            answer_type="integer",
            grader="exact_int",
            response_schema="profiles/answer_integer.schema.json",
            ground_truth=gt_donor_to_donor_path,
        ),
        "across_linker_path": NavigationTask(
            id="across_linker_path",
            question_template="",
            answer_type="integer",
            grader="exact_int",
            response_schema="profiles/answer_integer.schema.json",
            ground_truth=gt_across_linker_path,
        ),
        "coordination_number": NavigationTask(
            id="coordination_number",
            question_template="",
            answer_type="integer",
            grader="exact_int",
            response_schema="profiles/answer_integer.schema.json",
            needs_paths=False,
            ground_truth=gt_coordination_number,
        ),
        "node_nuclearity": NavigationTask(
            id="node_nuclearity",
            question_template="",
            answer_type="integer",
            grader="exact_int",
            response_schema="profiles/answer_integer.schema.json",
            needs_paths=False,
            needs_coords=False,
            ground_truth=gt_node_nuclearity,
        ),
        "node_connectivity": NavigationTask(
            id="node_connectivity",
            question_template="",
            answer_type="integer",
            grader="exact_int",
            response_schema="profiles/answer_integer.schema.json",
            needs_paths=False,
            needs_coords=False,
            ground_truth=gt_node_connectivity,
        ),
        "distinct_linkers": NavigationTask(
            id="distinct_linkers",
            question_template="",
            answer_type="integer",
            grader="exact_int",
            response_schema="profiles/answer_integer.schema.json",
            needs_paths=False,
            needs_coords=False,
            ground_truth=gt_distinct_linkers,
        ),
        "metals_per_cell": NavigationTask(
            id="metals_per_cell",
            question_template="",
            answer_type="integer",
            grader="exact_int",
            response_schema="profiles/answer_integer.schema.json",
            needs_paths=False,
            ground_truth=gt_metals_per_cell,
        ),
        "periodic_bond": NavigationTask(
            id="periodic_bond",
            question_template="",
            answer_type="boolean",
            grader="exact_bool",
            response_schema="profiles/answer_boolean.schema.json",
            ground_truth=gt_periodic_bond,
        ),
        "topology_from_connectivity": NavigationTask(
            id="topology_from_connectivity",
            question_template="",
            answer_type="string",
            grader="canonical_string",
            response_schema="profiles/answer_string.schema.json",
            needs_paths=False,
            needs_coords=False,
            ground_truth=gt_topology_from_connectivity,
        ),
    }


NAVIGATION_TASKS: dict[str, NavigationTask] = _make_tasks()


def _source_from_doc(doc: MOFJSONDocument) -> ItemSource:
    ids = doc.identifiers
    return ItemSource(
        mofjson_id=ids.mofjson_id if ids else None,
        odac_name=ids.odac_name if ids else None,
        csd_refcode=ids.csd_refcode if ids else None,
        structure_stem=ids.structure_stem if ids else None,
        is_hypothetical=bool(ids.is_hypothetical) if ids else False,
    )


def _strata_from_doc(doc: MOFJSONDocument, extra: dict[str, Any] | None = None) -> ItemStrata:
    n_atoms = None
    n_metals = None
    if doc.crystal is not None:
        n_atoms = doc.crystal.n_atoms or len(doc.crystal.sites)
        n_metals = len(_metal_sites(doc))
    topo = None
    node_conn = None
    n_linkers = None
    if doc.reticular is not None:
        ret = doc.reticular
        if hasattr(ret, "topology") and ret.topology:
            topo = ret.topology.rcsr_symbol
        elif isinstance(ret, dict):
            topo = (ret.get("topology") or {}).get("rcsr_symbol")
        nodes = ret.nodes if hasattr(ret, "nodes") else (ret.get("nodes") if isinstance(ret, dict) else None)
        linkers = ret.linkers if hasattr(ret, "linkers") else (ret.get("linkers") if isinstance(ret, dict) else None)
        if nodes:
            node_conn = nodes[0].connectivity if hasattr(nodes[0], "connectivity") else nodes[0].get("connectivity")
        if linkers is not None:
            n_linkers = len(linkers)
    data = {
        "n_atoms": n_atoms,
        "n_metals": n_metals,
        "topology": topo,
        "node_connectivity": node_conn,
        "n_distinct_linkers": n_linkers,
    }
    if extra:
        data.update(extra)
    return ItemStrata(**{k: v for k, v in data.items() if v is not None or k in ItemStrata.model_fields})


def GenerateNavigationItems(
    documents: Sequence[MOFJSONDocument],
    tasks: Sequence[str] | None = None,
    *,
    representations: Sequence[str] | None = None,
    cap_per_answer: int = 200,
) -> Iterator[TaskItem]:
    """Generate navigation TaskItems from documents."""
    task_ids = list(tasks) if tasks is not None else list(NAVIGATION_TASKS)
    reps = list(representations) if representations is not None else [
        "mofjson_crystal",
        "mofjson_crystal_columnar",
        "lattice_coords_text",
        "cif",
        "mofjson_reticular",
        "mofjson_reticular_nogeom",
    ]
    counters: dict[tuple[str, str, str], int] = {}
    seq = 0
    render_cache: dict[tuple[int, str], str] = {}

    for doc in documents:
        try:
            doc.ensure_mofjson_id()
        except Exception:
            pass
        try:
            doc = _ensure_bonds(doc)
        except Exception:
            pass
        doc_key = id(doc)
        for tid in task_ids:
            spec = NAVIGATION_TASKS[tid]
            assert spec.ground_truth is not None
            try:
                gts = spec.ground_truth(doc)
            except Exception:
                continue
            for gt in gts:
                ans_key = str(gt["answer"])
                for rep in reps:
                    key = (tid, ans_key, rep)
                    if counters.get(key, 0) >= cap_per_answer:
                        continue
                    if not representation_can_ask(
                        rep, needs_paths=spec.needs_paths, needs_coords=spec.needs_coords
                    ):
                        continue
                    # Reticular-only tasks can use reticular reps; path tasks need crystal-like.
                    if spec.needs_paths and rep.startswith("mofjson_reticular"):
                        continue
                    if tid in ("node_nuclearity", "node_connectivity", "distinct_linkers", "topology_from_connectivity"):
                        if rep in ("cif", "lattice_coords_text") and doc.reticular is None:
                            continue
                    cache_key = (doc_key, rep)
                    if cache_key not in render_cache:
                        try:
                            render_cache[cache_key] = Render(doc, rep)
                        except Exception:
                            render_cache[cache_key] = ""
                    block = render_cache[cache_key]
                    if not block:
                        continue
                    question = gt["question"]
                    prompt = RenderPrompt(
                        "structure_question",
                        structure_block=block,
                        question=question,
                    )
                    # Identifier leak scan is applied once in BuildTaskSuite (hot-path).
                    seq += 1
                    counters[key] = counters.get(key, 0) + 1
                    answer_type = spec.answer_type
                    yield TaskItem(
                        item_id=f"nav-{seq:05d}",
                        family="navigation",
                        task=tid,
                        representation=rep,
                        prompt=prompt,
                        response_schema=spec.response_schema,
                        answer=gt["answer"],
                        answer_type=answer_type,  # type: ignore[arg-type]
                        grader=spec.grader,
                        source=_source_from_doc(doc),
                        strata=_strata_from_doc(doc, gt.get("strata")),
                        provenance=ItemProvenance(
                            ground_truth_method=gt.get("method"),
                            generator=GENERATOR_VERSION,
                        ),
                    )


__all__ = [
    "GENERATOR_VERSION",
    "NAVIGATION_TASKS",
    "GenerateNavigationItems",
    "NavigationTask",
]
