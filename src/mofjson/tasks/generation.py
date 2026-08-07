"""Family C -- constrained generation: write a structure satisfying constraints.

Phase 12.4.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from mofjson.llm.prompts import RenderPrompt
from mofjson.schema.document import MOFJSONDocument
from mofjson.tasks.contamination import ScanForLeaks
from mofjson.tasks.item import ItemProvenance, ItemSource, ItemStrata, TaskItem

GENERATOR_VERSION = "tasks/generation.py@v1"


@dataclass
class Constraint:
    field: str
    relation: str  # eq | ge | le | contains | in
    value: Any
    checker: Callable[[dict[str, Any], Any], bool] | None = None
    description: str = ""

    def check(self, doc: dict[str, Any]) -> bool:
        if self.checker is not None:
            return bool(self.checker(doc, self.value))
        return _default_check(doc, self.field, self.relation, self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "relation": self.relation,
            "value": self.value,
            "description": self.description or f"{self.field} {self.relation} {self.value}",
        }


@dataclass
class ConstraintSet:
    constraints: list[Constraint]
    satisfying_source: ItemSource = field(default_factory=ItemSource)
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "constraints": [c.to_dict() for c in self.constraints],
            "satisfying_source": self.satisfying_source.model_dump(mode="json"),
        }


def _get_path(doc: dict[str, Any], path: str) -> Any:
    cur: Any = doc
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def _default_check(doc: dict[str, Any], path: str, relation: str, value: Any) -> bool:
    got = _get_path(doc, path)
    if relation == "eq":
        return got == value
    if relation == "ge":
        return got is not None and got >= value
    if relation == "le":
        return got is not None and got <= value
    if relation == "contains":
        if isinstance(got, (list, tuple, set)):
            return value in got
        if isinstance(got, str):
            return str(value) in got
        return False
    if relation == "in":
        return got in value
    return False


def _first_node(doc: dict[str, Any]) -> dict[str, Any] | None:
    nodes = (doc.get("reticular") or {}).get("nodes") or []
    return nodes[0] if nodes else None


def _first_linker(doc: dict[str, Any]) -> dict[str, Any] | None:
    linkers = (doc.get("reticular") or {}).get("linkers") or []
    return linkers[0] if linkers else None


def _check_metal(doc: dict[str, Any], value: Any) -> bool:
    node = _first_node(doc)
    if not node:
        return False
    metals = node.get("metals") or []
    return any(m.get("element") == value for m in metals)


def _check_archetype(doc: dict[str, Any], value: Any) -> bool:
    node = _first_node(doc)
    return bool(node and node.get("archetype") == value)


def _check_node_connectivity(doc: dict[str, Any], value: Any) -> bool:
    node = _first_node(doc)
    return bool(node and node.get("connectivity") == value)


def _check_nuclearity(doc: dict[str, Any], value: Any) -> bool:
    node = _first_node(doc)
    return bool(node and node.get("nuclearity") == value)


def _check_linker_connectivity(doc: dict[str, Any], value: Any) -> bool:
    linker = _first_linker(doc)
    return bool(linker and linker.get("connectivity") == value)


def _check_topology(doc: dict[str, Any], value: Any) -> bool:
    topo = (doc.get("reticular") or {}).get("topology") or {}
    return topo.get("rcsr_symbol") == value


def _check_geometry(doc: dict[str, Any], value: Any) -> bool:
    node = _first_node(doc)
    if not node:
        return False
    for site in node.get("sites") or []:
        if site.get("geometry") == value:
            return True
    return False


def _check_fg_contains(doc: dict[str, Any], value: Any) -> bool:
    linker = _first_linker(doc)
    if not linker:
        return False
    donors = linker.get("donor_groups") or []
    smi = linker.get("smiles") or ""
    name = (linker.get("name") or "") + " " + (linker.get("common_abbreviation") or "")
    blob = " ".join(str(x) for x in donors) + " " + smi + " " + name
    return str(value).lower() in blob.lower()


def constraints_from_document(doc: MOFJSONDocument) -> ConstraintSet | None:
    """Derive a satisfiable constraint set from a real document."""
    if doc.reticular is None:
        return None
    data = doc.model_dump(mode="json", exclude_none=True)
    node = _first_node(data)
    linker = _first_linker(data)
    if not node or not linker:
        return None
    metals = node.get("metals") or []
    if not metals:
        return None
    metal = metals[0].get("element")
    topo = ((data.get("reticular") or {}).get("topology") or {}).get("rcsr_symbol")
    constraints: list[Constraint] = [
        Constraint("metal", "eq", metal, checker=_check_metal, description=f"metal = {metal}"),
        Constraint(
            "node.connectivity",
            "eq",
            node.get("connectivity"),
            checker=_check_node_connectivity,
            description=f"node connectivity = {node.get('connectivity')}",
        ),
        Constraint(
            "node.archetype",
            "eq",
            node.get("archetype"),
            checker=_check_archetype,
            description=f"node archetype = {node.get('archetype')}",
        ),
        Constraint(
            "node.nuclearity",
            "eq",
            node.get("nuclearity"),
            checker=_check_nuclearity,
            description=f"nuclearity = {node.get('nuclearity')}",
        ),
        Constraint(
            "linker.connectivity",
            "eq",
            linker.get("connectivity"),
            checker=_check_linker_connectivity,
            description=f"linker connectivity = {linker.get('connectivity')}",
        ),
    ]
    if topo:
        constraints.append(
            Constraint(
                "topology",
                "eq",
                topo,
                checker=_check_topology,
                description=f"topology = {topo}",
            )
        )
    sites = node.get("sites") or []
    if sites and sites[0].get("geometry"):
        geom = sites[0]["geometry"]
        constraints.append(
            Constraint(
                "geometry",
                "eq",
                geom,
                checker=_check_geometry,
                description=f"coordination geometry = {geom}",
            )
        )
    # Functional group hint from donor groups / abbreviation.
    fg = None
    donors = linker.get("donor_groups") or []
    if "NH2" in str(linker.get("smiles") or "") or "amino" in str(linker.get("name") or "").lower():
        fg = "NH2"
    elif donors:
        fg = donors[0]
    if fg:
        constraints.append(
            Constraint(
                "functional_group",
                "contains",
                fg,
                checker=_check_fg_contains,
                description=f"linker contains {fg}",
            )
        )
    # Drop constraints with None values.
    constraints = [c for c in constraints if c.value is not None]
    if len(constraints) < 3:
        return None
    ids = doc.identifiers
    return ConstraintSet(
        constraints=constraints,
        satisfying_source=ItemSource(
            mofjson_id=ids.mofjson_id if ids else None,
            odac_name=ids.odac_name if ids else None,
            csd_refcode=ids.csd_refcode if ids else None,
            structure_stem=ids.structure_stem if ids else None,
            is_hypothetical=bool(ids.is_hypothetical) if ids else False,
        ),
        label=topo or (ids.structure_stem if ids else "") or "constraint_set",
    )


def GenerateConstraintSets(
    corpus: Sequence[MOFJSONDocument] | Sequence[dict[str, Any]],
    *,
    n: int = 500,
) -> list[ConstraintSet]:
    """Derive up to *n* unique constraint sets from real corpus combinations."""
    seen: set[str] = set()
    out: list[ConstraintSet] = []
    for entry in corpus:
        if len(out) >= n:
            break
        if isinstance(entry, MOFJSONDocument):
            doc = entry
        else:
            try:
                doc = MOFJSONDocument.model_validate(entry)
            except Exception:
                continue
        cs = constraints_from_document(doc)
        if cs is None:
            continue
        key = "|".join(f"{c.field}:{c.value}" for c in cs.constraints)
        if key in seen:
            continue
        # Verify satisfiable by the source document.
        data = doc.model_dump(mode="json", exclude_none=True)
        if not all(c.check(data) for c in cs.constraints):
            continue
        seen.add(key)
        out.append(cs)
    return out


def CheckConstraints(doc: dict[str, Any], constraints: Any) -> dict[str, Any]:
    """Evaluate constraints against a parsed document dict; report per-constraint."""
    # Accept document or profile-shaped output.
    if "reticular" not in doc and "nodes" in doc:
        doc = {"reticular": doc}

    # Schema validity
    schema_ok = True
    schema_error = None
    try:
        MOFJSONDocument.model_validate(doc)
    except Exception as exc:
        # Profile-only reticular is ok
        try:
            from mofjson.schema.reticular import ReticularStructure

            if "reticular" in doc:
                ReticularStructure.model_validate(doc["reticular"])
            else:
                raise exc
        except Exception as exc2:
            schema_ok = False
            schema_error = str(exc2)

    parsed_constraints: list[Constraint] = []
    if isinstance(constraints, list):
        for c in constraints:
            if isinstance(c, Constraint):
                parsed_constraints.append(c)
            elif isinstance(c, dict):
                parsed_constraints.append(
                    Constraint(
                        field=c["field"],
                        relation=c.get("relation", "eq"),
                        value=c["value"],
                        description=c.get("description", ""),
                        checker=_checker_for_field(c["field"]),
                    )
                )
    per: dict[str, bool] = {}
    for c in parsed_constraints:
        try:
            per[c.field] = bool(c.check(doc)) if schema_ok else False
        except Exception:
            per[c.field] = False

    # Soft chemistry / invariant hooks (Phase 11.4 / 11.7): nuclearity == sum(metals).
    invariant_ok = True
    try:
        node = _first_node(doc)
        if node and node.get("metals") is not None and node.get("nuclearity") is not None:
            s = sum(int(m.get("count") or 0) for m in node["metals"])
            if s != int(node["nuclearity"]):
                invariant_ok = False
        if node:
            for m in node.get("metals") or []:
                from mofjson.vocab.elements import METALS

                if m.get("element") not in METALS:
                    invariant_ok = False
    except Exception:
        invariant_ok = False

    all_ok = schema_ok and invariant_ok and (bool(per) and all(per.values()))
    return {
        "all_satisfied": all_ok,
        "schema_ok": schema_ok,
        "schema_error": schema_error,
        "invariant_ok": invariant_ok,
        "per_constraint": per,
    }


def _checker_for_field(field: str) -> Callable[[dict[str, Any], Any], bool] | None:
    return {
        "metal": _check_metal,
        "node.connectivity": _check_node_connectivity,
        "node.archetype": _check_archetype,
        "node.nuclearity": _check_nuclearity,
        "linker.connectivity": _check_linker_connectivity,
        "topology": _check_topology,
        "geometry": _check_geometry,
        "functional_group": _check_fg_contains,
    }.get(field)


def GenerateGenerationItems(
    constraint_sets: Sequence[ConstraintSet],
    *,
    output_representations: Sequence[str] | None = None,
) -> Iterator[TaskItem]:
    """Ask for a MOFJSON document satisfying each constraint set."""
    reps = list(output_representations) if output_representations is not None else [
        "mofjson_reticular",
        "mofjson_reticular_nogeom",
        "mofjson_crystal",
    ]
    seq = 0
    for cs in constraint_sets:
        lines = [c.description or f"{c.field} {c.relation} {c.value}" for c in cs.constraints]
        constraints_text = "\n".join(f"- {ln}" for ln in lines)
        prompt = RenderPrompt("constrained_generation", constraints=constraints_text)
        if ScanForLeaks(prompt):
            continue
        for rep in reps:
            seq += 1
            schema = (
                "profiles/reticular.schema.json"
                if "reticular" in rep
                else "profiles/crystal.schema.json"
            )
            yield TaskItem(
                item_id=f"gen-{seq:05d}",
                family="generation",
                task="constrained_generation",
                representation=rep,
                prompt=prompt,
                response_schema=schema,
                answer={"constraints": [c.to_dict() for c in cs.constraints]},
                answer_type="constraint_set",
                grader="constraint_set",
                source=cs.satisfying_source,
                strata=ItemStrata(
                    topology=next((c.value for c in cs.constraints if c.field == "topology"), None),
                    node_connectivity=next(
                        (c.value for c in cs.constraints if c.field == "node.connectivity"),
                        None,
                    ),
                ),
                provenance=ItemProvenance(
                    ground_truth_method="corpus_derived_constraints",
                    generator=GENERATOR_VERSION,
                ),
            )


__all__ = [
    "CheckConstraints",
    "Constraint",
    "ConstraintSet",
    "GENERATOR_VERSION",
    "GenerateConstraintSets",
    "GenerateGenerationItems",
    "constraints_from_document",
]
