"""Every grader, in one registry. Deterministic, programmatic, no judge models.

Phase 12.x -- built alongside each family.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from mofjson.tasks.item import TaskItem

FailureCategory = Literal[
    "correct",
    "wrong",
    "unparseable",
    "truncated",
    "refused",
    "schema_invalid",
]

# Published StructureMatcher defaults — part of the metric, not incidental config.
STRUCTURE_MATCHER_LTOL = 0.2
STRUCTURE_MATCHER_STOL = 0.3
STRUCTURE_MATCHER_ANGLE_TOL = 5.0


@dataclass
class GradeResult:
    correct: bool
    category: FailureCategory
    detail: dict[str, Any] = field(default_factory=dict)
    parsed: Any = None


Grader = Callable[[TaskItem, str], GradeResult]


def _looks_truncated(text: str) -> bool:
    s = text.rstrip()
    if not s:
        return False
    # Heuristic: opens a JSON object/array but never closes, or ends mid-token.
    if s[0] in "{[" and s[-1] not in "}]":
        return True
    if s.count("{") > s.count("}"):
        return True
    if s.count("[") > s.count("]"):
        return True
    return False


def _extract_json(text: str) -> Any | None:
    s = text.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Try fenced or embedded object.
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _unwrap_answer(obj: Any, keys: tuple[str, ...] = ("answer",)) -> Any:
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
    return obj


def grade_exact_int(item: TaskItem, response: str) -> GradeResult:
    if not response or not response.strip():
        return GradeResult(False, "unparseable", detail={"reason": "empty"})
    if _looks_truncated(response):
        return GradeResult(False, "truncated")
    obj = _extract_json(response)
    if obj is None:
        # bare integer
        try:
            val = int(response.strip())
        except ValueError:
            return GradeResult(False, "unparseable")
    else:
        val = _unwrap_answer(obj)
        try:
            val = int(val)
        except (TypeError, ValueError):
            return GradeResult(False, "unparseable", parsed=obj)
    gold = int(item.answer)
    return GradeResult(val == gold, "correct" if val == gold else "wrong", parsed=val)


def grade_exact_bool(item: TaskItem, response: str) -> GradeResult:
    if not response or not response.strip():
        return GradeResult(False, "unparseable", detail={"reason": "empty"})
    if _looks_truncated(response):
        return GradeResult(False, "truncated")
    obj = _extract_json(response)
    if obj is None:
        raw = response.strip().lower()
        if raw in ("yes", "true", "1"):
            val = True
        elif raw in ("no", "false", "0"):
            val = False
        else:
            return GradeResult(False, "unparseable")
    else:
        val = _unwrap_answer(obj)
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("yes", "true", "1"):
                val = True
            elif low in ("no", "false", "0"):
                val = False
            else:
                return GradeResult(False, "unparseable", parsed=val)
        elif not isinstance(val, bool):
            return GradeResult(False, "unparseable", parsed=val)
    gold = bool(item.answer)
    return GradeResult(val == gold, "correct" if val == gold else "wrong", parsed=val)


def grade_canonical_string(item: TaskItem, response: str) -> GradeResult:
    if not response or not response.strip():
        return GradeResult(False, "unparseable", detail={"reason": "empty"})
    if _looks_truncated(response):
        return GradeResult(False, "truncated")
    obj = _extract_json(response)
    if obj is None:
        val = response.strip()
    else:
        val = _unwrap_answer(obj, keys=("answer", "mofid", "topology", "net"))
        if not isinstance(val, str):
            return GradeResult(False, "unparseable", parsed=obj)
        val = val.strip()
    gold = str(item.answer).strip()
    # Canonicalise: lowercase for net symbols; MOFid compare case-sensitive after strip.
    if item.answer_type == "string" and item.task == "topology_from_connectivity":
        ok = val.lower() == gold.lower()
    else:
        ok = val == gold
    return GradeResult(ok, "correct" if ok else "wrong", parsed=val)


def _composition_from_structure(structure: Any) -> Counter[str]:
    return Counter(str(s.specie.symbol) for s in structure)


def grade_structure_match(item: TaskItem, response: str) -> GradeResult:
    if not response or not response.strip():
        return GradeResult(False, "unparseable", detail={"reason": "empty"})
    if _looks_truncated(response):
        return GradeResult(False, "truncated")

    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
    except ImportError as exc:
        raise ImportError(
            "structure_match grader requires pymatgen; install mofjson[crystal]"
        ) from exc

    gold_doc = item.answer
    # Gold may be stored as a crystal dict or full document dict.
    try:
        pred_doc, gold_structure = _structures_for_match(response, gold_doc)
    except Exception as exc:
        return GradeResult(False, "unparseable", detail={"reason": str(exc)})

    matcher = StructureMatcher(
        ltol=STRUCTURE_MATCHER_LTOL,
        stol=STRUCTURE_MATCHER_STOL,
        angle_tol=STRUCTURE_MATCHER_ANGLE_TOL,
    )
    try:
        fit = bool(matcher.fit(gold_structure, pred_doc))
    except Exception as exc:
        return GradeResult(False, "unparseable", detail={"reason": f"matcher: {exc}"})

    gold_comp = _composition_from_structure(gold_structure)
    pred_comp = _composition_from_structure(pred_doc)
    comp_ok = gold_comp == pred_comp
    ok = fit and comp_ok
    return GradeResult(
        ok,
        "correct" if ok else "wrong",
        detail={
            "structure_fit": fit,
            "composition_equal": comp_ok,
            "ltol": STRUCTURE_MATCHER_LTOL,
            "stol": STRUCTURE_MATCHER_STOL,
            "angle_tol": STRUCTURE_MATCHER_ANGLE_TOL,
        },
        parsed={"composition": dict(pred_comp)},
    )


def _structures_for_match(response: str, gold: Any) -> tuple[Any, Any]:
    from mofjson.convert.pymatgen_bridge import MOFToStructure
    from mofjson.schema.document import MOFJSONDocument
    from mofjson.tasks.representations import Parse

    # Parse prediction
    obj = _extract_json(response)
    if obj is None:
        # Treat as CIF
        pred = Parse(response, "cif")
        pred_struct = MOFToStructure(pred)
    elif isinstance(obj, dict) and "cif" in obj:
        pred = Parse(str(obj["cif"]), "cif")
        pred_struct = MOFToStructure(pred)
    elif isinstance(obj, dict) and "lattice_coords_text" in obj:
        pred = Parse(str(obj["lattice_coords_text"]), "lattice_coords_text")
        pred_struct = MOFToStructure(pred)
    else:
        doc = MOFJSONDocument.model_validate(obj if "crystal" in obj else {"crystal": obj})
        pred_struct = MOFToStructure(doc)

    # Gold
    if isinstance(gold, dict):
        if "crystal" in gold or "reticular" in gold:
            gold_doc = MOFJSONDocument.model_validate(gold)
        else:
            gold_doc = MOFJSONDocument.model_validate({"crystal": gold})
        gold_struct = MOFToStructure(gold_doc)
    elif isinstance(gold, str):
        gold_doc = Parse(gold, "cif")
        gold_struct = MOFToStructure(gold_doc)
    else:
        raise ValueError("unsupported gold answer for structure_match")
    return pred_struct, gold_struct


def _reticular_signature(reticular: Any) -> dict[str, Any]:
    if hasattr(reticular, "model_dump"):
        data = reticular.model_dump(mode="json", exclude_none=True)
    else:
        data = reticular
    nodes = data.get("nodes") or []
    linkers = data.get("linkers") or []
    node_keys = sorted(
        (
            n.get("archetype"),
            n.get("nuclearity"),
            n.get("connectivity"),
            tuple(sorted((m.get("element"), m.get("count")) for m in (n.get("metals") or []))),
        )
        for n in nodes
        if isinstance(n, dict)
    )
    linker_keys = sorted(
        (
            l.get("smiles") or (l.get("fragment") or {}).get("smiles"),
            l.get("connectivity"),
            l.get("count_per_cell"),
        )
        for l in linkers
        if isinstance(l, dict)
    )
    topo = (data.get("topology") or {}).get("rcsr_symbol")
    return {"nodes": node_keys, "linkers": linker_keys, "topology": topo}


def grade_reticular_match(item: TaskItem, response: str) -> GradeResult:
    if not response or not response.strip():
        return GradeResult(False, "unparseable", detail={"reason": "empty"})
    if _looks_truncated(response):
        return GradeResult(False, "truncated")
    obj = _extract_json(response)
    if not isinstance(obj, dict):
        return GradeResult(False, "unparseable")
    ret = obj.get("reticular", obj)
    try:
        pred_sig = _reticular_signature(ret)
        gold_sig = _reticular_signature(item.answer if not isinstance(item.answer, dict) or "nodes" in item.answer or "linkers" in item.answer else item.answer.get("reticular", item.answer))
    except Exception as exc:
        return GradeResult(False, "unparseable", detail={"reason": str(exc)})
    ok = pred_sig == gold_sig
    return GradeResult(ok, "correct" if ok else "wrong", detail={"pred": pred_sig, "gold": gold_sig}, parsed=pred_sig)


def grade_fragment_isomorphism(item: TaskItem, response: str) -> GradeResult:
    if not response or not response.strip():
        return GradeResult(False, "unparseable", detail={"reason": "empty"})
    if _looks_truncated(response):
        return GradeResult(False, "truncated")
    obj = _extract_json(response)
    if not isinstance(obj, dict):
        return GradeResult(False, "unparseable")
    try:
        from mofjson.convert.moljson_interop import FragmentToMol
        from mofjson.schema.molecular import MolFragment
    except Exception as exc:
        return GradeResult(False, "unparseable", detail={"reason": str(exc)})

    try:
        frag_data = obj.get("fragment", obj)
        pred = MolFragment.model_validate(frag_data)
        gold = MolFragment.model_validate(
            item.answer if "atoms" in (item.answer or {}) else item.answer.get("fragment", item.answer)
        )
        mol_a = FragmentToMol(pred)
        mol_b = FragmentToMol(gold)
        # Element+bond-order aware graph match via RDKit if available.
        try:
            from rdkit.Chem import rdFMCS

            mcs = rdFMCS.FindMCS([mol_a, mol_b], bondCompare=rdFMCS.BondCompare.CompareOrder)
            ok = mcs.numAtoms == mol_a.GetNumAtoms() == mol_b.GetNumAtoms()
        except Exception:
            # Fallback: atom element multiset + bond count.
            ea = Counter(a.GetSymbol() for a in mol_a.GetAtoms())
            eb = Counter(a.GetSymbol() for a in mol_b.GetAtoms())
            ok = ea == eb and mol_a.GetNumBonds() == mol_b.GetNumBonds()
    except Exception as exc:
        return GradeResult(False, "unparseable", detail={"reason": str(exc)})
    return GradeResult(ok, "correct" if ok else "wrong", parsed={"ok": ok})


def grade_constraint_set(item: TaskItem, response: str) -> GradeResult:
    if not response or not response.strip():
        return GradeResult(False, "unparseable", detail={"reason": "empty"})
    if _looks_truncated(response):
        return GradeResult(False, "truncated")
    obj = _extract_json(response)
    if not isinstance(obj, dict):
        return GradeResult(False, "unparseable")

    from mofjson.tasks.generation import CheckConstraints

    constraints = item.answer
    if isinstance(constraints, dict) and "constraints" in constraints:
        constraints = constraints["constraints"]
    try:
        report = CheckConstraints(obj, constraints)
    except Exception as exc:
        return GradeResult(False, "schema_invalid", detail={"reason": str(exc)})
    ok = bool(report.get("all_satisfied"))
    return GradeResult(
        ok,
        "correct" if ok else "wrong",
        detail=report,
        parsed=report.get("per_constraint"),
    )


GRADERS: dict[str, Grader] = {
    "exact_int": grade_exact_int,
    "exact_bool": grade_exact_bool,
    "canonical_string": grade_canonical_string,
    "structure_match": grade_structure_match,
    "reticular_match": grade_reticular_match,
    "fragment_isomorphism": grade_fragment_isomorphism,
    "constraint_set": grade_constraint_set,
}


def Grade(item: TaskItem, response: str) -> GradeResult:
    """Dispatch to the grader named on the item."""
    grader = GRADERS.get(item.grader)
    if grader is None:
        raise ValueError(f"Unknown grader {item.grader!r}; known: {sorted(GRADERS)}")
    # Never recompute gold from the pipeline — only compare to stored answer.
    return grader(item, response)


__all__ = [
    "GRADERS",
    "STRUCTURE_MATCHER_ANGLE_TOL",
    "STRUCTURE_MATCHER_LTOL",
    "STRUCTURE_MATCHER_STOL",
    "FailureCategory",
    "Grade",
    "GradeResult",
    "Grader",
]
