"""Parse, validate, diagnose and best-effort repair model output.

Phase 10.5.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from mofjson.exceptions import SchemaError
from mofjson.llm.profiles import ProfileModel
from mofjson.util.diagnostics import Diagnostic, DiagnosticCollector, Severity
from mofjson.vocab.geometry import POLYHEDRA

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_POLYHEDRA_LOOKUP = {p.lower(): p for p in POLYHEDRA}
# Near-miss spellings that map unambiguously.
_VOCAB_NEAR_MISS = {
    "octohedral": "octahedral",
    "octahedal": "octahedral",
    "tetrahedal": "tetrahedral",
    "square antiprismatic": "square_antiprismatic",
    "square-antiprismatic": "square_antiprismatic",
    "trigonal bipyramidal": "trigonal_bipyramidal",
    "square pyramidal": "square_pyramidal",
}


@dataclass(slots=True)
class RepairReport:
    """What repair attempted, changed, and left broken."""

    first_pass_valid: bool
    post_repair_valid: bool
    changes: list[str] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    dative_bond_order_issues: int = 0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_pass_valid": self.first_pass_valid,
            "post_repair_valid": self.post_repair_valid,
            "changes": list(self.changes),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "dative_bond_order_issues": self.dative_bond_order_issues,
            "truncated": self.truncated,
        }


def _looks_truncated(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    # Unbalanced braces/brackets strongly suggest truncation mid-JSON.
    opens = s.count("{") + s.count("[")
    closes = s.count("}") + s.count("]")
    if opens > closes:
        return True
    if s.startswith("{") and not s.rstrip().endswith("}"):
        return True
    if s.startswith("[") and not s.rstrip().endswith("]"):
        return True
    return False


def ParseResponse(text: str) -> tuple[dict[str, Any] | None, list[Diagnostic]]:
    """Extract a JSON object from raw model text (fences / preamble allowed)."""
    collector = DiagnosticCollector()
    if not isinstance(text, str) or not text.strip():
        collector.add(
            "llm_parse_invalid_json",
            "Empty model output",
            severity=Severity.error,
            stage="llm.parse",
        )
        return None, collector.to_list()

    truncated = _looks_truncated(text)
    if truncated:
        collector.add(
            "llm_parse_truncated",
            "Output appears truncated",
            severity=Severity.error,
            stage="llm.parse",
            value=text[-80:],
        )

    candidates: list[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1))
        collector.add(
            "llm_parse_fenced",
            "Extracted JSON from markdown fence",
            severity=Severity.info,
            stage="llm.parse",
        )
    stripped = text.strip()
    if stripped not in candidates:
        candidates.append(stripped)
    # Prose preamble: take from first { or [
    for opener in ("{", "["):
        idx = text.find(opener)
        if idx > 0:
            candidates.append(text[idx:].strip())
            collector.add(
                "llm_parse_fenced",
                "Extracted JSON after prose preamble",
                severity=Severity.info,
                stage="llm.parse",
            )
            break

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if truncated:
                # Truncation takes priority over a lucky partial parse.
                return None, collector.to_list()
            return obj, collector.to_list()

    if not truncated:
        collector.add(
            "llm_parse_invalid_json",
            "Could not parse model output as a JSON object",
            severity=Severity.error,
            stage="llm.parse",
            value=text[:120],
        )
    return None, collector.to_list()


def ValidateAgainstProfile(obj: dict[str, Any], profile: str) -> list[Diagnostic]:
    """Validate a parsed object against a named profile; return diagnostics."""
    collector = DiagnosticCollector()
    model = ProfileModel(profile)
    try:
        model.model_validate(obj)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            collector.add(
                "llm_validate_failed",
                err.get("msg", "validation failed"),
                severity=Severity.error,
                field_path=loc or None,
                value=err.get("input"),
                stage="llm.validate",
            )
    except SchemaError as exc:
        collector.add(
            "llm_validate_failed",
            exc.message,
            severity=Severity.error,
            field_path=exc.field_path,
            value=exc.value,
            stage="llm.validate",
        )
    return collector.to_list()


def _walk_repair(
    node: Any,
    *,
    path: str,
    changes: list[str],
    collector: DiagnosticCollector,
    dative_count: list[int],
) -> Any:
    """Best-effort mechanical repairs; never invent chemistry."""
    if isinstance(node, dict):
        out = dict(node)

        # Mechanical n_atoms / n_sites agreement.
        if "n_atoms" in out and "atoms" in out and isinstance(out["atoms"], list):
            if out["n_atoms"] != len(out["atoms"]):
                out["n_atoms"] = len(out["atoms"])
                changes.append(f"{path}.n_atoms <- len(atoms)")
                collector.add(
                    "llm_repair_n_atoms",
                    "Set n_atoms from atoms length",
                    severity=Severity.warning,
                    field_path=f"{path}.n_atoms".lstrip("."),
                    stage="llm.repair",
                )
        if "n_atoms" in out and "sites" in out and isinstance(out["sites"], list):
            if out["n_atoms"] != len(out["sites"]):
                out["n_atoms"] = len(out["sites"])
                changes.append(f"{path}.n_atoms <- len(sites)")
                collector.add(
                    "llm_repair_n_atoms",
                    "Set n_atoms from sites length",
                    severity=Severity.warning,
                    field_path=f"{path}.n_atoms".lstrip("."),
                    stage="llm.repair",
                )

        # Vocab near-miss on geometry.
        if "geometry" in out and isinstance(out["geometry"], str):
            g = out["geometry"]
            key = g.strip().lower().replace(" geometry", "")
            mapped = _VOCAB_NEAR_MISS.get(key) or _POLYHEDRA_LOOKUP.get(key)
            if mapped is not None and mapped != g:
                out["geometry"] = mapped
                changes.append(f"{path}.geometry {g!r} -> {mapped!r}")
                collector.add(
                    "llm_repair_vocab",
                    f"Mapped geometry {g!r} -> {mapped!r}",
                    severity=Severity.warning,
                    field_path=f"{path}.geometry".lstrip("."),
                    value=g,
                    stage="llm.repair",
                )

        # Bond order dative issues: count only, do not rewrite.
        if "order" in out:
            order = out["order"]
            if isinstance(order, str) and order.strip().lower() == "dative":
                dative_count[0] += 1
                collector.add(
                    "llm_bond_order_dative",
                    "Bond order 'dative' left unrepaired (report only)",
                    severity=Severity.warning,
                    field_path=f"{path}.order".lstrip("."),
                    value=order,
                    stage="llm.repair",
                )
            elif order == 1 and out.get("dative_to") is not None:
                dative_count[0] += 1
                collector.add(
                    "llm_bond_order_dative",
                    "Bond order 1 with dative_to left unrepaired (report only)",
                    severity=Severity.warning,
                    field_path=f"{path}.order".lstrip("."),
                    value=order,
                    stage="llm.repair",
                )

        # Wrong-typed numbers that are digit strings.
        for k, v in list(out.items()):
            if isinstance(v, str) and k in (
                "coordination_number",
                "nuclearity",
                "connectivity",
                "n_atoms",
                "answer",
            ):
                try:
                    coerced = int(v.strip())
                except ValueError:
                    continue
                out[k] = coerced
                changes.append(f"{path}.{k} coerced str->int")
                collector.add(
                    "llm_repair_type_coerce",
                    f"Coerced {k} from string to int",
                    severity=Severity.warning,
                    field_path=f"{path}.{k}".lstrip("."),
                    value=v,
                    stage="llm.repair",
                )

        # Recurse.
        for k, v in list(out.items()):
            child_path = f"{path}.{k}" if path else k
            out[k] = _walk_repair(
                v,
                path=child_path,
                changes=changes,
                collector=collector,
                dative_count=dative_count,
            )

        # Dangling atom_id in donors/bonds against local atoms list.
        atom_ids: set[str] | None = None
        if isinstance(out.get("atoms"), list):
            atom_ids = {
                str(a["id"])
                for a in out["atoms"]
                if isinstance(a, dict) and isinstance(a.get("id"), str)
            }
        if atom_ids is not None:
            for list_key in ("bonds", "donors", "charges", "attachment_points"):
                items = out.get(list_key)
                if not isinstance(items, list):
                    continue
                kept = []
                for i, item in enumerate(items):
                    if not isinstance(item, dict):
                        kept.append(item)
                        continue
                    bad = False
                    for ref_key in ("source", "target", "atom_id"):
                        ref = item.get(ref_key)
                        if isinstance(ref, str) and ref not in atom_ids:
                            bad = True
                            collector.add(
                                "llm_repair_dangling_ref",
                                f"Removed {list_key}[{i}] with dangling {ref_key}={ref!r}",
                                severity=Severity.warning,
                                field_path=f"{path}.{list_key}[{i}].{ref_key}".lstrip(
                                    "."
                                ),
                                value=ref,
                                stage="llm.repair",
                            )
                            changes.append(
                                f"removed {path}.{list_key}[{i}] dangling {ref_key}"
                            )
                            break
                    if not bad:
                        kept.append(item)
                out[list_key] = kept

        return out

    if isinstance(node, list):
        return [
            _walk_repair(
                x,
                path=f"{path}[{i}]",
                changes=changes,
                collector=collector,
                dative_count=dative_count,
            )
            for i, x in enumerate(node)
        ]
    return node


def _fill_missing_required(
    obj: dict[str, Any],
    model: type[BaseModel],
    *,
    changes: list[str],
    collector: DiagnosticCollector,
) -> dict[str, Any]:
    """Insert null for missing required keys that are optional/nullable in the model."""
    out = dict(obj)
    try:
        schema = model.model_json_schema(mode="validation")
    except Exception:  # noqa: BLE001
        return out
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    for key in required:
        if key in out:
            continue
        # Only fill when the schema allows null — inventing values is fabrication.
        prop = props.get(key) or {}
        allows_null = False
        if prop.get("type") == "null":
            allows_null = True
        t = prop.get("type")
        if isinstance(t, list) and "null" in t:
            allows_null = True
        if "anyOf" in prop:
            allows_null = any(
                isinstance(x, dict) and x.get("type") == "null" for x in prop["anyOf"]
            )
        if allows_null:
            out[key] = None
            changes.append(f"inserted missing nullable required key {key!r}=null")
            collector.add(
                "llm_repair_missing_key",
                f"Inserted missing required key {key!r} as null",
                severity=Severity.warning,
                field_path=key,
                stage="llm.repair",
            )
    return out


def Repair(
    obj: dict[str, Any],
    profile: str,
    *,
    budget: int = 1,
) -> tuple[dict[str, Any] | None, RepairReport]:
    """Best-effort repair. Never changes semantically meaningful chemistry values.

    Reports first-pass and post-repair validity separately.
    """
    model = ProfileModel(profile)
    first_diags = ValidateAgainstProfile(obj, profile)
    first_valid = not first_diags
    collector = DiagnosticCollector()
    changes: list[str] = []
    dative_count = [0]
    current: dict[str, Any] | None = dict(obj)

    attempts = max(0, budget)
    for _ in range(attempts):
        assert current is not None
        current = _fill_missing_required(
            current, model, changes=changes, collector=collector
        )
        current = _walk_repair(
            current,
            path="",
            changes=changes,
            collector=collector,
            dative_count=dative_count,
        )
        assert isinstance(current, dict)

    post_diags = ValidateAgainstProfile(current, profile) if current is not None else first_diags
    post_valid = not post_diags
    all_diags = list(first_diags) + collector.to_list() + list(post_diags)

    report = RepairReport(
        first_pass_valid=first_valid,
        post_repair_valid=post_valid,
        changes=changes,
        diagnostics=all_diags,
        dative_bond_order_issues=dative_count[0],
    )

    if not post_valid:
        collector.add(
            "llm_repair_budget_exhausted",
            "Still invalid after repair budget",
            severity=Severity.error,
            stage="llm.repair",
        )
        report.diagnostics = list(first_diags) + collector.to_list() + list(post_diags)
        # Return best-effort object; callers must check post_repair_valid separately
        # from first_pass_valid so repair is never conflated with format success.
        return current, report

    return current, report


__all__ = [
    "ParseResponse",
    "Repair",
    "RepairReport",
    "ValidateAgainstProfile",
]
