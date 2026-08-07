"""The Diagnostic type and its collector. The backbone of "never crash, never lie".

Phase 06.4 -- used by every stage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


DIAGNOSTIC_CODES: MappingProxyType[str, str] = MappingProxyType(
    {
        "unmapped_property": "Property name not in PROPERTY_ALIASES",
        "unmapped_unit": "Unit string not in UNIT_ALIASES",
        "ambiguous_unit": "Unit conversion refused because the mapping is ambiguous",
        "sentinel_replaced": "Upstream sentinel replaced with null",
        "refcode_warn": "CSD refcode failed the 6+2 pattern; kept with a warning",
        "value_conflict": "Conflicting values for the same property key kept as separate records",
        "ml_split_collapsed": "ML-split duplicate properties collapsed into one record",
        "implausible_energy": "Free/strain energy per atom flagged as physically suspect",
        "implausible_unit_for_field": (
            "Unit is implausible for the field (e.g. time in temperature)"
        ),
        "extra_property": "Rare property routed to extra_properties",
        "symmetry_route": "Structural descriptor routed to Symmetry, not properties",
        "float_noise_collapsed": "Float-noise duplicates collapsed by unit-aware rounding",
        "unregistered_code": "Internal: a diagnostic used an unregistered code",
        # Phase 10.5 repair / parse diagnostics
        "llm_parse_fenced": "JSON extracted from markdown fences or prose wrapper",
        "llm_parse_truncated": "Model output appears truncated (unbalanced braces/brackets)",
        "llm_parse_invalid_json": "Could not parse model output as JSON",
        "llm_validate_failed": "Document failed profile validation",
        "llm_repair_missing_key": "Inserted missing required key with null",
        "llm_repair_type_coerce": "Coerced a wrong-typed scalar to the expected type",
        "llm_repair_vocab": "Mapped an unambiguous vocabulary near-miss",
        "llm_repair_n_atoms": "Set n_atoms from sites/atoms length (mechanical)",
        "llm_repair_dangling_ref": "Removed or nullified a dangling atom_id reference",
        "llm_bond_order_dative": (
            "Bond order emitted as 'dative' or 1 where 0.5 expected; counted, not rewritten"
        ),
        "llm_repair_budget_exhausted": "Repair budget exhausted; document still invalid",
    }
)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    severity: Severity
    message: str
    field_path: str | None = None
    value: Any = None
    stage: str | None = None

    def __post_init__(self) -> None:
        if self.code not in DIAGNOSTIC_CODES:
            object.__setattr__(
                self,
                "message",
                f"[unregistered code {self.code!r}] {self.message}",
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class DiagnosticCollector:
    """Accumulate structured diagnostics for a document or batch."""

    _items: list[Diagnostic] = field(default_factory=list)

    def add(
        self,
        code: str,
        message: str,
        *,
        severity: Severity | str = Severity.warning,
        field_path: str | None = None,
        value: Any = None,
        stage: str | None = None,
    ) -> Diagnostic:
        if isinstance(severity, str):
            severity = Severity(severity)
        if code not in DIAGNOSTIC_CODES:
            # Still emit, but mark as unregistered so audits catch it.
            diag = Diagnostic(
                code="unregistered_code",
                severity=Severity.error,
                message=f"Unregistered diagnostic code {code!r}: {message}",
                field_path=field_path,
                value={"original_code": code, "value": value},
                stage=stage,
            )
            self._items.append(diag)
            return diag
        diag = Diagnostic(
            code=code,
            severity=severity,
            message=message,
            field_path=field_path,
            value=value,
            stage=stage,
        )
        self._items.append(diag)
        return diag

    def extend(self, items: Iterable[Diagnostic]) -> None:
        for item in items:
            if item.code not in DIAGNOSTIC_CODES:
                self.add(
                    "unregistered_code",
                    f"Unregistered diagnostic code {item.code!r}: {item.message}",
                    severity=Severity.error,
                    field_path=item.field_path,
                    value={"original_code": item.code, "value": item.value},
                    stage=item.stage,
                )
            else:
                self._items.append(item)

    def counts_by_code(self) -> Mapping[str, int]:
        return dict(Counter(d.code for d in self._items))

    def to_list(self) -> list[Diagnostic]:
        return list(self._items)

    def messages(self) -> list[str]:
        return [d.message for d in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self._items)


__all__ = [
    "DIAGNOSTIC_CODES",
    "Diagnostic",
    "DiagnosticCollector",
    "Severity",
]
