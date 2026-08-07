"""The placeholder values upstream data uses to mean "missing", and how to detect them.

Phase 06.5 / 06.7.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

# Exact strings that mean nothing / missing. Comparison is case-sensitive for the
# MOFid-specific tokens and case-insensitive for the generic NA-like set via is_sentinel.
STRING_SENTINELS: frozenset[str] = frozenset(
    {
        "",
        "NA",
        "N/A",
        "na",
        "n/a",
        "null",
        "None",
        "none",
        "nan",
        "NaN",
        "-",
        "?",
        "unknown",
        "Unknown",
        "UNKNOWN",
        "n.d.",
        "not reported",
        "ERROR",
        "TIMEOUT",
        "NO_REF",
        "unstable",
        "error_unknown",
        "unknown_sql",
        "sql_unknown",
        "abstract_not_found",
        "CHEM_None",
        "new_entry",
    }
)

# Case-insensitive membership set derived from STRING_SENTINELS.
_STRING_SENTINELS_LOWER: frozenset[str] = frozenset(s.lower() for s in STRING_SENTINELS)

# MOFid topology-field corruption tokens (Phase 00.4).
MOFID_SENTINELS: frozenset[str] = frozenset(
    {
        "ERROR",
        "TIMEOUT",
        "UNKNOWN",
        "NO_REF",
        "unstable",
        "error_unknown",
        "unknown_sql",
        "sql_unknown",
    }
)

# Numeric placeholders that are only sentinels *in context*.
# Keyed by canonical field / property name.
NUMERIC_SENTINELS: MappingProxyType[str, frozenset[float]] = MappingProxyType(
    {
        "void_fraction_accessible": frozenset({0.0}),
        "void_fraction_non_accessible": frozenset({0.0}),
        "pore_volume_fraction_accessible": frozenset({0.0}),
        "pore_volume_fraction_non_accessible": frozenset({0.0}),
        # Generic temperature contamination: 0 alone is not always a sentinel, but
        # synthesis temperatures of {0.5,1,2,3,4} are flagged elsewhere as suspect.
    }
)

# PubChem placeholder name pattern: "PubChem CID <n>"
_PUBCHEM_CID_PREFIX = "pubchem cid "


def is_sentinel(field: str | None, value: Any) -> bool:
    """Return True if *value* is an upstream placeholder for *field*."""
    if value is None:
        return True

    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _STRING_SENTINELS_LOWER:
            return True
        if text in STRING_SENTINELS:
            return True
        # PubChem CID placeholder used as a chemical name
        if text.lower().startswith(_PUBCHEM_CID_PREFIX):
            return True
        return text.upper() in MOFID_SENTINELS

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # NaN
        try:
            import math

            if math.isnan(float(value)):
                return True
        except (TypeError, ValueError):
            pass
        if field is None:
            return False
        sentinels = NUMERIC_SENTINELS.get(field)
        if sentinels is not None and float(value) in sentinels:
            return True
        return field in NUMERIC_SENTINELS and float(value) in {999.0, 9999.0, 1e30, -1.0}

    return False


def replace_sentinel(
    field: str | None, value: Any
) -> tuple[Any, str | None]:
    """If *value* is a sentinel, return (None, warning_code); else (value, None)."""
    if is_sentinel(field, value):
        return None, "sentinel_replaced"
    return value, None


def scan_for_sentinels(obj: Any, *, path: str = "") -> list[tuple[str, Any]]:
    """Recursively find surviving sentinel strings/numbers in a JSON-like object.

    Returns a list of (field_path, value) hits. Used by Phase 11.6 corpus scans.
    """
    hits: list[tuple[str, Any]] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{p}.{k}" if p else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{p}[{i}]")
        else:
            # For scan: any string that is a sentinel value is a hit, regardless of field.
            if isinstance(node, str) and is_sentinel(None, node):
                # Empty string / None-like in optional fields — still flag
                hits.append((p, node))
            elif isinstance(node, (int, float)) and is_sentinel(p.rsplit(".", 1)[-1], node):
                hits.append((p, node))

    walk(obj, path)
    return hits


__all__ = [
    "MOFID_SENTINELS",
    "NUMERIC_SENTINELS",
    "STRING_SENTINELS",
    "is_sentinel",
    "replace_sentinel",
    "scan_for_sentinels",
]
