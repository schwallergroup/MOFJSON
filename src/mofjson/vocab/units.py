"""Canonical unit strings, their aliases, and the conversions between them.

Phase 06.3.
"""

from __future__ import annotations

from types import MappingProxyType

from mofjson.util.diagnostics import Diagnostic, Severity
from mofjson.util.text import NormalizeKey, NormalizeUnicode

CANONICAL_UNITS: tuple[str, ...] = (
    "Angstrom",
    "Angstrom^3",
    "g/cm^3",
    "m^2/g",
    "m^2/cm^3",
    "m^2/m^3",
    "cm^3/g",
    "cm^3/cm^3",
    "cm^3(STP)/cm^3",
    "mol/kg",
    "mol/(kg.Pa)",
    "eV",
    "kJ/mol",
    "K",
    "degC",
    "degree",
    "g/mol",
    "nm",
    "cm^-1",
    "mm/s",
    "mm^-1",
    "ms",
    "muB",
    "e",
    "percent",
    "dimensionless",
)

DIMENSIONLESS: frozenset[str] = frozenset({"dimensionless", "count", "number"})

# Exact KG raw string -> (canonical_unit, factor, offset) where canonical = raw * factor + offset
_UNIT_ALIAS_EXACT: dict[str, tuple[str, float, float]] = {
    '%': ('percent', 1.0, 0.0),
    'K': ('K', 1.0, 0.0),
    'cm³(STP)/cm³': ('cm^3(STP)/cm^3', 1.0, 0.0),
    'cm³/cm³': ('cm^3/cm^3', 1.0, 0.0),
    'cm³/g': ('cm^3/g', 1.0, 0.0),
    'cm⁻¹': ('cm^-1', 1.0, 0.0),
    'count': ('dimensionless', 1.0, 0.0),
    'degrees': ('degree', 1.0, 0.0),
    'dimensionless': ('dimensionless', 1.0, 0.0),
    'e': ('e', 1.0, 0.0),
    'eV': ('eV', 1.0, 0.0),
    'g cm–3': ('g/cm^3', 1.0, 0.0),
    'g cm⁻³': ('g/cm^3', 1.0, 0.0),
    'g cm−3': ('g/cm^3', 1.0, 0.0),
    'g/cm³': ('g/cm^3', 1.0, 0.0),
    'g/mol': ('g/mol', 1.0, 0.0),
    'kJ/mol': ('kJ/mol', 1.0, 0.0),
    'kg/m³': ('g/cm^3', 0.001, 0.0),
    'm2/g': ('m^2/g', 1.0, 0.0),
    'mm s−1': ('mm/s', 1.0, 0.0),
    'mm–1': ('mm^-1', 1.0, 0.0),
    'mol/(kg·Pa)': ('mol/(kg.Pa)', 1.0, 0.0),
    'mol/kg': ('mol/kg', 1.0, 0.0),
    'ms': ('ms', 1.0, 0.0),
    'm²/cm³': ('m^2/cm^3', 1.0, 0.0),
    'm²/g': ('m^2/g', 1.0, 0.0),
    'm²/m³': ('m^2/m^3', 1.0, 0.0),
    'nm': ('nm', 1.0, 0.0),
    'number': ('dimensionless', 1.0, 0.0),
    '°C': ('degC', 1.0, 0.0),
    'Å': ('Angstrom', 1.0, 0.0),
    'Å, degrees, Å³': ('Angstrom^3', 1.0, 0.0),
    'Å3': ('Angstrom^3', 1.0, 0.0),
    'Å³': ('Angstrom^3', 1.0, 0.0),
    'μB': ('muB', 1.0, 0.0),
}

# Lookup also by NormalizeKey for unicode-variant resilience
_by_key: dict[str, tuple[str, float, float]] = {}
for _raw, _trip in _UNIT_ALIAS_EXACT.items():
    _by_key[NormalizeKey(_raw)] = _trip
    _by_key[_raw] = _trip
# Extra normalized forms that may appear after NormalizeUnicode on density variants
for _extra_raw, _trip in (
    ("g cm-3", ("g/cm^3", 1.0, 0.0)),
    ("g/cm3", ("g/cm^3", 1.0, 0.0)),
    ("kg/m3", ("g/cm^3", 0.001, 0.0)),
    ("a3", ("Angstrom^3", 1.0, 0.0)),
    ("angstrom", ("Angstrom", 1.0, 0.0)),
    ("angstrom^3", ("Angstrom^3", 1.0, 0.0)),
    ("m^2/g", ("m^2/g", 1.0, 0.0)),
    ("mm s-1", ("mm/s", 1.0, 0.0)),
    ("mm-1", ("mm^-1", 1.0, 0.0)),
    ("cm-1", ("cm^-1", 1.0, 0.0)),
    ("mol/(kg.pa)", ("mol/(kg.Pa)", 1.0, 0.0)),
    ("c", ("degC", 1.0, 0.0)),
    ("degc", ("degC", 1.0, 0.0)),
):
    _by_key.setdefault(NormalizeKey(_extra_raw), _trip)

UNIT_ALIASES: MappingProxyType[str, tuple[str, float, float]] = MappingProxyType(_by_key)

# (from_canonical, to_canonical) -> (factor, offset) with to = from * factor + offset
UNIT_CONVERSIONS: MappingProxyType[tuple[str, str], tuple[float, float]] = MappingProxyType(
    {
        ("degC", "K"): (1.0, 273.15),
        ("K", "degC"): (1.0, -273.15),
        ("g/cm^3", "g/cm^3"): (1.0, 0.0),
        ("kg/m^3", "g/cm^3"): (0.001, 0.0),  # if ever stored as kg/m^3 canonical
    }
)


def resolve_unit(raw_unit: str | None) -> tuple[str, float, float] | None:
    """Map a raw unit string to (canonical, factor, offset), or None if unknown."""
    if raw_unit is None:
        return None
    text = str(raw_unit).strip()
    if text == "":
        return None
    if text in UNIT_ALIASES:
        return UNIT_ALIASES[text]
    key = NormalizeKey(text)
    if key in UNIT_ALIASES:
        return UNIT_ALIASES[key]
    # Try NormalizeUnicode then key
    key2 = NormalizeKey(NormalizeUnicode(text))
    return UNIT_ALIASES.get(key2)


def convert_value(
    value: float,
    raw_unit: str | None,
    *,
    target_unit: str | None = None,
    field: str | None = None,
) -> tuple[float | None, str | None, Diagnostic | None]:
    """Convert *value* from *raw_unit* toward a canonical (or explicit target) unit.

    Returns (converted_value, canonical_unit, diagnostic).
    Refuses ambiguous conversions rather than guessing.
    """
    if raw_unit is None or str(raw_unit).strip() == "":
        return value, None, Diagnostic(
            code="unmapped_unit",
            severity=Severity.warning,
            message="Missing unit string",
            field_path=field,
            value=raw_unit,
            stage="unit_normalize",
        )

    resolved = resolve_unit(raw_unit)
    if resolved is None:
        return value, None, Diagnostic(
            code="unmapped_unit",
            severity=Severity.warning,
            message=f"Unrecognised unit {raw_unit!r}",
            field_path=field,
            value=raw_unit,
            stage="unit_normalize",
        )

    canonical, factor, offset = resolved
    if canonical in DIMENSIONLESS or raw_unit.strip().lower() in DIMENSIONLESS:
        converted = float(value)
        out_unit = "dimensionless"
    else:
        converted = float(value) * factor + offset
        out_unit = canonical

    if target_unit is not None and target_unit != out_unit:
        pair = (out_unit, target_unit)
        if pair not in UNIT_CONVERSIONS:
            return None, out_unit, Diagnostic(
                code="ambiguous_unit",
                severity=Severity.error,
                message=f"Refusing conversion from {out_unit!r} to {target_unit!r}",
                field_path=field,
                value={"from": out_unit, "to": target_unit, "raw_unit": raw_unit},
                stage="unit_normalize",
            )
        f2, o2 = UNIT_CONVERSIONS[pair]
        converted = converted * f2 + o2
        out_unit = target_unit

    # Implausible unit-for-field: temperature field with time-like unit
    if field and "temperature" in field.lower() and (
        out_unit in {"ms", "mm/s"} or (raw_unit and NormalizeKey(raw_unit) == "s")
    ):
        return converted, out_unit, Diagnostic(
            code="implausible_unit_for_field",
            severity=Severity.warning,
            message=f"Unit {raw_unit!r} is implausible for field {field!r}",
            field_path=field,
            value={"unit": raw_unit, "value": value},
            stage="unit_normalize",
        )

    return converted, out_unit, None


__all__ = [
    "CANONICAL_UNITS",
    "DIMENSIONLESS",
    "UNIT_ALIASES",
    "UNIT_CONVERSIONS",
    "convert_value",
    "resolve_unit",
]
