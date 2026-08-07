"""Identifiers, properties, conditions and provenance. Where upstream mess is quarantined.

Phase 06.

Field names follow project_notes/phases/06_schema_metadata_and_provenance.md.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from mofjson.exceptions import SchemaError
from mofjson.schema.base import ContentHash, MOFJSONModel
from mofjson.util.diagnostics import Diagnostic, DiagnosticCollector, Severity
from mofjson.util.ids import parse_mofid_line
from mofjson.util.text import NormalizeKey
from mofjson.version import MOFJSON_VERSION, __version__
from mofjson.vocab.properties import PropertyAlias, lookup_property_alias
from mofjson.vocab.sentinels import replace_sentinel
from mofjson.vocab.units import convert_value

_REFCODE_RE = re.compile(r"^[A-Za-z]{6}(\d{2})?$")
_ML_SPLIT_FILES = frozenset({"test.csv", "train.csv", "validation.csv", "val.csv"})
_CONDITIONS_KV_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9 /_-]*?)\s*:\s*(?P<val>[^;]+)"
)
_SUSPECT_ENERGY_NAMES = frozenset({"free_energy_per_atom", "strain_energy_per_atom"})


class Identifiers(MOFJSONModel):
    """Every name a MOF might have, each in its own typed field."""

    mofjson_id: ContentHash | None = None
    csd_refcode: str | None = None
    qmof_id: str | None = None
    mp_id: str | None = None
    odac_name: str | None = None
    kg_uri: str | None = None
    mofid: str | None = None
    mofkey: str | None = None
    common_names: list[str] = Field(default_factory=list)
    structure_stem: str | None = None
    source_filename: str | None = None
    parent_id: str | None = None
    is_functionalized: bool = False
    is_hypothetical: bool = False
    mofid_corrupted: bool | None = None

    @model_validator(mode="after")
    def _derive_mofid_flags(self) -> Identifiers:
        if self.mofid is not None and self.mofid_corrupted is None:
            parts = parse_mofid_line(self.mofid)
            object.__setattr__(self, "mofid_corrupted", parts.has_sentinel)
            if self.structure_stem is None and parts.structure_stem:
                object.__setattr__(self, "structure_stem", parts.structure_stem)
        return self


class Conditions(MOFJSONModel):
    """Structured measurement/simulation conditions (not free-text KG conditions)."""

    temperature_k: float | None = None
    pressure_bar: float | None = None
    gas: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


Confidence = Literal["high", "medium", "low"]
Plausibility = Literal["ok", "suspect", "unknown"]


class PropertyProvenance(MOFJSONModel):
    """Per-property provenance. A single MOF routinely mixes sources."""

    source: str | None = None
    source_dataset: str | None = None
    source_id: str | None = None
    source_file: str | None = None
    extraction: str | None = None
    confidence: Confidence | None = None
    conflict: bool = False
    plausibility: Plausibility | None = None
    note: str | None = None
    warnings: list[str] = Field(default_factory=list)
    ml_splits: list[str] = Field(default_factory=list)


class Property(MOFJSONModel):
    """Canonical property with raw retention and required provenance."""

    name: str
    value: float | int | str | None = None
    unit: str | None = None
    method: str | None = None
    functional: str | None = None
    conditions: Conditions | None = None
    raw_name: str | None = None
    raw_value: float | int | str | None = None
    raw_unit: str | None = None
    provenance: PropertyProvenance

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise SchemaError("Property.name must be non-empty", field_path="name", value=v)
        return v


class ExtraProperty(MOFJSONModel):
    """Passthrough for rare / unmapped property names."""

    raw_name: str
    value: float | int | str | None = None
    unit: str | None = None
    provenance: PropertyProvenance
    note: str | None = None


class SymmetryHint(MOFJSONModel):
    """Structural descriptor that belongs on Symmetry, not in properties[]."""

    kind: Literal["space_group_symbol", "space_group_number", "crystal_system"]
    value: str | float | int | None = None
    raw_name: str
    provenance: PropertyProvenance | None = None


class SourceRef(MOFJSONModel):
    name: str
    kind: str | None = None
    file: str | None = None
    doi: str | None = None
    version: str | None = None


class PipelineStep(MOFJSONModel):
    step: str
    tool: str | None = None
    version: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class DocumentProvenance(MOFJSONModel):
    """Document-level provenance (sources, pipeline, warnings)."""

    mofjson_version: str = MOFJSON_VERSION
    generated_by: str = Field(default_factory=lambda: f"mofjson {__version__}")
    generated_at: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    pipeline: list[PipelineStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_timestamp(self) -> DocumentProvenance:
        if self.generated_at is None:
            # Bypass validate_assignment to avoid re-entering this validator.
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        return self


class ConflictRecord(MOFJSONModel):
    """One entry in the document-level conflicts summary."""

    name: str
    method: str | None = None
    functional: str | None = None
    values: list[float | int | str | None] = Field(default_factory=list)
    note: str | None = None


def validate_refcode(refcode: str | None) -> tuple[str | None, Diagnostic | None]:
    """Warn (do not reject) on non-standard CSD refcodes."""
    if refcode is None:
        return None, None
    text = str(refcode).strip()
    if _REFCODE_RE.match(text):
        return text, None
    return text, Diagnostic(
        code="refcode_warn",
        severity=Severity.warning,
        message=f"CSD refcode {text!r} failed the 6-letter + optional 2-digit pattern",
        field_path="identifiers.csd_refcode",
        value=text,
        stage="identifiers",
    )


def parse_conditions_text(text: str | None) -> tuple[Conditions | None, dict[str, str]]:
    """Parse KG ``:propertyConditions`` free text into provenance bits + Conditions.

    Useful parts like ``Source File`` / ``Match Method`` go to the returned dict
    (for provenance). Genuinely unparseable residue stays out of Conditions.
    """
    if text is None or not str(text).strip():
        return None, {}
    raw = str(text).strip()
    provenance_bits: dict[str, str] = {}
    extras: dict[str, Any] = {}
    temperature_k = None
    pressure_bar = None
    gas = None

    for m in _CONDITIONS_KV_RE.finditer(raw):
        key = m.group("key").strip()
        val = m.group("val").strip()
        key_l = key.lower()
        if key_l in {"source file", "source_file"}:
            provenance_bits["source_file"] = val
        elif key_l in {"match method", "match_method"}:
            provenance_bits["note"] = f"Match Method: {val}"
        elif key_l in {"temperature", "temperature_k", "t"}:
            try:
                temperature_k = float(val.replace("K", "").strip())
            except ValueError:
                extras[key] = val
        elif key_l in {"pressure", "pressure_bar", "p"}:
            try:
                pressure_bar = float(val.replace("bar", "").strip())
            except ValueError:
                extras[key] = val
        elif key_l in {"gas", "adsorbate"}:
            gas = val
        else:
            extras[key] = val

    # If nothing structured was found, stash the whole string as an extra note key
    if (
        temperature_k is None
        and pressure_bar is None
        and gas is None
        and not extras
        and not provenance_bits
    ):
        provenance_bits["note"] = raw
        return None, provenance_bits

    built = Conditions(
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        gas=gas,
        extras=extras,
    )
    # Empty conditions object -> None
    if (
        built.temperature_k is None
        and built.pressure_bar is None
        and built.gas is None
        and not built.extras
    ):
        return None, provenance_bits
    return built, provenance_bits


def _unit_sig_figs(unit: str | None) -> int:
    """Significant digits implied by the unit for float-noise collapse."""
    if unit is None:
        return 8
    u = unit.lower()
    if u in {"dimensionless", "percent"}:
        return 8
    if "angstrom" in u or u in {"nm", "degree"}:
        return 6
    if u in {"eV", "ev"}:
        return 6
    if "g/cm" in u or "m^2" in u or "mol" in u:
        return 6
    return 8


def round_for_unit(value: float | int | str | None, unit: str | None) -> float | int | str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value
    figs = _unit_sig_figs(unit)
    if value == 0:
        return 0.0
    return float(f"{float(value):.{figs}g}")


def freeze_conditions(conditions: Conditions | None) -> tuple[Any, ...]:
    if conditions is None:
        return ()
    data = conditions.model_dump(mode="json", exclude_none=True)
    extras = tuple(sorted((data.pop("extras", {}) or {}).items()))
    return (
        data.get("temperature_k"),
        data.get("pressure_bar"),
        data.get("gas"),
        extras,
    )


@dataclass
class NormalizeResult:
    property: Property | None = None
    extra: ExtraProperty | None = None
    symmetry: SymmetryHint | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)


def normalize_property(
    raw_name: str,
    raw_value: float | int | str | None,
    raw_unit: str | None = None,
    *,
    conditions_text: str | None = None,
    conditions: Conditions | None = None,
    provenance: PropertyProvenance | None = None,
    method: str | None = None,
    functional: str | None = None,
    source: str | None = "MOFology-KG",
) -> NormalizeResult:
    """Canonicalise one upstream property triple into Property / Extra / SymmetryHint."""
    result = NormalizeResult()
    collector = DiagnosticCollector()

    # Sentinel on the name itself
    _cleaned_name, name_warn = replace_sentinel("property_name", raw_name)
    if name_warn:
        collector.add(
            name_warn,
            f"Sentinel property name {raw_name!r} dropped",
            field_path="raw_name",
            value=raw_name,
            stage="property_normalize",
        )
        result.diagnostics = collector.to_list()
        return result

    alias = lookup_property_alias(str(raw_name))
    if alias is None:
        collector.add(
            "unmapped_property",
            f"Unmapped property name {raw_name!r}; routing to extra_properties",
            field_path="raw_name",
            value=raw_name,
            stage="property_normalize",
        )
        alias = PropertyAlias(None, None, None, None, None, "extra_properties")

    # Parse conditions text into provenance bits
    parsed_cond, prov_bits = parse_conditions_text(conditions_text)
    if conditions is None:
        conditions = parsed_cond

    # Value sentinel
    field_for_sentinel = alias.canonical_name or NormalizeKey(str(raw_name))
    cleaned_value, val_warn = replace_sentinel(field_for_sentinel, raw_value)
    if val_warn:
        collector.add(
            val_warn,
            f"Sentinel value for {raw_name!r} replaced with null",
            field_path="value",
            value=raw_value,
            stage="property_normalize",
        )

    # Unit conversion
    unit_out: str | None = None
    value_out = cleaned_value
    if cleaned_value is not None and isinstance(cleaned_value, (int, float)):
        converted, unit_out, unit_diag = convert_value(
            float(cleaned_value),
            raw_unit,
            target_unit=alias.canonical_unit,
            field=alias.canonical_name,
        )
        if unit_diag is not None:
            collector.extend([unit_diag])
            if unit_diag.code == "ambiguous_unit":
                # Keep raw value/unit; do not invent a conversion
                value_out = cleaned_value
                unit_out = None
            elif unit_diag.code == "unmapped_unit":
                value_out = cleaned_value
                unit_out = raw_unit
            else:
                value_out = converted if converted is not None else cleaned_value
                unit_out = unit_out or alias.canonical_unit
        else:
            value_out = converted
            if unit_out is None:
                unit_out = alias.canonical_unit
    elif raw_unit:
        resolved = convert_value(0.0, raw_unit)
        # just to detect unmapped; ignore the 0
        _c, unit_out, unit_diag = resolved[0], resolved[1], resolved[2]
        if unit_diag and unit_diag.code == "unmapped_unit":
            collector.extend([unit_diag])
            unit_out = raw_unit
        elif alias.canonical_unit:
            unit_out = alias.canonical_unit

    # Build provenance
    prov = provenance or PropertyProvenance(source=source)
    if prov_bits.get("source_file") and not prov.source_file:
        prov = prov.model_copy(update={"source_file": prov_bits["source_file"]})
    if prov_bits.get("note"):
        note = prov.note
        merged = prov_bits["note"] if not note else f"{note}; {prov_bits['note']}"
        prov = prov.model_copy(update={"note": merged})
    if collector.messages():
        prov = prov.model_copy(
            update={"warnings": list(prov.warnings) + collector.messages()}
        )

    method_out = method if method is not None else alias.method
    functional_out = functional if functional is not None else alias.functional

    if alias.route == "symmetry":
        kind: Literal["space_group_symbol", "space_group_number", "crystal_system"]
        rn = str(raw_name)
        if rn.lower().startswith("space group number"):
            kind = "space_group_number"
            sym_val: Any = cleaned_value
        elif rn.lower().startswith("space group"):
            kind = "space_group_symbol"
            # "Space group: P21/c" -> symbol
            sym_val = rn.split(":", 1)[1].strip() if ":" in rn else cleaned_value
        else:
            kind = "crystal_system"
            sym_val = rn.split(":", 1)[1].strip() if ":" in rn else cleaned_value
        collector.add(
            "symmetry_route",
            f"Routed {raw_name!r} to Symmetry",
            field_path="raw_name",
            value=raw_name,
            stage="property_normalize",
            severity=Severity.info,
        )
        result.symmetry = SymmetryHint(
            kind=kind, value=sym_val, raw_name=str(raw_name), provenance=prov
        )
        result.diagnostics = collector.to_list()
        return result

    if alias.route == "extra_properties" or alias.canonical_name is None:
        collector.add(
            "extra_property",
            f"Routed {raw_name!r} to extra_properties",
            field_path="raw_name",
            value=raw_name,
            stage="property_normalize",
            severity=Severity.info,
        )
        result.extra = ExtraProperty(
            raw_name=str(raw_name),
            value=value_out if value_out is not None else cleaned_value,
            unit=unit_out or raw_unit,
            provenance=prov,
        )
        result.diagnostics = collector.to_list()
        return result

    # Suspect energies
    if alias.canonical_name in _SUSPECT_ENERGY_NAMES:
        prov = prov.model_copy(
            update={"confidence": "low", "plausibility": "suspect"}
        )
        collector.add(
            "implausible_energy",
            f"{alias.canonical_name} values are often physically implausible",
            field_path="name",
            value=value_out,
            stage="property_normalize",
        )

    result.property = Property(
        name=alias.canonical_name,
        value=value_out,
        unit=unit_out or alias.canonical_unit,
        method=method_out,
        functional=functional_out,
        conditions=conditions,
        raw_name=str(raw_name),
        raw_value=raw_value,
        raw_unit=raw_unit,
        provenance=prov,
    )
    result.diagnostics = collector.to_list()
    return result


@dataclass
class DedupReport:
    input_count: int = 0
    output_count: int = 0
    collapsed: int = 0
    float_noise_collapsed: int = 0
    ml_split_collapsed: int = 0
    conflicts: int = 0
    conflict_records: list[ConflictRecord] = field(default_factory=list)


def _property_key(
    prop: Property, *, mofjson_id: str | None, ignore_source_file: bool = False
) -> tuple[Any, ...]:
    return (
        mofjson_id,
        prop.name,
        prop.method,
        prop.functional,
        freeze_conditions(prop.conditions),
        None if ignore_source_file else (prop.provenance.source_file or None),
    )


def _values_agree(a: Property, b: Property) -> bool:
    ra = round_for_unit(a.value, a.unit)
    rb = round_for_unit(b.value, b.unit)
    if ra is None and rb is None:
        return True
    if type(ra) is not type(rb) and not (
        isinstance(ra, (int, float)) and isinstance(rb, (int, float))
    ):
        return False
    if isinstance(ra, (int, float)) and isinstance(rb, (int, float)):
        return float(ra) == float(rb)
    return ra == rb


def _union_provenance(a: PropertyProvenance, b: PropertyProvenance) -> PropertyProvenance:
    warnings = list(dict.fromkeys([*a.warnings, *b.warnings]))
    splits = list(dict.fromkeys([*a.ml_splits, *b.ml_splits]))
    note_parts = [x for x in [a.note, b.note] if x]
    return a.model_copy(
        update={
            "warnings": warnings,
            "ml_splits": splits,
            "source_file": a.source_file or b.source_file,
            "note": "; ".join(note_parts) if note_parts else None,
            "conflict": a.conflict or b.conflict,
            "confidence": a.confidence or b.confidence,
            "plausibility": a.plausibility or b.plausibility,
        }
    )


def deduplicate_properties(
    properties: list[Property],
    *,
    mofjson_id: str | None = None,
) -> tuple[list[Property], DedupReport]:
    """Apply Phase 06.5 deduplication policy."""
    report = DedupReport(input_count=len(properties))
    if not properties:
        return [], report

    # First pass: group by full key including source_file
    from collections import defaultdict

    groups: dict[tuple[Any, ...], list[Property]] = defaultdict(list)
    for p in properties:
        groups[_property_key(p, mofjson_id=mofjson_id, ignore_source_file=False)].append(p)

    # Collapse float-noise / agreeing values within identical keys
    intermediate: list[Property] = []
    for _key, items in groups.items():
        if len(items) == 1:
            intermediate.append(items[0])
            continue
        # Partition by rounded value
        by_val: dict[Any, list[Property]] = defaultdict(list)
        for p in items:
            by_val[round_for_unit(p.value, p.unit)].append(p)
        if len(by_val) == 1:
            base = items[0]
            prov = base.provenance
            for other in items[1:]:
                prov = _union_provenance(prov, other.provenance)
            intermediate.append(base.model_copy(update={"provenance": prov}))
            report.collapsed += len(items) - 1
            report.float_noise_collapsed += len(items) - 1
        else:
            # Conflicts within same key — keep all, mark conflict
            for p in items:
                prov = p.provenance.model_copy(update={"conflict": True})
                intermediate.append(p.model_copy(update={"provenance": prov}))
            report.conflicts += 1
            report.conflict_records.append(
                ConflictRecord(
                    name=items[0].name,
                    method=items[0].method,
                    functional=items[0].functional,
                    values=[p.value for p in items],
                    note="values disagree within identical key",
                )
            )

    # Second pass: ML-split collapse — same key ignoring source_file when files are ML splits
    groups2: dict[tuple[Any, ...], list[Property]] = defaultdict(list)
    for p in intermediate:
        groups2[_property_key(p, mofjson_id=mofjson_id, ignore_source_file=True)].append(p)

    output: list[Property] = []
    for _key, items in groups2.items():
        if len(items) == 1:
            output.append(items[0])
            continue

        files = [(p.provenance.source_file or "") for p in items]
        ml_files = [f for f in files if f.split("/")[-1].lower() in _ML_SPLIT_FILES]
        all_ml = len(ml_files) == len(items) and len(items) > 1

        if all_ml:
            # Rule 5: collapse regardless of value disagreement; list splits
            base = items[0]
            splits = sorted({(p.provenance.source_file or "").split("/")[-1] for p in items})
            prov = base.provenance
            for other in items[1:]:
                prov = _union_provenance(prov, other.provenance)
            values = [p.value for p in items]
            agree = all(_values_agree(items[0], p) for p in items[1:])
            note = prov.note
            split_note = f"ML splits collapsed: {', '.join(splits)}"
            if not agree:
                split_note += f"; values were {values}"
            prov = prov.model_copy(
                update={
                    "ml_splits": splits,
                    "note": f"{note}; {split_note}" if note else split_note,
                    "source_file": None,
                    "conflict": False if agree else prov.conflict,
                }
            )
            output.append(base.model_copy(update={"provenance": prov}))
            report.ml_split_collapsed += len(items) - 1
            report.collapsed += len(items) - 1
            continue

        # Non-ML: if values agree, merge; else keep all as conflicts
        if all(_values_agree(items[0], p) for p in items[1:]):
            base = items[0]
            prov = base.provenance
            for other in items[1:]:
                prov = _union_provenance(prov, other.provenance)
            output.append(base.model_copy(update={"provenance": prov}))
            report.collapsed += len(items) - 1
            report.float_noise_collapsed += len(items) - 1
        else:
            for p in items:
                if not p.provenance.conflict:
                    p = p.model_copy(
                        update={
                            "provenance": p.provenance.model_copy(update={"conflict": True})
                        }
                    )
                output.append(p)
            report.conflicts += 1
            report.conflict_records.append(
                ConflictRecord(
                    name=items[0].name,
                    method=items[0].method,
                    functional=items[0].functional,
                    values=[p.value for p in items],
                )
            )

    report.output_count = len(output)
    return output, report


__all__ = [
    "Conditions",
    "ConflictRecord",
    "DedupReport",
    "DocumentProvenance",
    "ExtraProperty",
    "Identifiers",
    "PipelineStep",
    "Property",
    "PropertyProvenance",
    "SourceRef",
    "SymmetryHint",
    "deduplicate_properties",
    "freeze_conditions",
    "normalize_property",
    "parse_conditions_text",
    "round_for_unit",
    "validate_refcode",
]
