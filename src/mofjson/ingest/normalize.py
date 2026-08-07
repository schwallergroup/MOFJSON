"""Apply the Phase 00 quirk table. One function per quirk, each individually tested.

Phase 09.5.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mofjson.ingest.kg_csv import KgRecord
from mofjson.schema.crystal import normalize_space_group_symbol
from mofjson.schema.metadata import (
    ExtraProperty,
    Property,
    PropertyProvenance,
    SymmetryHint,
    deduplicate_properties,
    normalize_property,
)
from mofjson.util.ids import parse_mofid_line
from mofjson.util.text import NormalizeUnicode
from mofjson.vocab.elements import METALS
from mofjson.vocab.sentinels import MOFID_SENTINELS, is_sentinel, replace_sentinel
from mofjson.vocab.topology import TOPOLOGY_ALIASES

# Non-metals that leak into KG metal-element fields (Phase 00.5).
_LEAKED_NON_METALS: frozenset[str] = frozenset({"O", "Br", "Cl", "Se"})

# Synthesis temperatures that are almost certainly reaction times in days.
_SUSPECT_TEMPERATURES: frozenset[float] = frozenset({0.5, 1.0, 2.0, 3.0, 4.0})

_PUBCHEM_CID_RE = re.compile(r"(?:PubChem\s+CID\s+|SOLVENT_CID_|CID_)(\d+)", re.I)
_TITLE_SEP = "<|>"

# All known element symbols longest-first for greedy split of concatenations.
_ELEMENT_SYMBOLS: tuple[str, ...] = tuple(
    sorted(
        {
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
        },
        key=len,
        reverse=True,
    )
)


@dataclass
class NormalizedRecord:
    """Normalised view of a KgRecord ready for merge / document assembly."""

    mof_uri: str
    chemical_formula: str | None = None
    mofid: str | None = None
    mofid_corrupted: bool = False
    structure_stem: str | None = None
    topology_symbols: list[str] | None = None
    n_nets: int | None = None
    metal_elements: list[dict[str, str]] = field(default_factory=list)
    space_group_symbol: str | None = None
    crystal_system: str | None = None
    linker_smiles: list[str] = field(default_factory=list)
    # Linkers with names suppressed (quirk: chemical name is parent MOF)
    linker_names_suppressed: bool = False
    properties: list[Property] = field(default_factory=list)
    extra_properties: list[ExtraProperty] = field(default_factory=list)
    symmetry_hints: list[SymmetryHint] = field(default_factory=list)
    formula_label: str | None = None
    compound_label: str | None = None
    synthesis_methods: list[str] = field(default_factory=list)
    synthesis_temperature_k: float | None = None
    synthesis_temperature_suspect: bool = False
    solvents: list[dict[str, Any]] = field(default_factory=list)
    functional_group_name: str | None = None
    abstract: str | None = None
    is_hypothetical: bool | None = None
    is_experimental: bool | None = None
    is_functionalized: bool | None = None
    parent_mof_uri: str | None = None
    canonical_name: str | None = None
    warnings: list[str] = field(default_factory=list)
    quirk_counts: Counter[str] = field(default_factory=Counter)


QuirkHandler = Callable[[KgRecord, NormalizedRecord], None]


def split_concatenated_elements(raw: str) -> tuple[list[str], list[str]]:
    """Split a concatenated element string like ``AgLaW`` into element symbols.

    Returns (elements, warnings).
    """
    warnings: list[str] = []
    text = raw.strip()
    if not text:
        return [], warnings

    # Semicolon-separated form: "Ba; Cu; CuBa"
    if ";" in text:
        tokens = [t.strip() for t in text.split(";") if t.strip()]
        # Drop tokens that are concatenations of other tokens in the same cell
        kept: list[str] = []
        for tok in tokens:
            others = [t for t in tokens if t != tok]
            # If tok equals concatenation of a subset of others (order-insensitive join)
            if _is_concatenation_of(tok, others):
                warnings.append(f"dropped concatenation token {tok!r} from metal_cluster_elements")
                continue
            kept.append(tok)
        # Expand any remaining concatenated tokens
        elements: list[str] = []
        for tok in kept:
            if tok in METALS or tok in _LEAKED_NON_METALS or len(tok) <= 2:
                elements.append(tok)
            else:
                parts, w = _greedy_element_split(tok)
                elements.extend(parts)
                warnings.extend(w)
        return elements, warnings

    return _greedy_element_split(text)


def _is_concatenation_of(token: str, others: list[str]) -> bool:
    """True if *token* is a concatenation of two or more *others* (any order)."""
    if len(others) < 2:
        return False
    # Try all permutations is expensive; check if sorting char-bags match joins
    # Simpler: if token == a+b for distinct a,b in others
    for i, a in enumerate(others):
        for b in others[i + 1 :]:
            if token == a + b or token == b + a:
                return True
            for c in others:
                if c in (a, b):
                    continue
                if token in {a + b + c, a + c + b, b + a + c, b + c + a, c + a + b, c + b + a}:
                    return True
    return False


def _greedy_element_split(text: str) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    remaining = text
    parts: list[str] = []
    while remaining:
        matched = False
        for sym in _ELEMENT_SYMBOLS:
            if remaining.startswith(sym):
                parts.append(sym)
                remaining = remaining[len(sym) :]
                matched = True
                break
        if not matched:
            warnings.append(f"unparseable residue {remaining!r} in metal element string")
            break
    return parts, warnings


def filter_metal_elements(elements: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    """Drop leaked non-metals; return [{element: X}, ...] and warnings."""
    warnings: list[str] = []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for el in elements:
        if el in _LEAKED_NON_METALS:
            warnings.append(f"dropped non-metal {el!r} from metal element set")
            continue
        if el not in METALS:
            warnings.append(f"dropped non-metal/unknown symbol {el!r} from metal element set")
            continue
        if el in seen:
            continue
        seen.add(el)
        out.append({"element": el})
    return out, warnings


def normalize_topology_code(raw: str | None) -> tuple[list[str] | None, int | None, list[str]]:
    """Normalise topology; sentinels → null. Multi-net → list + n_nets."""
    warnings: list[str] = []
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return None, None, warnings
    cleaned, warn = replace_sentinel("topology", raw)
    if warn:
        warnings.append(f"topology sentinel {raw!r} mapped to null")
        return None, None, warnings

    text = str(cleaned).strip()
    # Strip MOFid-style .catN suffix pieces when present alone
    tokens = [t.strip() for t in re.split(r"[,;]", text) if t.strip()]
    symbols: list[str] = []
    for tok in tokens:
        # Drop .cat0 etc
        base = tok.split(".")[0].strip()
        if not base:
            continue
        lower = base.lower()
        if lower in TOPOLOGY_ALIASES or base.upper() in MOFID_SENTINELS or lower in {
            s.lower() for s in MOFID_SENTINELS
        }:
            warnings.append(f"topology sentinel token {base!r} dropped")
            continue
        if is_sentinel("topology", base):
            warnings.append(f"topology sentinel token {base!r} dropped")
            continue
        symbols.append(base)
    if not symbols:
        return None, None, warnings
    return symbols, len(symbols), warnings


def parse_title_labels(title: str | None) -> tuple[str | None, str | None, list[str]]:
    """Split ``formula<|>compound`` titles."""
    warnings: list[str] = []
    if title is None:
        return None, None, warnings
    cleaned, warn = replace_sentinel("title", title)
    if warn or cleaned is None:
        warnings.append(f"title sentinel {title!r} mapped to null")
        return None, None, warnings
    text = str(cleaned)
    if _TITLE_SEP in text:
        left, right = text.split(_TITLE_SEP, 1)
        return left.strip() or None, right.strip() or None, warnings
    return text.strip() or None, None, warnings


def parse_solvent_cid(raw: str | None) -> dict[str, Any] | None:
    """Extract pubchem_cid from Solvent_SOLVENT_CID_962 / PubChem CID 962."""
    if raw is None:
        return None
    m = _PUBCHEM_CID_RE.search(str(raw))
    if not m:
        return None
    return {"pubchem_cid": int(m.group(1)), "name": None}


def parse_synthesis_methods(raw: str | None) -> list[str]:
    if raw is None or str(raw).strip() == "":
        return []
    return [p.strip() for p in str(raw).split(";") if p.strip()]


# ---------------------------------------------------------------------------
# Quirk handlers
# ---------------------------------------------------------------------------


def _quirk_metal_elements(record: KgRecord, out: NormalizedRecord) -> None:
    raw = record.metal_cluster_elements
    if raw is None and record.clusters:
        # Prefer :hasMetalElement from clusters; never invert CLUSTER_* URI
        parts: list[str] = []
        for c in record.clusters:
            if c.metal_element:
                parts.append(c.metal_element)
            elif c.uri:
                out.warnings.append(
                    f"cluster URI {c.uri!r} not inverted; missing hasMetalElement"
                )
                out.quirk_counts["cluster_uri_not_inverted"] += 1
        if not parts:
            return
        raw = "; ".join(parts)
    if raw is None:
        return
    elements, warns = split_concatenated_elements(str(raw))
    out.warnings.extend(warns)
    if warns:
        out.quirk_counts["metal_concatenation"] += 1
    metals, drop_warns = filter_metal_elements(elements)
    out.warnings.extend(drop_warns)
    if drop_warns:
        out.quirk_counts["metal_nonmetal_leak"] += 1
    out.metal_elements = metals
    out.quirk_counts["metal_elements"] += 1


def _quirk_topology(record: KgRecord, out: NormalizedRecord) -> None:
    symbols, n_nets, warns = normalize_topology_code(record.topology)
    out.warnings.extend(warns)
    if any("sentinel" in w for w in warns):
        out.quirk_counts["topology_sentinel"] += 1
    out.topology_symbols = symbols
    out.n_nets = n_nets
    if symbols and len(symbols) > 1:
        out.quirk_counts["topology_multinet"] += 1
    out.quirk_counts["topology"] += 1


def _quirk_space_group(record: KgRecord, out: NormalizedRecord) -> None:
    raw = record.space_group
    if raw is None:
        return
    cleaned, warn = replace_sentinel("space_group", raw)
    if warn or cleaned is None:
        out.warnings.append(f"space_group sentinel {raw!r} mapped to null")
        out.quirk_counts["space_group_sentinel"] += 1
        return
    try:
        out.space_group_symbol = normalize_space_group_symbol(str(cleaned))
        if str(cleaned) != out.space_group_symbol:
            out.warnings.append(
                f"space_group {cleaned!r} normalised to {out.space_group_symbol!r}"
            )
            out.quirk_counts["space_group_alias"] += 1
    except Exception as exc:
        out.warnings.append(f"space_group {cleaned!r} could not be normalised: {exc}")
        out.space_group_symbol = str(cleaned)
        out.quirk_counts["space_group_unmapped"] += 1
    out.quirk_counts["space_group"] += 1


def _quirk_crystal_system(record: KgRecord, out: NormalizedRecord) -> None:
    raw = record.crystal_system
    if raw is None:
        return
    cleaned, warn = replace_sentinel("crystal_system", raw)
    if warn:
        out.warnings.append(f"crystal_system sentinel {raw!r} mapped to null")
        out.quirk_counts["crystal_system_sentinel"] += 1
        return
    out.crystal_system = str(cleaned).strip().lower() if cleaned is not None else None
    out.quirk_counts["crystal_system"] += 1


def _quirk_sentinels_formula_mofid(record: KgRecord, out: NormalizedRecord) -> None:
    if record.chemical_formula is not None:
        cleaned, warn = replace_sentinel("chemical_formula", record.chemical_formula)
        if warn:
            out.warnings.append(
                f"chemical_formula sentinel {record.chemical_formula!r} mapped to null"
            )
            out.quirk_counts["formula_sentinel"] += 1
            out.chemical_formula = None
        else:
            out.chemical_formula = str(cleaned) if cleaned is not None else None

    if record.mofid is not None:
        parts = parse_mofid_line(record.mofid)
        out.mofid = record.mofid
        out.mofid_corrupted = parts.has_sentinel
        out.structure_stem = parts.structure_stem
        if parts.has_sentinel:
            out.warnings.append(
                f"MOFid sentinel tokens {list(parts.sentinel_tokens)}; not used as primary key"
            )
            out.quirk_counts["mofid_sentinel"] += 1
    out.quirk_counts["identifiers"] += 1


def _quirk_properties(record: KgRecord, out: NormalizedRecord) -> None:
    props: list[Property] = []
    extras: list[ExtraProperty] = []
    hints: list[SymmetryHint] = []
    for raw_name, raw_value in record.properties.items():
        # Unit cell parameters is a volume despite the name (handled by PROPERTY_ALIASES)
        # Space group: X in the name
        unit = None
        # Unicode-normalise unit-like strings if the value is a unit-bearing name —
        # actual units come from TTL; CSV has bare values.
        result = normalize_property(
            raw_name,
            raw_value,
            unit,
            provenance=PropertyProvenance(source="MOFology-KG", extraction="kg_ingest_v1"),
        )
        for d in result.diagnostics:
            if d.code == "sentinel_replaced":
                out.quirk_counts["property_sentinel"] += 1
            if d.code == "symmetry_route":
                out.quirk_counts["space_group_as_property"] += 1
            if "band_gap" in (result.property.name if result.property else ""):
                out.quirk_counts["band_gap_canonical"] += 1
            out.warnings.append(d.message)
        if result.property is not None:
            props.append(result.property)
        if result.extra is not None:
            extras.append(result.extra)
        if result.symmetry is not None:
            hints.append(result.symmetry)
            if result.symmetry.kind == "space_group_symbol" and result.symmetry.value:
                try:
                    sym = normalize_space_group_symbol(str(result.symmetry.value))
                    if out.space_group_symbol is None:
                        out.space_group_symbol = sym
                except Exception:
                    pass

    deduped, report = deduplicate_properties(props)
    if report.ml_split_collapsed:
        out.quirk_counts["ml_split_collapsed"] += report.ml_split_collapsed
        out.warnings.append(
            f"collapsed {report.ml_split_collapsed} ML-split duplicate properties"
        )
    if report.float_noise_collapsed:
        out.quirk_counts["float_noise_collapsed"] += report.float_noise_collapsed
    out.properties = deduped
    out.extra_properties = extras
    out.symmetry_hints = hints
    out.quirk_counts["properties"] += 1


def _quirk_linker_name(record: KgRecord, out: NormalizedRecord) -> None:
    smiles_list: list[str] = []
    suppressed = False
    for linker in record.linkers:
        if linker.smiles and not is_sentinel("linker_smiles", linker.smiles):
            smiles_list.append(str(linker.smiles))
        if linker.chemical_name:
            # Quirk: hasChemicalName is the parent MOF name — never populate Linker.name
            suppressed = True
            out.warnings.append(
                "linker hasChemicalName is parent MOF name; Linker.name left null"
            )
            out.quirk_counts["linker_name_is_mof"] += 1
    if record.linker_smiles and record.linker_smiles not in smiles_list:
        if not is_sentinel("linker_smiles", record.linker_smiles):
            smiles_list.append(record.linker_smiles)
    out.linker_smiles = smiles_list
    out.linker_names_suppressed = suppressed
    out.quirk_counts["linkers"] += 1


def _quirk_synthesis(record: KgRecord, out: NormalizedRecord) -> None:
    syn = record.synthesis
    if syn is None:
        # CSV may carry Synthesis Temperature as a property
        temp = record.properties.get("Synthesis Temperature")
        if temp is not None:
            try:
                t = float(temp)
            except (TypeError, ValueError):
                t = None
            if t is not None:
                if t in _SUSPECT_TEMPERATURES:
                    out.synthesis_temperature_k = t
                    out.synthesis_temperature_suspect = True
                    out.warnings.append(
                        f"synthesis temperature {t} flagged as suspect (likely reaction time)"
                    )
                    out.quirk_counts["suspect_temperature"] += 1
                else:
                    out.synthesis_temperature_k = t
        return

    out.synthesis_methods = list(syn.methods) or []
    if syn.temperature is not None:
        t = float(syn.temperature)
        out.synthesis_temperature_k = t
        if t in _SUSPECT_TEMPERATURES:
            out.synthesis_temperature_suspect = True
            out.warnings.append(
                f"synthesis temperature {t} flagged as suspect (likely reaction time)"
            )
            out.quirk_counts["suspect_temperature"] += 1

    for sol in syn.solvents:
        raw_name = sol.get("name") or sol.get("uri") or sol.get("chemical_name")
        parsed = parse_solvent_cid(str(raw_name) if raw_name else None)
        if parsed is not None:
            out.solvents.append(parsed)
            out.quirk_counts["solvent_cid"] += 1
        elif raw_name:
            cleaned, warn = replace_sentinel("solvent", raw_name)
            if warn:
                out.quirk_counts["solvent_sentinel"] += 1
            else:
                out.solvents.append({"name": cleaned, "pubchem_cid": None})

    if syn.methods:
        # Multi-valued methods already split by TTL reader; also handle ";" in one string
        expanded: list[str] = []
        for m in syn.methods:
            parts = parse_synthesis_methods(m)
            expanded.extend(parts)
            if len(parts) > 1:
                out.quirk_counts["multi_synthesis_method"] += 1
        out.synthesis_methods = expanded
    out.quirk_counts["synthesis"] += 1


def _quirk_functionalisation(record: KgRecord, out: NormalizedRecord) -> None:
    name = record.functional_group_name
    if name is None:
        return
    cleaned, warn = replace_sentinel("functional_group", name)
    if warn or (isinstance(cleaned, str) and cleaned.strip().lower() in {"none", "chem_none"}):
        out.functional_group_name = None
        out.warnings.append(f"functionalGroupName {name!r} mapped to null")
        out.quirk_counts["functional_none"] += 1
        return
    out.functional_group_name = str(cleaned) if cleaned is not None else None
    out.quirk_counts["functionalisation"] += 1


def _quirk_abstract_title(record: KgRecord, out: NormalizedRecord) -> None:
    for abs_ in record.abstracts:
        if abs_.abstract is not None:
            cleaned, warn = replace_sentinel("abstract", abs_.abstract)
            if warn:
                out.warnings.append(f"abstract sentinel {abs_.abstract!r} mapped to null")
                out.quirk_counts["abstract_not_found"] += 1
            else:
                out.abstract = str(cleaned) if cleaned is not None else None
        if abs_.title is not None:
            formula_label, compound_label, warns = parse_title_labels(abs_.title)
            out.warnings.extend(warns)
            out.formula_label = formula_label
            out.compound_label = compound_label
            if compound_label is not None:
                out.quirk_counts["title_split"] += 1
    out.quirk_counts["abstract"] += 1


def _quirk_hypothetical(record: KgRecord, out: NormalizedRecord) -> None:
    # Prefer syn:isHypothetical / synthesis flag over rdf:type conflict
    syn_flag = None
    if record.synthesis is not None and record.synthesis.is_hypothetical is not None:
        syn_flag = record.synthesis.is_hypothetical
    if record.is_hypothetical is not None:
        out.is_hypothetical = record.is_hypothetical
    if syn_flag is not None:
        if (
            out.is_hypothetical is not None
            and out.is_hypothetical != syn_flag
        ):
            out.warnings.append(
                "ExperimentalMOF/HypotheticalMOF conflict; preferring syn:isHypothetical"
            )
            out.quirk_counts["type_conflict"] += 1
        out.is_hypothetical = syn_flag
    # Type assertions on the record
    types = {t.lower() for t in record.types}
    has_exp = "experimentalmof" in types or record.is_experimental is True
    has_hyp = "hypotheticalmof" in types or record.is_hypothetical is True
    if has_exp and has_hyp and out.is_hypothetical is None:
        out.warnings.append(
            "both ExperimentalMOF and HypotheticalMOF asserted; leaving is_hypothetical unset"
        )
        out.quirk_counts["type_conflict"] += 1
    elif has_hyp and out.is_hypothetical is None:
        out.is_hypothetical = True
    elif has_exp and out.is_hypothetical is None:
        out.is_hypothetical = False
        out.is_experimental = True
    if record.is_experimental is not None:
        out.is_experimental = record.is_experimental
    if record.is_functionalized is not None:
        out.is_functionalized = record.is_functionalized
    out.parent_mof_uri = record.parent_mof_uri
    out.canonical_name = record.canonical_name
    out.quirk_counts["type_flags"] += 1


def _quirk_unicode_units(record: KgRecord, out: NormalizedRecord) -> None:
    """NFKC + dash table on any string property units surviving in extras.

    Primary unit normalisation happens inside normalize_property; this handler
    ensures raw unicode quirks are counted when units appear as string values.
    """
    for prop in out.properties:
        if prop.raw_unit:
            normed = NormalizeUnicode(prop.raw_unit)
            if normed != prop.raw_unit:
                out.quirk_counts["unicode_unit"] += 1
                out.warnings.append(f"unit {prop.raw_unit!r} unicode-normalised to {normed!r}")
    out.quirk_counts["unicode"] += 1


# Ordered registry — enumerable, countable, reviewable.
QUIRK_HANDLERS: OrderedDict[str, QuirkHandler] = OrderedDict(
    [
        ("metal_elements", _quirk_metal_elements),
        ("topology", _quirk_topology),
        ("space_group", _quirk_space_group),
        ("crystal_system", _quirk_crystal_system),
        ("sentinels_formula_mofid", _quirk_sentinels_formula_mofid),
        ("properties", _quirk_properties),
        ("linker_name", _quirk_linker_name),
        ("synthesis", _quirk_synthesis),
        ("functionalisation", _quirk_functionalisation),
        ("abstract_title", _quirk_abstract_title),
        ("hypothetical", _quirk_hypothetical),
        ("unicode_units", _quirk_unicode_units),
    ]
)


def NormalizeRecord(record: KgRecord) -> tuple[NormalizedRecord, list[str]]:
    """Apply every quirk handler; return normalised record and warning list."""
    out = NormalizedRecord(mof_uri=record.mof_uri)
    for quirk_id, handler in QUIRK_HANDLERS.items():
        try:
            handler(record, out)
        except Exception as exc:  # never crash on bad data
            out.warnings.append(f"quirk {quirk_id} failed: {exc}")
            out.quirk_counts[f"{quirk_id}_error"] += 1
    return out, list(out.warnings)


# PascalCase alias matching the stub public API.
normalize_kg_record = NormalizeRecord

__all__ = [
    "QUIRK_HANDLERS",
    "NormalizedRecord",
    "NormalizeRecord",
    "filter_metal_elements",
    "normalize_kg_record",
    "normalize_topology_code",
    "parse_solvent_cid",
    "parse_synthesis_methods",
    "parse_title_labels",
    "split_concatenated_elements",
]
