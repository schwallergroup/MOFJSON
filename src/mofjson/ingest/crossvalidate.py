"""Compare what the KG asserts against what we computed, and report every disagreement.

Phase 09.7.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from mofjson.ingest.normalize import NormalizedRecord
from mofjson.schema.document import MOFJSONDocument

Severity = Literal["info", "warning", "error"]


@dataclass
class Disagreement:
    field: str
    asserted: Any
    computed: Any
    severity: Severity = "warning"
    asserted_provenance: str | None = "MOFology-KG"
    computed_provenance: str | None = "mofjson"
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "asserted": self.asserted,
            "computed": self.computed,
            "severity": self.severity,
            "asserted_provenance": self.asserted_provenance,
            "computed_provenance": self.computed_provenance,
            "note": self.note,
        }


@dataclass
class CrossValidationReport:
    """Aggregated rates per field."""

    n_compared: Counter[str] = field(default_factory=Counter)
    n_agree: Counter[str] = field(default_factory=Counter)
    n_disagree: Counter[str] = field(default_factory=Counter)
    disagreements: list[Disagreement] = field(default_factory=list)

    def rate(self, field: str) -> float | None:
        n = self.n_compared.get(field, 0)
        if not n:
            return None
        return self.n_agree.get(field, 0) / n

    def to_dict(self) -> dict[str, Any]:
        fields = sorted(set(self.n_compared) | set(self.n_agree) | set(self.n_disagree))
        return {
            "fields": {
                f: {
                    "n_compared": self.n_compared.get(f, 0),
                    "n_agree": self.n_agree.get(f, 0),
                    "n_disagree": self.n_disagree.get(f, 0),
                    "agreement_rate": self.rate(f),
                }
                for f in fields
            },
            "n_disagreements": len(self.disagreements),
            "disagreements_sample": [d.to_dict() for d in self.disagreements[:100]],
        }

    def extend(self, other: CrossValidationReport) -> None:
        self.n_compared.update(other.n_compared)
        self.n_agree.update(other.n_agree)
        self.n_disagree.update(other.n_disagree)
        self.disagreements.extend(other.disagreements)


def _computed_metal_elements(document: MOFJSONDocument) -> set[str]:
    out: set[str] = set()
    reticular = document.reticular
    if reticular is not None:
        nodes = getattr(reticular, "metal_nodes", None) or []
        for node in nodes:
            metals = getattr(node, "metals", None) or []
            for m in metals:
                el = m.get("element") if isinstance(m, dict) else getattr(m, "element", None)
                if el:
                    out.add(str(el))
    if not out and document.crystal is not None:
        from mofjson.vocab.elements import METALS

        for site in document.crystal.sites:
            el = getattr(site, "element", None) or getattr(site, "species", None)
            if el and str(el) in METALS:
                out.add(str(el))
    return out


def _computed_oms_count(document: MOFJSONDocument) -> int | None:
    reticular = document.reticular
    if reticular is None:
        return None
    nodes = getattr(reticular, "metal_nodes", None) or []
    total = 0
    found = False
    for node in nodes:
        sites = getattr(node, "sites", None) or []
        for site in sites:
            oms = getattr(site, "open_metal_site", None)
            if oms is not None:
                found = True
            if oms:
                total += 1
    return total if found else None


def _computed_topology(document: MOFJSONDocument) -> list[str] | None:
    reticular = document.reticular
    if reticular is None:
        return None
    topo = getattr(reticular, "topology", None)
    if topo is None:
        return None
    sym = getattr(topo, "rcsr_symbol", None)
    if sym is None:
        return None
    if isinstance(sym, list):
        return [str(s) for s in sym]
    return [str(sym)]


def _kg_oms(normalized: NormalizedRecord) -> int | None:
    for prop in normalized.properties:
        if prop.name in {"number_of_open_metal_sites", "n_open_metal_sites"} or (
            prop.raw_name and "open metal site" in prop.raw_name.lower()
        ):
            if prop.value is None:
                return None
            try:
                return int(prop.value)
            except (TypeError, ValueError):
                return None
    return None


def BuildCrossValidationReport(
    document: MOFJSONDocument,
    kg_record: NormalizedRecord,
) -> CrossValidationReport:
    """Compare and return an aggregated per-document report."""
    report = CrossValidationReport()

    # 1. Metal elements
    asserted_metals = {m["element"] for m in kg_record.metal_elements if "element" in m}
    computed_metals = _computed_metal_elements(document)
    if asserted_metals and computed_metals:
        report.n_compared["metal_elements"] += 1
        if asserted_metals == computed_metals:
            report.n_agree["metal_elements"] += 1
        else:
            report.n_disagree["metal_elements"] += 1
            report.disagreements.append(
                Disagreement(
                    field="metal_elements",
                    asserted=sorted(asserted_metals),
                    computed=sorted(computed_metals),
                    note="concatenation / non-metal leakage may explain KG value",
                )
            )

    # 2. Open metal sites
    kg_oms = _kg_oms(kg_record)
    computed_oms = _computed_oms_count(document)
    if kg_oms is not None and computed_oms is not None:
        report.n_compared["open_metal_sites"] += 1
        if kg_oms == computed_oms:
            report.n_agree["open_metal_sites"] += 1
        else:
            report.n_disagree["open_metal_sites"] += 1
            report.disagreements.append(
                Disagreement(
                    field="open_metal_sites",
                    asserted=kg_oms,
                    computed=computed_oms,
                )
            )

    # 3. Topology (skip when KG topology is null / sentinel-cleaned)
    kg_topo = kg_record.topology_symbols
    computed_topo = _computed_topology(document)
    if kg_topo and computed_topo:
        report.n_compared["topology"] += 1
        kg_set = {s.lower() for s in kg_topo}
        comp_set = {s.lower() for s in computed_topo}
        if kg_set & comp_set or kg_set == comp_set:
            report.n_agree["topology"] += 1
        else:
            report.n_disagree["topology"] += 1
            report.disagreements.append(
                Disagreement(
                    field="topology",
                    asserted=kg_topo,
                    computed=computed_topo,
                )
            )

    # 4. Formula (informational)
    if kg_record.chemical_formula and document.crystal and document.crystal.formula_sum:
        report.n_compared["formula"] += 1
        a = "".join(kg_record.chemical_formula.split()).lower()
        b = "".join(document.crystal.formula_sum.split()).lower()
        if a == b:
            report.n_agree["formula"] += 1
        else:
            report.n_disagree["formula"] += 1
            report.disagreements.append(
                Disagreement(
                    field="formula",
                    asserted=kg_record.chemical_formula,
                    computed=document.crystal.formula_sum,
                    severity="info",
                )
            )

    return report


def CrossValidate(
    document: MOFJSONDocument,
    kg_record: NormalizedRecord,
) -> list[Disagreement]:
    """Compare computed document fields against a normalised KG record."""
    return list(BuildCrossValidationReport(document, kg_record).disagreements)


__all__ = [
    "BuildCrossValidationReport",
    "CrossValidate",
    "CrossValidationReport",
    "Disagreement",
]
