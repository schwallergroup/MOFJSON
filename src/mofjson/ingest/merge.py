"""Deduplicate and merge records that describe the same material.

Phase 09.6.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from mofjson.ingest.normalize import NormalizedRecord
from mofjson.schema.document import MOFJSONDocument
from mofjson.schema.metadata import (
    ConflictRecord,
    DocumentProvenance,
    Identifiers,
    Property,
    SourceRef,
)

MergeRule = Literal[
    "identical_mofjson_id",
    "same_stem",
    "same_refcode",
    "prefer_computed_formula",
    "prefer_computed_structure",
    "keep_both_properties",
]


@dataclass(frozen=True)
class MergePolicy:
    """Conflict-resolution rules as data."""

    prefer_computed_structural: bool = True
    merge_on_mofjson_id: bool = True
    merge_on_stem: bool = True
    merge_on_refcode: bool = True
    merge_on_name_similarity: bool = False  # hard no


DEFAULT_POLICY = MergePolicy()


@dataclass
class MergeResult:
    document: MOFJSONDocument
    warnings: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)
    formula_disagreement: bool = False


def GroupDuplicates(
    records: list[NormalizedRecord],
    *,
    policy: MergePolicy = DEFAULT_POLICY,
    stem_of: dict[str, str] | None = None,
    mofjson_id_of: dict[str, str] | None = None,
) -> list[list[NormalizedRecord]]:
    """Group records that describe the same material.

    Matching order: identical mofjson_id → same stem → same refcode.
    Never merges on name similarity.
    """
    _ = policy.merge_on_name_similarity  # explicitly unused
    groups_map: dict[str, list[NormalizedRecord]] = defaultdict(list)
    singleton_keys: list[str] = []

    for i, rec in enumerate(records):
        key = None
        mid = (mofjson_id_of or {}).get(rec.mof_uri)
        if policy.merge_on_mofjson_id and mid:
            key = f"id:{mid}"
        if key is None and policy.merge_on_stem:
            stem = (stem_of or {}).get(rec.mof_uri) or rec.structure_stem
            if stem:
                key = f"stem:{stem}"
        if key is None and policy.merge_on_refcode:
            # bare local name if it looks like a refcode — left to caller via stem_of
            pass
        if key is None:
            key = f"uri:{rec.mof_uri}"
            singleton_keys.append(key)
        groups_map[key].append(rec)

    return list(groups_map.values())


def MergeGroup(
    group: list[NormalizedRecord],
    *,
    structure_doc: MOFJSONDocument | None = None,
    identifiers: Identifiers | None = None,
    policy: MergePolicy = DEFAULT_POLICY,
) -> MergeResult:
    """Merge a duplicate group into one MOFJSONDocument.

    When *structure_doc* is provided, its crystal/reticular layers win for
    structural fields; KG values are kept in provenance on disagreement.
    """
    warnings: list[str] = []
    rules: list[str] = []
    if not group and structure_doc is None:
        raise ValueError("MergeGroup requires at least one record or a structure_doc")

    base = structure_doc or MOFJSONDocument()
    primary = group[0] if group else None

    ids = identifiers or base.identifiers
    if primary is not None:
        if ids.kg_uri is None:
            ids = ids.model_copy(update={"kg_uri": primary.mof_uri})
        if ids.mofid is None and primary.mofid:
            ids = ids.model_copy(
                update={
                    "mofid": primary.mofid,
                    "mofid_corrupted": primary.mofid_corrupted,
                    "structure_stem": primary.structure_stem,
                }
            )
        if primary.is_hypothetical is not None:
            ids = ids.model_copy(update={"is_hypothetical": primary.is_hypothetical})
        if primary.is_functionalized is not None:
            ids = ids.model_copy(update={"is_functionalized": primary.is_functionalized})

    # Properties: union, keep conflicts
    props: list[Property] = list(base.properties or [])
    extras = list(base.extra_properties or [])
    for rec in group:
        props.extend(rec.properties)
        extras.extend(rec.extra_properties)
        warnings.extend(rec.warnings)

    formula_disagreement = False
    kg_formula = primary.chemical_formula if primary else None
    computed_formula = None
    if base.crystal is not None:
        computed_formula = getattr(base.crystal, "formula_sum", None) or getattr(
            base.crystal, "formula", None
        )

    if (
        policy.prefer_computed_structural
        and kg_formula
        and computed_formula
        and _formula_disagree(kg_formula, str(computed_formula))
    ):
        formula_disagreement = True
        warnings.append(
            f"formula disagreement: KG={kg_formula!r} computed={computed_formula!r}; "
            "preferring computed"
        )
        rules.append("prefer_computed_formula")
        # Keep KG value in provenance note
        prov = base.provenance or DocumentProvenance()
        prov = prov.model_copy(
            update={
                "warnings": list(prov.warnings)
                + [f"kg_chemical_formula={kg_formula}"]
            }
        )
        base = base.model_copy(update={"provenance": prov})

    # Symmetry from KG if crystal lacks it
    if primary and primary.space_group_symbol and base.crystal is not None:
        sym = base.crystal.symmetry
        declared = getattr(sym, "declared_space_group_symbol", None)
        if declared in {None, "P1", "P 1"} and primary.space_group_symbol:
            warnings.append(
                f"KG space_group {primary.space_group_symbol!r} recorded in provenance "
                "(crystal layer keeps computed/declared symmetry)"
            )

    sources = list((base.provenance.sources if base.provenance else []) or [])
    if primary:
        sources.append(
            SourceRef(name="MOFology-KG", kind="metadata", file="mof_kg / chemcial_properties.csv")
        )
    if structure_doc is not None:
        rules.append("prefer_computed_structure")

    prov = base.provenance or DocumentProvenance()
    prov = prov.model_copy(
        update={
            "sources": sources,
            "warnings": list(dict.fromkeys([*(prov.warnings or []), *warnings])),
        }
    )

    conflicts: list[ConflictRecord] = list(base.conflicts or [])
    if formula_disagreement and kg_formula and computed_formula:
        conflicts.append(
            ConflictRecord(
                name="chemical_formula",
                values=[kg_formula, str(computed_formula)],
                note="prefer_computed",
            )
        )

    # Topology from KG into reticular if present and reticular missing topology
    reticular = base.reticular
    if primary and primary.topology_symbols and reticular is not None:
        try:
            topo = getattr(reticular, "topology", None)
            net = getattr(topo, "net", None) if topo is not None else None
            if net is None and hasattr(reticular, "model_copy"):
                # leave reticular as-is; crossvalidate will compare
                pass
        except Exception:
            pass

    doc = base.model_copy(
        update={
            "identifiers": ids,
            "properties": props,
            "extra_properties": extras,
            "provenance": prov,
            "warnings": list(dict.fromkeys([*(base.warnings or []), *warnings])),
            "conflicts": conflicts,
        }
    )
    return MergeResult(
        document=doc,
        warnings=warnings,
        rules_fired=rules,
        formula_disagreement=formula_disagreement,
    )


def MergeNormalizedWithStructure(
    normalized: NormalizedRecord,
    structure_doc: MOFJSONDocument,
    *,
    identifiers: Identifiers | None = None,
) -> MergeResult:
    """Convenience: merge one normalised KG record with a structure document."""
    return MergeGroup(
        [normalized],
        structure_doc=structure_doc,
        identifiers=identifiers,
    )


def _formula_disagree(a: str, b: str) -> bool:
    def norm(s: str) -> str:
        return "".join(s.split()).lower().replace(" ", "")

    return norm(a) != norm(b)


__all__ = [
    "DEFAULT_POLICY",
    "GroupDuplicates",
    "MergeGroup",
    "MergeNormalizedWithStructure",
    "MergePolicy",
    "MergeResult",
]
