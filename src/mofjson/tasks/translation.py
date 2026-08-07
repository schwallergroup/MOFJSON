"""Family A -- translation: read a structure in one representation, write it in another.

Phase 12.2.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from mofjson.schema.document import MOFJSONDocument
from mofjson.schema.reticular import ReticularStructure
from mofjson.tasks.item import ItemProvenance, ItemSource, ItemStrata, TaskItem
from mofjson.tasks.representations import REPRESENTATIONS, Render

GENERATOR_VERSION = "tasks/translation.py@v1"

# Crystal-like representations that can be graded with StructureMatcher.
_CRYSTAL_REPS = frozenset(
    {
        "cif",
        "lattice_coords_text",
        "mofjson_crystal",
        "mofjson_crystal_columnar",
    }
)
_RETICULAR_REPS = frozenset(
    {
        "mofjson_reticular",
        "mofjson_reticular_full",
        "mofjson_reticular_nogeom",
    }
)

# Exploratory all-pairs matrix (source, target).
ALL_PAIRS_MATRIX: list[tuple[str, str]] = [
    (a, b)
    for a in sorted(_CRYSTAL_REPS | {"mofjson_reticular_full", "mofid"})
    for b in sorted(_CRYSTAL_REPS | {"mofjson_reticular", "mofid"})
    if a != b
]

# Pairs worth scaling after the exploratory matrix.
TRANSLATION_PAIRS: list[tuple[str, str]] = [
    ("cif", "mofjson_crystal"),
    ("mofjson_crystal", "cif"),
    ("mofjson_crystal", "mofjson_crystal_columnar"),
    ("mofjson_crystal_columnar", "mofjson_crystal"),
    ("lattice_coords_text", "mofjson_crystal"),
    ("mofjson_crystal", "lattice_coords_text"),
    ("cif", "mofjson_reticular"),
    ("mofjson_crystal", "mofjson_reticular"),
    ("mofjson_crystal", "mofid"),
    ("cif", "mofid"),
    # Full reticular → crystal is included (phase 12.2); partial → crystal is excluded.
    ("mofjson_reticular_full", "cif"),
    ("mofjson_reticular_full", "mofjson_crystal"),
]


def _is_full_reticular(doc: MOFJSONDocument) -> bool:
    if doc.reticular is None:
        return False
    ret = doc.reticular
    if isinstance(ret, ReticularStructure):
        return bool(ret.assembly and ret.assembly.completeness == "full")
    if isinstance(ret, dict):
        assembly = ret.get("assembly") or {}
        return assembly.get("completeness") == "full"
    return False


def _assert_full_if_needed(doc: MOFJSONDocument, source: str) -> None:
    if source == "mofjson_reticular_full" and not _is_full_reticular(doc):
        raise ValueError("source mofjson_reticular_full requires assembly.completeness == 'full'")


def _pair_meta(source: str, target: str) -> dict[str, str]:
    # Exclude partial reticular → crystal.
    if source in ("mofjson_reticular", "mofjson_reticular_nogeom") and target in _CRYSTAL_REPS:
        raise ValueError(f"pair {source}->{target} excluded (partial reticular → crystal)")
    if target == "mofid":
        return {
            "grader": "canonical_string",
            "answer_type": "string",
            "response_schema": "profiles/answer_mofid.schema.json",
        }
    if target in _RETICULAR_REPS:
        return {
            "grader": "reticular_match",
            "answer_type": "mofjson",
            "response_schema": REPRESENTATIONS[target].answer_schema,
        }
    if target in _CRYSTAL_REPS:
        schema = REPRESENTATIONS[target].answer_schema
        return {
            "grader": "structure_match",
            "answer_type": "structure",
            "response_schema": schema,
        }
    raise ValueError(f"unsupported target {target!r}")


def _gold_answer(doc: MOFJSONDocument, target: str) -> Any:
    if target == "mofid":
        from mofjson.tasks.representations import _canonical_mofid

        raw = doc.identifiers.mofid if doc.identifiers else None
        if raw is None and isinstance(doc.reticular, dict):
            raw = doc.reticular.get("mofid")
        elif doc.reticular is not None and hasattr(doc.reticular, "mofid"):
            raw = getattr(doc.reticular, "mofid", None)
        # Also try golden reticular documents that store mofid at top level of reticular dump.
        canon = _canonical_mofid(raw)
        if not canon:
            raise ValueError("no mofid for gold answer")
        return canon
    if target in _RETICULAR_REPS:
        if doc.reticular is None:
            raise ValueError("no reticular layer")
        if hasattr(doc.reticular, "model_dump"):
            return doc.reticular.model_dump(mode="json", exclude_none=True)
        return doc.reticular
    if target in _CRYSTAL_REPS:
        if doc.crystal is None:
            raise ValueError("no crystal layer")
        # Store full document-shaped gold for structure_match.
        return {"crystal": doc.crystal.model_dump(mode="json", exclude_none=True)}
    raise ValueError(target)


def _source_from_doc(doc: MOFJSONDocument) -> ItemSource:
    ids = doc.identifiers
    return ItemSource(
        mofjson_id=ids.mofjson_id if ids else None,
        odac_name=ids.odac_name if ids else None,
        csd_refcode=ids.csd_refcode if ids else None,
        structure_stem=ids.structure_stem if ids else None,
        is_hypothetical=bool(ids.is_hypothetical) if ids else False,
    )


def _strata(doc: MOFJSONDocument) -> ItemStrata:
    from mofjson.vocab.elements import METALS

    n_atoms = n_metals = None
    if doc.crystal is not None:
        n_atoms = doc.crystal.n_atoms or len(doc.crystal.sites)
        n_metals = sum(1 for s in doc.crystal.sites if s.element in METALS)
    topo = None
    n_linkers = None
    if doc.reticular is not None:
        ret = doc.reticular
        if hasattr(ret, "topology") and ret.topology:
            topo = ret.topology.rcsr_symbol
        elif isinstance(ret, dict):
            topo = (ret.get("topology") or {}).get("rcsr_symbol")
        linkers = ret.linkers if hasattr(ret, "linkers") else (ret.get("linkers") if isinstance(ret, dict) else None)
        if linkers is not None:
            n_linkers = len(linkers)
    return ItemStrata(
        n_atoms=n_atoms,
        n_metals=n_metals,
        topology=topo,
        n_distinct_linkers=n_linkers,
    )


def GenerateTranslationItems(
    documents: Sequence[MOFJSONDocument],
    pairs: Sequence[tuple[str, str]] | None = None,
    *,
    n_per_stratum: int | None = None,
) -> Iterator[TaskItem]:
    """Generate translation items for the given (source, target) pairs."""
    _ = n_per_stratum
    use_pairs = list(pairs) if pairs is not None else list(TRANSLATION_PAIRS)
    seq = 0
    for doc in documents:
        try:
            doc.ensure_mofjson_id()
        except Exception:
            pass
        for source, target in use_pairs:
            try:
                _assert_full_if_needed(doc, source)
                meta = _pair_meta(source, target)
                block = Render(doc, source)
                gold = _gold_answer(doc, target)
            except Exception:
                continue
            question = (
                f"Translate the following structure from {source} into {target}. "
                f"Return JSON matching the provided answer schema."
            )
            prompt = (
                f"Given the following structure representation, answer the question.\n"
                f"Return JSON only matching the provided answer schema.\n\n"
                f"Structure:\n{block}\n\nQuestion:\n{question}\n"
            )
            seq += 1
            yield TaskItem(
                item_id=f"tr-{seq:05d}",
                family="translation",
                task=f"{source}__to__{target}",
                representation=source,
                prompt=prompt,
                response_schema=meta["response_schema"],
                answer=gold,
                answer_type=meta["answer_type"],  # type: ignore[arg-type]
                grader=meta["grader"],
                source=_source_from_doc(doc),
                strata=_strata(doc),
                provenance=ItemProvenance(
                    ground_truth_method="stored_source_document",
                    generator=GENERATOR_VERSION,
                ),
            )


__all__ = [
    "ALL_PAIRS_MATRIX",
    "GENERATOR_VERSION",
    "GenerateTranslationItems",
    "TRANSLATION_PAIRS",
]
