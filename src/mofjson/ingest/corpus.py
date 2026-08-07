"""Drive the whole ingestion and emit the corpus with its audit.

Phase 09.8.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mofjson.ingest.crossvalidate import BuildCrossValidationReport, CrossValidationReport
from mofjson.ingest.identifiers import ResolveIdentifiers
from mofjson.ingest.kg_csv import EXPECTED_ROW_COUNT, KgRecord, ReadKgCsv
from mofjson.ingest.kg_ttl import ReadKgTtl
from mofjson.ingest.manifest import Manifest
from mofjson.ingest.merge import MergeNormalizedWithStructure
from mofjson.ingest.normalize import NormalizeRecord
from mofjson.schema.document import MOFJSONDocument
from mofjson.schema.metadata import DocumentProvenance, PipelineStep, SourceRef
from mofjson.util.io import ResumableWriter
from mofjson.version import __version__

KgSourceKind = Literal["csv", "ttl"]


@dataclass
class CorpusReport:
    """Audit for a corpus build."""

    input_count: int = 0
    output_count: int = 0
    buckets: Counter[str] = field(default_factory=Counter)
    quirk_counts: Counter[str] = field(default_factory=Counter)
    source_tag_counts: Counter[str] = field(default_factory=Counter)
    family_counts: Counter[str] = field(default_factory=Counter)
    formula_disagreements: int = 0
    crossvalidation: CrossValidationReport = field(default_factory=CrossValidationReport)
    failures: list[dict[str, Any]] = field(default_factory=list)
    no_output: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        bucket_sum = sum(self.buckets.values())
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "buckets": dict(self.buckets),
            "bucket_sum": bucket_sum,
            "buckets_balance": bucket_sum == self.input_count,
            "quirk_counts": dict(self.quirk_counts),
            "source_tag_counts": dict(self.source_tag_counts),
            "family_counts": dict(self.family_counts),
            "formula_disagreements": self.formula_disagreements,
            "crossvalidation": self.crossvalidation.to_dict(),
            "n_failures": len(self.failures),
            "failures_sample": self.failures[:50],
            "n_no_output": len(self.no_output),
            "no_output_sample": self.no_output[:50],
            "elapsed_s": self.elapsed_s,
            "config": self.config,
            "expected_input_count": EXPECTED_ROW_COUNT,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def _detect_kg_kind(path: Path) -> KgSourceKind:
    suf = path.suffix.lower()
    if suf in {".csv"}:
        return "csv"
    if suf in {".ttl", ".turtle", ".nt"}:
        return "ttl"
    # peek
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(200)
    if head.lstrip().startswith("@prefix") or head.lstrip().startswith("#"):
        return "ttl"
    return "csv"


def _iter_kg(path: Path, kind: KgSourceKind, limit: int | None) -> Iterator[KgRecord]:
    if kind == "csv":
        yield from ReadKgCsv(path, limit=limit)
    else:
        yield from ReadKgTtl(path, limit=limit)


def _load_structure(
    loc: Any,
    *,
    structure_root: Path | None,
) -> MOFJSONDocument | None:
    if loc is None or not loc.resolved:
        return None
    if loc.kind == "cif" and (loc.path or loc.cif_relpath):
        from mofjson.convert.cif import MOFFromCif

        path = Path(loc.path) if loc.path else None
        if path is None and loc.cif_relpath and structure_root is not None:
            path = structure_root / loc.cif_relpath
        if path is None or not path.is_file():
            return None
        return MOFFromCif(path, derive_bonds=True)
    if loc.kind == "aselmdb" and loc.path is not None:
        from mofjson.convert.ase_bridge import MOFFromAtoms
        from mofjson.ingest.odac import OpenAseLmdb

        reader = OpenAseLmdb(loc.path)
        try:
            if loc.sid is not None:
                atoms = reader.get_by_sid(int(loc.sid))
            elif loc.record_index is not None:
                atoms = reader.get(int(loc.record_index))
            else:
                return None
            doc = MOFFromAtoms(atoms, derive_bonds=True, source_path=str(loc.path))
            # Provenance: inferred bonds
            prov = doc.provenance or DocumentProvenance()
            step = PipelineStep(
                step="odac_lmdb",
                tool="ase.db",
                params={"path": loc.path, "sid": loc.sid, "bonds": "inferred"},
            )
            doc.provenance = prov.model_copy(
                update={
                    "pipeline": list(prov.pipeline) + [step],
                    "warnings": list(prov.warnings)
                    + ["ODAC structure: bonds inferred (no deposited bond graph)"],
                    "sources": list(prov.sources)
                    + [SourceRef(name="ODAC25", kind="structure", file=str(loc.path))],
                }
            )
            return doc
        finally:
            reader.close()
    return None


def BuildCorpus(
    *,
    kg_source: str | Path,
    manifest: Manifest | str | Path,
    out: str | Path,
    limit: int | None = None,
    workers: int | None = None,
    deconstruct: bool = True,
    geometry: bool = False,
    resume: bool = True,
    structure_root: str | Path | None = None,
    kg_kind: KgSourceKind | None = None,
) -> CorpusReport:
    """Run the full ingestion pipeline and emit sharded/resumable NDJSON."""
    _ = workers  # reserved for parallel runs
    t0 = time.perf_counter()
    kg_path = Path(kg_source)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kind = kg_kind or _detect_kg_kind(kg_path)

    if isinstance(manifest, Manifest):
        man = manifest
    else:
        man = Manifest.load(manifest)

    root = Path(structure_root) if structure_root else Path(
        man.meta.get("structure_root") or "/data/matthart/mof_structures"
    )

    report = CorpusReport(
        config={
            "kg_source": str(kg_path),
            "kg_kind": kind,
            "out": str(out_path),
            "limit": limit,
            "deconstruct": deconstruct,
            "geometry": geometry,
            "resume": resume,
            "mofjson_version": __version__,
        }
    )

    ndjson_path = out_path if out_path.suffix else out_path / "corpus.ndjson"
    if out_path.suffix == "":
        out_path.mkdir(parents=True, exist_ok=True)
        ndjson_path = out_path / "corpus.ndjson"
        audit_path = out_path / "corpus_report.json"
    else:
        audit_path = out_path.with_suffix(".report.json")

    def _key(rec: dict[str, Any]) -> str:
        ids = rec.get("identifiers") or {}
        return str(ids.get("kg_uri") or ids.get("structure_stem") or rec.get("mofjson_version"))

    with ResumableWriter(ndjson_path, _key, resume=resume) as writer:
        for raw in _iter_kg(kg_path, kind, limit):
            report.input_count += 1
            try:
                normalized, warns = NormalizeRecord(raw)
                report.quirk_counts.update(normalized.quirk_counts)

                resolved = ResolveIdentifiers(normalized, man)
                report.family_counts[resolved.family.value] += 1
                if resolved.source_tag:
                    report.source_tag_counts[resolved.source_tag] += 1

                loc = man.lookup_uri(raw.mof_uri)
                if loc is None and resolved.structure_stem:
                    loc = man.lookup(resolved.structure_stem)

                structure_doc = _load_structure(loc, structure_root=root)

                if structure_doc is None:
                    # Metadata-only document
                    doc = MOFJSONDocument(
                        identifiers=resolved.identifiers,
                        properties=list(normalized.properties),
                        extra_properties=list(normalized.extra_properties),
                        warnings=list(
                            dict.fromkeys([*normalized.warnings, *resolved.warnings, *warns])
                        ),
                        provenance=DocumentProvenance(
                            sources=[
                                SourceRef(
                                    name="MOFology-KG",
                                    kind="metadata",
                                    file=str(kg_path.name),
                                )
                            ],
                            warnings=list(normalized.warnings),
                        ),
                    )
                    if loc is None:
                        bucket = "metadata_unresolved"
                    elif loc.kind == "ccdc_required" or (
                        loc.source_tag and "ccdc_required" in loc.source_tag
                    ):
                        bucket = "metadata_ccdc_required"
                    else:
                        bucket = "metadata_no_structure"
                    report.buckets[bucket] += 1
                    payload = doc.model_dump(mode="json", exclude_none=True)
                    if writer.write(payload):
                        report.output_count += 1
                    continue

                # Deconstruct
                if deconstruct:
                    try:
                        from mofjson.deconstruct import Deconstruct

                        structure_doc = Deconstruct(structure_doc, geometry=geometry)
                    except Exception as exc:
                        report.failures.append(
                            {"mof_uri": raw.mof_uri, "stage": "deconstruct", "error": str(exc)}
                        )
                        report.buckets["deconstruct_failed"] += 1
                        # Still emit crystal-only merge
                        pass

                merge = MergeNormalizedWithStructure(
                    normalized,
                    structure_doc,
                    identifiers=resolved.identifiers,
                )
                doc = merge.document
                if merge.formula_disagreement:
                    report.formula_disagreements += 1

                xval = BuildCrossValidationReport(doc, normalized)
                report.crossvalidation.extend(xval)

                # Attach crossval disagreements into warnings
                if xval.disagreements:
                    doc = doc.model_copy(
                        update={
                            "warnings": list(doc.warnings)
                            + [
                                f"crossvalidate:{d.field} asserted={d.asserted!r} "
                                f"computed={d.computed!r}"
                                for d in xval.disagreements
                            ]
                        }
                    )

                report.buckets["emitted_with_structure"] += 1
                payload = doc.model_dump(mode="json", exclude_none=True)
                if writer.write(payload):
                    report.output_count += 1
                else:
                    report.buckets["skipped_resume"] += 1
            except Exception as exc:
                report.failures.append(
                    {"mof_uri": getattr(raw, "mof_uri", None), "stage": "pipeline", "error": str(exc)}
                )
                report.buckets["pipeline_error"] += 1
                report.no_output.append(str(getattr(raw, "mof_uri", "")))

    report.elapsed_s = time.perf_counter() - t0
    # Ensure every input is in exactly one bucket — fold resume skips into accounting
    accounted = sum(report.buckets.values())
    if accounted < report.input_count:
        report.buckets["unaccounted"] += report.input_count - accounted

    report.save(audit_path)
    return report


__all__ = [
    "BuildCorpus",
    "CorpusReport",
]
