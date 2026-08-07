"""Corpus-scale driving of the converters: directories, NDJSON streaming.

Phase 07.6.
"""

from __future__ import annotations

import time
import traceback
from collections import Counter
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mofjson.convert.cif import MOFFromCif
from mofjson.exceptions import ConversionError
from mofjson.schema.document import MOFJSONDocument
from mofjson.util.io import ResumableWriter


@dataclass
class BatchFailure:
    path: str
    reason: str
    category: str
    traceback: str | None = None


@dataclass
class BatchReport:
    n_ok: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    elapsed_s: float = 0.0
    failures: list[BatchFailure] = field(default_factory=list)
    categories: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "n_skipped": self.n_skipped,
            "elapsed_s": self.elapsed_s,
            "categories": dict(self.categories),
            "failures": [
                {
                    "path": f.path,
                    "reason": f.reason,
                    "category": f.category,
                    "traceback": f.traceback,
                }
                for f in self.failures
            ],
        }


def _categorise(exc: BaseException) -> str:
    name = type(exc).__name__
    msg = str(exc).lower()
    if "occupancy" in msg or "disorder" in msg:
        return "disorder_occupancy"
    if "parse" in msg or "cif" in msg:
        return "cif_parse"
    if "bond" in msg:
        return "bond_derivation"
    if "symmetry" in msg or "spglib" in msg:
        return "symmetry"
    if isinstance(exc, ConversionError):
        return "conversion"
    if isinstance(exc, ImportError):
        return "missing_dependency"
    return name


def iter_documents(
    paths: list[Path],
    *,
    derive_bonds: bool = True,
    charge_source: str = "pbe_ddec",
    strict: bool = False,
) -> Iterator[MOFJSONDocument | BatchFailure]:
    """Yield documents or failures for each CIF path (sequential)."""
    for path in paths:
        try:
            doc = MOFFromCif(
                path,
                derive_bonds=derive_bonds,
                charge_source=charge_source,
                strict=strict,
            )
            yield doc
        except Exception as exc:
            yield BatchFailure(
                path=str(path),
                reason=str(exc),
                category=_categorise(exc),
                traceback=traceback.format_exc(),
            )


def ConvertDirectory(
    path: str | Path,
    *,
    pattern: str = "*.cif",
    workers: int | None = None,
    out: str | Path | None = None,
    resume: bool = True,
    derive_bonds: bool = True,
    charge_source: str = "pbe_ddec",
    strict: bool = False,
    progress: bool = True,
    error_log: str | Path | None = None,
) -> BatchReport:
    """Walk a directory of CIFs, stream NDJSON, never hold more than one structure.

    ``workers`` is accepted for API compatibility; parallel execution is not used
    in v0.1 (deterministic ordering). Resume skips stems already present in ``out``.
    """
    _ = workers  # reserved; sequential for determinism
    root = Path(path)
    if not root.is_dir():
        raise ConversionError(f"Not a directory: {root}", source_path=str(root))
    cif_paths = sorted(root.rglob(pattern))
    report = BatchReport()
    t0 = time.perf_counter()
    out_path = Path(out) if out is not None else None
    err_path = (
        Path(error_log)
        if error_log is not None
        else (out_path.with_suffix(".errors.ndjson") if out_path is not None else None)
    )
    cat: Counter[str] = Counter()

    def key_fn(rec: dict[str, Any]) -> str:
        ids = rec.get("identifiers") or {}
        stem = ids.get("structure_stem") or ids.get("source_filename")
        if stem:
            return str(Path(str(stem)).stem)
        for src in (rec.get("provenance") or {}).get("sources") or []:
            if src.get("file"):
                return Path(src["file"]).stem
        return str(rec.get("_stem") or "")

    writer: ResumableWriter | None = None
    err_writer: ResumableWriter | None = None
    writer_cm: Any = nullcontext()
    err_cm: Any = nullcontext()

    if out_path is not None:
        writer = ResumableWriter(out_path, key_fn=key_fn, resume=resume)
        writer_cm = writer
    if err_path is not None:
        err_writer = ResumableWriter(
            err_path, key_fn=lambda r: str(r.get("path", "")), resume=resume
        )
        err_cm = err_writer

    with writer_cm, err_cm:
        for i, cif_path in enumerate(cif_paths, start=1):
            stem = cif_path.stem
            if writer is not None and stem in writer.seen:
                report.n_skipped += 1
                continue
            if progress and (i == 1 or i % 50 == 0 or i == len(cif_paths)):
                print(f"[{i}/{len(cif_paths)}] {cif_path.name}", flush=True)
            try:
                doc = MOFFromCif(
                    cif_path,
                    derive_bonds=derive_bonds,
                    charge_source=charge_source,
                    strict=strict,
                )
                if doc.identifiers.structure_stem is None:
                    doc.identifiers = doc.identifiers.model_copy(
                        update={
                            "structure_stem": stem,
                            "source_filename": cif_path.name,
                        }
                    )
                rec = doc.model_dump(mode="json", exclude_none=True)
                rec["_stem"] = stem
                if writer is not None:
                    writer.write(rec)
                report.n_ok += 1
            except Exception as exc:
                fail = BatchFailure(
                    path=str(cif_path),
                    reason=str(exc),
                    category=_categorise(exc),
                    traceback=traceback.format_exc(),
                )
                report.failures.append(fail)
                report.n_failed += 1
                cat[fail.category] += 1
                if err_writer is not None:
                    err_writer.write(
                        {
                            "path": fail.path,
                            "reason": fail.reason,
                            "category": fail.category,
                        }
                    )

    report.categories = dict(cat)
    report.elapsed_s = time.perf_counter() - t0
    return report


__all__ = [
    "BatchFailure",
    "BatchReport",
    "ConvertDirectory",
    "iter_documents",
]
