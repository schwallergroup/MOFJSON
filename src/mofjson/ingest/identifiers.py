"""Parse and resolve every identifier family. The join key problem, in one module.

Phase 09.3, implementing the rules from Phase 00.4 and 00.5.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from mofjson.ingest.kg_csv import KgRecord
from mofjson.ingest.normalize import NormalizedRecord
from mofjson.schema.metadata import Identifiers, validate_refcode
from mofjson.util.ids import MofidParts, parse_mofid_line

if TYPE_CHECKING:
    from mofjson.ingest.manifest import Manifest

_STAB_RE = re.compile(r"^STAB_[0-9a-fA-F]{8}$")
_QMOF_RE = re.compile(r"^qmof-[0-9a-fA-F]{7}$")
_REFCODE_RE = re.compile(r"^[A-Za-z]{6}(\d{2})?$")
_FUNCMOF_RE = re.compile(r"^FuncMOF_(.+)$")

KG_NS = "http://emmo.info/domain-mof/mof-ontology#"


class IdentifierFamily(str, Enum):
    stab = "stab"
    csd_refcode = "csd_refcode"
    qmof = "qmof"
    tobacco_sr = "tobacco_sr"
    functionalized = "functionalized"
    other = "other"


@dataclass
class ResolvedIdentifiers:
    """Result of ResolveIdentifiers."""

    identifiers: Identifiers
    family: IdentifierFamily
    local_name: str
    structure_stem: str | None = None
    resolved: bool = False
    source_tag: str | None = None
    cif_relpath: str | None = None
    warnings: list[str] = field(default_factory=list)


def UriToLocalName(uri: str) -> str:
    """Strip the namespace / fragment to get the URI local name."""
    text = uri.strip()
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    path = urlparse(text).path
    return path.rsplit("/", 1)[-1] if path else text


def ClassifyLocalName(local_name: str) -> IdentifierFamily:
    """Classify a MOF URI local name into an identifier family."""
    name = local_name
    if name.startswith("MOF_"):
        name = name[4:]
    if name.startswith("FuncMOF_"):
        return IdentifierFamily.functionalized
    if _STAB_RE.match(name):
        return IdentifierFamily.stab
    if _QMOF_RE.match(name):
        return IdentifierFamily.qmof
    if _REFCODE_RE.match(name):
        return IdentifierFamily.csd_refcode
    # Bare SR_ stems sometimes appear as local names
    if name.startswith("SR_"):
        return IdentifierFamily.tobacco_sr
    return IdentifierFamily.other


def ParseMofid(raw: str | None) -> MofidParts:
    """Parse a full MOFid line; flag sentinel corruption."""
    return parse_mofid_line(raw)


def ResolveIdentifiers(
    record: KgRecord | NormalizedRecord,
    manifest: Manifest | None = None,
    *,
    structure_root: str | None = None,
) -> ResolvedIdentifiers:
    """Build Identifiers and resolve against the structure manifest.

    Join key is the MOFid structure stem looked up in the manifest, OR the
    precomputed ``mof_uri`` row in the KG structure manifest. Never reconstructs
    a filename from a URI.
    """
    _ = structure_root  # reserved; resolution goes through Manifest only
    warnings: list[str] = []

    if isinstance(record, NormalizedRecord):
        mof_uri = record.mof_uri
        mofid = record.mofid
        stem = record.structure_stem
        is_hyp = record.is_hypothetical
        is_func = record.is_functionalized
        parent = record.parent_mof_uri
        canonical = record.canonical_name
        mofid_corrupted = record.mofid_corrupted
    else:
        mof_uri = record.mof_uri
        mofid = record.mofid
        parts = parse_mofid_line(mofid)
        stem = parts.structure_stem
        mofid_corrupted = parts.has_sentinel
        is_hyp = record.is_hypothetical
        is_func = record.is_functionalized
        parent = record.parent_mof_uri
        canonical = record.canonical_name

    local = UriToLocalName(mof_uri)
    family = ClassifyLocalName(local)

    # Strip MOF_ / FuncMOF_ prefix for typed fields
    bare = local[4:] if local.startswith("MOF_") else local
    func_m = _FUNCMOF_RE.match(local)

    csd_refcode: str | None = None
    qmof_id: str | None = None
    odac_name: str | None = None

    if family == IdentifierFamily.csd_refcode:
        code, diag = validate_refcode(bare)
        csd_refcode = code
        if diag is not None:
            warnings.append(diag.message)
    elif family == IdentifierFamily.qmof:
        qmof_id = bare if bare.startswith("qmof-") else f"qmof-{bare}"
    elif family == IdentifierFamily.stab:
        odac_name = stem or canonical
    elif family == IdentifierFamily.functionalized and func_m:
        odac_name = func_m.group(1)
        is_func = True
    elif family == IdentifierFamily.tobacco_sr:
        odac_name = bare
    elif family == IdentifierFamily.other:
        code, diag = validate_refcode(bare)
        if code is not None:
            csd_refcode = code
        if diag is not None:
            warnings.append(diag.message)

    if stem and stem.startswith("SR_"):
        odac_name = odac_name or stem
        if family == IdentifierFamily.other:
            family = IdentifierFamily.tobacco_sr

    parent_id = None
    if parent:
        parent_id = UriToLocalName(parent)

    ids = Identifiers(
        csd_refcode=csd_refcode,
        qmof_id=qmof_id,
        odac_name=odac_name,
        kg_uri=mof_uri,
        mofid=mofid,
        structure_stem=stem,
        parent_id=parent_id,
        is_functionalized=bool(is_func),
        is_hypothetical=bool(is_hyp) if is_hyp is not None else family
        in {IdentifierFamily.stab, IdentifierFamily.tobacco_sr},
        mofid_corrupted=mofid_corrupted,
        common_names=[canonical] if canonical else [],
    )

    resolved = False
    source_tag: str | None = None
    cif_relpath: str | None = None

    if manifest is not None:
        # Prefer URI join (KG structure manifest), fall back to stem lookup
        loc = manifest.lookup_uri(mof_uri)
        if loc is None and stem:
            loc = manifest.lookup(stem)
        if loc is not None:
            resolved = loc.resolved if loc.resolved is not None else True
            source_tag = loc.source_tag
            cif_relpath = loc.cif_relpath or (
                loc.path if loc.kind == "cif" else None
            )
            if cif_relpath:
                ids = ids.model_copy(update={"source_filename": cif_relpath})
            if loc.kind == "aselmdb" and odac_name is None:
                ids = ids.model_copy(update={"odac_name": stem})
        else:
            warnings.append(f"no structure location for uri/stem {mof_uri!r}/{stem!r}")

    return ResolvedIdentifiers(
        identifiers=ids,
        family=family,
        local_name=local,
        structure_stem=stem,
        resolved=resolved,
        source_tag=source_tag,
        cif_relpath=cif_relpath,
        warnings=warnings,
    )


def FamilyHistogram(records: list[KgRecord | NormalizedRecord]) -> dict[str, int]:
    """Count identifier families over a list of records."""
    counts: Counter[str] = Counter()
    for rec in records:
        uri = rec.mof_uri
        counts[ClassifyLocalName(UriToLocalName(uri)).value] += 1
    return dict(counts)


# Expected Phase 00.3 counts for regression tests.
EXPECTED_FAMILY_COUNTS: dict[str, int] = {
    "stab": 209_799,
    "csd_refcode": 38_687,
    "qmof": 3_509,
    "other": 1_296,
}


__all__ = [
    "EXPECTED_FAMILY_COUNTS",
    "FamilyHistogram",
    "IdentifierFamily",
    "ParseMofid",
    "ResolveIdentifiers",
    "ResolvedIdentifiers",
    "UriToLocalName",
    "ClassifyLocalName",
]
