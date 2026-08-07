"""The structure manifest: map a filename stem / mof_uri to where the structure lives.

Phase 09.3.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from mofjson.exceptions import IngestError

Kind = Literal["cif", "aselmdb", "unresolved", "ccdc_required"]

# Phase 00.8 expected resolution histogram (structure manifest.csv).
EXPECTED_SOURCE_TAG_COUNTS: dict[str, int] = {
    "unresolved": 209_799,
    "ccdc_required": 21_985,
    "qmof_csd": 11_309,
    "qmof_id": 8_347,
    "ccdc_required+parent": 2_012,
    "qmof_csd+parent": 590,
    "qmof_id+parent": 44,
    "local_gen": 8,
    "local_gen+parent": 3,
    "qmof_mpcontribs+parent": 1,
    "qmof_mpcontribs": 1,
}
EXPECTED_RESOLVED_COUNT = 20_301
EXPECTED_MANIFEST_ROWS = 254_099


@dataclass
class StructureLocation:
    """Where a structure can be loaded from."""

    kind: Kind
    path: str | None = None
    # For ASE-LMDB: record index / sid
    record_index: int | None = None
    sid: int | None = None
    mof_uri: str | None = None
    stem: str | None = None
    source_tag: str | None = None
    resolved: bool | None = None
    cif_relpath: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructureLocation:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass
class Manifest:
    """In-memory stem/uri → StructureLocation index with disk cache."""

    by_stem: dict[str, StructureLocation] = field(default_factory=dict)
    by_uri: dict[str, StructureLocation] = field(default_factory=dict)
    collisions: list[dict[str, Any]] = field(default_factory=list)
    source_tag_counts: Counter[str] = field(default_factory=Counter)
    meta: dict[str, Any] = field(default_factory=dict)

    def lookup(self, stem: str) -> StructureLocation | None:
        """Exact stem lookup. Never fuzzy-matches."""
        return self.by_stem.get(stem)

    def lookup_uri(self, mof_uri: str) -> StructureLocation | None:
        return self.by_uri.get(mof_uri)

    def coverage(self) -> dict[str, Any]:
        resolved = sum(1 for loc in self.by_uri.values() if loc.resolved)
        by_kind: Counter[str] = Counter(loc.kind for loc in self.by_uri.values())
        return {
            "n_uri": len(self.by_uri),
            "n_stem": len(self.by_stem),
            "resolved": resolved,
            "by_kind": dict(by_kind),
            "source_tag_counts": dict(self.source_tag_counts),
            "collisions": len(self.collisions),
        }

    def save(self, path: str | Path) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "by_stem": {k: v.to_dict() for k, v in self.by_stem.items()},
            "by_uri": {k: v.to_dict() for k, v in self.by_uri.items()},
            "collisions": self.collisions,
            "source_tag_counts": dict(self.source_tag_counts),
        }
        dest.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        p = Path(path)
        if not p.is_file():
            raise IngestError(f"manifest cache not found: {p}", source_path=str(p))
        payload = json.loads(p.read_text(encoding="utf-8"))
        m = cls(
            by_stem={k: StructureLocation.from_dict(v) for k, v in payload.get("by_stem", {}).items()},
            by_uri={k: StructureLocation.from_dict(v) for k, v in payload.get("by_uri", {}).items()},
            collisions=list(payload.get("collisions", [])),
            source_tag_counts=Counter(payload.get("source_tag_counts", {})),
            meta=dict(payload.get("meta", {})),
        )
        # Stale-cache check
        expected = m.meta.get("input_fingerprint")
        if expected:
            current = _fingerprint_inputs(m.meta.get("inputs", {}))
            if current and current != expected:
                raise IngestError(
                    "stale manifest cache: input fingerprint mismatch; rebuild with build_manifest.py",
                    source_path=str(p),
                    value={"expected": expected, "current": current},
                )
        return m


def _fingerprint_inputs(inputs: dict[str, Any]) -> str | None:
    if not inputs:
        return None
    parts: list[str] = []
    for key in sorted(inputs):
        raw = inputs[key]
        if isinstance(raw, list):
            for item in raw:
                p = Path(str(item))
                if p.exists():
                    st = p.stat()
                    parts.append(f"{p}:{st.st_mtime_ns}:{st.st_size}")
                else:
                    parts.append(f"{p}:missing")
        else:
            p = Path(str(raw))
            if p.exists():
                st = p.stat()
                parts.append(f"{p}:{st.st_mtime_ns}:{st.st_size}")
            else:
                parts.append(f"{p}:missing")
    if not parts:
        return None
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def LoadKgStructureManifest(
    path: str | Path,
    *,
    structure_root: str | Path | None = None,
) -> Manifest:
    """Load the Phase 00.8 ``manifest.csv`` (one row per KG MOF)."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise IngestError(
            "pandas is required; install with `pip install mofjson[kg]`",
            source_path=str(path),
        ) from exc

    p = Path(path)
    if not p.is_file():
        raise IngestError(f"structure manifest not found: {p}", source_path=str(p))

    root = Path(structure_root) if structure_root else p.parent
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    m = Manifest()
    m.meta = {
        "inputs": {"kg_manifest_csv": str(p)},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "structure_root": str(root),
    }
    m.meta["input_fingerprint"] = _fingerprint_inputs(m.meta["inputs"])

    for _, row in df.iterrows():
        uri = (row.get("mof_uri") or "").strip()
        if not uri:
            continue
        source_tag = (row.get("source_tag") or "").strip() or "unresolved"
        resolved_raw = (row.get("resolved") or "").strip().lower()
        resolved = resolved_raw in {"1", "true", "yes"}
        cif_rel = (row.get("cif_relpath") or "").strip() or None
        mofid = (row.get("mofid") or "").strip() or None
        stem = None
        if mofid and ";" in mofid:
            stem = mofid.rsplit(";", 1)[-1].strip() or None

        if resolved and cif_rel:
            kind: Kind = "cif"
            abs_path = str((root / cif_rel).resolve()) if not Path(cif_rel).is_absolute() else cif_rel
        elif "ccdc_required" in source_tag:
            kind = "ccdc_required"
            abs_path = None
        else:
            kind = "unresolved"
            abs_path = None

        loc = StructureLocation(
            kind=kind,
            path=abs_path,
            mof_uri=uri,
            stem=stem,
            source_tag=source_tag,
            resolved=resolved,
            cif_relpath=cif_rel,
            note=(row.get("note") or "").strip() or None,
        )
        if uri in m.by_uri:
            m.collisions.append({"type": "uri", "key": uri, "existing": m.by_uri[uri].to_dict(), "new": loc.to_dict()})
        m.by_uri[uri] = loc
        if stem:
            if stem in m.by_stem and m.by_stem[stem].mof_uri != uri:
                m.collisions.append(
                    {
                        "type": "stem",
                        "key": stem,
                        "existing": m.by_stem[stem].to_dict(),
                        "new": loc.to_dict(),
                    }
                )
            m.by_stem[stem] = loc
        m.source_tag_counts[source_tag] += 1

    return m


def BuildManifest(
    cif_dirs: Iterable[str | Path] | None = None,
    lmdb_dirs: Iterable[str | Path] | None = None,
    *,
    kg_manifest_csv: str | Path | None = None,
    structure_root: str | Path | None = None,
) -> Manifest:
    """Build a manifest from the KG CSV plus optional CIF / LMDB directory scans."""
    if kg_manifest_csv is not None:
        m = LoadKgStructureManifest(kg_manifest_csv, structure_root=structure_root)
    else:
        m = Manifest()
        m.meta = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "inputs": {}}

    inputs = dict(m.meta.get("inputs", {}))

    if cif_dirs:
        cif_list = [str(p) for p in cif_dirs]
        inputs["cif_dirs"] = cif_list
        for d in cif_list:
            root = Path(d)
            if not root.is_dir():
                continue
            for cif in root.rglob("*.cif"):
                stem = cif.stem
                # MOF_HIHWED.cif → stem variants
                loc = StructureLocation(
                    kind="cif",
                    path=str(cif.resolve()),
                    stem=stem,
                    cif_relpath=str(cif),
                    resolved=True,
                    source_tag="cif_scan",
                )
                if stem in m.by_stem and m.by_stem[stem].path not in {None, loc.path}:
                    m.collisions.append(
                        {
                            "type": "stem",
                            "key": stem,
                            "existing": m.by_stem[stem].to_dict(),
                            "new": loc.to_dict(),
                        }
                    )
                m.by_stem.setdefault(stem, loc)

    if lmdb_dirs:
        from mofjson.ingest.odac import IndexAseLmdbDirs

        lmdb_list = [str(p) for p in lmdb_dirs]
        inputs["lmdb_dirs"] = lmdb_list
        index = IndexAseLmdbDirs(lmdb_list)
        for name, (db_path, sid) in index.items():
            loc = StructureLocation(
                kind="aselmdb",
                path=db_path,
                sid=sid,
                record_index=sid,
                stem=name,
                resolved=True,
                source_tag="odac25",
            )
            if name in m.by_stem and m.by_stem[name].kind == "cif":
                m.collisions.append(
                    {
                        "type": "stem",
                        "key": name,
                        "existing": m.by_stem[name].to_dict(),
                        "new": loc.to_dict(),
                    }
                )
            # Prefer keeping CIF if already resolved; otherwise set ODAC
            existing = m.by_stem.get(name)
            if existing is None or not existing.resolved or existing.kind != "cif":
                m.by_stem[name] = loc
            # Also attach to unresolved STAB URIs whose stem matches
            for uri, uloc in list(m.by_uri.items()):
                if uloc.stem == name and not uloc.resolved:
                    m.by_uri[uri] = StructureLocation(
                        kind="aselmdb",
                        path=db_path,
                        sid=sid,
                        record_index=sid,
                        mof_uri=uri,
                        stem=name,
                        resolved=True,
                        source_tag="odac25",
                    )
                    m.source_tag_counts["odac25"] += 1
                    if uloc.source_tag:
                        m.source_tag_counts[uloc.source_tag] -= 1

    m.meta["inputs"] = inputs
    m.meta["input_fingerprint"] = _fingerprint_inputs(inputs)
    return m


__all__ = [
    "EXPECTED_MANIFEST_ROWS",
    "EXPECTED_RESOLVED_COUNT",
    "EXPECTED_SOURCE_TAG_COUNTS",
    "BuildManifest",
    "LoadKgStructureManifest",
    "Manifest",
    "StructureLocation",
]
