"""ODAC25 ASE-LMDB access: the 83% of the corpus that has no CIF.

Phase 09.4.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mofjson.exceptions import IngestError


def _require_ase_db() -> Any:
    try:
        from ase.db import connect
    except ImportError as exc:
        raise IngestError(
            "ase is required for ODAC LMDB access; install with `pip install mofjson[crystal]`",
        ) from exc
    return connect


def _connect(path: Path) -> Any:
    """Open an ASE / fairchem ``.aselmdb`` read-only."""
    connect = _require_ase_db()
    kwargs: dict[str, Any] = {}
    if "aselmdb" in path.name or path.suffix == ".aselmdb":
        kwargs["readonly"] = True
        kwargs["use_lock_file"] = False
    return connect(str(path), **kwargs)


@dataclass
class AseLmdbReader:
    """Read-only wrapper around an ASE LMDB / ase.db database."""

    path: Path
    _db: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._db = _connect(self.path)

    def __len__(self) -> int:
        return int(len(self._db))

    def get(self, index: int) -> Any:
        """Return ase.Atoms for a 1-based ase.db id."""
        return self._db.get_atoms(id=index)

    def get_by_sid(self, sid: int) -> Any:
        """Return the first frame for a given sid."""
        for row in self._db.select(sid=sid):
            return row.toatoms()
        # Fallback: some backends store sid only in data dict
        for row in self._db.select():
            data = dict(getattr(row, "data", {}) or {})
            if int(data.get("sid", -1)) == int(sid):
                return row.toatoms()
        raise KeyError(f"sid {sid} not found in {self.path}")

    def iter(self) -> Iterator[Any]:
        for row in self._db.select():
            yield row.toatoms()

    def iter_rows(self) -> Iterator[Any]:
        yield from self._db.select()

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass


def OpenAseLmdb(path: str | Path) -> AseLmdbReader:
    return AseLmdbReader(Path(path))


def IndexAseLmdb(path: str | Path, *, dedupe_sid: bool = True) -> dict[str, int]:
    """Build ``mof_name → sid`` for one LMDB file.

    Rows repeat per frame (fid); when *dedupe_sid* is True, keep one entry per sid.
    """
    reader = OpenAseLmdb(path)
    index: dict[str, int] = {}
    seen_sid: set[int] = set()
    try:
        for row in reader.iter_rows():
            data = dict(getattr(row, "data", {}) or {})
            mof_name = data.get("mof_name")
            if mof_name is None:
                name = data.get("name")
                if name:
                    mof_name = str(name).split("_w_")[0]
            if mof_name is None:
                continue
            mof_name = str(mof_name)
            sid_raw = data.get("sid", getattr(row, "id", None))
            if sid_raw is None:
                continue
            sid = int(sid_raw)
            if dedupe_sid:
                if sid in seen_sid:
                    continue
                seen_sid.add(sid)
            if mof_name not in index:
                index[mof_name] = sid
    finally:
        reader.close()
    return index


def IndexAseLmdbDirs(dirs: list[str] | list[Path]) -> dict[str, tuple[str, int]]:
    """Index every ``*.aselmdb`` under *dirs* → ``mof_name → (file, sid)``."""
    out: dict[str, tuple[str, int]] = {}
    for d in dirs:
        root = Path(d)
        if not root.exists():
            continue
        files: list[Path] = []
        if root.is_file():
            files = [root]
        else:
            files = sorted(root.rglob("*.aselmdb"))
        for f in files:
            try:
                partial = IndexAseLmdb(f)
            except Exception:
                continue
            for name, sid in partial.items():
                if name not in out:
                    out[name] = (str(f.resolve()), sid)
    return out


def IterOdacStructures(
    root: str | Path,
    *,
    stems: set[str] | None = None,
) -> Iterator[tuple[str, Any]]:
    """Yield ``(mof_name, ase.Atoms)`` for structures under *root*."""
    root_p = Path(root)
    index = IndexAseLmdbDirs([root_p])
    by_file: dict[str, list[tuple[str, int]]] = {}
    for name, (path, sid) in index.items():
        if stems is not None and name not in stems:
            continue
        by_file.setdefault(path, []).append((name, sid))
    for path, items in by_file.items():
        reader = OpenAseLmdb(path)
        try:
            # Prefer scanning once and filtering by sid set
            wanted = {sid: name for name, sid in items}
            for row in reader.iter_rows():
                data = dict(getattr(row, "data", {}) or {})
                sid_raw = data.get("sid")
                if sid_raw is None:
                    continue
                sid = int(sid_raw)
                if sid not in wanted:
                    continue
                name = wanted[sid]
                atoms = row.toatoms()
                info = dict(getattr(atoms, "info", {}) or {})
                info.setdefault("odac_name", name)
                info.setdefault("mof_name", name)
                info.setdefault("structure_stem", name)
                atoms.info = info
                yield name, atoms
                del wanted[sid]
                if not wanted:
                    break
        finally:
            reader.close()


@dataclass
class OdacMatchReport:
    """Match-rate report for the 209,799 STAB names against ODAC."""

    n_stab_names: int = 0
    n_matched: int = 0
    n_unmatched: int = 0
    by_partition: dict[str, int] = field(default_factory=dict)
    unmatched_sample: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_stab_names": self.n_stab_names,
            "n_matched": self.n_matched,
            "n_unmatched": self.n_unmatched,
            "match_rate": self.n_matched / self.n_stab_names if self.n_stab_names else 0.0,
            "by_partition": dict(self.by_partition),
            "unmatched_sample": list(self.unmatched_sample),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def MatchStabNamesToOdac(
    stab_names: Iterable[str],
    lmdb_dirs: list[str | Path],
    *,
    unmatched_sample_size: int = 50,
) -> OdacMatchReport:
    """Join STAB canonical / MOFid stems to ODAC ``mof_name`` values."""
    index = IndexAseLmdbDirs([str(p) for p in lmdb_dirs])
    report = OdacMatchReport()
    names = list(stab_names)
    report.n_stab_names = len(names)
    for name in names:
        hit = index.get(name)
        if hit is None:
            report.n_unmatched += 1
            if len(report.unmatched_sample) < unmatched_sample_size:
                report.unmatched_sample.append(name)
            continue
        report.n_matched += 1
        part = str(Path(hit[0]).parent)
        report.by_partition[part] = report.by_partition.get(part, 0) + 1
    return report


__all__ = [
    "AseLmdbReader",
    "IndexAseLmdb",
    "IndexAseLmdbDirs",
    "IterOdacStructures",
    "MatchStabNamesToOdac",
    "OdacMatchReport",
    "OpenAseLmdb",
]
