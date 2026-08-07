"""Streaming NDJSON, sharding, atomic writes, and resumable output.

Phase 07.6 / 09.8 / 12.8 -- used by every corpus-scale stage.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def CountLines(path: str | Path) -> int:
    """Count lines in a (possibly gzipped) text file."""
    p = Path(path)
    opener: Callable[..., Any]
    if p.suffix == ".gz" or p.name.endswith(".jsonl.gz") or p.name.endswith(".ndjson.gz"):
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"
    n = 0
    with opener(p, mode, encoding="utf-8") as fh:  # type: ignore[arg-type]
        for _ in fh:
            n += 1
    return n


def ReadNdjson(path_or_paths: str | Path | Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from one or more NDJSON files (``.gz`` supported)."""
    if isinstance(path_or_paths, (str, Path)):
        paths = [Path(path_or_paths)]
    else:
        paths = [Path(p) for p in path_or_paths]
    for path in paths:
        opener: Callable[..., Any]
        if path.suffix == ".gz" or str(path).endswith(".gz"):
            opener = gzip.open
            mode = "rt"
        else:
            opener = open
            mode = "r"
        with opener(path, mode, encoding="utf-8") as fh:  # type: ignore[arg-type]
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected object in {path} line {line_no}")
                yield obj


@contextmanager
def AtomicWrite(path: str | Path, *, encoding: str = "utf-8"):
    """Write to a temp file in the same directory, rename on success."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            yield fh
        os.replace(tmp_path, dest)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def WriteNdjson(
    records: Iterable[dict[str, Any]],
    path: str | Path,
    *,
    shard_size: int | None = None,
) -> list[Path]:
    """Write records as NDJSON. Optionally shard into ``stem-0001.ndjson`` files."""
    dest = Path(path)
    records_list = list(records)
    if shard_size is None or shard_size <= 0:
        with AtomicWrite(dest) as fh:
            for rec in records_list:
                fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        return [dest]

    written: list[Path] = []
    stem = dest.stem
    suffix = dest.suffix or ".ndjson"
    parent = dest.parent
    shard_idx = 0
    for i in range(0, len(records_list), shard_size):
        shard_idx += 1
        shard_path = parent / f"{stem}-{shard_idx:04d}{suffix}"
        chunk = records_list[i : i + shard_size]
        with AtomicWrite(shard_path) as fh:
            for rec in chunk:
                fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        written.append(shard_path)
    return written


class ResumableWriter:
    """Append NDJSON records, skipping keys already present in the output file."""

    def __init__(
        self,
        path: str | Path,
        key_fn: Callable[[dict[str, Any]], str],
        *,
        resume: bool = True,
    ) -> None:
        self.path = Path(path)
        self.key_fn = key_fn
        self.resume = resume
        self._seen: set[str] = set()
        self._fh: Any = None
        if self.resume and self.path.is_file():
            for obj in ReadNdjson(self.path):
                try:
                    self._seen.add(self.key_fn(obj))
                except Exception:
                    continue

    def __enter__(self) -> ResumableWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def seen(self) -> set[str]:
        return set(self._seen)

    def write(self, record: dict[str, Any]) -> bool:
        """Write *record* unless its key was already seen. Returns True if written."""
        key = self.key_fn(record)
        if key in self._seen:
            return False
        if self._fh is None:
            raise RuntimeError("ResumableWriter must be used as a context manager")
        self._fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()
        self._seen.add(key)
        return True


__all__ = [
    "AtomicWrite",
    "CountLines",
    "ReadNdjson",
    "ResumableWriter",
    "WriteNdjson",
]
