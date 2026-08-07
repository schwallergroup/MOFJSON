"""The TaskItem record: one self-contained, re-gradable benchmark question.

Phase 12.1.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from mofjson.schema.base import MOFJSONModel
from mofjson.util.io import AtomicWrite, ReadNdjson, WriteNdjson

Family = Literal["translation", "navigation", "generation", "editing"]
AnswerType = Literal[
    "integer",
    "boolean",
    "string",
    "structure",
    "mofjson",
    "constraint_set",
]


class ItemSource(MOFJSONModel):
    """Provenance of the structure that backs a task item."""

    mofjson_id: str | None = None
    odac_name: str | None = None
    csd_refcode: str | None = None
    structure_stem: str | None = None
    is_hypothetical: bool = False


class ItemStrata(MOFJSONModel):
    """Breakdown keys for reporting; never regenerate to recover these."""

    n_atoms: int | None = None
    path_length: int | None = None
    n_metals: int | None = None
    topology: str | None = None
    node_connectivity: int | None = None
    n_distinct_linkers: int | None = None
    answer_value: str | None = None


class ItemProvenance(MOFJSONModel):
    ground_truth_method: str | None = None
    verified_by: str | None = None
    generator: str = "tasks/build.py@v1"


class TaskItem(MOFJSONModel):
    """One self-contained, re-gradable benchmark question."""

    item_id: str = Field(min_length=1)
    family: Family
    task: str = Field(min_length=1)
    representation: str = Field(min_length=1)
    prompt: str
    response_schema: str
    answer: Any
    answer_type: AnswerType
    grader: str
    source: ItemSource = Field(default_factory=ItemSource)
    strata: ItemStrata = Field(default_factory=ItemStrata)
    provenance: ItemProvenance = Field(default_factory=ItemProvenance)
    # Optional control-arm marker: structure withheld from prompt.
    no_structure: bool = False

    @field_validator("family")
    @classmethod
    def _family(cls, v: str) -> str:
        allowed = ("translation", "navigation", "generation", "editing")
        if v not in allowed:
            raise ValueError(f"family must be one of {allowed}; got {v!r}")
        return v

    def semantic_key(self) -> tuple[str, str, str]:
        """Dedup key: (task, source identity, answer)."""
        src = (
            self.source.mofjson_id
            or self.source.structure_stem
            or self.source.odac_name
            or self.source.csd_refcode
            or ""
        )
        answer_s = json.dumps(self.answer, sort_keys=True, default=str)
        return (self.task, src, answer_s)

    def to_record(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)


class ItemManifest(MOFJSONModel):
    """Counts, hashes and distributions for a frozen item file."""

    path: str
    n_items: int
    content_hash: str
    family_counts: dict[str, int] = Field(default_factory=dict)
    task_counts: dict[str, int] = Field(default_factory=dict)
    representation_counts: dict[str, int] = Field(default_factory=dict)
    hypothetical_count: int = 0
    experimental_count: int = 0

    @classmethod
    def from_items(cls, items: Iterable[TaskItem], path: str | Path) -> ItemManifest:
        items_list = list(items)
        family_c: Counter[str] = Counter()
        task_c: Counter[str] = Counter()
        rep_c: Counter[str] = Counter()
        hyp = 0
        exp = 0
        for it in items_list:
            family_c[it.family] += 1
            task_c[it.task] += 1
            rep_c[it.representation] += 1
            if it.source.is_hypothetical:
                hyp += 1
            else:
                exp += 1
        digest = _hash_items(items_list)
        return cls(
            path=str(path),
            n_items=len(items_list),
            content_hash=digest,
            family_counts=dict(sorted(family_c.items())),
            task_counts=dict(sorted(task_c.items())),
            representation_counts=dict(sorted(rep_c.items())),
            hypothetical_count=hyp,
            experimental_count=exp,
        )


def _hash_items(items: list[TaskItem]) -> str:
    h = hashlib.sha256()
    for it in items:
        line = json.dumps(it.to_record(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return f"sha256:{h.hexdigest()}"


def WriteItems(items: Iterable[TaskItem], path: str | Path) -> Path:
    """Write items as JSONL (sorted keys, one object per line)."""
    records = [it.to_record() for it in items]
    written = WriteNdjson(records, path)
    return written[0]


def ReadItems(path: str | Path) -> Iterator[TaskItem]:
    """Yield TaskItem records from a JSONL file."""
    for obj in ReadNdjson(path):
        yield TaskItem.model_validate(obj)


def LoadItems(path: str | Path) -> list[TaskItem]:
    return list(ReadItems(path))


def WriteManifest(manifest: ItemManifest, path: str | Path) -> Path:
    dest = Path(path)
    with AtomicWrite(dest) as fh:
        fh.write(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return dest


__all__ = [
    "AnswerType",
    "Family",
    "ItemManifest",
    "ItemProvenance",
    "ItemSource",
    "ItemStrata",
    "LoadItems",
    "ReadItems",
    "TaskItem",
    "WriteItems",
    "WriteManifest",
]
