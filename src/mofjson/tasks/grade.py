"""Score stored responses and produce the result tables.

Phase 12.9.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mofjson.tasks.graders import Grade, GradeResult
from mofjson.tasks.item import LoadItems, TaskItem
from mofjson.util.io import AtomicWrite, ReadNdjson


@dataclass
class ResponseRecord:
    item_id: str
    raw: str
    parsed: Any = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    model: str | None = None
    model_version: str | None = None
    run_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> ResponseRecord:
        return cls(
            item_id=str(obj["item_id"]),
            raw=str(obj.get("raw") or obj.get("response") or obj.get("text") or ""),
            parsed=obj.get("parsed"),
            output_tokens=obj.get("output_tokens"),
            reasoning_tokens=obj.get("reasoning_tokens"),
            model=obj.get("model"),
            model_version=obj.get("model_version"),
            run_config=dict(obj.get("run_config") or {}),
        )


def WilsonInterval(
    successes: int,
    n: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    # z for common confidence levels
    z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}.get(
        confidence, 1.959963984540054
    )
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    low = (centre - margin) / denom
    high = (centre + margin) / denom
    return (max(0.0, low), min(1.0, high))


@dataclass
class ResultTable:
    rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"rows": self.rows}

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        if not self.rows:
            return "_empty_\n"
        keys = list(self.rows[0].keys())
        lines = [
            "| " + " | ".join(keys) + " |",
            "| " + " | ".join("---" for _ in keys) + " |",
        ]
        for row in self.rows:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
        return "\n".join(lines) + "\n"


@dataclass
class RunReport:
    results: list[dict[str, Any]]
    n_items: int
    n_graded: int
    n_missing_responses: int = 0

    def table(self, by: Sequence[str] = ("task", "representation")) -> ResultTable:
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for r in self.results:
            key = tuple(r.get(k) for k in by)
            buckets[key].append(r)
        rows: list[dict[str, Any]] = []
        for key in sorted(buckets):
            group = buckets[key]
            n = len(group)
            successes = sum(1 for g in group if g.get("correct"))
            lo, hi = WilsonInterval(successes, n)
            row: dict[str, Any] = {k: key[i] for i, k in enumerate(by)}
            row.update(
                {
                    "n": n,
                    "correct": successes,
                    "accuracy": round(successes / n, 6) if n else 0.0,
                    "wilson_low": round(lo, 6),
                    "wilson_high": round(hi, 6),
                    "wrong": sum(1 for g in group if g.get("category") == "wrong"),
                    "unparseable": sum(1 for g in group if g.get("category") == "unparseable"),
                    "truncated": sum(1 for g in group if g.get("category") == "truncated"),
                    "mean_output_tokens": _mean([g.get("output_tokens") for g in group]),
                    "mean_reasoning_tokens": _mean([g.get("reasoning_tokens") for g in group]),
                    "hypothetical_accuracy": _subset_acc(group, True),
                    "experimental_accuracy": _subset_acc(group, False),
                    "no_structure_accuracy": _ns_acc(group),
                }
            )
            rows.append(row)
        return ResultTable(rows=rows)

    def save(self, path: str | Path) -> None:
        payload = {
            "n_items": self.n_items,
            "n_graded": self.n_graded,
            "n_missing_responses": self.n_missing_responses,
            "results": self.results,
            "table": self.table().to_dict(),
        }
        with AtomicWrite(path) as fh:
            fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _mean(vals: Iterable[Any]) -> float | None:
    nums = [float(v) for v in vals if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 3)


def _subset_acc(group: list[dict[str, Any]], hypothetical: bool) -> float | None:
    sub = [g for g in group if g.get("is_hypothetical") is hypothetical and not g.get("no_structure")]
    if not sub:
        return None
    return round(sum(1 for g in sub if g["correct"]) / len(sub), 6)


def _ns_acc(group: list[dict[str, Any]]) -> float | None:
    sub = [g for g in group if g.get("no_structure")]
    if not sub:
        return None
    return round(sum(1 for g in sub if g["correct"]) / len(sub), 6)


def GradeRun(items_path: str | Path, responses_path: str | Path) -> RunReport:
    """Grade stored responses against an item file; deterministic offline."""
    items = {it.item_id: it for it in LoadItems(items_path)}
    responses: dict[str, ResponseRecord] = {}
    for obj in ReadNdjson(responses_path):
        rec = ResponseRecord.from_dict(obj)
        responses[rec.item_id] = rec

    results: list[dict[str, Any]] = []
    missing = 0
    for item_id, item in sorted(items.items()):
        rec = responses.get(item_id)
        if rec is None:
            missing += 1
            results.append(
                {
                    "item_id": item_id,
                    "task": item.task,
                    "family": item.family,
                    "representation": item.representation,
                    "is_hypothetical": item.source.is_hypothetical,
                    "no_structure": item.no_structure,
                    "correct": False,
                    "category": "unparseable",
                    "detail": {"reason": "missing_response"},
                }
            )
            continue
        grade: GradeResult = Grade(item, rec.raw)
        results.append(
            {
                "item_id": item_id,
                "task": item.task,
                "family": item.family,
                "representation": item.representation,
                "is_hypothetical": item.source.is_hypothetical,
                "no_structure": item.no_structure,
                "correct": grade.correct,
                "category": grade.category,
                "detail": grade.detail,
                "output_tokens": rec.output_tokens,
                "reasoning_tokens": rec.reasoning_tokens,
                "model": rec.model,
            }
        )

    return RunReport(
        results=results,
        n_items=len(items),
        n_graded=len(items) - missing,
        n_missing_responses=missing,
    )


__all__ = [
    "GradeRun",
    "ResponseRecord",
    "ResultTable",
    "RunReport",
    "WilsonInterval",
]
