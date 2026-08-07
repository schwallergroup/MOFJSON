"""Stratified sampling from the corpus, and answer-distribution balancing.

Phase 12.3 and 12.8.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mofjson.schema.document import MOFJSONDocument
from mofjson.tasks.item import TaskItem


@dataclass
class DistributionReport:
    requested: dict[str, Any] = field(default_factory=dict)
    achieved: dict[str, int] = field(default_factory=dict)
    underfilled: list[str] = field(default_factory=list)
    discarded: int = 0
    discarded_by_stratum: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "achieved": dict(sorted(self.achieved.items())),
            "underfilled": list(self.underfilled),
            "discarded": self.discarded,
            "discarded_by_stratum": dict(sorted(self.discarded_by_stratum.items())),
        }


def _atom_bucket(n: int) -> str:
    if n < 50:
        return "lt50"
    if n < 150:
        return "50-149"
    if n < 300:
        return "150-299"
    if n < 600:
        return "300-599"
    return "ge600"


def default_strata_key(doc: MOFJSONDocument) -> str:
    """Composite stratum key used when no custom strata_spec is provided."""
    n_atoms = 0
    n_metals = 0
    if doc.crystal is not None:
        n_atoms = doc.crystal.n_atoms or len(doc.crystal.sites)
        n_metals = sum(1 for s in doc.crystal.sites if s.is_metal or s.element)
        # recount metals properly
        from mofjson.vocab.elements import METALS

        n_metals = sum(1 for s in doc.crystal.sites if s.element in METALS)
    topo = None
    node_conn = None
    if doc.reticular is not None:
        ret = doc.reticular
        if hasattr(ret, "topology"):
            topo = ret.topology.rcsr_symbol if ret.topology else None
            if ret.nodes:
                node_conn = ret.nodes[0].connectivity
        elif isinstance(ret, dict):
            topo = (ret.get("topology") or {}).get("rcsr_symbol")
            nodes = ret.get("nodes") or []
            if nodes:
                node_conn = nodes[0].get("connectivity")
    hyp = bool(doc.identifiers.is_hypothetical) if doc.identifiers else False
    return "|".join(
        [
            _atom_bucket(n_atoms),
            f"m{n_metals}",
            f"t{topo or 'null'}",
            f"c{node_conn if node_conn is not None else 'na'}",
            "hyp" if hyp else "exp",
        ]
    )


def StratifiedSample(
    corpus: Sequence[MOFJSONDocument],
    strata_spec: Callable[[MOFJSONDocument], str] | None = None,
    *,
    n_per_stratum: int = 10,
    seed: int = 0,
) -> list[MOFJSONDocument]:
    """Seeded stratified sample; report under-filled strata via DistributionReport side channel."""
    key_fn = strata_spec or default_strata_key
    rng = random.Random(seed)
    buckets: dict[str, list[MOFJSONDocument]] = defaultdict(list)
    for doc in corpus:
        buckets[key_fn(doc)].append(doc)
    out: list[MOFJSONDocument] = []
    for key in sorted(buckets):
        group = list(buckets[key])
        rng.shuffle(group)
        out.extend(group[:n_per_stratum])
    # Stable order by mofjson_id / stem for reproducibility after shuffle within stratum.
    def _sort_key(d: MOFJSONDocument) -> str:
        ids = d.identifiers
        return (
            (ids.mofjson_id if ids else None)
            or (ids.structure_stem if ids else None)
            or (ids.odac_name if ids else None)
            or ""
        )

    out.sort(key=_sort_key)
    StratifiedSample.last_report = DistributionReport(  # type: ignore[attr-defined]
        requested={"n_per_stratum": n_per_stratum, "n_strata": len(buckets)},
        achieved={k: min(len(v), n_per_stratum) for k, v in sorted(buckets.items())},
        underfilled=[k for k, v in buckets.items() if len(v) < n_per_stratum],
    )
    return out


StratifiedSample.last_report = DistributionReport()  # type: ignore[attr-defined]


def BalanceByAnswer(
    items: Iterable[TaskItem],
    *,
    cap_per_answer: int = 200,
    seed: int = 0,
) -> list[TaskItem]:
    """Cap items per (task, answer, representation); never duplicate. Discards excess randomly."""
    rng = random.Random(seed)
    by_key: dict[tuple[str, str, str], list[TaskItem]] = defaultdict(list)
    for it in items:
        ans = json_answer_key(it.answer)
        by_key[(it.task, ans, it.representation)].append(it)

    kept: list[TaskItem] = []
    discarded = 0
    discarded_by: Counter[str] = Counter()
    for (task, ans, rep), group in sorted(by_key.items()):
        group = list(group)
        rng.shuffle(group)
        take = group[:cap_per_answer]
        drop = group[cap_per_answer:]
        kept.extend(take)
        discarded += len(drop)
        if drop:
            discarded_by[task] += len(drop)

    # Deterministic order
    kept.sort(key=lambda it: it.item_id)
    BalanceByAnswer.last_report = DistributionReport(  # type: ignore[attr-defined]
        requested={"cap_per_answer": cap_per_answer},
        discarded=discarded,
        discarded_by_stratum=dict(discarded_by),
    )
    rep_report = BalanceByAnswer.last_report  # type: ignore[attr-defined]
    rep_report.achieved = {
        f"{task}|{ans}|{representation}": n
        for (task, ans, representation), n in Counter(
            (it.task, json_answer_key(it.answer), it.representation) for it in kept
        ).items()
    }
    return kept


BalanceByAnswer.last_report = DistributionReport()  # type: ignore[attr-defined]


def json_answer_key(answer: Any) -> str:
    import json

    return json.dumps(answer, sort_keys=True, default=str)


def ReportDistribution(items: Iterable[TaskItem]) -> DistributionReport:
    items_list = list(items)
    hist = Counter((it.task, json_answer_key(it.answer)) for it in items_list)
    return DistributionReport(
        achieved={f"{t}|{a}": n for (t, a), n in sorted(hist.items())},
    )


__all__ = [
    "BalanceByAnswer",
    "DistributionReport",
    "ReportDistribution",
    "StratifiedSample",
    "default_strata_key",
    "json_answer_key",
]
