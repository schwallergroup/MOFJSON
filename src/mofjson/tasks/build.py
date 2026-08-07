"""Assemble the dataset: generate, dedupe, balance, split, freeze, audit.

Phase 12.8.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mofjson.schema.document import MOFJSONDocument
from mofjson.tasks.contamination import BuildControlArm, ScanForLeaks
from mofjson.tasks.generation import GenerateConstraintSets, GenerateGenerationItems
from mofjson.tasks.item import (
    ItemManifest,
    TaskItem,
    WriteItems,
    WriteManifest,
)
from mofjson.tasks.navigation import GenerateNavigationItems
from mofjson.tasks.sampling import BalanceByAnswer, StratifiedSample
from mofjson.tasks.translation import ALL_PAIRS_MATRIX, GenerateTranslationItems, TRANSLATION_PAIRS
from mofjson.util.io import AtomicWrite


@dataclass
class SuiteReport:
    """Audit: counts, histograms, contamination stats."""

    n_items: int = 0
    family_counts: dict[str, int] = field(default_factory=dict)
    task_counts: dict[str, int] = field(default_factory=dict)
    answer_histograms: dict[str, dict[str, int]] = field(default_factory=dict)
    hypothetical_count: int = 0
    experimental_count: int = 0
    atom_count_hist: dict[str, int] = field(default_factory=dict)
    topology_hist: dict[str, int] = field(default_factory=dict)
    mean_prompt_chars_by_rep: dict[str, float] = field(default_factory=dict)
    n_leaks_rejected: int = 0
    n_deduped: int = 0
    n_balanced_discarded: int = 0
    control_arm_count: int = 0
    hand_verification: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str | None = None
    seed: int = 0
    families: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "family_counts": self.family_counts,
            "task_counts": self.task_counts,
            "answer_histograms": self.answer_histograms,
            "hypothetical_count": self.hypothetical_count,
            "experimental_count": self.experimental_count,
            "hypothetical_ratio": (
                self.hypothetical_count / self.n_items if self.n_items else 0.0
            ),
            "atom_count_hist": self.atom_count_hist,
            "topology_hist": self.topology_hist,
            "mean_prompt_chars_by_rep": self.mean_prompt_chars_by_rep,
            "n_leaks_rejected": self.n_leaks_rejected,
            "n_deduped": self.n_deduped,
            "n_balanced_discarded": self.n_balanced_discarded,
            "control_arm_count": self.control_arm_count,
            "hand_verification": self.hand_verification,
            "manifest_hash": self.manifest_hash,
            "seed": self.seed,
            "families": self.families,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def markdown_audit(self) -> str:
        lines = [
            "",
            "## 12.8 Suite audit",
            "",
            f"- Items: **{self.n_items}**",
            f"- Seed: `{self.seed}`",
            f"- Families: {', '.join(self.families)}",
            f"- Hypothetical / experimental: {self.hypothetical_count} / {self.experimental_count}",
            f"- Manifest hash: `{self.manifest_hash}`",
            f"- Leaks rejected during build: {self.n_leaks_rejected}",
            f"- Deduped: {self.n_deduped}; balance discarded: {self.n_balanced_discarded}",
            f"- Control-arm items: {self.control_arm_count}",
            "",
            "### Family counts",
            "",
        ]
        for k, v in sorted(self.family_counts.items()):
            lines.append(f"- `{k}`: {v}")
        lines.extend(["", "### Task counts", ""])
        for k, v in sorted(self.task_counts.items()):
            lines.append(f"- `{k}`: {v}")
        lines.append("")
        return "\n".join(lines)


def _load_corpus(corpus: str | Path | Sequence[MOFJSONDocument]) -> list[MOFJSONDocument]:
    if isinstance(corpus, (str, Path)):
        path = Path(corpus)
        docs: list[MOFJSONDocument] = []
        if path.is_dir():
            from mofjson.util.io import ReadNdjson

            files = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.ndjson"))
            if not files:
                files = sorted(path.glob("**/*.jsonl"))
            for f in files:
                for obj in ReadNdjson(f):
                    try:
                        docs.append(MOFJSONDocument.model_validate(obj))
                    except Exception:
                        continue
        elif path.is_file():
            from mofjson.util.io import ReadNdjson

            for obj in ReadNdjson(path):
                try:
                    docs.append(MOFJSONDocument.model_validate(obj))
                except Exception:
                    continue
        else:
            raise FileNotFoundError(f"corpus not found: {path}")
        return docs
    return list(corpus)


def _dedupe(items: Iterable[TaskItem]) -> tuple[list[TaskItem], int]:
    seen: set[tuple[Any, ...]] = set()
    out: list[TaskItem] = []
    dropped = 0
    for it in items:
        key = it.semantic_key()
        # Include representation so same GT asked in different formats remains.
        full: tuple[Any, ...] = (*key, it.representation, it.family)
        if full in seen:
            dropped += 1
            continue
        seen.add(full)
        out.append(it)
    return out, dropped


def Split(
    items: Sequence[TaskItem],
    *,
    dev_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[TaskItem], list[TaskItem]]:
    rng = random.Random(seed)
    items_list = list(items)
    rng.shuffle(items_list)
    n_dev = max(1, int(len(items_list) * dev_fraction)) if items_list else 0
    # Prefer keeping no_structure control items with their parents in same split:
    # split by base item_id prefix.
    main = [it for it in items_list if not it.no_structure]
    rng.shuffle(main)
    n_dev = max(1, int(len(main) * dev_fraction)) if main else 0
    dev_ids = {it.item_id for it in main[:n_dev]}
    test_ids = {it.item_id for it in main[n_dev:]}
    # Attach control-arm variants to the same split as their parent.
    dev: list[TaskItem] = []
    test: list[TaskItem] = []
    for it in items_list:
        base = it.item_id[:-3] if it.item_id.endswith("-ns") else it.item_id
        if base in dev_ids or it.item_id in dev_ids:
            dev.append(it)
        elif base in test_ids or it.item_id in test_ids:
            test.append(it)
        else:
            test.append(it)
    dev.sort(key=lambda x: x.item_id)
    test.sort(key=lambda x: x.item_id)
    return dev, test


def FreezeSplit(items: Sequence[TaskItem], path: str | Path) -> ItemManifest:
    path = Path(path)
    WriteItems(items, path)
    manifest = ItemManifest.from_items(items, path)
    WriteManifest(manifest, path.with_suffix(path.suffix + ".manifest.json"))
    return manifest


def _build_report(items: Sequence[TaskItem], *, seed: int, families: list[str]) -> SuiteReport:
    report = SuiteReport(seed=seed, families=families, n_items=len(items))
    fam: Counter[str] = Counter()
    task: Counter[str] = Counter()
    hyp = exp = 0
    atom_hist: Counter[str] = Counter()
    topo_hist: Counter[str] = Counter()
    chars: dict[str, list[int]] = {}
    ans_hist: dict[str, Counter[str]] = {}
    for it in items:
        fam[it.family] += 1
        task[it.task] += 1
        if it.source.is_hypothetical:
            hyp += 1
        else:
            exp += 1
        if it.strata.n_atoms is not None:
            n = it.strata.n_atoms
            bucket = (
                "lt50"
                if n < 50
                else "50-149"
                if n < 150
                else "150-299"
                if n < 300
                else "300-599"
                if n < 600
                else "ge600"
            )
            atom_hist[bucket] += 1
        if it.strata.topology:
            topo_hist[str(it.strata.topology)] += 1
        chars.setdefault(it.representation, []).append(len(it.prompt))
        # Keep histogram keys short — structure gold answers are huge JSON blobs.
        if it.answer_type in ("structure", "mofjson", "constraint_set"):
            ans_key = f"<{it.answer_type}>"
        else:
            ans_key = json.dumps(it.answer, sort_keys=True, default=str)
            if len(ans_key) > 80:
                ans_key = ans_key[:77] + "..."
        ans_hist.setdefault(it.task, Counter())[ans_key] += 1
    report.family_counts = dict(sorted(fam.items()))
    report.task_counts = dict(sorted(task.items()))
    report.hypothetical_count = hyp
    report.experimental_count = exp
    report.atom_count_hist = dict(sorted(atom_hist.items()))
    report.topology_hist = dict(sorted(topo_hist.items()))
    report.mean_prompt_chars_by_rep = {
        k: sum(v) / len(v) for k, v in sorted(chars.items()) if v
    }
    report.answer_histograms = {
        t: dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:20])
        for t, c in sorted(ans_hist.items())
    }
    report.control_arm_count = sum(1 for it in items if it.no_structure)
    return report


def BuildTaskSuite(
    corpus: str | Path | Sequence[MOFJSONDocument],
    *,
    families: Sequence[str] | None = None,
    out_dir: str | Path = "tasks",
    seed: int = 0,
    freeze: bool = False,
    include_control_arm: bool = True,
    n_per_stratum: int = 50,
    cap_per_answer: int = 200,
    max_documents: int | None = None,
    translation_pairs: Sequence[tuple[str, str]] | None = None,
    append_audit_to_phase_notes: bool = False,
) -> SuiteReport:
    """Sample corpus, generate items, dedupe, balance, scan leaks, split, optionally freeze."""
    docs = _load_corpus(corpus)
    if max_documents is not None:
        docs = docs[:max_documents]
    fams = list(families) if families else ["navigation", "translation", "generation"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled = StratifiedSample(docs, n_per_stratum=n_per_stratum, seed=seed)
    items: list[TaskItem] = []

    if "navigation" in fams:
        items.extend(GenerateNavigationItems(sampled, cap_per_answer=cap_per_answer))
    if "translation" in fams:
        pairs = list(translation_pairs) if translation_pairs is not None else TRANSLATION_PAIRS
        # Prefer all-pairs only on a small exploratory subset (Phase 12.2).
        if translation_pairs is None and len(sampled) <= 250:
            pairs = ALL_PAIRS_MATRIX
        elif translation_pairs is None:
            # Scale with the curated pair list; still run a small all-pairs pilot.
            pilot = sampled[: min(50, len(sampled))]
            items.extend(GenerateTranslationItems(pilot, ALL_PAIRS_MATRIX))
        items.extend(GenerateTranslationItems(sampled, pairs))
    if "generation" in fams:
        csets = GenerateConstraintSets(sampled, n=max(400, len(sampled)))
        items.extend(GenerateGenerationItems(csets))
    if "editing" in fams:
        raise NotImplementedError(
            "Family D (editing) is deferred — see tasks/editing.py"
        )

    items, n_dup = _dedupe(items)
    items = BalanceByAnswer(items, cap_per_answer=cap_per_answer, seed=seed)
    n_bal = getattr(BalanceByAnswer, "last_report", None)
    n_bal_disc = n_bal.discarded if n_bal else 0

    # Final leak scan — drop any that slipped through.
    clean: list[TaskItem] = []
    n_leaks = 0
    for it in items:
        if ScanForLeaks(it.prompt):
            n_leaks += 1
            continue
        clean.append(it)
    items = clean

    if include_control_arm:
        # Control arm on a sample of navigation items (cheap memorisation check).
        nav = [it for it in items if it.family == "navigation" and not it.no_structure]
        rng = random.Random(seed)
        sample_nav = nav if len(nav) <= 500 else rng.sample(nav, 500)
        items.extend(BuildControlArm(sample_nav))

    items.sort(key=lambda it: it.item_id)
    # Re-number is not required; freeze uses content hash.

    dev, test = Split(items, dev_fraction=0.1, seed=seed)
    WriteItems(dev, out_dir / "dev.jsonl")
    test_path = out_dir / "test.jsonl"
    if freeze:
        manifest = FreezeSplit(test, test_path)
        manifest_hash = manifest.content_hash
    else:
        WriteItems(test, test_path)
        manifest = ItemManifest.from_items(test, test_path)
        WriteManifest(manifest, out_dir / "test.jsonl.manifest.json")
        manifest_hash = manifest.content_hash

    report = _build_report(items, seed=seed, families=fams)
    report.n_deduped = n_dup
    report.n_balanced_discarded = n_bal_disc
    report.n_leaks_rejected = n_leaks
    report.manifest_hash = manifest_hash
    report.save(out_dir / "suite_report.json")

    if append_audit_to_phase_notes:
        phase = (
            Path(__file__).resolve().parents[3]
            / "project_notes"
            / "phases"
            / "12_structure_task_suite.md"
        )
        if phase.is_file():
            text = phase.read_text(encoding="utf-8")
            marker = "## 12.8 Suite audit"
            block = report.markdown_audit()
            if marker in text:
                # Replace previous audit section through EOF or next ## at end
                head, _, _rest = text.partition(marker)
                # Keep content before marker; drop old audit.
                text = head.rstrip() + "\n" + block
            else:
                text = text.rstrip() + "\n" + block
            with AtomicWrite(phase) as fh:
                fh.write(text if text.endswith("\n") else text + "\n")

    return report


__all__ = [
    "BuildTaskSuite",
    "FreezeSplit",
    "Split",
    "SuiteReport",
]
