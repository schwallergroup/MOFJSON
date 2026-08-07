"""Contamination control: identifier leak scan, famous-MOF list, no-structure control.

Phase 12.6.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from mofjson.schema.document import MOFJSONDocument
from mofjson.tasks.item import TaskItem

SourceClass = Literal["hypothetical", "experimental_famous", "experimental_obscure"]


@dataclass(frozen=True, slots=True)
class Leak:
    kind: str
    token: str
    span: tuple[int, int]


# ~50 names a model is most likely to have memorised, with common aliases.
FAMOUS_MOFS: frozenset[str] = frozenset(
    {
        "UiO-66",
        "UiO-67",
        "UiO-68",
        "MOF-5",
        "IRMOF-1",
        "IRMOF-3",
        "IRMOF-8",
        "IRMOF-10",
        "MOF-74",
        "MOF-177",
        "MOF-808",
        "HKUST-1",
        "Cu-BTC",
        "ZIF-8",
        "ZIF-67",
        "ZIF-90",
        "MIL-53",
        "MIL-100",
        "MIL-101",
        "MIL-88",
        "MIL-125",
        "NU-1000",
        "NU-901",
        "PCN-222",
        "PCN-224",
        "PCN-333",
        "DUT-67",
        "CAU-10",
        "MUV-10",
        "SIFSIX-3",
        "CPL-1",
        "NOTT-100",
        "UMCM-1",
        "Bio-MOF-1",
        "MOF-210",
        "MOF-11",
        "MOF-14",
        "MOF-69",
        "MOF-2",
        "MOF-3",
        "MFU-4l",
        "MAF-6",
        "CPO-27",
        "DOBDC",
        "Mg-MOF-74",
        "Ni-MOF-74",
        "Zn-MOF-74",
        "Co-MOF-74",
        "Fe-MOF-74",
        "UiO66",
        "HKUST1",
        "ZIF8",
        "MIL53",
        "MIL101",
    }
)

_FAMOUS_REFCODES: frozenset[str] = frozenset(
    {
        "SAHYIK",  # MOF-5
        "FIQCEN",  # HKUST-1
        "VEJYUF",  # ZIF-8-ish
        "RUBTAK",  # UiO-66
        "GITTOT",  # MIL-53
    }
)

_REFCODE_RE = re.compile(r"\b([A-Z]{6}\d{0,2})\b")
_TOBACCO_STEM_RE = re.compile(r"\bSR_[A-Za-z0-9._-]+\b")
_DATA_BLOCK_RE = re.compile(r"\bdata_([A-Za-z0-9._-]+)\b")
_MOFID_TAIL_RE = re.compile(r"MOFid-v1\.[^;\s]*;([A-Za-z0-9._-]+)")


def _famous_name_patterns() -> list[re.Pattern[str]]:
    pats: list[re.Pattern[str]] = []
    for name in sorted(FAMOUS_MOFS, key=len, reverse=True):
        # Allow optional hyphen variants already in the set; escape literally.
        pats.append(re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE))
    # Generic family patterns
    pats.extend(
        [
            re.compile(r"\bUiO-?\d+\b", re.IGNORECASE),
            re.compile(r"\bMOF-?\d+\b", re.IGNORECASE),
            re.compile(r"\bZIF-?\d+\b", re.IGNORECASE),
            re.compile(r"\bHKUST-?\d+\b", re.IGNORECASE),
            re.compile(r"\bMIL-?\d+\b", re.IGNORECASE),
            re.compile(r"\bIRMOF-?\d+\b", re.IGNORECASE),
            re.compile(r"\bPCN-?\d+\b", re.IGNORECASE),
            re.compile(r"\bNU-?\d+\b", re.IGNORECASE),
        ]
    )
    return pats


_FAMOUS_PATTERNS = _famous_name_patterns()


# Combined famous-name pattern for a single scan pass (hot path).
_FAMOUS_COMBINED = re.compile(
    r"\b(?:UiO-?\d+|MOF-?\d+|ZIF-?\d+|HKUST-?\d+|MIL-?\d+|IRMOF-?\d+|PCN-?\d+|NU-?\d+"
    r"|Cu-BTC|CPO-27|DOBDC|MFU-4l|MAF-6|DUT-67|CAU-10|MUV-10|SIFSIX-3|CPL-1|NOTT-100"
    r"|UMCM-1|Bio-MOF-1|Mg-MOF-74|Ni-MOF-74|Zn-MOF-74|Co-MOF-74|Fe-MOF-74)\b",
    re.IGNORECASE,
)


def ScanForLeaks(prompt: str) -> list[Leak]:
    """Find identifiers that identify the structure rather than describe it."""
    hits: list[Leak] = []
    seen: set[tuple[str, str, int, int]] = set()

    def _add(kind: str, token: str, start: int, end: int) -> None:
        key = (kind, token, start, end)
        if key in seen:
            return
        seen.add(key)
        hits.append(Leak(kind=kind, token=token, span=(start, end)))

    for m in _FAMOUS_COMBINED.finditer(prompt):
        _add("famous_name", m.group(0), m.start(), m.end())

    for m in _REFCODE_RE.finditer(prompt):
        tok = m.group(1)
        # Skip element-like false positives of length 1-2; refcodes are 6 letters.
        if len(tok) >= 6:
            _add("refcode", tok, m.start(), m.end())

    for m in _TOBACCO_STEM_RE.finditer(prompt):
        _add("tobacco_stem", m.group(0), m.start(), m.end())

    for m in _DATA_BLOCK_RE.finditer(prompt):
        name = m.group(1)
        if name.lower() not in {"structure", "mofjson", "global", "image"}:
            _add("cif_data_block", m.group(0), m.start(), m.end())

    for m in _MOFID_TAIL_RE.finditer(prompt):
        _add("mofid_tail", m.group(1), m.start(1), m.end(1))

    return hits


def ClassifySource(document: MOFJSONDocument) -> SourceClass:
    ids = document.identifiers
    if ids is not None and ids.is_hypothetical:
        return "hypothetical"
    names = set()
    if ids is not None:
        names.update(ids.common_names or [])
        if ids.csd_refcode:
            names.add(ids.csd_refcode.upper())
        if ids.structure_stem:
            names.add(ids.structure_stem)
        if ids.odac_name and str(ids.odac_name).startswith("SR_"):
            return "hypothetical"
    for n in names:
        if n in FAMOUS_MOFS or n.upper() in _FAMOUS_REFCODES:
            return "experimental_famous"
        for fam in FAMOUS_MOFS:
            if fam.lower() in str(n).lower():
                return "experimental_famous"
    return "experimental_obscure"


_STRUCTURE_BLOCK_RE = re.compile(
    r"(Structure:\n)(.*?)(\n\nQuestion:)",
    re.DOTALL | re.IGNORECASE,
)


def BuildControlArm(items: list[TaskItem]) -> list[TaskItem]:
    """Same questions with the structure block removed (memorisation control)."""
    out: list[TaskItem] = []
    for it in items:
        prompt = it.prompt
        m = _STRUCTURE_BLOCK_RE.search(prompt)
        if m:
            new_prompt = prompt[: m.start(2)] + "[STRUCTURE WITHHELD]\n" + prompt[m.end(2) :]
        else:
            new_prompt = (
                "Structure:\n[STRUCTURE WITHHELD]\n\n"
                + prompt
            )
        out.append(
            it.model_copy(
                update={
                    "item_id": f"{it.item_id}-ns",
                    "prompt": new_prompt,
                    "no_structure": True,
                }
            )
        )
    return out


__all__ = [
    "FAMOUS_MOFS",
    "BuildControlArm",
    "ClassifySource",
    "Leak",
    "ScanForLeaks",
    "SourceClass",
]
