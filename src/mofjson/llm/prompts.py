"""Prompt templates and the format-explainer text.

Phase 10.4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_LIST_RE = re.compile(r"^\[(.*)\]$")

# Famous names / refcode-like tokens that must not appear in benchmark templates.
_LEAK_PATTERNS = (
    re.compile(r"\bUiO-?\d+\b", re.IGNORECASE),
    re.compile(r"\bMOF-?\d+\b", re.IGNORECASE),
    re.compile(r"\bZIF-?\d+\b", re.IGNORECASE),
    re.compile(r"\bHKUST-?\d+\b", re.IGNORECASE),
    re.compile(r"\bMIL-?\d+\b", re.IGNORECASE),
    re.compile(r"\bIRMOF-?\d+\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{6}\d{0,2}\b"),  # CSD-like refcodes
)


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    id: str
    version: int
    text: str
    variables: tuple[str, ...]
    purpose: str
    path: str


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if key == "version":
            meta[key] = int(val)
        elif key == "variables":
            lm = _LIST_RE.match(val)
            if lm:
                items = [x.strip() for x in lm.group(1).split(",") if x.strip()]
                meta[key] = items
            else:
                meta[key] = []
        else:
            meta[key] = val
    return meta, m.group(2)


def _load_templates() -> dict[str, PromptTemplate]:
    out: dict[str, PromptTemplate] = {}
    root = Path(__file__).resolve().parent / "templates"
    files: list[Path] = sorted(root.glob("*.md"))
    if not files:
        # Wheel / installed package fallback via importlib.resources.
        try:
            pkg = resources.files("mofjson.llm") / "templates"
            for entry in pkg.iterdir():
                name = entry.name
                if not name.endswith(".md"):
                    continue
                raw = entry.read_text(encoding="utf-8")
                meta, body = _parse_frontmatter(raw)
                tid = str(meta.get("id") or name.split(".")[0])
                out[tid] = PromptTemplate(
                    id=tid,
                    version=int(meta.get("version", 1)),
                    text=body.strip() + "\n",
                    variables=tuple(meta.get("variables") or ()),
                    purpose=str(meta.get("purpose") or ""),
                    path=name,
                )
        except (FileNotFoundError, TypeError, AttributeError):
            pass
        return out

    for path in files:
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        tid = str(meta.get("id") or path.stem.split(".")[0])
        out[tid] = PromptTemplate(
            id=tid,
            version=int(meta.get("version", 1)),
            text=body.strip() + "\n",
            variables=tuple(meta.get("variables") or ()),
            purpose=str(meta.get("purpose") or ""),
            path=str(path),
        )
    return out


PROMPTS: dict[str, PromptTemplate] = _load_templates()


def RenderPrompt(template_id: str, **vars: Any) -> str:
    """Render a versioned prompt template with the given variables."""
    if template_id not in PROMPTS:
        raise ValueError(f"Unknown template {template_id!r}; known: {sorted(PROMPTS)}")
    tmpl = PROMPTS[template_id]
    missing = [v for v in tmpl.variables if v not in vars]
    if missing:
        raise ValueError(f"Template {template_id!r} missing variables: {missing}")
    try:
        return tmpl.text.format(**{k: vars[k] for k in tmpl.variables})
    except KeyError as exc:
        raise ValueError(f"Template {template_id!r} render failed: {exc}") from exc


_PRIMERS: dict[str, dict[str, str]] = {
    "metal_node": {
        "brief": (
            "A MetalNode describes an inorganic SBU: id, kind, archetype, nuclearity, "
            "metals, connectivity, and optional per-site coordination geometry."
        ),
        "full": (
            "A MetalNode is the inorganic building unit. Required: id, kind, archetype, "
            "nuclearity (= sum of metals[].count), metals, connectivity (nullable). "
            "MetalSite records coordination_number, geometry name, ChemEnv symbol, CSM, "
            "and donors. Bond order 0.5 means a dative/coordination bond."
        ),
    },
    "reticular": {
        "brief": (
            "Reticular MOFJSON: topology (RCSR symbol), metal nodes, linkers "
            "(MolFragment), and optional assembly."
        ),
        "full": (
            "The reticular layer describes a framework as topology + nodes + linkers. "
            "Topology uses lowercase RCSR symbols (null for unknown, never ERROR). "
            "Linkers embed a MolFragment with atoms, bonds, and attachment_points."
        ),
    },
    "crystal": {
        "brief": "Crystal MOFJSON: lattice, symmetry, and per-site fractional coordinates.",
        "full": (
            "The crystal layer is lossless: lattice parameters, space group, sites with "
            "fractional coords, and a periodic bond graph with optional image translations."
        ),
    },
    "crystal_columnar": {
        "brief": (
            "Crystal MOFJSON in columnar layout: parallel arrays for elements, frac, ids."
        ),
        "full": (
            "Columnar crystal layout stores elements[], frac[], ids[] as parallel arrays "
            "instead of site objects, reducing tokens for large cells."
        ),
    },
    "properties": {
        "brief": "Identifiers plus a list of canonical properties with provenance.",
        "full": (
            "The properties profile carries identifiers (refcode, qmof_id, …) and "
            "properties[{name, value, unit, method, provenance}]."
        ),
    },
    "minimal": {
        "brief": "Identifiers plus topology only — for cheap classification.",
        "full": "Minimal profile: identifiers and reticular.topology only.",
    },
    "reticular_nogeom": {
        "brief": "Reticular without metal-site geometry fields (ablation control).",
        "full": (
            "Same as reticular but MetalSite omits geometry/donors/bond summaries — "
            "used only for the Phase 13.2 ablation."
        ),
    },
}


def FormatPrimer(profile: str, *, level: str = "brief") -> str:
    """Short explanation of the MOFJSON fields in play for a profile."""
    if profile not in _PRIMERS:
        raise ValueError(f"No primer for profile {profile!r}")
    if level not in _PRIMERS[profile]:
        raise ValueError(f"Unknown primer level {level!r}")
    return _PRIMERS[profile][level]


def ScanTemplateLeaks(text: str) -> list[str]:
    """Return identifier-like tokens that must not appear in benchmark templates."""
    hits: list[str] = []
    for pat in _LEAK_PATTERNS:
        for m in pat.finditer(text):
            hits.append(m.group(0))
    return hits


__all__ = [
    "PROMPTS",
    "FormatPrimer",
    "PromptTemplate",
    "RenderPrompt",
    "ScanTemplateLeaks",
]
