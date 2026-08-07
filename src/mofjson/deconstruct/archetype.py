"""Stage 3: classify each metal node against the named SBU archetypes.

Phase 08.3.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from mofjson.schema.metal_node import MetalNode
from mofjson.vocab.archetypes import ARCHETYPE_FINGERPRINTS


@dataclass
class ArchetypeMatch:
    """Result of :func:`ClassifyNode`."""

    archetype: str
    confidence: float
    matched_rules: list[str] = field(default_factory=list)
    near_misses: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: dict[str, Any] = field(default_factory=dict)


def _node_fingerprint(node: MetalNode) -> dict[str, Any]:
    metal_counts = {m.element: m.count for m in node.metals}
    cn_list: list[int] = []
    geometries: list[str] = []
    if node.sites:
        cn_list = [s.coordination_number for s in node.sites]
        geometries = [s.geometry for s in node.sites]
    bridging: Counter[str] = Counter()
    if node.bridging_ligands:
        for lig in node.bridging_ligands:
            key = f"{lig.bridging_mode or 'unknown'}:{lig.formula}"
            bridging[key] += lig.count
    return {
        "nuclearity": node.nuclearity,
        "metals": metal_counts,
        "n_metal_elements": len(metal_counts),
        "kind": node.kind,
        "cn_multiset": sorted(cn_list),
        "geometries": geometries,
        "bridging": dict(bridging),
        "connectivity": node.connectivity,
        "has_mu3_O": any(
            lig.formula == "O" and lig.bridging_mode == "mu3"
            for lig in (node.bridging_ligands or [])
        ),
        "has_mu3_OH": any(
            lig.formula in ("OH", "HO") and lig.bridging_mode == "mu3"
            for lig in (node.bridging_ligands or [])
        ),
        "has_mu4_O": any(
            lig.formula == "O" and lig.bridging_mode == "mu4"
            for lig in (node.bridging_ligands or [])
        ),
        "has_mu2_OH": any(
            lig.formula in ("OH", "HO") and lig.bridging_mode == "mu2"
            for lig in (node.bridging_ligands or [])
        ),
        "donor_groups": sorted(
            {d.donor_group for s in (node.sites or []) for d in (s.donors or [])}
        ),
    }


def _score_fingerprint(fp: dict[str, Any], rule: dict[str, Any]) -> tuple[float, list[str]]:
    """Return (score in 0..1, list of matched clause names)."""
    matched: list[str] = []
    required = 0
    hits = 0

    def req(name: str, ok: bool) -> None:
        nonlocal required, hits
        required += 1
        if ok:
            hits += 1
            matched.append(name)

    # Rod / layer: periodicity (kind) is decisive.
    if "periodicity_kinds" in rule:
        ok = fp.get("kind") in rule["periodicity_kinds"]
        req("periodicity_kinds", ok)
        if ok:
            return 1.0, matched
        return 0.0, matched

    if "nuclearity" in rule:
        req("nuclearity", fp["nuclearity"] == rule["nuclearity"])
    if "exact_metal_elements" in rule:
        req("exact_metal_elements", fp["n_metal_elements"] == rule["exact_metal_elements"])
    if "cn" in rule:
        req(
            "cn",
            bool(fp["cn_multiset"]) and all(c == rule["cn"] for c in fp["cn_multiset"]),
        )
    if "cn_any" in rule:
        allowed = set(rule["cn_any"])
        req(
            "cn_any",
            bool(fp["cn_multiset"]) and all(c in allowed for c in fp["cn_multiset"]),
        )
    if "geometry_any" in rule:
        allowed = set(rule["geometry_any"])
        geoms = [g for g in fp["geometries"] if g not in ("unknown",)]
        req(
            "geometry_any",
            (not geoms) or all(g in allowed or g == "other" for g in geoms),
        )
    if rule.get("require_mu3_O"):
        req("require_mu3_O", fp["has_mu3_O"])
    if rule.get("require_mu3_OH"):
        req("require_mu3_OH", fp["has_mu3_OH"])
    if rule.get("require_mu4_O"):
        req("require_mu4_O", fp["has_mu4_O"])
    if rule.get("require_mu2_OH"):
        req("require_mu2_OH", fp["has_mu2_OH"])
    if "donor_group_any" in rule:
        needed = set(rule["donor_group_any"])
        req("donor_group_any", bool(needed & set(fp["donor_groups"])))
    if "connectivity" in rule:
        req("connectivity", fp["connectivity"] == rule["connectivity"])
    if "connectivity_any" in rule:
        req("connectivity_any", fp["connectivity"] in rule["connectivity_any"])

    if required == 0:
        return 0.0, matched
    return hits / required, matched


def ClassifyNode(node: MetalNode) -> ArchetypeMatch:
    """Walk :data:`ARCHETYPE_FINGERPRINTS` and return the best match (or ``other``)."""
    fp = _node_fingerprint(node)
    if fp["n_metal_elements"] > 1:
        return ArchetypeMatch(
            archetype="mixed_metal_cluster",
            confidence=0.9,
            matched_rules=["n_metal_elements>1"],
            fingerprint=fp,
        )

    best_name = "other"
    best_score = 0.0
    best_matched: list[str] = []
    near: list[dict[str, Any]] = []

    for rule in ARCHETYPE_FINGERPRINTS:
        name = str(rule["archetype"])
        score, matched = _score_fingerprint(fp, rule)
        near.append({"archetype": name, "score": round(score, 3), "matched": matched})
        min_score = float(rule.get("min_score", 0.85))
        if score + 1e-9 >= min_score and score > best_score:
            best_score = score
            best_name = name
            best_matched = matched

    near.sort(key=lambda x: -x["score"])
    if best_name == "other":
        return ArchetypeMatch(
            archetype="other",
            confidence=0.0,
            matched_rules=[],
            near_misses=near[:5],
            fingerprint=fp,
        )
    return ArchetypeMatch(
        archetype=best_name,
        confidence=best_score,
        matched_rules=best_matched,
        near_misses=[n for n in near if n["archetype"] != best_name][:5],
        fingerprint=fp,
    )


__all__ = ["ArchetypeMatch", "ClassifyNode"]
