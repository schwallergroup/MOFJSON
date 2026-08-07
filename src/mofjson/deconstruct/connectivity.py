"""Stage 4: node connectivity, bridging modes, and open metal sites.

Phase 08.4.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from mofjson.deconstruct.partition import Component, PartitionResult
from mofjson.schema.metal_node import MetalNode, MetalSite, OpenMetalSiteEvidence
from mofjson.vocab.archetypes import ARCHETYPE_EXPECTED_CONNECTIVITY
from mofjson.vocab.donors import SOLVENT_DONORS
from mofjson.vocab.elements import EXPECTED_CN, METALS


@dataclass
class BridgingAssignment:
    """One bridging atom / ligand group assignment on a node."""

    atom_id: str
    element: str
    formula: str
    bridging_mode: str
    metal_site_ids: tuple[str, ...]
    count: int = 1


@dataclass
class OpenSiteJudgement:
    """Open-metal-site call for one metal centre."""

    site_id: str
    open_as_deposited: bool
    open_after_activation: bool
    method: str
    evidence: OpenMetalSiteEvidence
    removable_ligand_atom_ids: tuple[str, ...] = ()


def _attachment_key(
    donor_index: int,
    partition: PartitionResult,
) -> tuple[str, int]:
    """Identity of one linker attachment point on a node.

    Carboxylate donors collapse to their carboxylate carbon so a chelating / bridging
    RCO2 counts once. Other donors use the donor atom index.
    """
    elements = partition.elements
    adj = partition.adjacency
    # Only count donors that belong to linker-role components.
    donor_comp = next(
        (c for c in partition.organic_components if donor_index in c.site_indices),
        None,
    )
    if donor_comp is not None and donor_comp.role != "linker":
        return ("skip", -1)
    if elements[donor_index] == "O":
        for j, _ in adj.get(donor_index, []):
            if elements[j] == "C":
                o_on_c = [k for k, _ in adj.get(j, []) if elements[k] == "O"]
                if len(o_on_c) >= 2:
                    return ("carboxylate", j)
    return ("donor", donor_index)


def NodeConnectivity(partition: PartitionResult, node: Component | MetalNode) -> int:
    """Count distinct linker attachment points on the node (periodic images included).

    A chelating carboxylate counts once. Solvent / capping ligands are excluded.
    """
    if isinstance(node, MetalNode):
        if node.attachment_points is not None:
            return len(node.attachment_points)
        comp = next((c for c in partition.metal_components if c.id == node.id), None)
        if comp is None:
            return 0
    else:
        comp = node

    keys: set[tuple[str, int]] = set()
    for cb in comp.cut_bonds:
        key = _attachment_key(cb.donor_index, partition)
        if key[1] >= 0:
            keys.add(key)
    return len(keys)


def BridgingModes(
    partition: PartitionResult,
    node: Component | MetalNode,
) -> list[BridgingAssignment]:
    """Classify bridging ligands on the node from the partition graph."""
    if isinstance(node, MetalNode):
        comp = next((c for c in partition.metal_components if c.id == node.id), None)
        if comp is None:
            return []
    else:
        comp = node

    elements = partition.elements
    adj = partition.adjacency
    site_ids = partition.site_ids
    assignments: list[BridgingAssignment] = []

    for idx in sorted(comp.site_indices):
        if elements[idx] in METALS:
            continue
        metal_nbs = sorted(
            j for j, _ in adj.get(idx, []) if j in comp.metal_indices
        )
        if not metal_nbs:
            continue
        # Skip H
        if elements[idx] == "H":
            continue
        n_met = len(metal_nbs)
        mode = "terminal" if n_met <= 1 else f"mu{min(n_met, 6)}"
        # Formula: OH if bonded to H else element.
        formula = elements[idx]
        if elements[idx] == "O" and any(
            elements[j] == "H" for j, _ in adj.get(idx, []) if j in comp.site_indices
        ):
            formula = "OH"
        assignments.append(
            BridgingAssignment(
                atom_id=site_ids[idx],
                element=elements[idx],
                formula=formula,
                bridging_mode=mode,
                metal_site_ids=tuple(site_ids[m] for m in metal_nbs),
            )
        )
    return assignments


def _expected_cn(element: str) -> int:
    lo_hi = EXPECTED_CN.get(element)
    if lo_hi is None:
        return 6
    return lo_hi[1]


def OpenMetalSites(
    node: MetalNode,
    *,
    require_activation: bool = True,
) -> list[OpenSiteJudgement]:
    """Judge open metal sites as-deposited and after solvent removal."""
    out: list[OpenSiteJudgement] = []
    for site in node.sites or []:
        donors = site.donors or []
        observed = site.coordination_number
        expected = _expected_cn(site.element)
        removable = [d for d in donors if d.donor_group in SOLVENT_DONORS]
        # Terminal alkoxide_O only (bridging alkoxides are framework).
        removable = [
            d
            for d in removable
            if d.donor_group != "alkoxide_O" or d.bridging_mode == "terminal"
        ]
        cn_after = observed - len(removable)
        open_deposited = observed < expected and not removable
        # Also: unsaturated relative to typical max even with ligands present.
        if observed < expected and not any(d.donor_group in SOLVENT_DONORS for d in donors):
            open_deposited = True
        # Paddlewheel Cu: CN 4 as deposited, expected often 5–6 → open after activation
        # only if axial solvent present; without solvent, open as deposited for Cu paddlewheel.
        open_activation = cn_after < expected

        if require_activation:
            is_open = open_activation and (bool(removable) or open_deposited)
            # HKUST-1: no axial solvent in QMOF CIF → as deposited unsaturated (CN4 < 5),
            # and after activation still open. Distinguish:
            # - open_as_deposited: CN below expected WITHOUT counting removable solvent
            # - open_after_activation: CN after stripping SOLVENT_DONORS below expected
            pass

        evidence = OpenMetalSiteEvidence(
            observed_cn=observed,
            expected_cn=expected,
            removable_ligand=",".join(d.donor_group for d in removable) or None,
            solvent_removed=bool(removable),
        )
        method = "solvent_removal" if removable else "coordination_deficit"
        out.append(
            OpenSiteJudgement(
                site_id=site.site_id,
                open_as_deposited=observed < expected and not removable,
                open_after_activation=open_activation,
                method=method,
                evidence=evidence,
                removable_ligand_atom_ids=tuple(d.atom_id for d in removable),
            )
        )
    return out


def apply_oms_to_sites(node: MetalNode, *, require_activation: bool = True) -> MetalNode:
    """Return a copy of ``node`` with OMS fields filled on each site."""
    judgements = OpenMetalSites(node, require_activation=require_activation)
    by_id = {j.site_id: j for j in judgements}
    new_sites: list[MetalSite] = []
    for site in node.sites or []:
        j = by_id.get(site.site_id)
        if j is None:
            new_sites.append(site)
            continue
        # Benchmark convention (Phase 08.4 / 12.3): report open after activation.
        new_sites.append(
            site.model_copy(
                update={
                    "open_metal_site": j.open_after_activation,
                    "open_metal_site_method": j.method,
                    "open_metal_site_evidence": j.evidence,
                }
            )
        )
    return node.model_copy(update={"sites": new_sites})


def connectivity_archetype_warnings(node: MetalNode) -> list[str]:
    """Warn when computed connectivity disagrees with the archetype table."""
    expected = ARCHETYPE_EXPECTED_CONNECTIVITY.get(node.archetype)
    if expected is None or node.connectivity is None:
        return []
    if node.connectivity not in expected:
        return [
            f"connectivity_archetype_mismatch:{node.id}:{node.connectivity}:expected{list(expected)}"
        ]
    return []


def summarize_bridging_ligands(
    assignments: list[BridgingAssignment],
) -> list[dict[str, Any]]:
    """Collapse assignments into LigandCount-like dicts."""
    key_counts: Counter[tuple[str, str]] = Counter()
    for a in assignments:
        if a.bridging_mode == "terminal":
            continue
        key_counts[(a.formula, a.bridging_mode)] += 1
    return [
        {"formula": formula, "bridging_mode": mode, "count": count}
        for (formula, mode), count in sorted(key_counts.items())
    ]


__all__ = [
    "BridgingAssignment",
    "BridgingModes",
    "NodeConnectivity",
    "OpenMetalSites",
    "OpenSiteJudgement",
    "apply_oms_to_sites",
    "connectivity_archetype_warnings",
    "summarize_bridging_ligands",
]
