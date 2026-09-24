"""
DetectPolymerUnits -- automatic PolymerUnit (repeating-unit) detection.

atoms/bonds-dict -> RDKit RWMol construction is shared with linker.py via
molBuilder.fragment_to_rwmol (previously duplicated in both files).
"""
from __future__ import annotations

from typing import Any

from rdkit import Chem

from src.mofjson.molBuilder import fragment_to_rwmol
from src.mofjson.polymer import PolymerUnit

_AXES = ["a", "b", "c"]


def _self_loop_axes(block_id: str, net_edges: list[dict[str, Any]]) -> list[str]:
    axes: set[str] = set()
    for e in net_edges:
        if e["from_block"] == block_id and e["to_block"] == block_id:
            shift = e.get("shift", [0, 0, 0])
            for i, v in enumerate(shift):
                if v != 0:
                    axes.add(_AXES[i])
    return sorted(axes)


def _fragment_to_repeat_unit_smiles(fragment: dict[str, Any]) -> str | None:
    rw, _id_to_idx = fragment_to_rwmol(fragment)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def DetectPolymerUnits(
    building_blocks: list[dict[str, Any]], net_edges: list[dict[str, Any]]
) -> dict[str, PolymerUnit]:
    """Returns {block_id: PolymerUnit} for every block that bonds to its own
    next periodic copy. Blocks with no self-loop net_edge are simply
    absent from the result -- most linkers/nodes are NOT infinite repeats,
    so an empty result on an all-discrete structure (e.g. Cu-BTC, whose
    paddlewheel nodes are discrete, not rods) is the expected, correct
    output, not a failure."""
    result: dict[str, PolymerUnit] = {}
    for block in building_blocks:
        axes = _self_loop_axes(block["id"], net_edges)
        if not axes:
            continue
        repeat_unit_smiles = _fragment_to_repeat_unit_smiles(block)
        if repeat_unit_smiles is None:
            continue  # sanitization failed -- skip rather than guess
        result[block["id"]] = PolymerUnit(
            repeat_unit_smiles=repeat_unit_smiles, is_infinite=True, periodic_directions=axes
        )
    return result


__all__ = ["DetectPolymerUnits"]
