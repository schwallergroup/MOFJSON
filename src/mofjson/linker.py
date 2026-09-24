"""
IdentifyLinkers -- chemical identity (SMILES + molecular formula) for
linker_* building blocks, via RDKit.
"""
from __future__ import annotations

from typing import Any

from pymatgen.core import Lattice
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.mofjson import conversion as _conversion
from src.mofjson.molBuilder import fragment_to_rwmol


def _fragment_to_smiles_and_formula(fragment: dict[str, Any]) -> tuple[str | None, str | None, dict[str, Any]]:
    charge_by_id = {c["atom_id"]: c["formal_charge"] for c in (fragment.get("charges") or [])}
    rw, _id_to_idx = fragment_to_rwmol(fragment, charge_by_id)

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return None, None, {"ok": False, "reason": str(exc)}

    try:
        smiles = Chem.MolToSmiles(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception as exc:
        return None, None, {"ok": False, "reason": f"SMILES/formula extraction failed: {exc}"}

    return smiles, formula, {"ok": True}


def IdentifyLinkers(
    building_blocks: list[dict[str, Any]], lattice_params: dict[str, float]
) -> list[dict[str, Any]]:
    """Returns a NEW building_blocks list; every linker_* entry additionally
    carries smiles/formula/identity_provenance keys. Non-linker entries pass
    through with those three keys set to null. Never mutates the input list
    or its dicts in place."""
    lattice = Lattice.from_parameters(
        a=lattice_params["a"], b=lattice_params["b"], c=lattice_params["c"],
        alpha=lattice_params["alpha"], beta=lattice_params["beta"], gamma=lattice_params["gamma"],
    )

    out: list[dict[str, Any]] = []
    for block in building_blocks:
        block_out = dict(block)
        if not block["id"].startswith("linker_"):
            block_out.setdefault("smiles", None)
            block_out.setdefault("formula", None)
            block_out.setdefault("identity_provenance", None)
            out.append(block_out)
            continue

        refined, diagnostic = _conversion.InferOrganicBondOrders(block, lattice)
        if diagnostic.get("ok"):
            smiles, formula, extract_diag = _fragment_to_smiles_and_formula(refined)
            block_out["smiles"] = smiles
            block_out["formula"] = formula
            block_out["identity_provenance"] = extract_diag
        else:
            block_out["smiles"] = None
            block_out["formula"] = None
            block_out["identity_provenance"] = diagnostic
        out.append(block_out)
    return out


__all__ = ["IdentifyLinkers"]
