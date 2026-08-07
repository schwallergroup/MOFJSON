"""Derive the periodic bond graph from geometry, when the source has no connectivity.

Phase 07.4.
"""

from __future__ import annotations

from typing import Any, Literal

from mofjson.exceptions import ConversionError, LossyConversionError
from mofjson.schema.crystal import PeriodicBond
from mofjson.vocab.elements import METALS

BondingMethod = Literal["crystalnn", "cutoff", "covalent_radii", "voronoi"]

# Covalent radii (Å) for organic-order fallback / cutoff method.
_COVALENT_RADII: dict[str, float] = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


def _require_pymatgen() -> Any:
    try:
        import pymatgen  # noqa: F401
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ConversionError(
            "pymatgen is required for bond derivation; install with "
            "`pip install mofjson[crystal]`",
        ) from exc
    return Structure


def _site_ids_and_elements(structure: Any) -> tuple[list[str], list[str]]:
    from mofjson.schema.crystal import generate_site_ids

    elements = [str(site.specie.symbol) for site in structure]
    return generate_site_ids(elements), elements


def _is_metal_ligand(el_i: str, el_j: str) -> bool:
    mi = el_i in METALS
    mj = el_j in METALS
    return mi != mj  # exactly one metal


def _crystalnn_bonds(
    structure: Any,
    ids: list[str],
    elements: list[str],
) -> list[tuple[int, int, tuple[int, int, int], float]]:
    """Return (i, j, image, distance) from CrystalNN. Image is to_jimage of j."""
    from pymatgen.analysis.local_env import CrystalNN

    cnn = CrystalNN()
    seen: set[tuple[int, int, tuple[int, int, int]]] = set()
    out: list[tuple[int, int, tuple[int, int, int], float]] = []
    n = len(structure)
    for i in range(n):
        try:
            neighbors = cnn.get_nn_info(structure, i)
        except Exception:
            continue
        for nb in neighbors:
            j = int(nb["site_index"])
            image = tuple(int(x) for x in nb["image"])
            # Canonical undirected key: smaller index first; flip image if swapped.
            if i < j:
                key = (i, j, image)
                i_use, j_use, img_use = i, j, image
            elif j < i:
                # Neighbor of i is j with image; equivalent bond j→i has -image
                # when expressed as to_jimage from j. Store with min index as source.
                inv = (-image[0], -image[1], -image[2])
                key = (j, i, inv)
                i_use, j_use, img_use = j, i, inv
            else:
                continue
            if key in seen:
                continue
            seen.add(key)
            # Distance from structure
            try:
                dist = float(structure.get_distance(i_use, j_use, jimage=img_use))
            except Exception:
                dist = float(nb.get("weight", 0.0)) or 0.0
            out.append((i_use, j_use, img_use, dist))
    return out


def _organic_orders_rdkit(
    structure: Any,
    ids: list[str],
    elements: list[str],
    connectivity: list[tuple[int, int, tuple[int, int, int], float]],
) -> dict[tuple[int, int], float]:
    """Perceive organic bond orders with RDKit on non-metal connected components.

    Returns mapping (i,j) with i<j → order. Missing pairs stay unspecified.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return {}

    # Build undirected organic adjacency (both endpoints non-metal).
    adj: dict[int, list[int]] = {i: [] for i in range(len(elements))}
    organic_edges: list[tuple[int, int]] = []
    for i, j, _img, _d in connectivity:
        if elements[i] in METALS or elements[j] in METALS:
            continue
        adj[i].append(j)
        adj[j].append(i)
        organic_edges.append((i, j) if i < j else (j, i))

    visited: set[int] = set()
    orders: dict[tuple[int, int], float] = {}

    for start in range(len(elements)):
        if start in visited or elements[start] in METALS:
            continue
        # BFS component
        stack = [start]
        comp: list[int] = []
        visited.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        if len(comp) < 2:
            continue
        index_map = {old: new for new, old in enumerate(comp)}
        rw = Chem.RWMol()
        for old in comp:
            atomic_num = structure[old].specie.Z
            rw.AddAtom(Chem.Atom(int(atomic_num)))
        added: set[tuple[int, int]] = set()
        for i, j, _img, _d in connectivity:
            if i not in index_map or j not in index_map:
                continue
            a, b = index_map[i], index_map[j]
            key = (a, b) if a < b else (b, a)
            if key in added:
                continue
            added.add(key)
            rw.AddBond(a, b, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            old_i, old_j = comp[a], comp[b]
            key = (old_i, old_j) if old_i < old_j else (old_j, old_i)
            bt = bond.GetBondType()
            if bt == Chem.BondType.TRIPLE:
                orders[key] = 3.0
            elif bt == Chem.BondType.DOUBLE:
                orders[key] = 2.0
            elif bt == Chem.BondType.AROMATIC:
                orders[key] = 1.5
            else:
                orders[key] = 1.0
    return orders


def DeriveBonds(
    structure: Any,
    *,
    method: BondingMethod = "crystalnn",
    strict: bool = False,
    site_ids: list[str] | None = None,
) -> tuple[list[PeriodicBond], list[str]]:
    """Derive periodic bonds from a pymatgen Structure.

    Returns ``(bonds, warnings)``. Metal–ligand contacts get order ``0.5``.
    Organic bonds get RDKit-perceived orders when RDKit is available, else ``1.0``.

    Convention: ``PeriodicBond.image`` is pymatgen StructureGraph ``to_jimage`` —
    the lattice translation applied to ``target`` relative to ``source``.
    """
    _require_pymatgen()
    warnings: list[str] = []
    ids = site_ids or _site_ids_and_elements(structure)[0]
    elements = [str(site.specie.symbol) for site in structure]
    if len(ids) != len(structure):
        raise ConversionError(
            "site_ids length does not match structure",
            field_path="site_ids",
            value={"n_ids": len(ids), "n_sites": len(structure)},
        )

    if method == "crystalnn":
        raw = _crystalnn_bonds(structure, ids, elements)
    elif method in ("cutoff", "covalent_radii"):
        raw = _cutoff_bonds(structure, elements, scale=1.3 if method == "covalent_radii" else 1.0)
    elif method == "voronoi":
        raw = _voronoi_bonds(structure)
    else:
        raise ConversionError(f"Unknown bonding method: {method!r}", field_path="method", value=method)

    organic_orders = _organic_orders_rdkit(structure, ids, elements, raw)
    if not organic_orders and any(
        elements[i] not in METALS and elements[j] not in METALS for i, j, _, _ in raw
    ):
        msg = "RDKit organic bond-order perception unavailable or failed; organic orders default to 1.0"
        warnings.append(msg)
        if strict:
            raise LossyConversionError(msg, field_path="bonds")

    bonds: list[PeriodicBond] = []
    for i, j, image, dist in raw:
        el_i, el_j = elements[i], elements[j]
        if _is_metal_ligand(el_i, el_j):
            order = 0.5
        elif el_i in METALS and el_j in METALS:
            # Metal–metal: keep as single unless we decide otherwise; Phase 08 must be consistent.
            order = 1.0
        else:
            key = (i, j) if i < j else (j, i)
            order = organic_orders.get(key, 1.0)
        img_list = None if image == (0, 0, 0) else [image[0], image[1], image[2]]
        bonds.append(
            PeriodicBond(
                source=ids[i],
                target=ids[j],
                order=order,
                image=img_list,
                distance=round(dist, 4) if dist else None,
            )
        )
    return bonds, warnings


def _cutoff_bonds(
    structure: Any,
    elements: list[str],
    *,
    scale: float,
) -> list[tuple[int, int, tuple[int, int, int], float]]:
    from pymatgen.core.periodic_table import Element

    out: list[tuple[int, int, tuple[int, int, int], float]] = []
    seen: set[tuple[int, int, tuple[int, int, int]]] = set()
    n = len(structure)
    for i in range(n):
        ri = _COVALENT_RADII.get(elements[i])
        if ri is None:
            try:
                ri = float(Element(elements[i]).atomic_radius or 1.5)
            except Exception:
                ri = 1.5
        neighbors = structure.get_neighbors(structure[i], r=ri * scale * 2.5)
        for ns in neighbors:
            j = ns.index
            image = tuple(int(x) for x in ns.image)
            rj = _COVALENT_RADII.get(elements[j])
            if rj is None:
                try:
                    rj = float(Element(elements[j]).atomic_radius or 1.5)
                except Exception:
                    rj = 1.5
            cutoff = (ri + rj) * scale
            if ns.nn_distance > cutoff:
                continue
            if i < j:
                key = (i, j, image)
                i_use, j_use, img = i, j, image
            elif j < i:
                inv = (-image[0], -image[1], -image[2])
                key = (j, i, inv)
                i_use, j_use, img = j, i, inv
            else:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append((i_use, j_use, img, float(ns.nn_distance)))
    return out


def _voronoi_bonds(structure: Any) -> list[tuple[int, int, tuple[int, int, int], float]]:
    from pymatgen.analysis.local_env import VoronoiNN

    vnn = VoronoiNN()
    seen: set[tuple[int, int, tuple[int, int, int]]] = set()
    out: list[tuple[int, int, tuple[int, int, int], float]] = []
    for i in range(len(structure)):
        try:
            neighbors = vnn.get_nn_info(structure, i)
        except Exception:
            continue
        for nb in neighbors:
            j = int(nb["site_index"])
            image = tuple(int(x) for x in nb["image"])
            if i < j:
                key = (i, j, image)
                i_use, j_use, img = i, j, image
            elif j < i:
                inv = (-image[0], -image[1], -image[2])
                key = (j, i, inv)
                i_use, j_use, img = j, i, inv
            else:
                continue
            if key in seen:
                continue
            seen.add(key)
            try:
                dist = float(structure.get_distance(i_use, j_use, jimage=img))
            except Exception:
                dist = 0.0
            out.append((i_use, j_use, img, dist))
    return out


__all__ = ["BondingMethod", "DeriveBonds"]
