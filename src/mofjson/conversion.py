"""
MOFJSON <-> pymatgen conversion.

Public API:
- StructureToJSON(structure) -> dict
- StructureFromJSON(mofjson) -> pymatgen.core.Structure
- CheckRoundTrip(structure) -> (bool, dict)

It only solves the problem for a periodic structure: read a
crystal structure, losslessly represent its atoms/bonds/lattice as MOFJSON,
and reconstruct an equivalent structure from that MOFJSON. 

Bond perception uses covalent radii (Cordero et al., Dalton Trans. 2008,
2832-2838) with an added skin distance, the same convention referenced in
the MOF cheminformatics literature (e.g. Bucior et al., Cryst. Growth Des.
2019, 19, 6682-6697). Metal-involving contacts default to a zero-order
(coordination) bond, following the InChI/MOFid convention for classifying
metal vs. nonmetal connectivity; this reuses the exact `order: 0` slot
MolJSON's own bond-order enum already reserves for zero-order bonds.

"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element

_DUMMY_SYMBOL = "*"

_ALLOWED_BOND_ORDERS: List[float] = [0.0, 1.0, 1.5, 2.0, 3.0]

# Cordero et al. covalent radii (Angstrom), Dalton Trans. 2008, 2832-2838.
_COVALENT_RADII: Dict[str, float] = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42,
    "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Cs": 2.44, "Ba": 2.15,
    "La": 2.07, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44,
    "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Pb": 1.44, "Bi": 1.51,
}
_DEFAULT_RADIUS = 1.50  # fallback for elements not in the table above

_BOND_SKIN = 0.40  # Angstrom; within the 0.3-0.45 range common in MOF literature

# Elements treated as metals for default coordination-bond-order assignment,
# following the InChI convention referenced by MOFid.
_METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Pb", "Bi",
}


def _require_list_optional(data: Dict[str, Any], key: str) -> List[Any]:
    if key not in data or data[key] is None:
        return []
    v = data[key]
    if not isinstance(v, list):
        raise ValueError(f"mofjson['{key}'] must be a list when present")
    return v


def _is_valid_element_symbol(sym: Any) -> bool:
    if not isinstance(sym, str):
        return False
    s = sym.strip()
    if not s:
        return False
    if s == _DUMMY_SYMBOL:
        return True
    try:
        Element(s)
        return True
    except ValueError:
        return False


def _normalize_bond_order(order: Any) -> float:
    if not isinstance(order, (int, float)):
        raise ValueError(
            f"Bond order must be a number (0, 1, 1.5, 2, 3). Got: {type(order)}"
        )
    o = float(order)
    for allowed in _ALLOWED_BOND_ORDERS:
        if abs(o - allowed) < 1e-9:
            return allowed
    raise ValueError(f"Unsupported bond order: {order!r} (allowed: 0, 1, 1.5, 2, 3)")


def _default_bond_order(sym_a: str, sym_b: str) -> float:
    """Metal-involving contacts default to a zero-order (coordination) bond;
    all-nonmetal contacts default to a single bond."""
    if sym_a in _METALS or sym_b in _METALS:
        return 0.0
    return 1.0


def _require_ordered_sites(structure: Structure) -> None:
    for i, site in enumerate(structure):
        if not site.is_ordered:
            raise NotImplementedError(
                f"Site {i} has mixed/disordered occupancy ({site.species}); "
                "compositionally disordered sites are not yet supported by "
                "StructureToJSON. Representing genuine site disorder as an "
                "ensemble is a deliberate future extension, analogous to how "
                "BigSMILES represents polymer ensembles with a stochastic "
                "object, not something to silently approximate here."
            )


def StructureFromJSON(mofjson: Dict[str, Any]) -> Structure:
    """
    Build a pymatgen Structure from MOFJSON.

    Bond/charge/role entries are validated strictly (unknown atom ids, bad
    shapes, and conflicting duplicate edges all raise) even though
    pymatgen's Structure has no native bond-list container -- charges and
    node/linker roles are attached as site_properties so a round trip can
    recover them; the bond list itself is re-derivable from geometry via
    StructureToJSON and is not stored on the returned Structure.
    """
    lattice_obj = mofjson.get("lattice")
    if not isinstance(lattice_obj, dict):
        raise ValueError("mofjson['lattice'] is required and must be an object")

    for key in ("a", "b", "c", "alpha", "beta", "gamma"):
        if key not in lattice_obj:
            raise ValueError(f"mofjson['lattice'] missing required key: {key}")

    lattice = Lattice.from_parameters(
        a=float(lattice_obj["a"]),
        b=float(lattice_obj["b"]),
        c=float(lattice_obj["c"]),
        alpha=float(lattice_obj["alpha"]),
        beta=float(lattice_obj["beta"]),
        gamma=float(lattice_obj["gamma"]),
    )

    atoms = _require_list_optional(mofjson, "atoms")
    bonds = _require_list_optional(mofjson, "bonds")
    charges = _require_list_optional(mofjson, "charges")
    node_linker_roles = _require_list_optional(mofjson, "node_linker_roles")

    id_to_index: Dict[str, int] = {}
    species: List[str] = []
    frac_coords: List[List[float]] = []
    occupancies: List[float] = []

    for atom_obj in atoms:
        if not isinstance(atom_obj, dict):
            raise ValueError("Each atom must be an object")
        atom_id = atom_obj.get("id")
        element = atom_obj.get("element")
        frac_xyz = atom_obj.get("frac_xyz")
        occupancy = atom_obj.get("occupancy", 1.0)

        if not isinstance(atom_id, str):
            raise ValueError(f"Atom id must be a string. Got: {type(atom_id)}")
        if atom_id in id_to_index:
            raise ValueError(f"Duplicate atom id: {atom_id}")
        if not _is_valid_element_symbol(element):
            raise ValueError(f"Unsupported/invalid element symbol: {element!r}")
        if (
            not isinstance(frac_xyz, list)
            or len(frac_xyz) != 3
            or not all(isinstance(v, (int, float)) for v in frac_xyz)
        ):
            raise ValueError(
                f"atoms.frac_xyz must be a length-3 numeric list: {frac_xyz!r}"
            )
        if element == _DUMMY_SYMBOL:
            raise ValueError(
                "Dummy atoms ('*') are not yet supported when reconstructing "
                "a periodic pymatgen Structure."
            )

        id_to_index[atom_id] = len(species)
        species.append(str(element))
        frac_coords.append([float(v) for v in frac_xyz])
        occupancies.append(float(occupancy))

    # Bonds carry the connectivity/periodicity information a plain pymatgen
    # Structure does not retain; they are validated strictly here so
    # malformed input fails loudly rather than silently losing connectivity.
    seen_edges: Dict[Tuple[str, str, Tuple[int, int, int]], float] = {}
    for bond_obj in bonds:
        if not isinstance(bond_obj, dict):
            raise ValueError("Each bond must be an object")
        a1 = bond_obj.get("source")
        a2 = bond_obj.get("target")
        order_raw = bond_obj.get("order")
        shift = bond_obj.get("shift", [0, 0, 0])

        if not isinstance(a1, str) or not isinstance(a2, str):
            raise ValueError("Bond source/target must be strings")
        if a1 not in id_to_index or a2 not in id_to_index:
            raise ValueError(f"Bond references unknown atom id(s): {a1}, {a2}")
        if (
            not isinstance(shift, list)
            or len(shift) != 3
            or not all(isinstance(v, int) for v in shift)
        ):
            raise ValueError(f"bonds.shift must be a length-3 integer list: {shift!r}")

        order = _normalize_bond_order(order_raw)
        key = (a1, a2, tuple(shift))
        if key in seen_edges and abs(seen_edges[key] - order) > 1e-9:
            raise ValueError(
                f"Conflicting duplicated edge between {a1} and {a2} with "
                f"shift {shift}: previous order={seen_edges[key]}, new order={order}"
            )
        seen_edges[key] = order

    seen_charge_atom_ids = set()
    charge_by_id: Dict[str, int] = {}
    for charge_obj in charges:
        if not isinstance(charge_obj, dict):
            raise ValueError("charges entries must be objects")
        atom_id = charge_obj.get("atom_id")
        formal_charge = charge_obj.get("formal_charge")

        if not isinstance(atom_id, str):
            raise ValueError(f"charges.atom_id must be a string. Got: {type(atom_id)}")
        if atom_id not in id_to_index:
            raise ValueError(f"charges.atom_id references unknown atom id: {atom_id}")
        if not isinstance(formal_charge, int):
            raise ValueError(
                f"charges.formal_charge must be int. Got: {type(formal_charge)}"
            )
        if atom_id in seen_charge_atom_ids:
            raise ValueError(f"Duplicate charges entry for atom_id: {atom_id}")
        seen_charge_atom_ids.add(atom_id)
        if formal_charge != 0:
            charge_by_id[atom_id] = formal_charge

    role_by_id: Dict[str, str] = {}
    for role_obj in node_linker_roles:
        if not isinstance(role_obj, dict):
            raise ValueError("node_linker_roles entries must be objects")
        atom_id = role_obj.get("atom_id")
        role = role_obj.get("role")
        if atom_id not in id_to_index:
            raise ValueError(
                f"node_linker_roles.atom_id references unknown atom id: {atom_id}"
            )
        if role not in ("node", "linker", "guest"):
            raise ValueError(f"node_linker_roles.role must be node/linker/guest: {role!r}")
        role_by_id[atom_id] = role

    ordered_ids = list(id_to_index.keys())
    site_properties: Dict[str, List[Any]] = {
        "atom_id": ordered_ids,
        "charge": [charge_by_id.get(aid, 0) for aid in ordered_ids],
        "role": [role_by_id.get(aid) for aid in ordered_ids],
    }

    return Structure(
        lattice=lattice,
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False,
        site_properties=site_properties,
    )


def StructureToJSON(
    structure: Structure,
    *,
    bond_skin: float = _BOND_SKIN,
    include_empty_fields: bool = False,
) -> Dict[str, Any]:
    """
    Convert a pymatgen Structure to MOFJSON.

    Periodic bond perception uses a Cordero covalent-radius cutoff with a
    configurable skin distance. Bonds crossing a periodic boundary carry an
    explicit integer `shift` vector labeling which neighboring unit cell the
    target atom sits in, following the labeled-quotient-graph approach used
    by periodic string representations such as SLICES.

    If include_empty_fields is False (default), empty optional fields
    (charges, aromatic_n_h, node_linker_roles) are omitted from the returned
    dict. If True, they are emitted as null. `topology` is always emitted
    (as null unless the caller fills it in later) since it is a single
    nullable object, not a sparse list.
    """
    if structure is None:
        raise ValueError("structure is None")

    _require_ordered_sites(structure)

    lattice = structure.lattice
    atoms: List[Dict[str, Any]] = []
    atom_ids: List[str] = []
    element_counts: Dict[str, int] = {}

    for site in structure:
        sym = site.specie.symbol
        if not _is_valid_element_symbol(sym):
            raise ValueError(f"Invalid element symbol in structure: {sym!r}")
        element_counts[sym] = element_counts.get(sym, 0) + 1
        atom_id = f"{sym}{element_counts[sym]}"
        atom_ids.append(atom_id)

        occ = float(site.species.num_atoms)
        atom_entry: Dict[str, Any] = {
            "id": atom_id,
            "element": sym,
            "frac_xyz": [float(v) for v in site.frac_coords],
        }
        if abs(occ - 1.0) > 1e-6:
            atom_entry["occupancy"] = occ
        atoms.append(atom_entry)

    max_radius = (
        max(_COVALENT_RADII.get(a["element"], _DEFAULT_RADIUS) for a in atoms)
        if atoms
        else _DEFAULT_RADIUS
    )
    search_radius = 2 * max_radius + bond_skin

    bonds: List[Dict[str, Any]] = []
    seen_pairs: set = set()
    for i, site in enumerate(structure):
        sym_i = atoms[i]["element"]
        r_i = _COVALENT_RADII.get(sym_i, _DEFAULT_RADIUS)
        for neighbor in structure.get_neighbors(site, r=search_radius):
            j = neighbor.index
            sym_j = atoms[j]["element"]
            r_j = _COVALENT_RADII.get(sym_j, _DEFAULT_RADIUS)
            cutoff = r_i + r_j + bond_skin
            if neighbor.nn_distance > cutoff:
                continue

            image = tuple(int(round(v)) for v in neighbor.image)
            if j == i and image == (0, 0, 0):
                continue  # no self-bonds within the same periodic image

            pair_key = (i, j, image)
            reverse_key = (j, i, tuple(-v for v in image))
            if pair_key in seen_pairs or reverse_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            bond_entry: Dict[str, Any] = {
                "source": atom_ids[i],
                "target": atom_ids[j],
                "order": _default_bond_order(sym_i, sym_j),
            }
            if image != (0, 0, 0):
                bond_entry["shift"] = list(image)
            bonds.append(bond_entry)

    charges: List[Dict[str, Any]] = []
    if "charge" in structure.site_properties:
        for i, fc in enumerate(structure.site_properties["charge"]):
            if fc:
                charges.append({"atom_id": atom_ids[i], "formal_charge": int(fc)})

    node_linker_roles: List[Dict[str, Any]] = []
    if "role" in structure.site_properties:
        for i, role in enumerate(structure.site_properties["role"]):
            if role:
                node_linker_roles.append({"atom_id": atom_ids[i], "role": role})

    mofjson: Dict[str, Any] = {
        "lattice": {
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha": lattice.alpha,
            "beta": lattice.beta,
            "gamma": lattice.gamma,
        },
        "atoms": atoms,
        "bonds": bonds,
        "charges": charges,
        "aromatic_n_h": [],
        "node_linker_roles": node_linker_roles,
        "topology": None,
        "building_blocks": None,
        "net_edges": None,
    }

    for key in ("charges", "aromatic_n_h", "node_linker_roles"):
        if len(mofjson[key]) == 0:
            if include_empty_fields:
                mofjson[key] = None
            else:
                mofjson.pop(key, None)

    return mofjson


def _unwrap_same_role(
    atoms: List[Dict[str, Any]], bonds: List[Dict[str, Any]], role_by_id: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, Tuple[int, int, int]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a graph from every SAME-ROLE bond regardless of its raw
    shift value, and within each connected component run a spanning-tree
    walk that accumulates an integer cell-offset per atom from each
    traversed bond's own shift (offset[target] = offset[source] + shift).
    Any additional (non-tree) same-role bond within a component is then
    checked against the offsets the tree already assigned: if walking that
    bond's own shift from its source's offset lands exactly on the
    target's already-assigned offset, the bond is an ordinary part of one
    finite, wrapped molecule (kept internal, regardless of raw shift). If
    it does NOT land there, that mismatch is a genuine monodromy -- a real
    topological signal of periodicity, not wrapping -- and the bond is
    treated as external/periodic, same as a cross-role bond. This is
    exactly the same class of shift-accumulating graph walk topology.py
    uses for coordination sequences, applied one step earlier in the
    pipeline, before roles are collapsed into building blocks at all.

    Returns (component_of, offset_by_atom_id, internal_same_role_bonds,
    monodromy_bonds). Component membership is computed from ALL same-role
    bonds (a monodromy edge never changes connectivity -- by definition it
    closes a cycle that the tree already connects some other way).
    """
    atom_ids = [a["id"] for a in atoms]
    parent: Dict[str, str] = {aid: aid for aid in atom_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    same_role_bonds = [b for b in bonds if role_by_id[b["source"]] == role_by_id[b["target"]]]
    for b in same_role_bonds:
        union(b["source"], b["target"])

    adjacency: Dict[str, List[Tuple[str, Tuple[int, int, int]]]] = {aid: [] for aid in atom_ids}
    for b in same_role_bonds:
        s, t = b["source"], b["target"]
        shift = tuple(b.get("shift", (0, 0, 0)))
        adjacency[s].append((t, shift))
        adjacency[t].append((s, tuple(-v for v in shift)))

    offset: Dict[str, Tuple[int, int, int]] = {}
    visited: set = set()
    tree_edge_pairs: set = set()  # {(a, b), (b, a)} for every tree-traversal edge used
    for root in atom_ids:
        if root in visited:
            continue
        offset[root] = (0, 0, 0)
        visited.add(root)
        stack = [root]
        while stack:
            cur = stack.pop()
            for nb, shift in adjacency[cur]:
                if nb not in visited:
                    offset[nb] = tuple(o + s for o, s in zip(offset[cur], shift))
                    visited.add(nb)
                    tree_edge_pairs.add((cur, nb))
                    tree_edge_pairs.add((nb, cur))
                    stack.append(nb)

    internal_same_role: List[Dict[str, Any]] = []
    monodromy_bonds: List[Dict[str, Any]] = []
    for b in same_role_bonds:
        s, t = b["source"], b["target"]
        if (s, t) in tree_edge_pairs:
            internal_same_role.append(b)
            continue
        shift = tuple(b.get("shift", (0, 0, 0)))
        expected = tuple(o + v for o, v in zip(offset[s], shift))
        if expected == offset[t]:
            internal_same_role.append(b)
        else:
            monodromy_bonds.append(b)

    return parent, offset, internal_same_role, monodromy_bonds


def CollapseToBuildingBlocks(mofjson: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collapse a flat, atomistic MOFJSON object into the compact repeating-unit
    layer: {"building_blocks": {...}, "net_edges": [...]}.
    """
    atoms = _require_list_optional(mofjson, "atoms")
    bonds = _require_list_optional(mofjson, "bonds")
    charges = _require_list_optional(mofjson, "charges")
    aromatic_n_h = _require_list_optional(mofjson, "aromatic_n_h")
    roles = _require_list_optional(mofjson, "node_linker_roles")

    role_by_id: Dict[str, str] = {r["atom_id"]: r["role"] for r in roles}
    atom_by_id: Dict[str, Dict[str, Any]] = {a["id"]: a for a in atoms}

    missing = [a["id"] for a in atoms if a["id"] not in role_by_id]
    if missing:
        raise ValueError(
            "CollapseToBuildingBlocks requires node_linker_roles to cover "
            f"every atom; missing roles for: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    parent, offset_by_id, internal_bonds, monodromy_bonds = _unwrap_same_role(atoms, bonds, role_by_id)

    def find(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x

    cross_role_bonds = [b for b in bonds if role_by_id[b["source"]] != role_by_id[b["target"]]]
    external_bonds: List[Dict[str, Any]] = cross_role_bonds + monodromy_bonds

    # Assign block ids per connected component, numbered per role for readability.
    component_of: Dict[str, str] = {}
    role_counters: Dict[str, int] = {}
    block_id_of_root: Dict[str, str] = {}
    for atom_id in atom_by_id:
        root = find(atom_id)
        if root not in block_id_of_root:
            role = role_by_id[root] if root in role_by_id else role_by_id[atom_id]
            role = role_by_id[atom_id]  # role is constant within a component by construction
            role_counters[role] = role_counters.get(role, 0) + 1
            block_id_of_root[root] = f"{role}_{role_counters[role]}"
        component_of[atom_id] = block_id_of_root[root]

    building_blocks: Dict[str, Any] = {
        block_id: {"atoms": [], "bonds": [], "charges": [], "aromatic_n_h": []}
        for block_id in set(component_of.values())
    }
    for atom in atoms:
        block = building_blocks[component_of[atom["id"]]]
        # Unwrap: shift this atom's stored frac_xyz by its component-relative
        # integer cell offset, so every atom in a fragment that survived the
        # monodromy check above lands in ONE geometrically consistent image
        # otherwise a fragment could contain two atoms that are really
        # adjacent but stored ~1 cell apart, silently corrupting every
        # downstream Cartesian-distance calculation (bond-order inference)
        o = offset_by_id.get(atom["id"], (0, 0, 0))
        unwrapped_xyz = [v + o_i for v, o_i in zip(atom["frac_xyz"], o)]
        block["atoms"].append(
            {**{k: v for k, v in atom.items() if k in ("id", "element", "frac_xyz", "occupancy")},
             "frac_xyz": unwrapped_xyz}
        )
    for bond in internal_bonds:
        block = building_blocks[component_of[bond["source"]]]
        block["bonds"].append(
            {"source": bond["source"], "target": bond["target"], "order": bond["order"]}
        )
    charge_by_id = {c["atom_id"]: c["formal_charge"] for c in charges}
    for atom_id, block_id in component_of.items():
        if atom_id in charge_by_id:
            building_blocks[block_id]["charges"].append(
                {"atom_id": atom_id, "formal_charge": charge_by_id[atom_id]}
            )
    aromatic_by_id = {e["atom_id"]: e["hcount"] for e in aromatic_n_h}
    for atom_id, block_id in component_of.items():
        if atom_id in aromatic_by_id:
            building_blocks[block_id]["aromatic_n_h"].append(
                {"atom_id": atom_id, "hcount": aromatic_by_id[atom_id]}
            )

    # Cut every external bond, adding a dummy attachment atom on each side.
    next_connection_id: Dict[str, int] = {b: 0 for b in building_blocks}
    net_edges: List[Dict[str, Any]] = []
    for bond in sorted(external_bonds, key=lambda b: (b["source"], b["target"])):
        raw_shift = tuple(bond.get("shift", (0, 0, 0)))
        # Derivation: unwrapped_position[atom] = raw_position[atom] +
        # offset[atom]. A bond (source, target, raw_shift) physically
        # connects source's own stored image to target's image at
        # raw_shift-from-source. Re-anchoring both endpoints to their
        # NEW (unwrapped) per-block reference frames and keeping the
        # existing "shift is target relative to source's own local cell 0"
        # convention gives corrected_shift = raw_shift + offset[source] -
        # offset[target] (translate the whole bond by +offset[source] so
        # source lands back on its own new local zero).
        off_s = offset_by_id.get(bond["source"], (0, 0, 0))
        off_t = offset_by_id.get(bond["target"], (0, 0, 0))
        shift = [raw_shift[i] + off_s[i] - off_t[i] for i in range(3)]
        endpoints = []
        for atom_id in (bond["source"], bond["target"]):
            block_id = component_of[atom_id]
            next_connection_id[block_id] += 1
            cid = next_connection_id[block_id]
            dummy_id = f"{block_id}_pt{cid}"
            # Same unwrapping as the real atom loop above -- a dummy point
            # must sit at its real atom's UNWRAPPED position, or a fragment
            # whose own atoms were unwrapped (see _unwrap_same_role) would
            # have an internally-inconsistent attachment point sitting a
            # full cell away from the rest of its own fragment.
            o = offset_by_id.get(atom_id, (0, 0, 0))
            dummy_xyz = [v + o_i for v, o_i in zip(atom_by_id[atom_id]["frac_xyz"], o)]
            building_blocks[block_id]["atoms"].append(
                {"id": dummy_id, "element": "*", "frac_xyz": dummy_xyz, "connection_id": cid}
            )
            building_blocks[block_id]["bonds"].append(
                {"source": atom_id, "target": dummy_id, "order": bond["order"]}
            )
            endpoints.append((block_id, cid))
        (fb, fp), (tb, tp) = endpoints
        net_edges.append(
            {"from_block": fb, "from_point": fp, "to_block": tb, "to_point": tp,
             "order": bond["order"], "shift": shift}
        )

    for block in building_blocks.values():
        if not block["charges"]:
            block["charges"] = None
        if not block["aromatic_n_h"]:
            block["aromatic_n_h"] = None

    # Emit as a sorted array of {id, atoms, bonds, charges, aromatic_n_h}
    building_blocks_array = [
        {"id": block_id, **block} for block_id, block in sorted(building_blocks.items())
    ]

    return {"building_blocks": building_blocks_array, "net_edges": net_edges}


def ExpandNet(
    building_blocks: List[Dict[str, Any]], net_edges: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Expand the compact repeating-unit layer back into a flat atoms/bonds core.

    Mechanical inverse of CollapseToBuildingBlocks: real atoms from every
    block are pooled as-is (ids are already globally unique by construction),
    dummy '*' attachment atoms are dropped.

    `building_blocks` is a list of {id, atoms, bonds, charges, aromatic_n_h}
    objects (matching the schema's array shape); block ids are looked up
    from each item's own "id" field, the same way bonds look up atom ids.
    """
    atoms: List[Dict[str, Any]] = []
    bonds: List[Dict[str, Any]] = []
    charges: List[Dict[str, Any]] = []
    aromatic_n_h: List[Dict[str, Any]] = []

    real_endpoint: Dict[Tuple[str, int], str] = {}  # (block_id, connection_id) -> real atom id

    for block in building_blocks:
        block_id = block["id"]
        dummy_ids = {
            a["id"]: a.get("connection_id")
            for a in block["atoms"]
            if a["element"] == _DUMMY_SYMBOL
        }
        for bond in block["bonds"]:
            src_is_dummy = bond["source"] in dummy_ids
            tgt_is_dummy = bond["target"] in dummy_ids
            if src_is_dummy and tgt_is_dummy:
                raise ValueError(f"Block {block_id} has a bond between two dummy atoms")
            if src_is_dummy:
                real_endpoint[(block_id, dummy_ids[bond["source"]])] = bond["target"]
            elif tgt_is_dummy:
                real_endpoint[(block_id, dummy_ids[bond["target"]])] = bond["source"]
            else:
                bonds.append(dict(bond))  # genuine internal bond, no shift

        for atom in block["atoms"]:
            if atom["element"] != _DUMMY_SYMBOL:
                atoms.append(
                    {k: v for k, v in atom.items() if k in ("id", "element", "frac_xyz", "occupancy")}
                )
        for c in (block.get("charges") or []):
            charges.append(c)
        for a in (block.get("aromatic_n_h") or []):
            aromatic_n_h.append(a)

    for edge in net_edges:
        source = real_endpoint.get((edge["from_block"], edge["from_point"]))
        target = real_endpoint.get((edge["to_block"], edge["to_point"]))
        if source is None or target is None:
            raise ValueError(f"net_edge references an unresolved connection point: {edge}")
        bond_entry: Dict[str, Any] = {"source": source, "target": target, "order": edge["order"]}
        shift = edge.get("shift", [0, 0, 0])
        if shift != [0, 0, 0]:
            bond_entry["shift"] = shift
        bonds.append(bond_entry)

    return {"atoms": atoms, "bonds": bonds, "charges": charges, "aromatic_n_h": aromatic_n_h}


def _canonical_bond_set(atoms: List[Dict[str, Any]], bonds: List[Dict[str, Any]]) -> set:
    """Bond identity independent of direction and independent of which
    endpoint the shift vector is expressed relative to"""
    elem_by_id = {a["id"]: a["element"] for a in atoms}
    canon = set()
    for b in bonds:
        shift = tuple(b.get("shift", (0, 0, 0)))
        forward = (b["source"], b["target"], b["order"], shift)
        backward = (b["target"], b["source"], b["order"], tuple(-v for v in shift))
        canon.add(min(forward, backward))
    return canon


def _gauge_fix_bonds(
    bonds: List[Dict[str, Any]], role_by_id: Dict[str, str]
) -> List[Dict[str, Any]]:
    _, offset_by_id, internal_same_role, monodromy_bonds = _unwrap_same_role(
        [{"id": aid} for aid in role_by_id], bonds, role_by_id
    )
    cross_role_bonds = [b for b in bonds if role_by_id[b["source"]] != role_by_id[b["target"]]]
    fixed: List[Dict[str, Any]] = []
    for b in internal_same_role:
        fixed.append({"source": b["source"], "target": b["target"], "order": b["order"]})
    for b in monodromy_bonds + cross_role_bonds:
        raw_shift = tuple(b.get("shift", (0, 0, 0)))
        off_s = offset_by_id.get(b["source"], (0, 0, 0))
        off_t = offset_by_id.get(b["target"], (0, 0, 0))
        corrected = [raw_shift[i] + off_s[i] - off_t[i] for i in range(3)]
        entry = {"source": b["source"], "target": b["target"], "order": b["order"]}
        if corrected != [0, 0, 0]:
            entry["shift"] = corrected
        fixed.append(entry)
    return fixed


def CheckNetConsistency(
    mofjson: Dict[str, Any], node_linker_roles: List[Dict[str, Any]] | None = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify that mofjson['building_blocks'] + mofjson['net_edges'], once
    expanded, describe the same atom/bond set as mofjson['atoms'] /
    mofjson['bonds']. Returns (ok, expanded_core).

    Pass `node_linker_roles` (the same list CollapseToBuildingBlocks was
    given) whenever the building_blocks/net_edges being checked were
    actually produced by CollapseToBuildingBlocks
    """
    building_blocks = mofjson.get("building_blocks")
    net_edges = mofjson.get("net_edges")
    if not building_blocks or net_edges is None:
        raise ValueError("mofjson has no building_blocks/net_edges to check")

    expanded = ExpandNet(building_blocks, net_edges)

    original_atom_ids = {a["id"] for a in mofjson["atoms"]}
    expanded_atom_ids = {a["id"] for a in expanded["atoms"]}
    atoms_match = original_atom_ids == expanded_atom_ids

    original_bonds = mofjson["bonds"]
    if node_linker_roles is not None:
        role_by_id = {r["atom_id"]: r["role"] for r in node_linker_roles}
        original_bonds = _gauge_fix_bonds(original_bonds, role_by_id)

    bonds_match = _canonical_bond_set(
        mofjson["atoms"], original_bonds
    ) == _canonical_bond_set(expanded["atoms"], expanded["bonds"])

    return (atoms_match and bonds_match), expanded

_BOND_LENGTH_RANGES: Dict[frozenset, List[Tuple[float, float, float]]] = {
    frozenset(["C", "C"]): [(1.16, 1.25, 3.0), (1.25, 1.36, 2.0), (1.36, 1.42, 1.5), (1.42, 1.60, 1.0)],
    frozenset(["C", "N"]): [(1.14, 1.24, 3.0), (1.24, 1.34, 2.0), (1.34, 1.42, 1.5), (1.42, 1.55, 1.0)],
    frozenset(["C", "O"]): [(1.10, 1.18, 3.0), (1.18, 1.28, 2.0), (1.28, 1.30, 1.5), (1.30, 1.48, 1.0)],
    frozenset(["N", "N"]): [(1.06, 1.16, 3.0), (1.16, 1.28, 2.0), (1.28, 1.36, 1.5), (1.36, 1.50, 1.0)],
    frozenset(["N", "O"]): [(1.14, 1.22, 2.0), (1.22, 1.30, 1.5), (1.30, 1.45, 1.0)],
}
_ORGANIC_NONMETALS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}


def _classify_bond_order_by_length(elem_a: str, elem_b: str, distance: float) -> float:
    if "H" in (elem_a, elem_b):
        return 1.0
    for lo, hi, order in _BOND_LENGTH_RANGES.get(frozenset([elem_a, elem_b]), []):
        if lo <= distance < hi:
            return order
    return 1.0


def _find_rings(edges: List[Tuple[str, str]], max_size: int = 7) -> List[List[str]]:
    """Simple, dependency-free ring finder over an undirected atom-id edge
    list: build a DFS spanning forest and, for every back edge found,
    recover the cycle it closes by walking parent pointers up to the
    common ancestor."""
    from collections import defaultdict

    adjacency: Dict[str, List[str]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    parent: Dict[str, Any] = {}
    depth: Dict[str, int] = {}
    visited: set = set()
    seen_rings: set = set()
    rings: List[List[str]] = []

    for root in adjacency:
        if root in visited:
            continue
        visited.add(root)
        parent[root] = None
        depth[root] = 0
        stack = [root]
        while stack:
            u = stack[-1]
            advanced = False
            for v in adjacency[u]:
                if v == parent[u]:
                    continue
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    stack.append(v)
                    advanced = True
                    break
                elif depth[v] < depth[u]:
                    cycle = [u]
                    cur = u
                    while cur != v:
                        cur = parent[cur]
                        cycle.append(cur)
                    if len(cycle) <= max_size:
                        key = frozenset(cycle)
                        if key not in seen_rings:
                            seen_rings.add(key)
                            rings.append(cycle)
            if not advanced:
                stack.pop()
    return rings


def InferOrganicBondOrders(
    fragment: Dict[str, Any], lattice: Lattice
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Infer bond order and aromaticity for a metal-free building-block
    fragment, from geometry alone, using established structural-chemistry
    fundamentals rather than requiring these to be supplied or guessed
    elsewhere in the pipeline.

    Returns (updated_fragment, diagnostic). `diagnostic` (e.g.
    {"ok": True} or {"ok": False, "reason": ...}) is deliberately NOT
    merged into the fragment dict: fragment_schema has
    additionalProperties=False, so any extra key would make the result
    fail its own schema the moment it is written into `building_blocks`.
    On failure, `updated_fragment` is returned unchanged (all-single-bond,
    original charges/aromatic_n_h) rather than a structure known to be
    chemically invalid.

    Mechanism, in order:
    1. Refuse (return the fragment unchanged, diagnostic {"ok": None}) if
       any real atom is a metal. Bond-order assignment via valence theory
       does not apply reliably to coordination bonds -- this stays exactly
       as ambiguous as it already is for MOFid, deliberately not "solved"
       here.
    2. Classify every real-atom-to-real-atom internal bond by length,
       against standard Pauling bond-order/length ranges. X-H bonds are
       always order 1 (no ambiguity).
    3. Apply one narrow, explicitly-scoped correction: a carbon bonded to
       exactly two terminal oxygens with near-equal C-O distances, NEITHER
       of which already has an explicit hydrogen bonded to it in this
       fragment, is a genuinely delocalized (anionic) carboxylate. It is
       set to one C=O double bond and one C-O single bond with an explicit
       -1 charge (and implicit-H suppressed) on the single-bonded oxygen --
       the standard resonance-structure convention -- rather than the raw
       per-bond classification, which independently misclassifies both
       bonds as C=O and over-values the carbon once its third substituent
       is counted (verified directly). If an explicit H IS already present
       on one of the two oxygens, this is an ordinary protonated -COOH and
       the correction is skipped entirely; protonation state is a
       geometric fact once hydrogens are explicitly present, not a
       property to infer. Carboxylates are targeted by name because they
       are the single most common linker motif in MOF chemistry.
    3b. Any small ring (size <= 7) containing at least one bond the raw
       per-bond classifier called aromatic (1.5) is resolved as its own
       unit via exhaustive search over just that ring's bonds, rather
       than trusting the independent per-bond guesses or handing an
       AROMATIC-typed ring straight to RDKit's Kekulizer -- both were
       tried and disproven directly against a real structure (see this
       function's own inline comment for the full story: RDKit's
       electron-donor-type determination for an aromatic heteroatom is
       based on its DEGREE, so a metal-coordinated ring atom's own dummy
       bond -- despite correctly contributing zero to valence -- still
       reads as an extra substituent, and with more than one such atom in
       the same ring, no valid Kekule structure is reachable at all no
       matter how implicit-H handling is tuned). A ring nitrogen with no
       explicit H that comes up exactly one valence unit short is allowed
       the same anionic-deprotonation resonance treatment as the
       carboxylate correction above (formal charge -1, implicit-H
       suppressed) -- the azolate-anion picture these linkers actually
       are, since the metal occupies the position a proton would.
    4. Build an RDKit molecule from the resulting graph and sanitize it.
       This is the safety net, not a formality: it is what catches any
       remaining locally-plausible-but-globally-invalid bond assignment
       (verified directly for both an over-valent carboxyl carbon and a
       non-ring atom incorrectly marked aromatic). If sanitization fails,
       the fragment is returned with bond orders reset to the conservative
       all-single default.
    5. On success, aromatic bonds are relabeled 1.5, aromatic_n_h entries
       are (re)computed for any aromatic nitrogen bearing an implicit H,
       and any additional formal charges required to satisfy valence are
       (re)computed -- reusing exactly the same extraction logic MolJSON's
       own MolToJSON already performs.

    Dummy '*' attachment atoms are left untouched: their bond to the real
    atom they cap keeps whatever order the original external bond had, and
    they never participate in the distance classification.
    """
    from rdkit import Chem  
    real_atoms = [a for a in fragment["atoms"] if a["element"] != _DUMMY_SYMBOL]
    if any(a["element"] in _METALS for a in real_atoms):
        return fragment, {"ok": None, "reason": "fragment contains a metal atom; not attempted"}

    cart = {
        a["id"]: lattice.get_cartesian_coords(a["frac_xyz"]) for a in fragment["atoms"]
    }
    elem_by_id = {a["id"]: a["element"] for a in fragment["atoms"]}
    dummy_ids = {a["id"] for a in fragment["atoms"] if a["element"] == _DUMMY_SYMBOL}

    new_bonds: List[Dict[str, Any]] = []
    real_bond_lengths: Dict[Tuple[str, str], float] = {}
    for bond in fragment["bonds"]:
        if bond["source"] in dummy_ids or bond["target"] in dummy_ids:
            new_bonds.append(dict(bond))  # untouched, as documented
            continue
        d = float(
            (cart[bond["source"]] - cart[bond["target"]]) @ (cart[bond["source"]] - cart[bond["target"]])
        ) ** 0.5
        order = _classify_bond_order_by_length(
            elem_by_id[bond["source"]], elem_by_id[bond["target"]], d
        )
        new_bonds.append({"source": bond["source"], "target": bond["target"], "order": order})
        real_bond_lengths[(bond["source"], bond["target"])] = d

    # Narrow carboxylate-resonance correction (see docstring, step 3).
    # A terminal carbon bonded to exactly two near-equidistant oxygens is
    # only a genuinely delocalized (anionic) carboxylate if NEITHER oxygen
    # already has an explicit hydrogen bonded to it in this fragment's own
    # connectivity -- if one does, this is an ordinary protonated -COOH,
    # correctly left to the raw per-bond classification (one short C=O,
    # one longer C-OH), and no charge is needed at all. This mirrors the
    # earlier finding that protonation state is a geometric fact once
    # hydrogens are explicitly present, not something to infer separately.
    from collections import defaultdict

    o_neighbors: Dict[str, List[str]] = defaultdict(list)
    c_neighbor_count: Dict[str, int] = defaultdict(int)
    h_neighbors_of: Dict[str, List[str]] = defaultdict(list)
    for b in new_bonds:
        for a1, a2 in ((b["source"], b["target"]), (b["target"], b["source"])):
            c_neighbor_count[a1] += 1
            if elem_by_id.get(a1) == "C" and elem_by_id.get(a2) == "O":
                o_neighbors[a1].append(a2)
            if elem_by_id.get(a2) == "H":
                h_neighbors_of[a1].append(a2)

    # atom_id -> pre-assignment applied before sanitization (RDKit would
    # otherwise assume a missing bond means "add an implicit H", which is
    # only correct for the protonated case).
    presets: Dict[str, Dict[str, Any]] = {}

    for c_id, o_ids in o_neighbors.items():
        if len(o_ids) != 2 or c_neighbor_count[c_id] != 3:
            continue
        d1 = real_bond_lengths.get((c_id, o_ids[0]), real_bond_lengths.get((o_ids[0], c_id)))
        d2 = real_bond_lengths.get((c_id, o_ids[1]), real_bond_lengths.get((o_ids[1], c_id)))
        if d1 is None or d2 is None or abs(d1 - d2) >= 0.08:
            continue
        if h_neighbors_of[o_ids[0]] or h_neighbors_of[o_ids[1]]:
            continue  # a real H is present -- protonated, not delocalized; leave as-is

        double_o, single_o = sorted(o_ids)  # deterministic, arbitrary resonance-form choice
        for b in new_bonds:
            if {b["source"], b["target"]} == {c_id, double_o}:
                b["order"] = 2.0
            elif {b["source"], b["target"]} == {c_id, single_o}:
                b["order"] = 1.0
        presets[single_o] = {"no_implicit": True, "formal_charge": -1}

    # Ring-level aromaticity resolution (see docstring, step 3b). Two
    # earlier, simpler attempts at this were tried and DISPROVEN directly
    # against a real structure before landing on this one:
    #   1. "Promote the whole ring to AROMATIC and let RDKit's own
    #      Kekulizer pick a pattern" -- fails whenever more than one ring
    #      heteroatom coordinates a metal. RDKit's electron-donor-type
    #      determination for an aromatic heteroatom is based on its
    #      DEGREE (connection count), not its valence sum -- a metal-
    #      coordinated ring atom's dummy bond contributes zero to valence
    #      but still counts as a connection, so RDKit treats every such
    #      atom as already-saturated. With more than one in the same ring
    #      (a common azolate binding mode -- verified directly: a real
    #      linker's two 1,2,4-triazolate rings each coordinate a metal
    #      through all three ring nitrogens), no ring carbon can find a
    #      double-bond partner at all and Kekulization is unsatisfiable no
    #      matter what implicit-H handling is layered on top.
    #   2. "Just suppress implicit-H on every metal-coordinated atom" --
    #      also fails: exactly ONE ring nitrogen legitimately needs the
    #      implicit-valence-completion slot (the anionic azolate position
    #      the metal actually replaces), and blocking it on all of them
    #      leaves no valid Kekule structure at all.
    # The fix that actually works, verified directly against all 16 real
    # linker fragments of that same structure: solve each ring's Kekule
    # structure ourselves via exhaustive search over just that ring's own
    # bonds (small rings -- a handful of atoms, entirely tractable),
    # scoring each candidate against every ring atom's real target valence
    # (4 for C, 3 for N) computed from ONLY its bonds outside the ring
    # (dummy/metal bonds excluded -- those never count). A ring nitrogen
    # with no explicit H of its own that comes up exactly one short is
    # allowed the same anionic-deprotonation treatment already used for
    # the unprotonated carboxylate oxygen above: formal charge -1,
    # implicit-H suppressed -- exactly the azolate-anion picture these
    # linkers actually are (no ring N-H is ever present in the crystal;
    # the metal occupies that position instead). The first
    # fully-consistent assignment found is adopted, deterministically,
    # same "arbitrary but reproducible resonance-form choice" philosophy
    # as the carboxylate correction just above.
    metal_coordinated_ids = {
        (b["target"] if b["source"] in dummy_ids else b["source"])
        for b in fragment["bonds"]
        if (b["source"] in dummy_ids) != (b["target"] in dummy_ids)
        and float(b.get("order", 1.0)) == 0.0
    }

    real_edges = [
        (b["source"], b["target"])
        for b in new_bonds
        if b["source"] not in dummy_ids and b["target"] not in dummy_ids
    ]
    order_by_pair: Dict[Tuple[str, str], float] = {}
    for b in new_bonds:
        order_by_pair[(b["source"], b["target"])] = b["order"]
        order_by_pair[(b["target"], b["source"])] = b["order"]

    # Fused-ring systems (e.g. a benzimidazole/naphthalene-like linker,
    # two rings sharing an atom) MUST be resolved together, not one ring
    # at a time: resolving them independently lets two separate rings
    # each hand a shared atom its own double bond, over-valuing it
    # (verified directly: this is what produced a real "explicit valence
    # ... greater than permitted" error on a shared ring carbon). Any
    # rings found by _find_rings that share an atom are merged (union-
    # find) into one combined ring system before resolution, so a shared
    # atom/bond is only ever decided once.
    candidate_rings = [
        r for r in _find_rings(real_edges, max_size=7)
        if any(order_by_pair.get(p) == 1.5 for p in zip(r, r[1:] + r[:1]))
    ]
    ring_system_parent: Dict[str, str] = {}

    def _rs_find(x: str) -> str:
        ring_system_parent.setdefault(x, x)
        while ring_system_parent[x] != x:
            ring_system_parent[x] = ring_system_parent[ring_system_parent[x]]
            x = ring_system_parent[x]
        return x

    def _rs_union(x: str, y: str) -> None:
        rx, ry = _rs_find(x), _rs_find(y)
        if rx != ry:
            ring_system_parent[rx] = ry

    for ring in candidate_rings:
        for a in ring:
            _rs_find(a)
        for a in ring[1:]:
            _rs_union(ring[0], a)

    system_atoms: Dict[str, set] = defaultdict(set)
    system_bond_pairs: Dict[str, set] = defaultdict(set)
    for ring in candidate_rings:
        root = _rs_find(ring[0])
        system_atoms[root].update(ring)
        for pair in zip(ring, ring[1:] + ring[:1]):
            system_bond_pairs[root].add(frozenset(pair))

    for root, atom_set in system_atoms.items():
        ring_atoms = sorted(atom_set)
        ring_pair_set = system_bond_pairs[root]
        ring_pairs = [tuple(p) for p in ring_pair_set]
        n = len(ring_pairs)

        extra_valence: Dict[str, float] = {}
        for a in ring_atoms:
            total = 0.0
            for b in new_bonds:
                if b["source"] in dummy_ids or b["target"] in dummy_ids:
                    continue
                if a not in (b["source"], b["target"]):
                    continue
                other = b["target"] if b["source"] == a else b["source"]
                if frozenset((a, other)) not in ring_pair_set:
                    total += b["order"]
            extra_valence[a] = total
        target_valence = {a: (4.0 if elem_by_id[a] == "C" else 3.0) for a in ring_atoms}

        solution = None
        for combo in range(2 ** n):
            orders = {
                frozenset(ring_pairs[i]): (2.0 if (combo >> i) & 1 else 1.0)
                for i in range(n)
            }
            totals = dict(extra_valence)
            for pair, o in orders.items():
                for a in pair:
                    totals[a] += o
            ring_presets: Dict[str, Dict[str, Any]] = {}
            ok = True
            for a in ring_atoms:
                deficit = target_valence[a] - totals[a]
                if abs(deficit) < 1e-9:
                    continue
                if (
                    abs(deficit - 1.0) < 1e-9
                    and elem_by_id[a] == "N"
                    and not h_neighbors_of.get(a)
                    and a in metal_coordinated_ids
                ):
                    ring_presets[a] = {"no_implicit": True, "formal_charge": -1}
                    continue
                ok = False
                break
            if ok:
                solution = (orders, ring_presets)
                break

        if solution is None:
            # No valid Kekule structure exists for this ring even with the
            # anionic-azolate allowance -- report failure rather than
            # emit a bond graph known to be chemically inconsistent.
            return fragment, {
                "ok": False,
                "reason": f"no valid aromatic ring resolution for atoms {ring_atoms}",
            }
        ring_orders, ring_presets = solution
        for b in new_bonds:
            pair = frozenset((b["source"], b["target"]))
            if pair in ring_orders:
                b["order"] = ring_orders[pair]
        presets.update(ring_presets)

    # Any bond still at 1.5 here was never resolved above, meaning
    # _find_rings never placed it in an actual cycle -- either it is a
    # genuinely acyclic bond whose length coincidentally fell in the
    # aromatic range (e.g. a biaryl single bond linking two separate
    # rings, which can sit right at that boundary), or a ring larger than
    # the size-7 cap. Leaving it as order 1.5 would mark both its atoms
    # aromatic with no ring for RDKit to Kekulize them into, which RDKit
    # correctly rejects as "non-ring atom marked aromatic" (verified
    # directly -- this is a distinct failure from anything the ring
    # resolution above fixes, not the same bug recurring). A bond outside
    # any ring cannot truly be aromatic, so it is clamped to the same
    # conservative single-bond default _classify_bond_order_by_length
    # itself falls back to for any length outside its known ranges.
    for b in new_bonds:
        if b["order"] == 1.5:
            b["order"] = 1.0

    # RDKit is the mandatory gate, not a formality (see docstring, step 4).
    rw = Chem.RWMol()
    id_to_idx: Dict[str, int] = {}
    for a in fragment["atoms"]:
        atom = Chem.Atom(0) if a["element"] == _DUMMY_SYMBOL else Chem.Atom(a["element"])
        if a["id"] in presets:
            atom.SetNoImplicit(presets[a["id"]]["no_implicit"])
            atom.SetFormalCharge(presets[a["id"]]["formal_charge"])
        id_to_idx[a["id"]] = rw.AddAtom(atom)
    order_to_bondtype = {0.0: Chem.BondType.ZERO, 1.0: Chem.BondType.SINGLE,
                          1.5: Chem.BondType.AROMATIC, 2.0: Chem.BondType.DOUBLE,
                          3.0: Chem.BondType.TRIPLE}
    for b in new_bonds:
        rw.AddBond(id_to_idx[b["source"]], id_to_idx[b["target"]], order_to_bondtype[b["order"]])
    aromatic_atom_idxs = {
        idx for b in rw.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC
        for idx in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    }
    for idx in aromatic_atom_idxs:
        rw.GetAtomWithIdx(idx).SetIsAromatic(True)

    mol = rw.GetMol()
    result = dict(fragment)
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return fragment, {"ok": False, "reason": str(exc)}

    result["bonds"] = new_bonds
    charges = []
    for a_id, idx in id_to_idx.items():
        fc = int(mol.GetAtomWithIdx(idx).GetFormalCharge())
        if fc != 0:
            charges.append({"atom_id": a_id, "formal_charge": fc})
    aromatic_n_h = []
    for a_id, idx in id_to_idx.items():
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "N" and atom.GetIsAromatic() and atom.GetTotalNumHs() == 1:
            aromatic_n_h.append({"atom_id": a_id, "hcount": 1})
    result["charges"] = charges or None
    result["aromatic_n_h"] = aromatic_n_h or None
    return result, {"ok": True}


def CheckRoundTrip(
    structure: Structure, *, bond_skin: float = _BOND_SKIN
) -> Tuple[bool, Dict[str, Any]]:
    """
    Round trip check:
    pymatgen Structure -> MOFJSON -> pymatgen Structure

    Uses pymatgen's StructureMatcher (lattice + site comparison under
    symmetry operations and numerical tolerance) rather than a string
    equality check. Unlike MolJSON, which can compare canonical SMILES
    strings, MOFJSON has no single canonical serialization -- structural
    identity for a periodic crystal has to be checked geometrically, not
    lexically.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    mofjson = StructureToJSON(structure, bond_skin=bond_skin, include_empty_fields=True)
    structure2 = StructureFromJSON(mofjson)
    matcher = StructureMatcher()
    ok = matcher.fit(structure, structure2)
    return ok, mofjson