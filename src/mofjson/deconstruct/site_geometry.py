"""Stage 2: compute coordination number, polyhedron, CSM and donors per metal site.

Phase 08.2.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Literal

from mofjson.exceptions import GeometryError
from mofjson.schema.metal_node import (
    BondAngleSummary,
    BondLengthSummary,
    Donor,
    MetalSite,
)
from mofjson.vocab.donors import BRIDGING_MODES
from mofjson.vocab.elements import METALS
from mofjson.vocab.geometry import CSM_THRESHOLDS, map_chemenv_symbol

GeometryMethod = Literal["crystalnn", "chemenv", "both"]

_DEFAULT_TIMEOUT_S = 5.0


def _require_pymatgen() -> None:
    try:
        import pymatgen  # noqa: F401
    except ImportError as exc:
        raise GeometryError(
            "pymatgen is required for site geometry; install with `pip install mofjson[crystal]`",
        ) from exc


def _classify_donor_group(
    donor_idx: int,
    elements: list[str],
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]],
) -> str:
    """Heuristic donor-group assignment from the donor's local environment."""
    el = elements[donor_idx]
    nbs = [j for j, _ in adj.get(donor_idx, [])]
    nb_els = [elements[j] for j in nbs]
    metal_nbs = sum(1 for e in nb_els if e in METALS)
    nonmetal = [j for j in nbs if elements[j] not in METALS]
    nonmetal_els = [elements[j] for j in nonmetal]

    if el in ("F", "Cl", "Br", "I") and metal_nbs >= 1 and not any(
        e not in METALS and e != "H" for e in nb_els
    ):
        # Terminal halide (or only metals): still halide.
        return "halide"

    if el == "O":
        h_count = sum(1 for e in nonmetal_els if e == "H")
        c_nbs = [j for j in nonmetal if elements[j] == "C"]
        if metal_nbs >= 1 and not nonmetal:
            return "oxide_O"
        if metal_nbs >= 1 and set(nonmetal_els) <= {"H"} and h_count >= 1:
            if metal_nbs >= 2:
                return "hydroxide_O"
            return "aqua_O" if h_count >= 2 else "hydroxide_O"
        if c_nbs:
            # Carboxylate: C bonded to this O and another O.
            for c in c_nbs:
                o_on_c = [j for j, _ in adj.get(c, []) if elements[j] == "O"]
                if len(o_on_c) >= 2:
                    # Chelating if both O touch the same metal set size 1 exclusively.
                    return "carboxylate_O"
            return "alkoxide_O"
        return "other"

    if el == "N":
        c_nbs = [j for j in nonmetal if elements[j] == "C"]
        if c_nbs:
            # Imidazolate-like: N in a ring with C neighbours that have another N.
            for c in c_nbs:
                n_on_c = [j for j, _ in adj.get(c, []) if elements[j] == "N"]
                if len(n_on_c) >= 2:
                    return "imidazolate_N"
            # Pyridyl: aromatic N with two C neighbours.
            if sum(1 for e in nonmetal_els if e == "C") >= 2:
                return "pyridyl_N"
            return "amine_N"
        return "amine_N"

    if el == "S":
        return "thiolate_S"
    if el == "C":
        return "cyanide_C"
    return "other"


def _bridging_mode_for_donor(
    donor_idx: int,
    elements: list[str],
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]],
) -> str:
    n_metals = len({j for j, _ in adj.get(donor_idx, []) if elements[j] in METALS})
    if n_metals <= 1:
        return "terminal"
    mode = f"mu{n_metals}"
    if mode in BRIDGING_MODES:
        return mode
    return "mu6"


def _crystalnn_neighbors(structure: Any, site_index: int) -> list[dict[str, Any]]:
    from pymatgen.analysis.local_env import CrystalNN

    cnn = CrystalNN()
    try:
        return list(cnn.get_nn_info(structure, site_index))
    except Exception as exc:
        raise GeometryError(
            f"CrystalNN failed for site {site_index}: {exc}",
            field_path="sites",
            value=site_index,
        ) from exc


def _chemenv_geometry(
    structure: Any,
    site_index: int,
    *,
    distance_cutoff: float = 1.4,
    angle_cutoff: float = 0.3,
) -> tuple[str | None, float | None]:
    """Return (ce_symbol, csm) or (None, None) on failure."""
    try:
        from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
            SimplestChemenvStrategy,
        )
        from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
            LocalGeometryFinder,
        )
    except ImportError:
        return None, None

    try:
        lgf = LocalGeometryFinder()
        lgf.setup_structure(structure=structure)
        se = lgf.compute_structure_environments(
            only_indices=[site_index],
            max_cn=12,
            maximum_distance_factor=distance_cutoff,
        )
        strategy = SimplestChemenvStrategy(
            distance_cutoff=distance_cutoff,
            angle_cutoff=angle_cutoff,
        )
        strategy.set_structure_environments(se)
        env = strategy.get_site_coordination_environment(structure[site_index])
        if not env:
            return None, None
        # Modern pymatgen: (ce_symbol, details_dict) with symmetry_measure inside.
        if isinstance(env, (list, tuple)) and len(env) >= 2:
            symbol = str(env[0])
            second = env[1]
            if isinstance(second, dict):
                csm_raw = second.get("symmetry_measure", second.get("csm"))
                csm = float(csm_raw) if csm_raw is not None else None
            else:
                csm = float(second)
            return symbol, csm
        if isinstance(env, dict):
            symbol = str(env.get("ce_symbol") or env.get("symbol") or "")
            csm = env.get("csm", env.get("symmetry_measure"))
            return (symbol or None), (float(csm) if csm is not None else None)
        return str(env), None
    except Exception:
        return None, None


def _order_parameter_geometry(
    structure: Any,
    site_index: int,
    cn: int,
) -> tuple[str, float | None, str]:
    """Fallback via LocalStructOrderParams → geometry name + fake-ish CSM."""
    try:
        from pymatgen.analysis.local_env import LocalStructOrderParams
    except ImportError:
        return "unknown", None, "cn_heuristic"

    # Map CN to candidate OP types.
    op_types_by_cn: dict[int, list[str]] = {
        2: ["linear"],
        3: ["triangular", "tetrahedral"],
        4: ["square", "tetrahedral", "see_saw_rect"],
        5: ["square_pyramidal", "trigonal_bipyramidal"],
        6: ["octahedron", "trigonal_prismatic"],
        8: ["bcc", "square_antiprism_0_0_0"],
    }
    types = op_types_by_cn.get(cn)
    if not types:
        return "unknown", None, "cn_heuristic"
    try:
        lop = LocalStructOrderParams(types)
        vals = lop.get_order_parameters(structure, site_index)
    except Exception:
        return "unknown", None, "cn_heuristic"
    if not vals:
        return "unknown", None, "cn_heuristic"
    best_i = max(range(len(vals)), key=lambda i: (vals[i] if vals[i] is not None else -1.0))
    best_val = vals[best_i]
    if best_val is None or best_val < 0.3:
        return "unknown", None, "cn_heuristic"
    name = types[best_i]
    mapping = {
        "linear": ("linear", "L:2"),
        "triangular": ("trigonal_planar", "TL:3"),
        "tetrahedral": ("tetrahedral", "T:4"),
        "square": ("square_planar", "S:4"),
        "see_saw_rect": ("seesaw", "SS:4"),
        "square_pyramidal": ("square_pyramidal", "SP:5"),
        "trigonal_bipyramidal": ("trigonal_bipyramidal", "TB:5"),
        "octahedron": ("octahedral", "O:6"),
        "trigonal_prismatic": ("trigonal_prismatic", "PP:6"),
        "bcc": ("cubic", "C:8"),
        "square_antiprism_0_0_0": ("square_antiprismatic", "SA:8"),
    }
    geom, symbol = mapping.get(name, ("other", None))
    # Convert OP (1=perfect) to a CSM-like distortion measure.
    csm = float(max(0.0, (1.0 - float(best_val)) * 10.0))
    return geom, csm, "order_parameters"


def _bond_angle_summary(
    structure: Any,
    site_index: int,
    neighbor_indices: list[int],
    images: list[tuple[int, int, int]],
) -> BondAngleSummary | None:
    if len(neighbor_indices) < 2:
        return None
    angles: list[float] = []
    cart0 = structure[site_index].coords
    carts = []
    for j, img in zip(neighbor_indices, images, strict=True):
        carts.append(structure.lattice.get_cartesian_coords(
            structure[j].frac_coords + list(img)
        ))
    for a in range(len(carts)):
        for b in range(a + 1, len(carts)):
            v1 = carts[a] - cart0
            v2 = carts[b] - cart0
            n1 = math.sqrt(float(v1.dot(v1)))
            n2 = math.sqrt(float(v2.dot(v2)))
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            cosang = float(v1.dot(v2) / (n1 * n2))
            cosang = max(-1.0, min(1.0, cosang))
            angles.append(math.degrees(math.acos(cosang)))
    if not angles:
        return None
    return BondAngleSummary(
        min=min(angles),
        max=max(angles),
        mean=float(statistics.fmean(angles)),
    )


def _neighbors_from_adjacency(
    site_index: int,
    elements: list[str],
    adjacency: dict[int, list[tuple[int, tuple[int, int, int]]]],
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Non-metal neighbours from the existing bond graph (no CrystalNN re-run)."""
    neighbor_indices: list[int] = []
    images: list[tuple[int, int, int]] = []
    seen: set[tuple[int, tuple[int, int, int]]] = set()
    for j, img in adjacency.get(site_index, []):
        if elements[j] in METALS:
            continue
        key = (j, img)
        if key in seen:
            continue
        seen.add(key)
        neighbor_indices.append(j)
        images.append(img)
    return neighbor_indices, images


def AnalyseSite(
    structure: Any,
    site_index: int,
    *,
    method: GeometryMethod = "chemenv",
    site_id: str | None = None,
    elements: list[str] | None = None,
    adjacency: dict[int, list[tuple[int, tuple[int, int, int]]]] | None = None,
    distance_cutoff: float = 1.4,
    angle_cutoff: float = 0.3,
    use_bond_graph: bool = False,
) -> MetalSite:
    """Analyse one metal site; returns a :class:`MetalSite`."""
    _require_pymatgen()
    if site_index < 0 or site_index >= len(structure):
        raise GeometryError(
            f"site_index out of range: {site_index}",
            field_path="site_index",
            value=site_index,
        )

    el = str(structure[site_index].specie.symbol)
    sid = site_id or f"M{site_index + 1}"
    els = elements or [str(s.specie.symbol) for s in structure]

    if use_bond_graph and adjacency is not None:
        neighbor_indices, images = _neighbors_from_adjacency(site_index, els, adjacency)
    else:
        nn_info = _crystalnn_neighbors(structure, site_index)
        # Keep non-metal donors primarily; include metals only if no other neighbours.
        donor_infos = [
            nb
            for nb in nn_info
            if str(structure[nb["site_index"]].specie.symbol) not in METALS
        ]
        if not donor_infos:
            donor_infos = list(nn_info)
        neighbor_indices = [int(nb["site_index"]) for nb in donor_infos]
        images = [tuple(int(x) for x in nb["image"]) for nb in donor_infos]

    distances: list[float] = []
    for j, img in zip(neighbor_indices, images, strict=True):
        try:
            distances.append(float(structure.get_distance(site_index, j, jimage=img)))
        except Exception:
            distances.append(float("nan"))

    donors: list[Donor] = []
    adj = adjacency or {}
    for j, img, dist in zip(neighbor_indices, images, distances, strict=True):
        donor_el = els[j]
        if adj:
            group = _classify_donor_group(j, els, adj)
            mode = _bridging_mode_for_donor(j, els, adj)
        else:
            group = "unknown"
            mode = "terminal"
        donors.append(
            Donor(
                atom_id=f"{donor_el}{j + 1}",
                element=donor_el,
                donor_group=group,
                bridging_mode=mode,
                distance=None if math.isnan(dist) else round(dist, 4),
            )
        )

    cn = len(donors)
    donor_composition = dict(Counter(d.element for d in donors))
    finite_dists = [d for d in distances if not math.isnan(d)]
    bond_lengths = None
    if finite_dists:
        mean = float(statistics.fmean(finite_dists))
        std = float(statistics.pstdev(finite_dists)) if len(finite_dists) > 1 else 0.0
        bond_lengths = BondLengthSummary(
            min=min(finite_dists),
            max=max(finite_dists),
            mean=mean,
            std=std,
        )
    bond_angles = _bond_angle_summary(structure, site_index, neighbor_indices, images)

    geometry = "unknown"
    geometry_symbol: str | None = None
    geometry_csm: float | None = None
    geometry_source: str | None = None
    geometry_confidence: float | None = None

    want_chemenv = method in ("chemenv", "both")
    if want_chemenv:
        symbol, csm = _chemenv_geometry(
            structure,
            site_index,
            distance_cutoff=distance_cutoff,
            angle_cutoff=angle_cutoff,
        )
        if symbol:
            mapped = map_chemenv_symbol(symbol)
            thresh = float(CSM_THRESHOLDS.get("default", 5.0))
            geometry_symbol = symbol
            geometry_csm = csm
            geometry_source = "chemenv"
            if csm is not None and csm > thresh:
                geometry = "other"
                geometry_confidence = max(0.0, 1.0 - csm / (2 * thresh))
            else:
                geometry = mapped
                geometry_confidence = (
                    max(0.0, min(1.0, 1.0 - (csm / thresh))) if csm is not None else 0.5
                )
            # CN must match polyhedron ideal unless other/unknown.
            from mofjson.vocab.geometry import POLYHEDRON_CN

            ideal = POLYHEDRON_CN.get(geometry)
            if geometry not in ("other", "unknown") and ideal is not None and ideal != cn:
                geometry = "other"

    if geometry == "unknown" and method != "crystalnn":
        geom, csm, src = _order_parameter_geometry(structure, site_index, cn)
        if geom != "unknown":
            geometry = geom
            geometry_csm = csm if csm is not None else 0.0
            geometry_source = src
            geometry_confidence = 0.4
            from mofjson.vocab.geometry import POLYHEDRON_CN

            ideal = POLYHEDRON_CN.get(geometry)
            if ideal is not None and ideal != cn:
                geometry = "other"

    if geometry == "unknown" and method == "crystalnn":
        # CrystalNN-only: report CN with unknown polyhedron (allowed: no csm required).
        pass
    elif geometry == "unknown":
        # Still unknown after fallbacks — leave as unknown without csm.
        pass

    # When geometry is named/other, schema requires csm + source.
    if geometry != "unknown" and geometry_csm is None:
        geometry_csm = 0.0
    if geometry != "unknown" and geometry_source is None:
        geometry_source = "cn_heuristic"

    return MetalSite(
        site_id=sid,
        element=el,
        coordination_number=cn,
        geometry=geometry,
        geometry_symbol=geometry_symbol,
        geometry_csm=geometry_csm,
        geometry_source=geometry_source,
        geometry_confidence=geometry_confidence,
        donors=donors or None,
        donor_composition=donor_composition or None,
        bond_lengths=bond_lengths,
        bond_angles=bond_angles,
        open_metal_site=None,
    )


def AnalyseSites(
    structure: Any,
    site_indices: list[int],
    *,
    method: GeometryMethod = "chemenv",
    site_ids: list[str] | None = None,
    elements: list[str] | None = None,
    adjacency: dict[int, list[tuple[int, tuple[int, int, int]]]] | None = None,
    distance_cutoff: float = 1.4,
    angle_cutoff: float = 0.3,
    use_bond_graph: bool = False,
) -> list[MetalSite]:
    """Analyse several metal sites."""
    out: list[MetalSite] = []
    for k, idx in enumerate(site_indices):
        sid = site_ids[k] if site_ids is not None else None
        try:
            out.append(
                AnalyseSite(
                    structure,
                    idx,
                    method=method,
                    site_id=sid,
                    elements=elements,
                    adjacency=adjacency,
                    distance_cutoff=distance_cutoff,
                    angle_cutoff=angle_cutoff,
                    use_bond_graph=use_bond_graph,
                )
            )
        except GeometryError:
            el = str(structure[idx].specie.symbol)
            out.append(
                MetalSite(
                    site_id=sid or f"M{idx + 1}",
                    element=el,
                    coordination_number=0,
                    geometry="unknown",
                )
            )
    return out


__all__ = [
    "AnalyseSite",
    "AnalyseSites",
    "GeometryMethod",
]
