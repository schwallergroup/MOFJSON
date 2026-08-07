"""Render one MOFJSON document into each representation being compared.

Phase 12.2.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from mofjson.schema.document import MOFJSONDocument
from mofjson.schema.reticular import ReticularStructure
from mofjson.util.ids import parse_mofid_line

_METAL_SITE_GEOM_FIELDS = frozenset(
    {
        "geometry",
        "geometry_symbol",
        "geometry_csm",
        "geometry_source",
        "geometry_confidence",
        "donors",
        "donor_composition",
        "bond_lengths",
        "bond_angles",
    }
)

_DATA_BLOCK_RE = re.compile(r"^data_\S+", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Representation:
    """One format under comparison."""

    id: str
    description: str
    answer_schema: str
    supports_paths: bool
    supports_coords: bool
    supports_reticular: bool
    is_mofjson: bool


REPRESENTATIONS: dict[str, Representation] = {
    "cif": Representation(
        id="cif",
        description="Raw CIF text with identifiers stripped",
        answer_schema="profiles/answer_cif.schema.json",
        supports_paths=True,
        supports_coords=True,
        supports_reticular=False,
        is_mofjson=False,
    ),
    "lattice_coords_text": Representation(
        id="lattice_coords_text",
        description="Lattice parameters + fractional coordinates as plain text",
        answer_schema="profiles/answer_lattice_coords_text.schema.json",
        supports_paths=True,
        supports_coords=True,
        supports_reticular=False,
        is_mofjson=False,
    ),
    "mofjson_crystal": Representation(
        id="mofjson_crystal",
        description="MOFJSON crystal layer, verbose",
        answer_schema="profiles/crystal.schema.json",
        supports_paths=True,
        supports_coords=True,
        supports_reticular=False,
        is_mofjson=True,
    ),
    "mofjson_crystal_columnar": Representation(
        id="mofjson_crystal_columnar",
        description="MOFJSON crystal layer, columnar",
        answer_schema="profiles/crystal_columnar.schema.json",
        supports_paths=True,
        supports_coords=True,
        supports_reticular=False,
        is_mofjson=True,
    ),
    "mofjson_reticular": Representation(
        id="mofjson_reticular",
        description="MOFJSON reticular layer (any completeness)",
        answer_schema="profiles/reticular.schema.json",
        supports_paths=False,
        supports_coords=False,
        supports_reticular=True,
        is_mofjson=True,
    ),
    "mofjson_reticular_full": Representation(
        id="mofjson_reticular_full",
        description="MOFJSON reticular layer with completeness=full",
        answer_schema="profiles/reticular.schema.json",
        supports_paths=False,
        supports_coords=True,
        supports_reticular=True,
        is_mofjson=True,
    ),
    "mofjson_reticular_nogeom": Representation(
        id="mofjson_reticular_nogeom",
        description="Reticular without Phase 03 geometry fields",
        answer_schema="profiles/reticular_nogeom.schema.json",
        supports_paths=False,
        supports_coords=False,
        supports_reticular=True,
        is_mofjson=True,
    ),
    "mofid": Representation(
        id="mofid",
        description="MOFid string alone",
        answer_schema="profiles/answer_mofid.schema.json",
        supports_paths=False,
        supports_coords=False,
        supports_reticular=False,
        is_mofjson=False,
    ),
}


def _strip_identifiers_from_doc(document: MOFJSONDocument) -> MOFJSONDocument:
    """Return a copy with identifying metadata cleared for prompt rendering."""
    ids = document.identifiers
    if ids is None:
        clean_ids = None
    else:
        clean_ids = ids.model_copy(
            update={
                "csd_refcode": None,
                "qmof_id": None,
                "mp_id": None,
                "odac_name": None,
                "kg_uri": None,
                "common_names": [],
                "structure_stem": None,
                "source_filename": None,
                "mofid": None,
                "mofkey": None,
            }
        )
    ret = document.reticular
    if isinstance(ret, dict) and "mofid" in ret:
        ret = {k: v for k, v in ret.items() if k != "mofid"}
    return document.model_copy(update={"identifiers": clean_ids, "reticular": ret})


def _canonical_mofid(raw: str | None) -> str | None:
    if not raw:
        return None
    parts = parse_mofid_line(raw)
    # Drop structure stem after ';' for contamination control.
    if parts.smiles and parts.topology_field:
        return f"{parts.smiles} MOFid-v1.{parts.topology_field}"
    if parts.smiles:
        return parts.smiles
    # Already stem-free or unparseable: strip trailing ;stem if present.
    text = str(raw).strip()
    if ";" in text:
        return text.rsplit(";", 1)[0].strip()
    return text


def _strip_cif_identifiers(cif_text: str) -> str:
    text = _DATA_BLOCK_RE.sub("data_structure", cif_text)
    return text


def _render_lattice_coords_text(document: MOFJSONDocument, *, precision: int) -> str:
    if document.crystal is None:
        raise ValueError("document has no crystal layer")
    crystal = document.crystal.to_verbose() if hasattr(document.crystal, "to_verbose") else document.crystal
    lat = crystal.lattice
    lines = [
        f"a={lat.a:.{precision}f} b={lat.b:.{precision}f} c={lat.c:.{precision}f}",
        f"alpha={lat.alpha:.{precision}f} beta={lat.beta:.{precision}f} gamma={lat.gamma:.{precision}f}",
        "sites:",
    ]
    for site in crystal.sites:
        x, y, z = site.frac
        lines.append(
            f"{site.element} {site.id} "
            f"{x:.{precision}f} {y:.{precision}f} {z:.{precision}f}"
        )
    return "\n".join(lines) + "\n"


def _strip_geometry_reticular(reticular: Any) -> dict[str, Any]:
    if hasattr(reticular, "model_dump"):
        data = reticular.model_dump(mode="json", exclude_none=True)
    else:
        data = copy.deepcopy(reticular)
    nodes = data.get("nodes") or []
    stripped_nodes = []
    for n in nodes:
        if not isinstance(n, dict):
            stripped_nodes.append(n)
            continue
        nd = {k: v for k, v in n.items() if k != "sites"}
        sites = n.get("sites")
        if isinstance(sites, list):
            nd["sites"] = [
                {k: v for k, v in s.items() if k not in _METAL_SITE_GEOM_FIELDS}
                if isinstance(s, dict)
                else s
                for s in sites
            ]
        stripped_nodes.append(nd)
    data["nodes"] = stripped_nodes
    return data


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _json_dump(obj: Any) -> str:
    payload = _to_jsonable(obj)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def Render(document: MOFJSONDocument, representation: str) -> str:
    """Render *document* into the named representation with identifiers stripped."""
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation {representation!r}; known: {sorted(REPRESENTATIONS)}"
        )
    clean = _strip_identifiers_from_doc(document)
    prec = 4
    if clean.crystal is not None:
        prec = int(clean.crystal.coordinate_precision or 4)

    if representation == "cif":
        from mofjson.convert.cif import MOFToCif

        return _strip_cif_identifiers(MOFToCif(clean, precision=prec))

    if representation == "lattice_coords_text":
        return _render_lattice_coords_text(clean, precision=prec)

    if representation == "mofjson_crystal":
        if clean.crystal is None:
            raise ValueError("document has no crystal layer")
        crystal = clean.crystal.to_verbose()
        return _json_dump({"crystal": crystal})

    if representation == "mofjson_crystal_columnar":
        if clean.crystal is None:
            raise ValueError("document has no crystal layer")
        crystal = clean.crystal.to_columnar()
        return _json_dump({"crystal": crystal})

    if representation == "mofjson_reticular":
        if clean.reticular is None:
            raise ValueError("document has no reticular layer")
        return _json_dump({"reticular": clean.reticular})

    if representation == "mofjson_reticular_full":
        if clean.reticular is None:
            raise ValueError("document has no reticular layer")
        ret = clean.reticular
        if isinstance(ret, ReticularStructure):
            completeness = ret.assembly.completeness if ret.assembly else None
        else:
            assembly = (ret or {}).get("assembly") if isinstance(ret, dict) else None
            completeness = assembly.get("completeness") if isinstance(assembly, dict) else None
        if completeness != "full":
            raise ValueError(
                "mofjson_reticular_full requires assembly.completeness == 'full' "
                f"(got {completeness!r})"
            )
        return _json_dump({"reticular": ret})

    if representation == "mofjson_reticular_nogeom":
        if clean.reticular is None:
            raise ValueError("document has no reticular layer")
        return _json_dump({"reticular": _strip_geometry_reticular(clean.reticular)})

    if representation == "mofid":
        raw = None
        if document.identifiers and document.identifiers.mofid:
            raw = document.identifiers.mofid
        elif isinstance(document.reticular, dict):
            raw = document.reticular.get("mofid")
        elif document.reticular is not None and hasattr(document.reticular, "mofid"):
            raw = getattr(document.reticular, "mofid", None)
        canon = _canonical_mofid(raw)
        if not canon:
            raise ValueError("document has no mofid")
        return canon

    raise ValueError(f"Unhandled representation {representation!r}")


def Parse(text: str, representation: str) -> MOFJSONDocument | Any:
    """Parse rendered text back to a document or pymatgen Structure."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation {representation!r}")

    if representation == "cif":
        from mofjson.convert.cif import MOFFromCif

        return MOFFromCif(text, derive_bonds=False)

    if representation == "lattice_coords_text":
        return _parse_lattice_coords_text(text)

    if representation == "mofid":
        return text.strip()

    # JSON-shaped MOFJSON profiles
    obj = json.loads(text)
    if representation in (
        "mofjson_crystal",
        "mofjson_crystal_columnar",
        "mofjson_reticular",
        "mofjson_reticular_full",
        "mofjson_reticular_nogeom",
    ):
        return MOFJSONDocument.model_validate(obj)
    raise ValueError(f"Unhandled representation {representation!r}")


def _parse_lattice_coords_text(text: str) -> MOFJSONDocument:
    """Best-effort parse of lattice_coords_text into a crystal document."""
    from mofjson.schema.crystal import CrystalStructure, Lattice, Site, Symmetry

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError("lattice_coords_text too short")
    # a=... b=... c=...
    def _kv_floats(line: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for part in line.replace(",", " ").split():
            if "=" in part:
                k, _, v = part.partition("=")
                out[k.strip().lower()] = float(v)
        return out

    abc = _kv_floats(lines[0])
    angles = _kv_floats(lines[1])
    sites: list[Site] = []
    for ln in lines[3:] if lines[2].lower().startswith("sites") else lines[2:]:
        parts = ln.split()
        if len(parts) < 5:
            continue
        el, sid, x, y, z = parts[0], parts[1], float(parts[2]), float(parts[3]), float(parts[4])
        sites.append(Site(id=sid, element=el, frac=[x, y, z]))
    a, b, c = abc["a"], abc["b"], abc["c"]
    # Approximate volume for triclinic-unknown; use orthogonal estimate.
    import math

    alpha = math.radians(angles["alpha"])
    beta = math.radians(angles["beta"])
    gamma = math.radians(angles["gamma"])
    volume = (
        a
        * b
        * c
        * math.sqrt(
            max(
                0.0,
                1
                - math.cos(alpha) ** 2
                - math.cos(beta) ** 2
                - math.cos(gamma) ** 2
                + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma),
            )
        )
    )
    crystal = CrystalStructure(
        lattice=Lattice(
            a=a,
            b=b,
            c=c,
            alpha=angles["alpha"],
            beta=angles["beta"],
            gamma=angles["gamma"],
            volume=volume,
        ),
        symmetry=Symmetry(
            space_group_number=1,
            space_group_symbol="P1",
            is_p1_expanded=True,
            crystal_system="triclinic",
        ),
        sites=sites,
        n_atoms=len(sites),
    )
    return MOFJSONDocument(crystal=crystal)


def representation_can_ask(representation: str, *, needs_paths: bool = False, needs_coords: bool = False) -> bool:
    rep = REPRESENTATIONS[representation]
    if needs_paths and not rep.supports_paths:
        return False
    if needs_coords and not rep.supports_coords:
        return False
    return True


__all__ = [
    "REPRESENTATIONS",
    "Parse",
    "Render",
    "Representation",
    "representation_can_ask",
]
