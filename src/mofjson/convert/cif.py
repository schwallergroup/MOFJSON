"""CIF <-> MOFJSON crystal layer. The most-used converter and the strictest test of it.

Phase 07.1 (read) and 07.2 (write).
"""

from __future__ import annotations

import io
import re
import warnings as py_warnings
from pathlib import Path
from typing import Any

from mofjson.convert.pymatgen_bridge import MOFFromStructure, MOFToStructure
from mofjson.exceptions import ConversionError, LossyConversionError
from mofjson.schema.crystal import normalize_space_group_symbol
from mofjson.schema.document import MOFJSONDocument
from mofjson.schema.metadata import PipelineStep

_CHARGE_COL_RE = re.compile(
    r"^_atom_site_(?P<func>[a-z0-9]+)_ddec_charge$",
    re.IGNORECASE,
)
_DECLARED_HM_RE = re.compile(
    r"_symmetry_space_group_name_H-M\s+'?([^'\n]+)'?",
    re.IGNORECASE,
)
_DISORDER_RE = re.compile(r"_atom_site_disorder_group", re.IGNORECASE)


def _require_pymatgen() -> None:
    try:
        import pymatgen  # noqa: F401
    except ImportError as exc:
        raise ConversionError(
            "pymatgen is required for CIF I/O; install with `pip install mofjson[crystal]`",
        ) from exc


def _read_text(path_or_text: str | Path) -> tuple[str, str | None]:
    """Return (cif_text, source_path_or_None)."""
    if isinstance(path_or_text, Path):
        return path_or_text.read_text(encoding="utf-8", errors="replace"), str(path_or_text)
    text = str(path_or_text)
    # Heuristic: short path-like existing file
    p = Path(text)
    if len(text) < 512 and p.is_file() and not text.lstrip().startswith(("data_", "#")):
        return p.read_text(encoding="utf-8", errors="replace"), str(p)
    return text, None


def _parse_cif_float(raw: str) -> float | None:
    s = raw.strip().strip("'\"")
    if s in ("", ".", "?", "None"):
        return None
    # Strip uncertainty in parentheses: 12.345(6)
    s = re.sub(r"\([^)]*\)", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def extract_qmof_site_columns(
    cif_text: str,
    *,
    charge_source: str = "pbe_ddec",
) -> dict[str, Any]:
    """Parse `_atom_site_*` loop columns that pymatgen ignores.

    Returns dict with keys: charges, magmoms, ddec_sum_bond_orders, cm5_charges,
    bader_spin_densities, labels, disorder_groups, charge_method, has_disorder.
    """
    # Find atom_site loop
    lines = cif_text.splitlines()
    loop_idx = None
    headers: list[str] = []
    for i, line in enumerate(lines):
        if line.strip().lower() == "loop_":
            # Peek following header lines
            j = i + 1
            hdrs: list[str] = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                hdrs.append(lines[j].strip().split()[0])
                j += 1
            if any(h.lower().startswith("_atom_site_") for h in hdrs):
                loop_idx = i
                headers = hdrs
                data_start = j
                break
    if loop_idx is None:
        return {
            "charges": None,
            "magmoms": None,
            "ddec_sum_bond_orders": None,
            "cm5_charges": None,
            "bader_spin_densities": None,
            "labels": None,
            "disorder_groups": None,
            "charge_method": None,
            "has_disorder": False,
        }

    # Resolve charge column for requested functional
    # charge_source like "pbe_ddec" → _atom_site_pbe_ddec_charge
    func = charge_source.replace("_ddec", "").replace("_cm5", "")
    if charge_source.endswith("_ddec") or charge_source == "pbe_ddec":
        charge_header = f"_atom_site_{func}_ddec_charge"
        charge_method = f"{func}_ddec"
    elif charge_source.endswith("_cm5"):
        charge_header = f"_atom_site_{func}_cm5_charge"
        charge_method = f"{func}_cm5"
    else:
        charge_header = f"_atom_site_{charge_source}_charge"
        charge_method = charge_source

    col = {h.lower(): idx for idx, h in enumerate(headers)}

    def col_of(*names: str) -> int | None:
        for n in names:
            if n.lower() in col:
                return col[n.lower()]
            # prefix match
            for h, idx in col.items():
                if h.endswith(n.lower().lstrip("_")) or h == n.lower():
                    return idx
        return None

    charge_i = col_of(charge_header)
    if charge_i is None:
        # Fallback: any *_ddec_charge
        for h, idx in col.items():
            if h.endswith("_ddec_charge"):
                charge_i = idx
                m = _CHARGE_COL_RE.match(headers[idx])
                charge_method = f"{m.group('func')}_ddec" if m else "ddec"
                break

    magmom_i = None
    for h, idx in col.items():
        if h.endswith("_magmom"):
            magmom_i = idx
            break
    sbo_i = None
    for h, idx in col.items():
        if h.endswith("_ddec_sum_bond_order"):
            sbo_i = idx
            break
    cm5_i = None
    for h, idx in col.items():
        if h.endswith("_cm5_charge"):
            cm5_i = idx
            break
    spin_i = None
    for h, idx in col.items():
        if h.endswith("_bader_spin_density") or h.endswith("_ddec_spin_density"):
            spin_i = idx
            break
    label_i = col_of("_atom_site_label")
    disorder_i = col_of("_atom_site_disorder_group")

    # Read data rows until next loop_ / data_ / blank-only end
    rows: list[list[str]] = []
    i = data_start
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue
        if raw.lower() == "loop_" or raw.lower().startswith("data_"):
            break
        if raw.startswith("_"):
            break
        if raw.startswith("#"):
            i += 1
            continue
        # Tokenise respecting quotes
        tokens = re.findall(r"'(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\"|\S+", raw)
        tokens = [t.strip("'\"") for t in tokens]
        if len(tokens) < len(headers):
            # continuation lines — rare; try to merge
            while len(tokens) < len(headers) and i + 1 < len(lines):
                i += 1
                more = re.findall(r"'(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\"|\S+", lines[i].strip())
                tokens.extend(t.strip("'\"") for t in more)
        rows.append(tokens[: len(headers)])
        i += 1

    def series(idx: int | None) -> list[float | None] | None:
        if idx is None:
            return None
        out: list[float | None] = []
        for row in rows:
            if idx >= len(row):
                out.append(None)
            else:
                out.append(_parse_cif_float(row[idx]))
        return out

    labels = None
    if label_i is not None:
        labels = [row[label_i] if label_i < len(row) else None for row in rows]

    disorder_groups = None
    has_disorder = False
    if disorder_i is not None:
        disorder_groups = [row[disorder_i] if disorder_i < len(row) else None for row in rows]
        has_disorder = any(g not in (None, ".", "?", "1", "0", "") for g in disorder_groups)

    return {
        "charges": series(charge_i),
        "magmoms": series(magmom_i),
        "ddec_sum_bond_orders": series(sbo_i),
        "cm5_charges": series(cm5_i),
        "bader_spin_densities": series(spin_i),
        "labels": labels,
        "disorder_groups": disorder_groups,
        "charge_method": charge_method if charge_i is not None else None,
        "has_disorder": has_disorder or (_DISORDER_RE.search(cif_text) is not None),
    }


def _declared_hm_symbol(cif_text: str) -> str | None:
    m = _DECLARED_HM_RE.search(cif_text)
    if not m:
        return None
    try:
        return normalize_space_group_symbol(m.group(1))
    except Exception:
        return m.group(1).strip()


def MOFFromCif(
    path_or_text: str | Path,
    *,
    expand_to_p1: bool = False,
    derive_bonds: bool = True,
    charge_source: str = "pbe_ddec",
    symprec: float = 0.01,
    strict: bool = False,
) -> MOFJSONDocument:
    """Parse CIF text or path into a MOFJSONDocument."""
    _require_pymatgen()
    from pymatgen.io.cif import CifParser

    text, source_path = _read_text(path_or_text)
    warn: list[str] = []
    extras = extract_qmof_site_columns(text, charge_source=charge_source)
    if extras["has_disorder"]:
        msg = "CIF contains _atom_site_disorder_group; disorder preserved with warning"
        warn.append(msg)
        if strict:
            raise LossyConversionError(msg, source_path=source_path, field_path="disorder")

    declared = _declared_hm_symbol(text)

    with py_warnings.catch_warnings(record=True) as caught:
        py_warnings.simplefilter("always")
        try:
            parser = CifParser.from_str(text) if hasattr(CifParser, "from_str") else CifParser(io.StringIO(text))
        except TypeError:
            # pymatgen API variants
            try:
                parser = CifParser.from_file(source_path) if source_path else CifParser(io.StringIO(text))
            except Exception as exc:
                raise ConversionError(
                    f"Failed to parse CIF: {exc}",
                    source_path=source_path,
                ) from exc
        except Exception as exc:
            raise ConversionError(f"Failed to parse CIF: {exc}", source_path=source_path) from exc

        try:
            structures = parser.parse_structures(primitive=False)
        except Exception:
            try:
                structures = parser.get_structures(primitive=False)
            except Exception as exc:
                raise ConversionError(
                    f"Failed to extract structure from CIF: {exc}",
                    source_path=source_path,
                ) from exc

    for w in caught:
        warn.append(f"pymatgen: {w.message}")

    if not structures:
        raise ConversionError("CIF contained no structures", source_path=source_path)
    structure = structures[0]

    if expand_to_p1:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            structure = SpacegroupAnalyzer(structure, symprec=symprec).get_primitive_standard_structure()
            # Actually expand to conventional P1 sites already present in QMOF.
            structure = structure.get_sorted_structure()
        except Exception as exc:
            warn.append(f"expand_to_p1 failed: {exc}")

    # Align charge arrays to structure site count if possible
    charges = extras["charges"]
    magmoms = extras["magmoms"]
    n = len(structure)
    if charges is not None and len(charges) != n:
        warn.append(
            f"charge column length {len(charges)} != structure sites {n}; dropping charges"
        )
        if strict:
            raise LossyConversionError("charge column length mismatch", source_path=source_path)
        charges = None
    if magmoms is not None and len(magmoms) != n:
        warn.append(f"magmom column length {len(magmoms)} != structure sites {n}")
        magmoms = None

    # Heuristic is_p1_expanded: QMOF declares P1 with identity only.
    is_p1 = None
    if declared in ("P1",) and n > 0:
        is_p1 = True

    doc = MOFFromStructure(
        structure,
        derive_bonds=derive_bonds,
        charge_method=extras["charge_method"],
        symprec=symprec,
        declared_space_group_symbol=declared,
        is_p1_expanded=is_p1,
        site_charges=charges,
        site_magmoms=magmoms,
        source_path=source_path,
        warnings=warn,
    )
    # Attach extras into provenance params
    assert doc.crystal is not None
    step = PipelineStep(
        step="MOFFromCif",
        tool="pymatgen.CifParser",
        params={
            "charge_source": charge_source,
            "derive_bonds": derive_bonds,
            "symprec": symprec,
            "declared_space_group_symbol": declared,
        },
    )
    doc.provenance = doc.provenance.model_copy(
        update={"pipeline": list(doc.provenance.pipeline) + [step]}
    )
    # Note extracted DDEC bond-order sums when present (validation aid for 07.4).
    if extras["ddec_sum_bond_orders"] is not None:
        doc.warnings = list(doc.warnings) + [
            f"extracted ddec_sum_bond_order for {len(extras['ddec_sum_bond_orders'])} sites"
        ]
    return doc


def MOFToCif(
    document: MOFJSONDocument,
    *,
    precision: int | None = None,
    symmetry: str = "P1",
) -> str:
    """Write a P1 CIF that pymatgen can read back."""
    _require_pymatgen()
    if document.crystal is None:
        raise ConversionError("document has no crystal layer", field_path="crystal")
    crystal = document.crystal
    prec = precision if precision is not None else crystal.coordinate_precision
    structure = MOFToStructure(document)

    # Emit a simple P1 CIF manually for full control over precision / occupancy.
    lat = crystal.lattice
    lines: list[str] = [
        "data_mofjson",
        "_symmetry_space_group_name_H-M   'P 1'",
        f"_cell_length_a   {lat.a:.{4}f}",
        f"_cell_length_b   {lat.b:.{4}f}",
        f"_cell_length_c   {lat.c:.{4}f}",
        f"_cell_angle_alpha   {lat.alpha:.{3}f}",
        f"_cell_angle_beta   {lat.beta:.{3}f}",
        f"_cell_angle_gamma   {lat.gamma:.{3}f}",
        "_symmetry_Int_Tables_number   1",
        f"_cell_volume   {lat.volume:.6f}",
        "loop_",
        " _symmetry_equiv_pos_as_xyz",
        "  'x, y, z'",
        "loop_",
        " _atom_site_type_symbol",
        " _atom_site_label",
        " _atom_site_fract_x",
        " _atom_site_fract_y",
        " _atom_site_fract_z",
        " _atom_site_occupancy",
    ]
    for site in crystal.sites:
        x, y, z = site.frac
        label = site.label or site.id
        lines.append(
            f"{site.element}  {label}  "
            f"{x:.{prec}f}  {y:.{prec}f}  {z:.{prec}f}  {site.occupancy:.4f}"
        )
    _ = structure  # used for validation path; emission is manual for precision control
    _ = symmetry
    return "\n".join(lines) + "\n"


__all__ = ["MOFFromCif", "MOFToCif", "extract_qmof_site_columns"]
