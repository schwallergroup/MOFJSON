"""The crystal layer: a lossless, CIF-equivalent description of the periodic structure.

Phase 04.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Annotated, Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from mofjson.exceptions import SchemaError
from mofjson.schema.base import MOFJSONModel
from mofjson.schema.molecular import ALLOWED_BOND_ORDERS
from mofjson.vocab.elements import METALS

# ---------------------------------------------------------------------------
# Space-group helpers
# ---------------------------------------------------------------------------

CRYSTAL_SYSTEMS: tuple[str, ...] = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)

BOND_GRAPH_SOURCES: frozenset[str] = frozenset(
    {
        "crystalnn",
        "cif_geom_bond",
        "distance_cutoff",
        "ddec_bond_order",
        "manual",
        "covalent_radii",
        "voronoi",
        "cutoff",
    }
)

# International Tables number ranges → crystal system.
_SYSTEM_BY_NUMBER: list[tuple[int, int, str]] = [
    (1, 2, "triclinic"),
    (3, 15, "monoclinic"),
    (16, 74, "orthorhombic"),
    (75, 142, "tetragonal"),
    (143, 167, "trigonal"),
    (168, 194, "hexagonal"),
    (195, 230, "cubic"),
]

# Common KG URI-sanitised spellings → Hermann–Mauguin.
_KG_SYMBOL_ALIASES: dict[str, str] = {
    "P2_1_c": "P2_1/c",
    "P21_c": "P2_1/c",
    "P2_1/c": "P2_1/c",
    "P21/c": "P2_1/c",
    "P212121": "P2_12_12_1",
    "P2_12_12_1": "P2_12_12_1",
    "P2_1_2_1_2_1": "P2_12_12_1",
    "P-1": "P-1",
    "P1": "P1",
    "P 1": "P1",
    "Fm-3m": "Fm-3m",
    "Fm3m": "Fm-3m",
    "R-3m": "R-3m",
    "R3m": "R3m",
}

_VOLUME_TOL = 1e-3
_MATRIX_TOL = 1e-4
_ZERO_IMAGE = (0, 0, 0)


def crystal_system_from_number(number: int) -> str:
    """Derive crystal system from International Tables space-group number."""
    if not 1 <= number <= 230:
        raise SchemaError(
            f"space_group_number must be in 1..230; got {number}",
            field_path="space_group_number",
            value=number,
        )
    for lo, hi, system in _SYSTEM_BY_NUMBER:
        if lo <= number <= hi:
            return system
    raise SchemaError(  # pragma: no cover
        f"No crystal system for space_group_number={number}",
        field_path="space_group_number",
        value=number,
    )


def normalize_space_group_symbol(symbol: str) -> str:
    """Normalise Hermann–Mauguin / KG-sanitised spellings to one canonical form.

    Covers the Phase 00.6 / 04.2 KG collisions:
    ``P2_1_c``, ``P21_c`` → ``P2_1/c``; ``P212121`` / ``P2_12_12_1`` → ``P2_12_12_1``.
    """
    raw = symbol.strip().strip("'\"")
    if not raw:
        raise SchemaError(
            "space_group_symbol must be non-empty",
            field_path="space_group_symbol",
            value=symbol,
        )
    # Collapse internal whitespace for lookup (CIF often has 'P 1').
    collapsed = re.sub(r"\s+", " ", raw)
    compact = collapsed.replace(" ", "")
    if collapsed in _KG_SYMBOL_ALIASES:
        return _KG_SYMBOL_ALIASES[collapsed]
    if compact in _KG_SYMBOL_ALIASES:
        return _KG_SYMBOL_ALIASES[compact]
    # Soft normalisation: prefer underscore subscript form for screw axes.
    soft = compact.replace("P21/", "P2_1/").replace("P21", "P2_1")
    if soft in _KG_SYMBOL_ALIASES:
        return _KG_SYMBOL_ALIASES[soft]
    return collapsed


def _round_len(v: float) -> float:
    return round(float(v), 4)


def _round_angle(v: float) -> float:
    return round(float(v), 3)


def _lattice_volume(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> float:
    """Cell volume from parameters (Å³), matching pymatgen's formula."""
    aa = math.radians(alpha)
    bb = math.radians(beta)
    gg = math.radians(gamma)
    cos_a, cos_b, cos_g = math.cos(aa), math.cos(bb), math.cos(gg)
    # triclinic volume formula
    v2 = (
        1.0
        - cos_a * cos_a
        - cos_b * cos_b
        - cos_g * cos_g
        + 2.0 * cos_a * cos_b * cos_g
    )
    if v2 <= 0.0:
        raise SchemaError(
            "Lattice parameters do not describe a valid cell (volume^2 <= 0)",
            field_path="lattice",
            value={"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma},
        )
    return a * b * c * math.sqrt(v2)


def _matrix_from_params(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> list[list[float]]:
    """Standard crystallographic lattice matrix (row vectors)."""
    aa = math.radians(alpha)
    bb = math.radians(beta)
    gg = math.radians(gamma)
    cos_a, cos_b, cos_g = math.cos(aa), math.cos(bb), math.cos(gg)
    sin_g = math.sin(gg)
    c1 = c * cos_b
    c2 = c * (cos_a - cos_b * cos_g) / sin_g
    c3_sq = c * c - c1 * c1 - c2 * c2
    if c3_sq < 0 and abs(c3_sq) < 1e-8:
        c3_sq = 0.0
    if c3_sq < 0:
        raise SchemaError(
            "Lattice parameters do not describe a valid cell",
            field_path="matrix",
            value={"a": a, "b": b, "c": c},
        )
    c3 = math.sqrt(c3_sq)
    return [
        [a, 0.0, 0.0],
        [b * cos_g, b * sin_g, 0.0],
        [c1, c2, c3],
    ]


def _norm3(v: list[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dot3(u: list[float], v: list[float]) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _params_from_matrix(
    matrix: list[list[float]],
) -> tuple[float, float, float, float, float, float]:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise SchemaError(
            "lattice.matrix must be 3x3",
            field_path="matrix",
            value=matrix,
        )
    a_vec = [float(x) for x in matrix[0]]
    b_vec = [float(x) for x in matrix[1]]
    c_vec = [float(x) for x in matrix[2]]
    a = _norm3(a_vec)
    b = _norm3(b_vec)
    c = _norm3(c_vec)
    if a == 0 or b == 0 or c == 0:
        raise SchemaError(
            "lattice.matrix has a zero-length vector",
            field_path="matrix",
            value=matrix,
        )

    def ang(u: list[float], v: list[float]) -> float:
        cos = _dot3(u, v) / (_norm3(u) * _norm3(v))
        cos = max(-1.0, min(1.0, cos))
        return math.degrees(math.acos(cos))

    return a, b, c, ang(b_vec, c_vec), ang(a_vec, c_vec), ang(a_vec, b_vec)


def _vec3(v: list[float], field_path: str) -> list[float]:
    if len(v) != 3 or not all(isinstance(x, (int, float)) for x in v):
        raise SchemaError(
            f"{field_path} must be a length-3 numeric list",
            field_path=field_path,
            value=v,
        )
    return [float(x) for x in v]


def _image_tuple(image: list[int] | tuple[int, ...] | None) -> tuple[int, int, int]:
    if image is None:
        return _ZERO_IMAGE
    if len(image) != 3:
        raise SchemaError(
            "image must be length-3 [dx, dy, dz]",
            field_path="image",
            value=image,
        )
    return (int(image[0]), int(image[1]), int(image[2]))


def generate_site_ids(elements: list[str]) -> list[str]:
    """MolJSON-style ``Element + running index`` ids (1-based per element)."""
    counts: Counter[str] = Counter()
    ids: list[str] = []
    for el in elements:
        counts[el] += 1
        ids.append(f"{el}{counts[el]}")
    return ids


def formula_from_sites(sites: list[Site]) -> tuple[str, str, int]:
    """Return (formula_sum, formula_reduced, n_atoms) from site list.

    Occupancy is summed (partial occupancy contributes fractionally to the reduced
    formula string via rounded integer counts for the sum form when occupancy==1).
    """
    n_atoms = len(sites)
    counts: Counter[str] = Counter()
    for s in sites:
        counts[s.element] += float(s.occupancy)
    # Hill-ish order: C, H, then alphabetical.
    keys = sorted(counts.keys(), key=lambda e: (0 if e == "C" else 1 if e == "H" else 2, e))
    parts_sum: list[str] = []
    for el in keys:
        c = counts[el]
        # Prefer integers when close.
        if abs(c - round(c)) < 1e-6:
            n = int(round(c))
            parts_sum.append(el if n == 1 else f"{el}{n}")
        else:
            parts_sum.append(f"{el}{c:.2f}".rstrip("0").rstrip("."))
    formula_sum = " ".join(parts_sum) if parts_sum else ""

    # Reduced: divide by GCD of near-integer counts; else leave as-is with floats.
    ints: list[int] = []
    ok = True
    for el in keys:
        c = counts[el]
        if abs(c - round(c)) > 1e-3:
            ok = False
            break
        ints.append(int(round(c)))
    if ok and ints:
        g = ints[0]
        for n in ints[1:]:
            g = math.gcd(g, n)
        g = max(g, 1)
        parts_red: list[str] = []
        for el, n in zip(keys, ints, strict=True):
            r = n // g
            parts_red.append(el if r == 1 else f"{el}{r}")
        formula_reduced = "".join(parts_red)
    else:
        formula_reduced = "".join(
            f"{el}{counts[el]:.2f}".rstrip("0").rstrip(".") for el in keys
        )
    return formula_sum, formula_reduced, n_atoms


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Lattice(MOFJSONModel):
    """Unit-cell parameters (canonical) with optional orientation matrix."""

    a: Annotated[float, Field(gt=0)]
    b: Annotated[float, Field(gt=0)]
    c: Annotated[float, Field(gt=0)]
    alpha: Annotated[float, Field(gt=0, lt=180)]
    beta: Annotated[float, Field(gt=0, lt=180)]
    gamma: Annotated[float, Field(gt=0, lt=180)]
    volume: Annotated[float, Field(gt=0)]
    matrix: list[list[float]] | None = None

    @field_validator("a", "b", "c", mode="before")
    @classmethod
    def _round_abc(cls, v: float) -> float:
        return _round_len(v)

    @field_validator("alpha", "beta", "gamma", mode="before")
    @classmethod
    def _round_angles(cls, v: float) -> float:
        return _round_angle(v)

    @model_validator(mode="before")
    @classmethod
    def _fill_from_matrix(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        matrix = data.get("matrix")
        param_keys = ("a", "b", "c", "alpha", "beta", "gamma")
        has_params = all(k in data and data[k] is not None for k in param_keys)
        if matrix is not None and not has_params:
            a, b, c, alpha, beta, gamma = _params_from_matrix(matrix)
            data = {
                **data,
                "a": a,
                "b": b,
                "c": c,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            }
        if data.get("volume") is None and all(
            k in data and data[k] is not None for k in ("a", "b", "c", "alpha", "beta", "gamma")
        ):
            data = {
                **data,
                "volume": _lattice_volume(
                    float(data["a"]),
                    float(data["b"]),
                    float(data["c"]),
                    float(data["alpha"]),
                    float(data["beta"]),
                    float(data["gamma"]),
                ),
            }
        return data

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        expected = _lattice_volume(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)
        if abs(self.volume - expected) > max(_VOLUME_TOL, 1e-4 * expected):
            raise SchemaError(
                f"lattice.volume {self.volume} disagrees with parameters ({expected})",
                field_path="volume",
                value=self.volume,
            )
        if self.matrix is not None:
            if len(self.matrix) != 3 or any(len(row) != 3 for row in self.matrix):
                raise SchemaError(
                    "lattice.matrix must be 3x3",
                    field_path="matrix",
                    value=self.matrix,
                )
            a, b, c, alpha, beta, gamma = _params_from_matrix(self.matrix)
            checks = [
                (abs(a - self.a), "a"),
                (abs(b - self.b), "b"),
                (abs(c - self.c), "c"),
                (abs(alpha - self.alpha), "alpha"),
                (abs(beta - self.beta), "beta"),
                (abs(gamma - self.gamma), "gamma"),
            ]
            for delta, name in checks:
                tol = _MATRIX_TOL if name in ("a", "b", "c") else 1e-3
                if delta > tol:
                    raise SchemaError(
                        f"lattice.matrix disagrees with parameter {name}",
                        field_path="matrix",
                        value=self.matrix,
                    )
        return self

    def to_matrix(self) -> list[list[float]]:
        """Return orientation matrix (stored or reconstructed from parameters)."""
        if self.matrix is not None:
            return [list(row) for row in self.matrix]
        return _matrix_from_params(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)


class Symmetry(MOFJSONModel):
    """Space-group assignment and whether sites are the asymmetric unit or P1-expanded."""

    space_group_number: Annotated[int, Field(ge=1, le=230)]
    space_group_symbol: str
    crystal_system: str
    setting: str = "standard"
    is_p1_expanded: bool
    symmetry_operations: list[str] | None = None
    symprec: float | None = 0.01
    declared_space_group_symbol: str | None = None

    @field_validator("space_group_symbol", "declared_space_group_symbol", mode="before")
    @classmethod
    def _norm_symbol(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return normalize_space_group_symbol(str(v))

    @model_validator(mode="before")
    @classmethod
    def _derive_system(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        number = data.get("space_group_number")
        if number is not None and data.get("crystal_system") is None:
            data = {**data, "crystal_system": crystal_system_from_number(int(number))}
        return data

    @model_validator(mode="after")
    def _check_rules(self) -> Self:
        expected = crystal_system_from_number(self.space_group_number)
        if self.crystal_system != expected:
            raise SchemaError(
                f"crystal_system must be {expected!r} for number {self.space_group_number}",
                field_path="crystal_system",
                value=self.crystal_system,
            )
        if self.crystal_system not in CRYSTAL_SYSTEMS:
            raise SchemaError(
                f"crystal_system must be one of {CRYSTAL_SYSTEMS}",
                field_path="crystal_system",
                value=self.crystal_system,
            )
        if self.is_p1_expanded:
            # Operations optional / identity-only when expanded.
            pass
        else:
            if not self.symmetry_operations:
                raise SchemaError(
                    "symmetry_operations required when is_p1_expanded is false",
                    field_path="symmetry_operations",
                    value=self.symmetry_operations,
                )
        return self


class Site(MOFJSONModel):
    """One crystallographic site (fractional coordinates)."""

    id: str = Field(min_length=1)
    element: str = Field(min_length=1)
    frac: list[float]
    occupancy: Annotated[float, Field(gt=0, le=1)] = 1.0
    label: str | None = None
    wyckoff: str | None = None
    charge: float | None = None
    magmom: float | None = None
    is_metal: bool | None = None
    node_id: str | None = None
    linker_id: str | None = None

    @field_validator("frac")
    @classmethod
    def _frac3(cls, v: list[float]) -> list[float]:
        return _vec3(v, "frac")

    @model_validator(mode="after")
    def _defaults_and_refs(self) -> Self:
        if self.node_id is not None and self.linker_id is not None:
            raise SchemaError(
                "site cannot claim both node_id and linker_id",
                field_path="node_id",
                value={"node_id": self.node_id, "linker_id": self.linker_id},
            )
        if self.is_metal is None:
            object.__setattr__(self, "is_metal", self.element in METALS)
        return self


class PeriodicBond(MOFJSONModel):
    """Bond in a periodic structure; ``image`` is the lattice translation of target."""

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    order: float
    image: list[int] | None = None
    distance: Annotated[float, Field(ge=0)] | None = None

    @field_validator("order")
    @classmethod
    def _order_allowed(cls, v: float) -> float:
        fv = float(v)
        for allowed in ALLOWED_BOND_ORDERS:
            if abs(fv - allowed) < 1e-9:
                return allowed
        raise SchemaError(
            f"Unsupported bond order: {v!r} (allowed: 0, 0.5, 1, 1.5, 2, 3)",
            field_path="order",
            value=v,
        )

    @field_validator("image", mode="before")
    @classmethod
    def _norm_image(cls, v: Any) -> list[int] | None:
        if v is None:
            return None
        t = _image_tuple(v)
        if t == _ZERO_IMAGE:
            return None
        return [t[0], t[1], t[2]]

    def image_tuple(self) -> tuple[int, int, int]:
        return _image_tuple(self.image)


CrystalLayout = Literal["verbose", "columnar"]


class CrystalStructure(MOFJSONModel):
    """Assembled crystal layer: lattice + symmetry + sites + bonds."""

    lattice: Lattice
    symmetry: Symmetry
    sites: list[Site] = Field(default_factory=list)
    bonds: list[PeriodicBond] = Field(default_factory=list)
    formula_sum: str | None = None
    formula_reduced: str | None = None
    n_atoms: int | None = None
    density: float | None = None
    charge_method: str | None = None
    coordinate_precision: Annotated[int, Field(ge=1, le=8)] = 4
    bond_graph_source: str | None = None
    layout: CrystalLayout = "verbose"
    # Columnar parallel arrays (used when layout == "columnar"; mirrored into sites).
    elements: list[str] | None = None
    frac: list[list[float]] | None = None
    ids: list[str] | None = None
    occupancies: list[float] | None = None

    @field_validator("bond_graph_source")
    @classmethod
    def _bond_source(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v not in BOND_GRAPH_SOURCES:
            raise SchemaError(
                f"bond_graph_source must be one of {sorted(BOND_GRAPH_SOURCES)}; got {v!r}",
                field_path="bond_graph_source",
                value=v,
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def _expand_columnar(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        layout = data.get("layout", "verbose")
        if layout == "columnar":
            elements = data.get("elements")
            fracs = data.get("frac")
            if elements is None or fracs is None:
                raise SchemaError(
                    "columnar layout requires elements and frac",
                    field_path="layout",
                    value=layout,
                )
            if len(elements) != len(fracs):
                raise SchemaError(
                    "columnar elements and frac length mismatch",
                    field_path="frac",
                    value={"n_elements": len(elements), "n_frac": len(fracs)},
                )
            ids = data.get("ids")
            if ids is None:
                ids = generate_site_ids(list(elements))
            elif len(ids) != len(elements):
                raise SchemaError(
                    "columnar ids length mismatch",
                    field_path="ids",
                    value={"n_ids": len(ids), "n_elements": len(elements)},
                )
            occs = data.get("occupancies")
            if occs is None:
                occs = [1.0] * len(elements)
            elif len(occs) != len(elements):
                raise SchemaError(
                    "columnar occupancies length mismatch",
                    field_path="occupancies",
                    value=occs,
                )
            # Prefer explicit sites if already provided; else build from columns.
            if not data.get("sites"):
                sites = [
                    {
                        "id": ids[i],
                        "element": elements[i],
                        "frac": fracs[i],
                        "occupancy": occs[i],
                    }
                    for i in range(len(elements))
                ]
                data = {**data, "sites": sites, "ids": ids, "occupancies": occs}
        return data

    @model_validator(mode="after")
    def _derived_fields(self) -> Self:
        site_ids = [s.id for s in self.sites]
        if len(site_ids) != len(set(site_ids)):
            raise SchemaError(
                "site ids must be unique",
                field_path="sites",
                value=site_ids,
            )
        id_set = set(site_ids)
        for i, bond in enumerate(self.bonds):
            if bond.source not in id_set:
                raise SchemaError(
                    f"bond source {bond.source!r} not in sites",
                    field_path=f"bonds[{i}].source",
                    value=bond.source,
                )
            if bond.target not in id_set:
                raise SchemaError(
                    f"bond target {bond.target!r} not in sites",
                    field_path=f"bonds[{i}].target",
                    value=bond.target,
                )

        formula_sum, formula_reduced, n_atoms = formula_from_sites(self.sites)
        if self.n_atoms is None:
            object.__setattr__(self, "n_atoms", n_atoms)
        elif self.n_atoms != n_atoms:
            raise SchemaError(
                f"n_atoms {self.n_atoms} disagrees with len(sites)={n_atoms}",
                field_path="n_atoms",
                value=self.n_atoms,
            )
        if self.formula_sum is None:
            object.__setattr__(self, "formula_sum", formula_sum)
        elif self.formula_sum.replace(" ", "") != formula_sum.replace(" ", ""):
            # Allow spacing differences only if token multisets match after split.
            if sorted(self.formula_sum.split()) != sorted(formula_sum.split()):
                raise SchemaError(
                    f"formula_sum {self.formula_sum!r} disagrees with sites ({formula_sum!r})",
                    field_path="formula_sum",
                    value=self.formula_sum,
                )
        if self.formula_reduced is None:
            object.__setattr__(self, "formula_reduced", formula_reduced)

        # Density (g/cm³) if not set: requires atomic masses — soft, leave None if unset.
        return self

    def to_verbose(self) -> CrystalStructure:
        """Return a verbose-layout copy (sites objects; columnar arrays cleared)."""
        if self.layout == "verbose" and self.elements is None:
            return self
        return self.model_copy(
            update={
                "layout": "verbose",
                "elements": None,
                "frac": None,
                "ids": None,
                "occupancies": None,
            }
        )

    def to_columnar(self) -> CrystalStructure:
        """Return a columnar-layout copy with parallel arrays filled from sites."""
        elements = [s.element for s in self.sites]
        fracs = [list(s.frac) for s in self.sites]
        ids = [s.id for s in self.sites]
        occs = [float(s.occupancy) for s in self.sites]
        return self.model_copy(
            update={
                "layout": "columnar",
                "elements": elements,
                "frac": fracs,
                "ids": ids,
                "occupancies": occs,
            }
        )

    @classmethod
    def from_columnar(
        cls,
        *,
        lattice: Lattice,
        symmetry: Symmetry,
        elements: list[str],
        frac: list[list[float]],
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> CrystalStructure:
        """Build from columnar arrays."""
        return cls(
            lattice=lattice,
            symmetry=symmetry,
            layout="columnar",
            elements=elements,
            frac=frac,
            ids=ids,
            **kwargs,
        )


__all__ = [
    "BOND_GRAPH_SOURCES",
    "CRYSTAL_SYSTEMS",
    "CrystalLayout",
    "CrystalStructure",
    "Lattice",
    "PeriodicBond",
    "Site",
    "Symmetry",
    "crystal_system_from_number",
    "formula_from_sites",
    "generate_site_ids",
    "normalize_space_group_symbol",
]
