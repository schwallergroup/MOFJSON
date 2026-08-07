"""Read the MOFology KG from its flattened CSV export.

Phase 09.1. Build this before the Turtle parser.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mofjson.exceptions import IngestError

# Identity columns in header order (Phase 00.7 / 09.1).
CORE_COLUMNS: tuple[str, ...] = (
    "mof_uri",
    "chemical_formula",
    "mofid",
    "topology",
    "metal_cluster_elements",
    "linker_smiles",
    "space_group",
    "crystal_system",
)

EXPECTED_ROW_COUNT = 254_099


@dataclass
class KgLinker:
    """Raw linker assertion from the KG (TTL path fills this fully)."""

    uri: str | None = None
    smiles: str | None = None
    chemical_name: str | None = None


@dataclass
class KgCluster:
    """Raw metal-cluster assertion from the KG."""

    uri: str | None = None
    metal_element: str | None = None


@dataclass
class KgSynthesis:
    """Raw synthesis process / conditions from the KG."""

    uri: str | None = None
    methods: list[str] = field(default_factory=list)
    temperature: float | None = None
    solvents: list[dict[str, Any]] = field(default_factory=list)
    additives: list[dict[str, Any]] = field(default_factory=list)
    is_hypothetical: bool | None = None


@dataclass
class KgAbstract:
    """Raw publication abstract / title from the KG."""

    uri: str | None = None
    title: str | None = None
    abstract: str | None = None


@dataclass
class KgRecord:
    """One MOF's worth of raw KG fields, un-normalised."""

    mof_uri: str
    chemical_formula: str | None = None
    mofid: str | None = None
    topology: str | None = None
    metal_cluster_elements: str | None = None
    linker_smiles: str | None = None
    space_group: str | None = None
    crystal_system: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    # Relational slots (TTL enrichment)
    linkers: list[KgLinker] = field(default_factory=list)
    clusters: list[KgCluster] = field(default_factory=list)
    synthesis: KgSynthesis | None = None
    abstracts: list[KgAbstract] = field(default_factory=list)
    functional_group_name: str | None = None
    canonical_name: str | None = None
    types: list[str] = field(default_factory=list)
    is_experimental: bool | None = None
    is_hypothetical: bool | None = None
    is_functionalized: bool | None = None
    parent_mof_uri: str | None = None
    source: str = "csv"


def _is_empty_cell(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except Exception:
        pass
    return False


def _cell(value: Any) -> Any:
    """Map empty / NaN cells to None; preserve zeros and other values."""
    if _is_empty_cell(value):
        return None
    if isinstance(value, str):
        return value
    # pandas may give numpy scalars
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def validate_kg_csv_header(columns: list[str]) -> list[str]:
    """Ensure all CORE_COLUMNS are present; return property column names.

    Raises IngestError if an expected identity column is missing or renamed.
    """
    present = set(columns)
    missing = [c for c in CORE_COLUMNS if c not in present]
    if missing:
        raise IngestError(
            f"KG CSV missing expected identity columns: {missing}",
            field_path="header",
            value=missing,
        )
    prop_cols = [c for c in columns if c not in CORE_COLUMNS]
    return prop_cols


@dataclass
class KgCsvCoverage:
    """Field-coverage table for the Phase 09 audit."""

    n_rows: int = 0
    field_present: Counter[str] = field(default_factory=Counter)
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "columns": list(self.columns),
            "field_present": dict(self.field_present),
            "field_fraction": {
                k: (v / self.n_rows if self.n_rows else 0.0)
                for k, v in sorted(self.field_present.items())
            },
        }


def ReadKgCsv(
    path: str | Path,
    *,
    chunksize: int = 5_000,
    limit: int | None = None,
    collect_coverage: bool = False,
) -> Iterator[KgRecord]:
    """Stream the flattened KG CSV as KgRecord objects.

    When *collect_coverage* is True, the iterator's final attribute
    ``coverage`` (on the generator) is unreliable; use
    :func:`ReadKgCsvWithCoverage` instead.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise IngestError(
            "pandas is required for KG CSV ingest; install with `pip install mofjson[kg]`",
            source_path=str(path),
        ) from exc

    p = Path(path)
    if not p.is_file():
        raise IngestError(f"KG CSV not found: {p}", source_path=str(p))

    # Peek header
    header_df = pd.read_csv(p, nrows=0)
    columns = list(header_df.columns)
    prop_cols = validate_kg_csv_header(columns)

    yielded = 0
    for chunk in pd.read_csv(p, chunksize=chunksize, dtype=str, keep_default_na=False):
        # Re-read with default NA handling for numeric property columns would
        # convert empties to NaN; we used dtype=str so empties are "".
        for _, row in chunk.iterrows():
            if limit is not None and yielded >= limit:
                return
            uri = _cell(row.get("mof_uri"))
            if uri is None:
                continue
            props: dict[str, Any] = {}
            for col in prop_cols:
                val = _cell(row.get(col))
                if val is not None:
                    # Attempt numeric coercion for property values
                    if isinstance(val, str):
                        try:
                            if "." in val or "e" in val.lower():
                                props[col] = float(val)
                            else:
                                props[col] = int(val)
                        except ValueError:
                            props[col] = val
                    else:
                        props[col] = val

            # Seed linker list from the single CSV linker_smiles column
            linker_smiles = _cell(row.get("linker_smiles"))
            linkers: list[KgLinker] = []
            if linker_smiles is not None:
                linkers.append(KgLinker(smiles=str(linker_smiles)))

            yield KgRecord(
                mof_uri=str(uri),
                chemical_formula=_cell(row.get("chemical_formula")),
                mofid=_cell(row.get("mofid")),
                topology=_cell(row.get("topology")),
                metal_cluster_elements=_cell(row.get("metal_cluster_elements")),
                linker_smiles=str(linker_smiles) if linker_smiles is not None else None,
                space_group=_cell(row.get("space_group")),
                crystal_system=_cell(row.get("crystal_system")),
                properties=props,
                linkers=linkers,
                source="csv",
            )
            yielded += 1


def ReadKgCsvWithCoverage(
    path: str | Path,
    *,
    chunksize: int = 5_000,
    limit: int | None = None,
) -> tuple[list[KgRecord], KgCsvCoverage]:
    """Convenience: materialise records (honouring *limit*) and build coverage.

    Prefer :func:`ReadKgCsv` for full-corpus streaming; this helper is for
    tests and small slices.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise IngestError(
            "pandas is required for KG CSV ingest; install with `pip install mofjson[kg]`",
            source_path=str(path),
        ) from exc

    p = Path(path)
    header_df = pd.read_csv(p, nrows=0)
    columns = list(header_df.columns)
    validate_kg_csv_header(columns)
    records = list(ReadKgCsv(path, chunksize=chunksize, limit=limit))
    coverage = KgCsvCoverage(columns=columns, n_rows=len(records))
    for rec in records:
        if rec.mof_uri:
            coverage.field_present["mof_uri"] += 1
        if rec.chemical_formula is not None:
            coverage.field_present["chemical_formula"] += 1
        if rec.mofid is not None:
            coverage.field_present["mofid"] += 1
        if rec.topology is not None:
            coverage.field_present["topology"] += 1
        if rec.metal_cluster_elements is not None:
            coverage.field_present["metal_cluster_elements"] += 1
        if rec.linker_smiles is not None:
            coverage.field_present["linker_smiles"] += 1
        if rec.space_group is not None:
            coverage.field_present["space_group"] += 1
        if rec.crystal_system is not None:
            coverage.field_present["crystal_system"] += 1
        for col in rec.properties:
            coverage.field_present[col] += 1
    return records, coverage


__all__ = [
    "CORE_COLUMNS",
    "EXPECTED_ROW_COUNT",
    "KgAbstract",
    "KgCluster",
    "KgCsvCoverage",
    "KgLinker",
    "KgRecord",
    "KgSynthesis",
    "ReadKgCsv",
    "ReadKgCsvWithCoverage",
    "validate_kg_csv_header",
]
