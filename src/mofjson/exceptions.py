"""The exception hierarchy for the whole package.

Phase 01.1.

WHAT THIS FILE CONTAINS
    MOFJSONError                base of everything; callers can catch just this
    ├── SchemaError             a document violates the schema or a cross-layer invariant
    ├── ConversionError         CIF / pymatgen / ASE conversion failed
    │   └── LossyConversionError raised only when strict=True and information would be lost
    ├── DeconstructionError     Phase 08 could not partition or classify a structure
    │   ├── PartitionError        node/linker split failed or produced nonsense
    │   └── GeometryError         coordination analysis failed for a site
    ├── IngestError             Phase 09 could not read or reconcile an upstream record
    │   └── IdentifierError       an identifier could not be resolved to a structure
    └── LLMError                Phase 10 provider or repair-loop failure
        └── RepairFailedError     output still invalid after the repair budget

DESIGN RULES
    - Every exception carries structured context, not just a message: the mofjson_id or
      source path, the field path, and the offending value. Phase 09 processes ~250k
      records and a bare "ValueError: invalid" is useless at that scale.
    - Nothing in the corpus pipeline should raise on bad *data*. Bad data becomes a
      warning attached to the document's provenance (Phase 06.4); exceptions are for bad
      *code* and unusable inputs. This distinction is what lets Phase 09 run to
      completion over a noisy KG.
    - Include a `warnings`-style helper or a small `Diagnostic` dataclass here if it is
      needed by both schema and ingest, so neither has to import the other.

DONE WHEN
    The hierarchy exists and Phase 11 has a test asserting every custom exception
    subclasses MOFJSONError.
"""

from __future__ import annotations

from typing import Any


class MOFJSONError(Exception):
    """Base of every MOFJSON exception; catch this to handle the whole hierarchy."""

    def __init__(
        self,
        message: str,
        *,
        mofjson_id: str | None = None,
        source_path: str | None = None,
        field_path: str | None = None,
        value: Any = None,
    ) -> None:
        self.message = message
        self.mofjson_id = mofjson_id
        self.source_path = source_path
        self.field_path = field_path
        self.value = value
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.mofjson_id is not None:
            parts.append(f"mofjson_id={self.mofjson_id!r}")
        if self.source_path is not None:
            parts.append(f"source_path={self.source_path!r}")
        if self.field_path is not None:
            parts.append(f"field_path={self.field_path!r}")
        if self.value is not None:
            parts.append(f"value={self.value!r}")
        return "; ".join(parts)


class SchemaError(MOFJSONError):
    """A document violates the schema or a cross-layer invariant."""


class ConversionError(MOFJSONError):
    """CIF / pymatgen / ASE conversion failed."""


class LossyConversionError(ConversionError):
    """Raised only when strict=True and information would be lost."""


class DeconstructionError(MOFJSONError):
    """Phase 08 could not partition or classify a structure."""


class PartitionError(DeconstructionError):
    """Node/linker split failed or produced nonsense."""


class GeometryError(DeconstructionError):
    """Coordination analysis failed for a site."""


class IngestError(MOFJSONError):
    """Phase 09 could not read or reconcile an upstream record."""


class IdentifierError(IngestError):
    """An identifier could not be resolved to a structure."""


class LLMError(MOFJSONError):
    """Phase 10 provider or repair-loop failure."""


class RepairFailedError(LLMError):
    """Output still invalid after the repair budget."""


__all__ = [
    "ConversionError",
    "DeconstructionError",
    "GeometryError",
    "IdentifierError",
    "IngestError",
    "LLMError",
    "LossyConversionError",
    "MOFJSONError",
    "PartitionError",
    "RepairFailedError",
    "SchemaError",
]
