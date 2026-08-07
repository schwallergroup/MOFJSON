"""MOFJSONDocument: the top-level object, and the validators that span layers.

Phase 06.6.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from mofjson.exceptions import SchemaError
from mofjson.schema.base import MOFJSONModel
from mofjson.schema.crystal import CrystalStructure
from mofjson.schema.metadata import (
    ConflictRecord,
    DocumentProvenance,
    ExtraProperty,
    Identifiers,
    Property,
)
from mofjson.util.diagnostics import Diagnostic, DiagnosticCollector, Severity
from mofjson.util.ids import make_mofjson_id
from mofjson.version import MOFJSON_VERSION


class MOFJSONDocument(MOFJSONModel):
    """Top-level MOFJSON document.

    Metadata-only documents are allowed (STAB/KG rows with no structure yet).
    Cross-layer structural checks emit diagnostics when both layers are present;
    full composition/ref validation lands with Phases 04/05.
    """

    mofjson_version: str = MOFJSON_VERSION
    identifiers: Identifiers = Field(default_factory=Identifiers)
    crystal: CrystalStructure | None = None
    reticular: Any | None = None
    properties: list[Property] = Field(default_factory=list)
    extra_properties: list[ExtraProperty] = Field(default_factory=list)
    provenance: DocumentProvenance = Field(default_factory=DocumentProvenance)
    warnings: list[str] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_invariants(self) -> MOFJSONDocument:
        for i, prop in enumerate(self.properties):
            if prop.provenance is None:  # pragma: no cover - type forbids this
                raise SchemaError(
                    "Every property must carry provenance",
                    field_path=f"properties[{i}].provenance",
                    value=None,
                )
        # Soft cross-layer checks when both layers present (thin until 04/05).
        diags = DiagnosticCollector()
        # Placeholder: real checks need typed CrystalStructure / ReticularStructure.
        _ = self.crystal is not None and self.reticular is not None
        # Use object.__setattr__ to avoid validate_assignment re-entering this validator.
        object.__setattr__(self, "diagnostics", [d.to_dict() for d in diags.to_list()])
        return self

    def ensure_mofjson_id(self) -> str:
        """Set identifiers.mofjson_id from structural layers if missing; return it."""
        current = self.identifiers.mofjson_id
        if current is not None:
            return current
        mid = make_mofjson_id(crystal=self.crystal, reticular=self.reticular)
        self.identifiers = self.identifiers.model_copy(update={"mofjson_id": mid})
        return mid

    def structural_payload(self) -> dict[str, Any]:
        """Payload used for content hashing (excludes provenance/timestamps)."""
        return {"crystal": self.crystal, "reticular": self.reticular}

    def collect_diagnostics(self) -> list[Diagnostic]:
        """Return document diagnostics as Diagnostic objects."""
        out: list[Diagnostic] = []
        for d in self.diagnostics:
            try:
                out.append(
                    Diagnostic(
                        code=d.get("code", "unregistered_code"),
                        severity=Severity(d.get("severity", "warning")),
                        message=d.get("message", ""),
                        field_path=d.get("field_path"),
                        value=d.get("value"),
                        stage=d.get("stage"),
                    )
                )
            except Exception:
                continue
        return out


__all__ = ["MOFJSONDocument"]
