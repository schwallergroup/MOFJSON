"""Family D -- editing: apply a structural edit to a given document. Stretch goal.

Phase 12.5.

STATUS
    Deferred. Comparison tolerance for legitimate atom-ordering / id-naming freedom
    was not settled cleanly enough to ship an interpretable metric alongside Families
    A–C. Report editing separately if/when this family lands; do not fold scores into
    the main structure-suite tables.

PUBLIC API (placeholders)
    EDIT_TYPES, ApplyEdit, GenerateEditingItems — not implemented.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from mofjson.schema.document import MOFJSONDocument
from mofjson.tasks.item import TaskItem

EDIT_TYPES: dict[str, Any] = {}


def ApplyEdit(document: MOFJSONDocument, edit: Any) -> MOFJSONDocument:
    raise NotImplementedError(
        "phase-12.5: Family D (editing) is deferred — see module docstring"
    )


def GenerateEditingItems(
    documents: Sequence[MOFJSONDocument],
    edit_types: Sequence[str] | None = None,
) -> Iterator[TaskItem]:
    _ = documents, edit_types
    raise NotImplementedError(
        "phase-12.5: Family D (editing) is deferred — see module docstring"
    )
    yield  # pragma: no cover


__all__ = ["EDIT_TYPES", "ApplyEdit", "GenerateEditingItems"]
