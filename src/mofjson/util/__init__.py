"""Shared helpers with no domain logic.

Phase 01.1, extended as needed.
"""

from mofjson.util.diagnostics import (
    DIAGNOSTIC_CODES,
    Diagnostic,
    DiagnosticCollector,
    Severity,
)
from mofjson.util.ids import (
    ContentHash,
    MakeMofjsonId,
    MofidParts,
    ParseMofid,
    ShortId,
    content_hash,
    make_mofjson_id,
    parse_mofid_line,
    short_id,
)
from mofjson.util.io import (
    AtomicWrite,
    CountLines,
    ReadNdjson,
    ResumableWriter,
    WriteNdjson,
)
from mofjson.util.text import (
    DASH_TABLE,
    SUBSCRIPT_TABLE,
    SUPERSCRIPT_TABLE,
    CollapseWhitespace,
    NormalizeKey,
    NormalizeUnicode,
    collapse_whitespace,
    normalize_key,
    normalize_unicode,
)

__all__ = [
    "AtomicWrite",
    "CountLines",
    "DASH_TABLE",
    "DIAGNOSTIC_CODES",
    "SUBSCRIPT_TABLE",
    "SUPERSCRIPT_TABLE",
    "CollapseWhitespace",
    "ContentHash",
    "Diagnostic",
    "DiagnosticCollector",
    "MakeMofjsonId",
    "MofidParts",
    "NormalizeKey",
    "NormalizeUnicode",
    "ParseMofid",
    "ReadNdjson",
    "ResumableWriter",
    "Severity",
    "ShortId",
    "WriteNdjson",
    "collapse_whitespace",
    "content_hash",
    "make_mofjson_id",
    "normalize_key",
    "normalize_unicode",
    "parse_mofid_line",
    "short_id",
]
