"""Single source of truth for the package and format version numbers.

Phase 01.1.

WHAT THIS FILE CONTAINS
    __version__         package version, read by pyproject via dynamic metadata
    MOFJSON_VERSION     the *format* version written into every document's
                        `mofjson_version` field

WHY TWO VERSIONS
    The package and the format change at different rates. A bug fix in the deconstruction
    pipeline bumps __version__ but must not bump MOFJSON_VERSION, because that would
    invalidate every previously emitted document. MOFJSON_VERSION changes only when the
    schema changes in a way that affects parsing (Phase 13.6 freezes it at "0.1").

    Every document carries MOFJSON_VERSION so that a reader can refuse or migrate old
    files. Phase 11.4 asserts the field is present and parseable on every document.

DONE WHEN
    Both constants exist and pyproject reads __version__ dynamically.
"""

__version__ = "0.1.0"

MOFJSON_VERSION = "0.1"
