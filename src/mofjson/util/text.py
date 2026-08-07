"""Unicode and whitespace normalisation for upstream text fields.

Phase 06.3 / 09.5.

PUBLIC API
    NormalizeUnicode(s) -> str
    DASH_TABLE, SUPERSCRIPT_TABLE, SUBSCRIPT_TABLE
    CollapseWhitespace(s) -> str
    NormalizeKey(s) -> str        # aggressive: lowercase, strip, collapse, for alias lookup

RULES
    - NormalizeUnicode is conservative (fit for storage).
    - NormalizeKey is aggressive (lookup only; never store).
    - Order: NFKC -> custom tables -> whitespace collapse.
"""

from __future__ import annotations

import re
import unicodedata
from types import MappingProxyType

# Hyphen / minus / dash variants -> ASCII hyphen-minus for storage normalisation.
# Density traps from Phase 00.6: en dash, superscript minus, Unicode minus.
DASH_TABLE: MappingProxyType[str, str] = MappingProxyType(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
        "\ufe58": "-",  # small em dash
        "\ufe63": "-",  # small hyphen-minus
        "\uff0d": "-",  # fullwidth hyphen-minus
        "\u00ad": "",  # soft hyphen -> drop
        "\u00b7": ".",  # middle dot (mol/(kg·Pa)) -> period for storage
        "\u2022": ".",  # bullet
        "\u22c5": ".",  # dot operator
    }
)

SUPERSCRIPT_TABLE: MappingProxyType[str, str] = MappingProxyType(
    {
        "\u00b2": "2",  # ²
        "\u00b3": "3",  # ³
        "\u2070": "0",
        "\u00b9": "1",
        "\u2074": "4",
        "\u2075": "5",
        "\u2076": "6",
        "\u2077": "7",
        "\u2078": "8",
        "\u2079": "9",
        "\u207a": "+",
        "\u207b": "-",  # superscript minus
        "\u207c": "=",
        "\u207d": "(",
        "\u207e": ")",
        "\u2071": "i",
        "\u207f": "n",
    }
)

SUBSCRIPT_TABLE: MappingProxyType[str, str] = MappingProxyType(
    {
        "\u2080": "0",
        "\u2081": "1",
        "\u2082": "2",
        "\u2083": "3",
        "\u2084": "4",
        "\u2085": "5",
        "\u2086": "6",
        "\u2087": "7",
        "\u2088": "8",
        "\u2089": "9",
        "\u208a": "+",
        "\u208b": "-",
        "\u208c": "=",
        "\u208d": "(",
        "\u208e": ")",
    }
)

_WS_RE = re.compile(r"[\s\u00a0\u2000-\u200b\u202f\u205f\u3000]+")


def _apply_table(s: str, table: MappingProxyType[str, str]) -> str:
    if not s:
        return s
    return "".join(table.get(ch, ch) for ch in s)


def CollapseWhitespace(s: str) -> str:
    """Collapse all whitespace (including NBSP) to a single ASCII space and strip."""
    return _WS_RE.sub(" ", s).strip()


def NormalizeUnicode(s: str) -> str:
    """Conservative normalisation fit for storage.

    Order: NFKC -> dash/minus table -> superscript/subscript tables -> whitespace.
    """
    if s is None:
        raise TypeError("NormalizeUnicode requires a string")
    if not isinstance(s, str):
        s = str(s)
    out = unicodedata.normalize("NFKC", s)
    out = _apply_table(out, DASH_TABLE)
    out = _apply_table(out, SUPERSCRIPT_TABLE)
    out = _apply_table(out, SUBSCRIPT_TABLE)
    return CollapseWhitespace(out)


def NormalizeKey(s: str) -> str:
    """Aggressive lookup key: NormalizeUnicode then lowercase.

    Never store the result — it is lossy by design.
    """
    return NormalizeUnicode(s).lower()


# Snake_case aliases matching typical Python call sites in this package.
normalize_unicode = NormalizeUnicode
normalize_key = NormalizeKey
collapse_whitespace = CollapseWhitespace

__all__ = [
    "DASH_TABLE",
    "SUBSCRIPT_TABLE",
    "SUPERSCRIPT_TABLE",
    "CollapseWhitespace",
    "NormalizeKey",
    "NormalizeUnicode",
    "collapse_whitespace",
    "normalize_key",
    "normalize_unicode",
]
