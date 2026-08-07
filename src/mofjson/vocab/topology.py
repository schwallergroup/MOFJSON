"""RCSR net symbols and the connectivity signatures that imply them.

Phase 05.1 and Phase 12.3.

WHAT THIS FILE CONTAINS
    RCSR_NETS               the net symbols that actually occur in the corpus, taken from
                            the KG's `:topologyCode` distribution rather than invented:
                            pcu, fcu, sod, dia, rho, tbo, bcu, nbo, spn, lvt, ...
                            Store the observed frequency alongside each, so Phase 12.3 can
                            sample a realistic label space instead of a uniform one.
    NET_SIGNATURES          net -> (node connectivity, linker connectivity) pairs, e.g.
                            fcu -> (12, 2), pcu -> (6, 2), sod -> (4, 2), tbo -> (4, 3).
                            Used two ways: as a cross-check in Phase 08.6, and as the
                            ground truth for the `topology_from_connectivity` task in
                            Phase 12.3.
    UNAMBIGUOUS_SIGNATURES  the subset of NET_SIGNATURES where the mapping from
                            connectivity to net is one-to-one. Only these may be used to
                            generate benchmark questions -- asking a model to name the net
                            for a signature shared by three nets is an unanswerable
                            question, and a benchmark full of those measures nothing.
    TOPOLOGY_ALIASES        normalisation for the KG's variants: case differences,
                            whitespace, the `**`-suffixed and hyphenated forms, and the
                            sentinel strings that mean "no topology assigned".

CONTEXT FROM PHASE 00
    Topology codes in the KG are the single most reliable structural field available at
    scale, which makes them valuable -- but they are still dirty: sentinel values,
    inconsistent case, and the ToBaCCo-generated `SR_` names encode the net in the
    filename stem where it can be cross-checked against `:topologyCode`. That
    cross-check is Phase 09.7's job and disagreements must be recorded rather than
    silently resolved.

DONE WHEN
    Every distinct `:topologyCode` value in the KG maps to an RCSR_NETS member or to a
    recorded sentinel, with the residue counted in the Phase 09 audit.
"""

from __future__ import annotations

from types import MappingProxyType

from mofjson.vocab.sentinels import MOFID_SENTINELS

# Net symbol -> observed frequency in the KG's `:topologyCode` distribution (Phase 00.4).
# `None` means the net is known to occur in the corpus (named explicitly in the phase
# notes, including the hyphenated suffixes that must not be stripped) but a precise count
# was not part of the measured evidence -- Phase 09.7's audit fills these in from the full
# KG scan rather than this file guessing.
RCSR_NETS: MappingProxyType[str, int | None] = MappingProxyType(
    {
        "pcu": 2229,
        "sql": 1414,
        "fcu": None,
        "sod": None,
        "dia": None,
        "rho": None,
        "tbo": None,
        "bcu": None,
        "nbo": None,
        "spn": None,
        "lvt": None,
        "hmse": None,
        "tfz-d": None,
        "bcu-f": None,
        "npo-a": None,
        "cem-d": None,
        "llw-z": None,
    }
)

# net -> (node_connectivity, linker_connectivity). Only nets with a solid evidentiary basis
# (established RCSR/reticular-chemistry topology, or a signature directly measured in the
# local corpus) are listed -- guessing an unverified signature would corrupt the Phase 12.3
# benchmark's ground truth.
NET_SIGNATURES: MappingProxyType[str, tuple[int, int]] = MappingProxyType(
    {
        "pcu": (6, 2),
        "fcu": (12, 2),
        "sod": (4, 2),
        "dia": (4, 2),
        # HKUST-1: Cu2 paddlewheel (4-c) + BTC (3-c) on the tbo net.
        "tbo": (4, 3),
        "bcu": (8, 2),
        "nbo": (4, 2),
        # SR_hmse_v1-6c_..._v2-6c_Zr_1_Ch: both vertices are 6-connected (Phase 00.4).
        "hmse": (6, 6),
    }
)


def _build_unambiguous_signatures() -> MappingProxyType[tuple[int, int], str]:
    counts: dict[tuple[int, int], list[str]] = {}
    for net, sig in NET_SIGNATURES.items():
        counts.setdefault(sig, []).append(net)
    return MappingProxyType({sig: nets[0] for sig, nets in counts.items() if len(nets) == 1})


# The subset of NET_SIGNATURES that is injective (signature -> exactly one net). Only these
# may back a Phase 12.3 "name the net from this connectivity" question.
UNAMBIGUOUS_SIGNATURES: MappingProxyType[tuple[int, int], str] = _build_unambiguous_signatures()

# Sentinel tokens observed in the KG's MOFid topology field (Phase 00.4) and their KG-entity
# cousins (Phase 00.5's :Topology_error_unknown / :Topology_unstable / :Topology_unknown_sql /
# :Topology_sql_unknown). Every one maps to "no topology known", never to a string. Matched
# case-insensitively against a single (already comma-split) token. Sourced from the single
# canonical sentinel registry (vocab.sentinels.MOFID_SENTINELS, Phase 06.7) rather than a
# second hand-maintained list, so the two phases can never drift apart.
TOPOLOGY_ALIASES: MappingProxyType[str, None] = MappingProxyType(
    {token.lower(): None for token in MOFID_SENTINELS}
)

# How a Topology record's net was determined (Phase 05.1).
TOPOLOGY_SOURCES: tuple[str, ...] = (
    "mofid",
    "systre",
    "crystalnn",
    "kg",
    "literature",
    "llm",
    "manual",
)

# How much to trust it.
TOPOLOGY_CONFIDENCE: tuple[str, ...] = (
    "high",
    "medium",
    "low",
    "none",
)


__all__ = [
    "NET_SIGNATURES",
    "RCSR_NETS",
    "TOPOLOGY_ALIASES",
    "TOPOLOGY_CONFIDENCE",
    "TOPOLOGY_SOURCES",
    "UNAMBIGUOUS_SIGNATURES",
]
