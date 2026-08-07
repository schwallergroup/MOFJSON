"""Stream the full MOFology Turtle graph: 8.4 M triples, without loading it into memory.

Phase 09.2.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mofjson.exceptions import IngestError
from mofjson.ingest.kg_csv import (
    KgAbstract,
    KgCluster,
    KgLinker,
    KgRecord,
    KgSynthesis,
)

# Phase 00.2 subject-block counts by prefix (regression target).
EXPECTED_SUBJECT_BLOCK_COUNTS: dict[str, int] = {
    "PROP_": 606_993,
    "MOF_": 253_291,
    "StrainEnergy_": 158_616,
    "FreeEnergy_": 108_886,
    "LATTICE_": 20_523,
    "SYN_": 15_288,
    "ABSTRACT_": 15_143,
    "LINKER_": 5_169,
    "FuncMOF_": 2_650,
    "FUNC_": 2_650,
    "COND_": 2_349,
    "CLUSTER_": 1_388,
    "Topology_": 954,
    "SpaceGroup_": 365,
}

_PREFIX_RE = re.compile(
    r"@prefix\s+(\w*):?\s+<([^>]+)>\s*\.",
    re.I,
)
_BASE_RE = re.compile(r"@base\s+<([^>]+)>\s*\.", re.I)

# Predicates we care about when assembling MOF records
_HAS_LINKER = "hasLinker"
_USED_IN_MOF = "usedInMOF"
_HAS_METAL_NODE = "hasMetalNode"
_IS_COMPONENT_OF = "isComponentOf"
_HAS_METAL_ELEMENT = "hasMetalElement"
_HAS_TOPOLOGY = "hasTopology"
_TOPOLOGY_CODE = "topologyCode"
_HAS_MOFID = "hasMOFid"
_HAS_FORMULA = "hasFormula"
_HAS_CANONICAL_NAME = "hasCanonicalName"
_HAS_CSD = "hasCSDCode"
_HAS_MPID = "hasMPID"
_HAS_PROPERTY = "hasProperty"
_HAS_COMP_PROP = "hasComputationalProperty"
_HAS_STRUCT_PROP = "hasStructuralProperty"
_HAS_PHYS_PROP = "hasPhysicalProperty"
_HAS_ABSTRACT = "hasAbstract"
_HAS_SYN = "hasSynthesisProcess"
_DERIVED_FROM = "derivedFrom"
_TYPE = "type"


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    object_is_literal: bool = False
    datatype: str | None = None
    language: str | None = None


@dataclass
class SubjectBlock:
    """One subject with its predicates and objects."""

    subject: str
    triples: list[Triple] = field(default_factory=list)
    raw_text: str = ""

    def objects(self, predicate_local: str) -> list[str]:
        return [
            t.object
            for t in self.triples
            if _local(t.predicate) == predicate_local
        ]

    def literal(self, predicate_local: str) -> str | None:
        for t in self.triples:
            if _local(t.predicate) == predicate_local and t.object_is_literal:
                return t.object
        return None


def _local(term: str) -> str:
    if "#" in term:
        return term.rsplit("#", 1)[-1]
    if ":" in term and not term.startswith("http"):
        return term.split(":", 1)[-1]
    return term.rsplit("/", 1)[-1]


def _expand(term: str, prefixes: dict[str, str], base: str | None) -> str:
    term = term.strip()
    if term.startswith("<") and term.endswith(">"):
        return term[1:-1]
    if term.startswith("_:"):
        return term
    if term.startswith(":"):
        ns = prefixes.get("", prefixes.get("mof", "")) or (
            base or "http://emmo.info/domain-mof/mof-ontology#"
        )
        return ns + term[1:]
    if ":" in term:
        pref, local = term.split(":", 1)
        if pref in prefixes:
            return prefixes[pref] + local
    return term


class _TurtleLexer:
    """Accumulate subject blocks from pretty-printed Turtle."""

    def __init__(self, lines: Iterator[str]) -> None:
        self._lines = lines
        self.prefixes: dict[str, str] = {
            "": "http://emmo.info/domain-mof/mof-ontology#",
            "syn": "http://emmo.info/domain-mof/synthesis#",
            "emmo": "https://w3id.org/emmo#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }
        self.base: str | None = None

    def iter_raw_blocks(self) -> Iterator[str]:
        buf: list[str] = []
        in_literal = False
        literal_delim: str | None = None
        for line in self._lines:
            # Prefix / base directives
            stripped = line.strip()
            if not buf and not in_literal:
                if stripped.startswith("@prefix"):
                    m = _PREFIX_RE.match(stripped)
                    if m:
                        pref = m.group(1).rstrip(":")
                        self.prefixes[pref] = m.group(2)
                    continue
                if stripped.startswith("@base"):
                    m = _BASE_RE.match(stripped)
                    if m:
                        self.base = m.group(1)
                    continue
                if not stripped or stripped.startswith("#"):
                    continue

            # Track triple-quoted / quoted literals for depth
            i = 0
            while i < len(line):
                ch = line[i]
                if in_literal:
                    if literal_delim == '"""' or literal_delim == "'''":
                        if line[i : i + 3] == literal_delim:
                            in_literal = False
                            literal_delim = None
                            i += 3
                            continue
                    elif literal_delim in {'"', "'"}:
                        if ch == "\\" and i + 1 < len(line):
                            i += 2
                            continue
                        if ch == literal_delim:
                            in_literal = False
                            literal_delim = None
                    i += 1
                    continue
                # not in literal
                if line[i : i + 3] in ('"""', "'''"):
                    in_literal = True
                    literal_delim = line[i : i + 3]
                    i += 3
                    continue
                if ch in {'"', "'"}:
                    in_literal = True
                    literal_delim = ch
                    i += 1
                    continue
                i += 1

            buf.append(line)
            # Statement ends with '.' at depth zero (not in literal) at end of line-ish
            if not in_literal:
                s = "".join(buf)
                # Find terminating '.' outside of <iri> and not part of numbers awkwardly —
                # pretty Turtle puts the final '.' at end of the last predicate line.
                if _block_complete(s):
                    yield s
                    buf = []
        if buf:
            yield "".join(buf)


def _block_complete(text: str) -> bool:
    """True if *text* contains a statement-terminating '.' outside literals/IRIs."""
    in_lit = False
    lit_d: str | None = None
    in_iri = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_iri:
            if ch == ">":
                in_iri = False
            i += 1
            continue
        if in_lit:
            if lit_d in ('"""', "'''"):
                if text[i : i + 3] == lit_d:
                    in_lit = False
                    lit_d = None
                    i += 3
                    continue
            elif lit_d in {'"', "'"}:
                if ch == "\\" and i + 1 < n:
                    i += 2
                    continue
                if ch == lit_d:
                    in_lit = False
                    lit_d = None
            i += 1
            continue
        if text[i : i + 3] in ('"""', "'''"):
            in_lit = True
            lit_d = text[i : i + 3]
            i += 3
            continue
        if ch in {'"', "'"}:
            in_lit = True
            lit_d = ch
            i += 1
            continue
        if ch == "<":
            in_iri = True
            i += 1
            continue
        if ch == ".":
            # Terminating if rest is whitespace
            rest = text[i + 1 :]
            if rest.strip() == "":
                return True
        i += 1
    return False


def _parse_literal(token: str) -> tuple[str, bool, str | None, str | None]:
    """Return (value, is_literal, datatype, language)."""
    token = token.strip()
    if not token:
        return token, False, None, None
    # Triple-quoted
    for q in ('"""', "'''"):
        if token.startswith(q):
            end = token.find(q, 3)
            if end == -1:
                val = token[3:]
                rest = ""
            else:
                val = token[3:end]
                rest = token[end + 3 :]
            lang, dt = _parse_literal_suffix(rest)
            return val, True, dt, lang
    if token[0] in {'"', "'"}:
        q = token[0]
        i = 1
        chars: list[str] = []
        while i < len(token):
            if token[i] == "\\" and i + 1 < len(token):
                chars.append(token[i + 1])
                i += 2
                continue
            if token[i] == q:
                rest = token[i + 1 :]
                lang, dt = _parse_literal_suffix(rest)
                return "".join(chars), True, dt, lang
            chars.append(token[i])
            i += 1
        return "".join(chars), True, None, None
    # Numeric / boolean bare literals
    if re.fullmatch(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", token):
        return token, True, None, None
    if token in {"true", "false"}:
        return token, True, None, None
    return token, False, None, None


def _parse_literal_suffix(rest: str) -> tuple[str | None, str | None]:
    rest = rest.strip()
    if rest.startswith("@"):
        lang = rest[1:].split("^^")[0].strip()
        return lang, None
    if rest.startswith("^^"):
        return None, rest[2:].strip()
    return None, None


def _tokenize_objects(obj_text: str) -> list[str]:
    """Split a comma-separated object list, respecting quotes and IRIs."""
    objs: list[str] = []
    buf: list[str] = []
    in_lit = False
    lit_d: str | None = None
    in_iri = False
    i = 0
    text = obj_text.strip()
    while i < len(text):
        ch = text[i]
        if in_iri:
            buf.append(ch)
            if ch == ">":
                in_iri = False
            i += 1
            continue
        if in_lit:
            buf.append(ch)
            if lit_d in ('"""', "'''"):
                if text[i : i + 3] == lit_d and len(buf) >= 3:
                    # closing already partially appended? handle carefully
                    pass
                if text[i : i + 3] == lit_d:
                    # we already appended first char of delimiter
                    if lit_d[0] == ch and text[i : i + 3] == lit_d:
                        buf.extend(list(lit_d[1:]))
                        i += 3
                        in_lit = False
                        lit_d = None
                        continue
            elif lit_d in {'"', "'"}:
                if ch == "\\" and i + 1 < len(text):
                    buf.append(text[i + 1])
                    i += 2
                    continue
                if ch == lit_d:
                    in_lit = False
                    lit_d = None
            i += 1
            continue
        if text[i : i + 3] in ('"""', "'''"):
            in_lit = True
            lit_d = text[i : i + 3]
            buf.append(lit_d)
            i += 3
            continue
        if ch in {'"', "'"}:
            in_lit = True
            lit_d = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "<":
            in_iri = True
            buf.append(ch)
            i += 1
            continue
        if ch == ",":
            objs.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        objs.append("".join(buf).strip())
    return [o for o in objs if o]


def _parse_block(raw: str, prefixes: dict[str, str], base: str | None) -> SubjectBlock | None:
    text = raw.strip()
    if not text or text.startswith("@") or text.startswith("#"):
        return None
    # Strip terminating '.'
    if text.endswith("."):
        text = text[:-1].rstrip()

    # Subject is the first token
    # Pretty Turtle: subject at column 0 of first line
    lines = text.splitlines()
    if not lines:
        return None
    first = lines[0].strip()
    # subject [predicate objects ; ...]
    # Find subject token
    m = re.match(r"^(\S+)\s+(.*)$", first, re.DOTALL)
    if not m:
        # subject alone then predicates on following lines
        subject_tok = first
        rest = "\n".join(lines[1:])
    else:
        subject_tok = m.group(1)
        rest = m.group(2) + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")

    subject = _expand(subject_tok, prefixes, base)
    block = SubjectBlock(subject=subject, raw_text=raw)

    # Split predicates on ';' at depth 0
    pred_chunks = _split_top_level(rest, ";")
    for chunk in pred_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        pm = re.match(r"^(\S+)\s+(.*)$", chunk, re.DOTALL)
        if not pm:
            continue
        pred_tok, obj_text = pm.group(1), pm.group(2)
        predicate = _expand(pred_tok, prefixes, base)
        for obj_tok in _tokenize_objects(obj_text):
            val, is_lit, dt, lang = _parse_literal(obj_tok)
            if not is_lit:
                val = _expand(obj_tok, prefixes, base)
            block.triples.append(
                Triple(
                    subject=subject,
                    predicate=predicate,
                    object=val,
                    object_is_literal=is_lit,
                    datatype=dt,
                    language=lang,
                )
            )
    return block


def _split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    in_lit = False
    lit_d: str | None = None
    in_iri = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_iri:
            buf.append(ch)
            if ch == ">":
                in_iri = False
            i += 1
            continue
        if in_lit:
            buf.append(ch)
            if lit_d in ('"""', "'''"):
                if text[i : i + 3] == lit_d:
                    buf.extend(list(lit_d[1:]))
                    i += 3
                    in_lit = False
                    lit_d = None
                    continue
            elif lit_d in {'"', "'"}:
                if ch == "\\" and i + 1 < len(text):
                    buf.append(text[i + 1])
                    i += 2
                    continue
                if ch == lit_d:
                    in_lit = False
                    lit_d = None
            i += 1
            continue
        if text[i : i + 3] in ('"""', "'''"):
            in_lit = True
            lit_d = text[i : i + 3]
            buf.append(lit_d)
            i += 3
            continue
        if ch in {'"', "'"}:
            in_lit = True
            lit_d = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "<":
            in_iri = True
            buf.append(ch)
            i += 1
            continue
        if ch == sep:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))
    return parts


def IterSubjectBlocks(path: str | Path) -> Iterator[SubjectBlock]:
    """Yield one SubjectBlock per subject from a Turtle file."""
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"KG TTL not found: {p}", source_path=str(p))

    def _line_iter() -> Iterator[str]:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            yield from fh

    lexer = _TurtleLexer(_line_iter())
    for raw in lexer.iter_raw_blocks():
        block = _parse_block(raw, lexer.prefixes, lexer.base)
        if block is not None and block.triples:
            yield block


def CountSubjectBlocks(path: str | Path) -> dict[str, int]:
    """Count subject blocks by local-name prefix (Phase 00.2 regression)."""
    counts: Counter[str] = Counter()
    for block in IterSubjectBlocks(path):
        local = _local(block.subject)
        matched = False
        for prefix in EXPECTED_SUBJECT_BLOCK_COUNTS:
            if local.startswith(prefix):
                counts[prefix] += 1
                matched = True
                break
        if not matched:
            counts["_other"] += 1
    return dict(counts)


def ReadKgTtl(
    path: str | Path,
    *,
    limit: int | None = None,
) -> Iterator[KgRecord]:
    """Stream MOF KgRecords from Turtle, including relational content.

    Two-pass over the stream is avoided by buffering linker/cluster/property
    side tables while collecting MOF subjects, then emitting MOF records.
    For the full 513 MB file this keeps only the relational side maps in memory
    (linkers ~5k, clusters ~1k, syn ~15k) plus MOF stubs — not the full triple store.
    """
    p = Path(path)
    # Side tables
    linker_by_uri: dict[str, KgLinker] = {}
    cluster_by_uri: dict[str, KgCluster] = {}
    mof_linkers: dict[str, list[str]] = defaultdict(list)
    mof_clusters: dict[str, list[str]] = defaultdict(list)
    mof_props: dict[str, dict[str, Any]] = defaultdict(dict)
    prop_values: dict[str, tuple[str | None, Any, str | None]] = {}
    abstracts: dict[str, KgAbstract] = {}
    mof_abstracts: dict[str, list[str]] = defaultdict(list)
    syn_by_uri: dict[str, KgSynthesis] = {}
    mof_syn: dict[str, str] = {}
    mof_fields: dict[str, dict[str, Any]] = defaultdict(dict)
    mof_types: dict[str, list[str]] = defaultdict(list)
    topology_code: dict[str, str] = {}

    n_mof_seen = 0
    for block in IterSubjectBlocks(p):
        local = _local(block.subject)
        types = [_local(o) for o in block.objects(_TYPE)]

        if local.startswith("LINKER_") or "OrganicLinker" in types:
            linker = linker_by_uri.setdefault(block.subject, KgLinker(uri=block.subject))
            smi = block.literal("hasCanonicalSMILES")
            if smi:
                linker.smiles = smi
            name = block.literal("hasChemicalName")
            if name:
                linker.chemical_name = name
            for mof in block.objects(_USED_IN_MOF):
                mof_linkers[mof].append(block.subject)
            continue

        if local.startswith("CLUSTER_") or "MetalCluster" in types:
            cluster = cluster_by_uri.setdefault(block.subject, KgCluster(uri=block.subject))
            el = block.literal(_HAS_METAL_ELEMENT)
            if el:
                cluster.metal_element = el
            for mof in block.objects(_IS_COMPONENT_OF):
                mof_clusters[mof].append(block.subject)
            continue

        if local.startswith("Topology_") or "Topology" in types:
            code = block.literal(_TOPOLOGY_CODE)
            if code:
                topology_code[block.subject] = code
            continue

        if local.startswith("ABSTRACT_") or "Abstract" in types or "Publication" in types:
            abs_ = abstracts.setdefault(block.subject, KgAbstract(uri=block.subject))
            title = block.literal("publicationTitle") or block.literal("hasTitle")
            abstract = block.literal("publicationAbstract") or block.literal("hasAbstract")
            if title:
                abs_.title = title
            if abstract:
                abs_.abstract = abstract
            continue

        if local.startswith("SYN_") or "SynthesisProcess" in types:
            syn = syn_by_uri.setdefault(block.subject, KgSynthesis(uri=block.subject))
            for m in block.objects("hasSynthesisMethod") + [
                x for x in [block.literal("hasSynthesisMethod")] if x
            ]:
                if m and m not in syn.methods:
                    syn.methods.append(_local(m) if not m[0].isdigit() and " " not in m else m)
            lit_method = block.literal("hasSynthesisMethod")
            if lit_method and lit_method not in syn.methods:
                syn.methods.append(lit_method)
            t = block.literal("hasTemperature")
            if t is not None:
                try:
                    syn.temperature = float(t)
                except ValueError:
                    pass
            hyp = block.literal("isHypothetical")
            if hyp is not None:
                syn.is_hypothetical = str(hyp).lower() in {"true", "1"}
            for sol in block.objects("hasSolvent"):
                syn.solvents.append({"uri": sol, "name": _local(sol)})
            continue

        if local.startswith(("PROP_", "StrainEnergy_", "FreeEnergy_", "LATTICE_", "SpaceGroup_")):
            pname = block.literal("propertyName")
            pval = block.literal("propertyValue")
            punit = block.literal("propertyUnits") or block.literal("propertyUnit")
            if pval is not None:
                try:
                    if "." in str(pval) or "e" in str(pval).lower():
                        coerced: Any = float(pval)
                    else:
                        coerced = int(pval)
                except ValueError:
                    coerced = pval
            else:
                coerced = None
            prop_values[block.subject] = (pname, coerced, punit)
            # Inverse owner
            for owner in block.objects("hasStructuralPropertyOwner") + block.objects(
                "hasPropertyOwner"
            ) + block.objects("hasComputationalPropertyOwner") + block.objects(
                "hasPhysicalPropertyOwner"
            ):
                if pname:
                    mof_props[owner][pname] = coerced
            continue

        if local.startswith(("MOF_", "FuncMOF_")) or "MOF" in types or "HypotheticalMOF" in types or "ExperimentalMOF" in types:
            fields = mof_fields[block.subject]
            for pred, attr in [
                (_HAS_MOFID, "mofid"),
                (_HAS_FORMULA, "chemical_formula"),
                (_HAS_CANONICAL_NAME, "canonical_name"),
                (_HAS_CSD, "csd_code"),
                (_HAS_MPID, "mp_id"),
            ]:
                lit = block.literal(pred)
                if lit:
                    fields[attr] = lit
            for topo in block.objects(_HAS_TOPOLOGY):
                fields["topology_uri"] = topo
                if topo in topology_code:
                    fields["topology"] = topology_code[topo]
                else:
                    # sometimes topology code is on the MOF itself
                    code = block.literal(_TOPOLOGY_CODE)
                    if code:
                        fields["topology"] = code
            code = block.literal(_TOPOLOGY_CODE)
            if code:
                fields["topology"] = code
            for linker in block.objects(_HAS_LINKER):
                mof_linkers[block.subject].append(linker)
            for cluster in block.objects(_HAS_METAL_NODE):
                mof_clusters[block.subject].append(cluster)
            for prop_uri in (
                block.objects(_HAS_PROPERTY)
                + block.objects(_HAS_COMP_PROP)
                + block.objects(_HAS_STRUCT_PROP)
                + block.objects(_HAS_PHYS_PROP)
            ):
                if prop_uri in prop_values:
                    pname, pval, _punit = prop_values[prop_uri]
                    if pname:
                        mof_props[block.subject][pname] = pval
            for abs_uri in block.objects(_HAS_ABSTRACT):
                mof_abstracts[block.subject].append(abs_uri)
            for syn_uri in block.objects(_HAS_SYN):
                mof_syn[block.subject] = syn_uri
            for parent in block.objects(_DERIVED_FROM):
                fields["parent_mof_uri"] = parent
            mof_types[block.subject].extend(types)
            n_mof_seen += 1

    # Emit MOF records
    emitted = 0
    for uri, fields in mof_fields.items():
        if limit is not None and emitted >= limit:
            return
        linkers = [
            linker_by_uri.get(lu, KgLinker(uri=lu))
            for lu in dict.fromkeys(mof_linkers.get(uri, []))
        ]
        clusters = [
            cluster_by_uri.get(cu, KgCluster(uri=cu))
            for cu in dict.fromkeys(mof_clusters.get(uri, []))
        ]
        # metal_cluster_elements string for normalize compatibility
        metal_str = None
        els = [c.metal_element for c in clusters if c.metal_element]
        if els:
            metal_str = "; ".join(els)

        types = mof_types.get(uri, [])
        is_hyp = "HypotheticalMOF" in types
        is_exp = "ExperimentalMOF" in types
        is_func = "FunctionalizedMOF" in types or _local(uri).startswith("FuncMOF_")

        syn = syn_by_uri.get(mof_syn[uri]) if uri in mof_syn else None
        abs_list = [
            abstracts[a] for a in mof_abstracts.get(uri, []) if a in abstracts
        ]

        # topology may have been filled after MOF block if Topology_ came later —
        # second chance via uri
        topo = fields.get("topology")
        topo_uri = fields.get("topology_uri")
        if topo is None and topo_uri and topo_uri in topology_code:
            topo = topology_code[topo_uri]

        yield KgRecord(
            mof_uri=uri,
            chemical_formula=fields.get("chemical_formula"),
            mofid=fields.get("mofid"),
            topology=topo,
            metal_cluster_elements=metal_str,
            linker_smiles=linkers[0].smiles if linkers else None,
            space_group=None,
            crystal_system=None,
            properties=dict(mof_props.get(uri, {})),
            linkers=linkers,
            clusters=clusters,
            synthesis=syn,
            abstracts=abs_list,
            canonical_name=fields.get("canonical_name"),
            types=types,
            is_experimental=is_exp or None,
            is_hypothetical=is_hyp or None,
            is_functionalized=is_func or None,
            parent_mof_uri=fields.get("parent_mof_uri"),
            source="ttl",
        )
        emitted += 1


__all__ = [
    "EXPECTED_SUBJECT_BLOCK_COUNTS",
    "CountSubjectBlocks",
    "IterSubjectBlocks",
    "ReadKgTtl",
    "SubjectBlock",
    "Triple",
]
