"""Command-line entry point. One `mofjson` command, one subcommand per pipeline stage.

Phase 01.6, then filled in by each later phase.

WHAT THIS FILE CONTAINS
    An argparse (or typer, if added as a dependency) parser with subcommands, each a thin
    wrapper around a library function. No business logic lives here -- if a subcommand
    needs more than ~15 lines, the logic belongs in the module it calls.

SUBCOMMANDS AND OWNING PHASE
    mofjson from-cif <path> [-o out.json] [--profile P]        Phase 07.1
    mofjson to-cif <path> [-o out.cif]                          Phase 07.2
    mofjson deconstruct <path> [--no-geometry]                   Phase 08.8
    mofjson ingest-kg <ttl|csv> [--limit N] [-o corpus.jsonl]    Phase 09
    mofjson schema [--layer L] [--profile P] [--dialect D]       Phase 06 / 10.1
    mofjson validate <path> [--strict]                           Phase 11
    mofjson tasks build [--families ...] [--split dev|test]      Phase 12.8
    mofjson tasks grade <items.jsonl> <responses.jsonl>          Phase 12.9
    mofjson tokens <path> [--tokeniser NAME]                     Phase 13.1

CONVENTIONS
    - Every subcommand accepts `-` for stdin/stdout so the tools compose in a shell.
    - Corpus-scale subcommands (`ingest-kg`, `deconstruct` on a directory) stream NDJSON
      and never build a full list in memory; Phase 09 handles ~250k records.
    - Exit codes: 0 success, 1 usage error, 2 validation failure. `validate` returning a
      distinct code is what makes it usable in CI.
    - Register the entry point in pyproject as `mofjson = "mofjson.cli:main"`.

INITIAL STATE
    Every subcommand parses its arguments and raises NotImplementedError with the phase
    number in the message. This makes `mofjson --help` a live checklist of remaining work.

DONE WHEN
    `mofjson --help` lists every subcommand above and each stub raises with its phase
    number.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _cmd_from_cif(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from mofjson.convert.cif import MOFFromCif

    if args.path == "-":
        text = sys.stdin.read()
        doc = MOFFromCif(text)
    else:
        doc = MOFFromCif(Path(args.path))
    payload = doc.model_dump(mode="json", exclude_none=True)
    text_out = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(text_out)
    else:
        Path(args.output).write_text(text_out, encoding="utf-8")
    return 0


def _cmd_to_cif(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from mofjson.convert.cif import MOFToCif
    from mofjson.schema.document import MOFJSONDocument

    if args.path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(args.path).read_text(encoding="utf-8")
    doc = MOFJSONDocument.model_validate(json.loads(raw))
    cif = MOFToCif(doc)
    if args.output == "-":
        sys.stdout.write(cif)
    else:
        Path(args.output).write_text(cif, encoding="utf-8")
    return 0


def _cmd_deconstruct(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from mofjson.convert.cif import MOFFromCif
    from mofjson.deconstruct import Deconstruct
    from mofjson.schema.document import MOFJSONDocument

    if args.path == "-":
        raw = sys.stdin.read().lstrip()
        if raw.startswith("{"):
            doc = MOFJSONDocument.model_validate(json.loads(raw))
        else:
            doc = MOFFromCif(raw, derive_bonds=True)
    else:
        p = Path(args.path)
        if p.suffix.lower() == ".json":
            doc = MOFJSONDocument.model_validate(json.loads(p.read_text(encoding="utf-8")))
        else:
            doc = MOFFromCif(p, derive_bonds=True)

    doc = Deconstruct(doc, geometry=not args.no_geometry)
    text_out = json.dumps(doc.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True) + "\n"
    out = getattr(args, "output", "-")
    if out == "-" or out is None:
        sys.stdout.write(text_out)
    else:
        Path(out).write_text(text_out, encoding="utf-8")
    return 0


def _cmd_ingest_kg(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from mofjson.ingest.corpus import BuildCorpus
    from mofjson.ingest.manifest import LoadKgStructureManifest, Manifest

    kg_path = Path(args.path)
    out = args.output
    manifest_path = getattr(args, "manifest", None)
    if manifest_path:
        man: Manifest | Path = Path(manifest_path)
    else:
        default_csv = Path("/data/matthart/mof_structures/manifest.csv")
        if default_csv.is_file():
            man = LoadKgStructureManifest(default_csv)
        else:
            man = Manifest()

    if out == "-":
        # Stream a limited metadata-only preview to stdout
        from mofjson.ingest.identifiers import ResolveIdentifiers
        from mofjson.ingest.kg_csv import ReadKgCsv
        from mofjson.ingest.kg_ttl import ReadKgTtl
        from mofjson.ingest.normalize import NormalizeRecord
        from mofjson.schema.document import MOFJSONDocument
        from mofjson.schema.metadata import DocumentProvenance, SourceRef

        kind = "csv" if kg_path.suffix.lower() == ".csv" else "ttl"
        it = ReadKgCsv(kg_path, limit=args.limit) if kind == "csv" else ReadKgTtl(
            kg_path, limit=args.limit
        )
        n = 0
        for raw in it:
            normalized, _warns = NormalizeRecord(raw)
            resolved = ResolveIdentifiers(
                normalized, man if isinstance(man, Manifest) else None
            )
            doc = MOFJSONDocument(
                identifiers=resolved.identifiers,
                properties=list(normalized.properties),
                extra_properties=list(normalized.extra_properties),
                warnings=list(normalized.warnings),
                provenance=DocumentProvenance(
                    sources=[SourceRef(name="MOFology-KG", kind="metadata")]
                ),
            )
            sys.stdout.write(
                json.dumps(doc.model_dump(mode="json", exclude_none=True), sort_keys=True)
                + "\n"
            )
            n += 1
            if args.limit is not None and n >= args.limit:
                break
        return 0

    report = BuildCorpus(
        kg_source=kg_path,
        manifest=man,
        out=out,
        limit=args.limit,
        deconstruct=not getattr(args, "no_deconstruct", False),
        geometry=bool(getattr(args, "geometry", False)),
        resume=bool(getattr(args, "resume", True)),
    )
    sys.stderr.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


_SCHEMA_LAYERS = frozenset(
    {
        "metal_node",
        "crystal",
        "reticular",
        "identifiers",
        "property",
        "provenance",
        "property_provenance",
        "document",
        "metadata",
    }
)


def _cmd_schema(args: argparse.Namespace) -> int:
    import json

    from mofjson.llm.profiles import PROFILES
    from mofjson.schema.jsonschema import GetSchema

    if args.profile is not None and args.profile not in PROFILES:
        raise SystemExit(
            f"unknown profile {args.profile!r}; known: {sorted(PROFILES)}"
        )
    layer = args.layer
    if (
        args.profile is None
        and layer is not None
        and layer not in _SCHEMA_LAYERS
    ):
        raise NotImplementedError(f"phase-06/10.1: schema layer {layer!r} not implemented yet")
    schema = GetSchema(layer=layer, profile=args.profile, dialect=args.dialect)
    sys.stdout.write(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a MOFJSON document or a profile-shaped object. Exit 0 / 2."""
    import json
    from pathlib import Path

    from pydantic import ValidationError

    from mofjson.exceptions import SchemaError
    from mofjson.schema.document import MOFJSONDocument
    from mofjson.vocab.sentinels import scan_for_sentinels

    raw = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"invalid JSON: {exc}\n")
        return 2

    profile = getattr(args, "profile", None)
    errors: list[str] = []
    if profile:
        from mofjson.llm.profiles import PROFILES
        from mofjson.llm.repair import ValidateAgainstProfile

        if profile not in PROFILES:
            sys.stderr.write(
                f"unknown profile {profile!r}; known: {sorted(PROFILES)}\n"
            )
            return 1
        for diag in ValidateAgainstProfile(obj, profile):
            if diag.severity.value == "error" or args.strict:
                errors.append(f"{diag.code}: {diag.message}" + (
                    f" ({diag.field_path})" if diag.field_path else ""
                ))
    else:
        try:
            doc = MOFJSONDocument.model_validate(obj)
            payload = doc.model_dump(mode="json", exclude_none=True)
        except (ValidationError, SchemaError) as exc:
            sys.stderr.write(f"validation failed: {exc}\n")
            return 2
        if args.strict:
            for path, value in scan_for_sentinels(payload):
                errors.append(f"sentinel at {path}: {value!r}")
            if getattr(doc, "warnings", None):
                for w in doc.warnings:
                    errors.append(f"warning: {w}")

    if errors:
        for line in errors:
            sys.stderr.write(line + "\n")
        return 2
    sys.stdout.write("ok\n")
    return 0


def _cmd_tasks_build(args: argparse.Namespace) -> int:
    import json

    from mofjson.tasks.build import BuildTaskSuite

    corpus = getattr(args, "corpus", None) or "corpus"
    out_dir = getattr(args, "out", None) or "tasks"
    seed = int(getattr(args, "seed", 0) or 0)
    freeze = bool(getattr(args, "freeze", False))
    families = args.families
    report = BuildTaskSuite(
        corpus,
        families=families,
        out_dir=out_dir,
        seed=seed,
        freeze=freeze,
        append_audit_to_phase_notes=freeze,
    )
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


def _cmd_tasks_grade(args: argparse.Namespace) -> int:
    from mofjson.tasks.grade import GradeRun

    report = GradeRun(args.items, args.responses)
    out = getattr(args, "out", None)
    table = report.table()
    if out:
        report.save(out)
    sys.stdout.write(table.to_json())
    return 0


def _cmd_tokens(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from mofjson.schema.document import MOFJSONDocument
    from mofjson.util.tokens import CountDocumentTokens, CountTokens

    raw = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
    tokeniser = args.tokeniser or "cl100k_base"
    # Accept plain text (no JSON) for ad-hoc counting; otherwise treat as MOFJSON.
    text = raw.strip()
    if text.startswith("{") or text.startswith("["):
        try:
            doc = MOFJSONDocument.model_validate(json.loads(raw))
            count = CountDocumentTokens(doc, tokeniser=tokeniser)
        except Exception:
            count = CountTokens(raw, tokeniser=tokeniser)
    else:
        count = CountTokens(raw, tokeniser=tokeniser)
    sys.stdout.write(json.dumps({"tokeniser": tokeniser, "tokens": count}) + "\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mofjson",
        description="MOFJSON — JSON representation of metal-organic frameworks for LLMs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_from = sub.add_parser("from-cif", help="Convert a CIF to MOFJSON (Phase 07.1)")
    p_from.add_argument("path", help="Input CIF path, or - for stdin")
    p_from.add_argument("-o", "--output", default="-", help="Output JSON path, or - for stdout")
    p_from.add_argument("--profile", default=None, help="LLM answer schema profile")
    p_from.set_defaults(func=_cmd_from_cif)

    p_to = sub.add_parser("to-cif", help="Convert MOFJSON to a CIF (Phase 07.2)")
    p_to.add_argument("path", help="Input MOFJSON path, or - for stdin")
    p_to.add_argument("-o", "--output", default="-", help="Output CIF path, or - for stdout")
    p_to.set_defaults(func=_cmd_to_cif)

    p_de = sub.add_parser(
        "deconstruct",
        help="Deconstruct a structure into nodes/linkers (Phase 08.8)",
    )
    p_de.add_argument("path", help="Input CIF or MOFJSON path, or - for stdin")
    p_de.add_argument("-o", "--output", default="-", help="Output JSON path, or - for stdout")
    p_de.add_argument("--no-geometry", action="store_true", help="Skip coordination geometry")
    p_de.set_defaults(func=_cmd_deconstruct)

    p_kg = sub.add_parser("ingest-kg", help="Ingest the MOFology KG (Phase 09)")
    p_kg.add_argument("path", help="Input TTL or CSV path")
    p_kg.add_argument("--limit", type=int, default=None, help="Max records to ingest")
    p_kg.add_argument("-o", "--output", default="-", help="Output NDJSON path, or - for stdout")
    p_kg.add_argument("--manifest", default=None, help="Cached manifest.json path")
    p_kg.add_argument("--geometry", action="store_true", help="Enable ChemEnv geometry")
    p_kg.add_argument("--no-deconstruct", action="store_true", help="Skip deconstruction")
    p_kg.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip records already in the output",
    )
    p_kg.set_defaults(func=_cmd_ingest_kg)

    p_schema = sub.add_parser(
        "schema",
        help="Emit a JSON Schema layer or profile (Phase 06 / 10.1)",
    )
    p_schema.add_argument("--layer", default=None, help="Schema layer to emit")
    p_schema.add_argument("--profile", default=None, help="LLM answer schema profile")
    p_schema.add_argument("--dialect", default="jsonschema", help="Schema dialect")
    p_schema.set_defaults(func=_cmd_schema)

    p_val = sub.add_parser("validate", help="Validate a MOFJSON document (Phase 11)")
    p_val.add_argument("path", help="Input MOFJSON path, or - for stdin")
    p_val.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    p_val.add_argument("--profile", default=None, help="Validate against an LLM profile")
    p_val.set_defaults(func=_cmd_validate)

    p_tasks = sub.add_parser("tasks", help="Structure task suite (Phase 12)")
    tasks_sub = p_tasks.add_subparsers(dest="tasks_command", required=True)

    p_build = tasks_sub.add_parser("build", help="Build the structure task suite (Phase 12.8)")
    p_build.add_argument("--families", nargs="*", default=None, help="Task families to include")
    p_build.add_argument("--split", choices=["dev", "test"], default="dev", help="Split to build")
    p_build.add_argument("--corpus", default="corpus", help="Corpus dir or JSONL path")
    p_build.add_argument("--out", default="tasks", help="Output directory for item files")
    p_build.add_argument("--seed", type=int, default=0, help="RNG seed")
    p_build.add_argument("--freeze", action="store_true", help="Freeze the test split manifest")
    p_build.set_defaults(func=_cmd_tasks_build)

    p_grade = tasks_sub.add_parser("grade", help="Grade model responses (Phase 12.9)")
    p_grade.add_argument("items", help="Task items JSONL")
    p_grade.add_argument("responses", help="Model responses JSONL")
    p_grade.add_argument("-o", "--out", default=None, help="Write full RunReport JSON")
    p_grade.set_defaults(func=_cmd_tasks_grade)

    p_tok = sub.add_parser("tokens", help="Token-count a MOFJSON document (Phase 13.1)")
    p_tok.add_argument("path", help="Input MOFJSON path, or - for stdin")
    p_tok.add_argument("--tokeniser", default=None, help="Tokeniser name")
    p_tok.set_defaults(func=_cmd_tokens)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv and dispatch to a subcommand. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
