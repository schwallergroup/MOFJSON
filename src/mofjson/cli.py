"""Command-line interface for inspecting draft MOFJSON documents."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any, Sequence

from .canonical import canonical_dumps
from .validation import validate


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _validate_command(args: argparse.Namespace) -> int:
    try:
        document = _read_json(args.path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR {args.path}: {exc}", file=sys.stderr)
        return 2
    issues = validate(document)
    if issues:
        for issue in issues:
            print(f"ERROR {issue}", file=sys.stderr)
        return 1
    print(f"VALID {args.path}")
    return 0


def _canonicalize_command(args: argparse.Namespace) -> int:
    try:
        document = _read_json(args.path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR {args.path}: {exc}", file=sys.stderr)
        return 2
    issues = validate(document)
    if issues and not args.allow_invalid:
        for issue in issues:
            print(f"ERROR {issue}", file=sys.stderr)
        return 1
    output = canonical_dumps(document) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


def _stats_command(args: argparse.Namespace) -> int:
    try:
        document = _read_json(args.path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR {args.path}: {exc}", file=sys.stderr)
        return 2
    blocks = document.get("building_blocks", [])
    atoms = sum(len(block.get("atom_graph", {}).get("atoms", [])) for block in blocks)
    result = {
        "building_blocks": len(blocks),
        "atoms_in_unique_building_blocks": atoms,
        "vertices_in_quotient_graph": len(document.get("net", {}).get("vertices", [])),
        "edges_in_quotient_graph": len(document.get("net", {}).get("edges", [])),
        "has_realization": "realization" in document,
        "validation_issues": len(validate(document)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _schema_command(_: argparse.Namespace) -> int:
    schema = files("mofjson").joinpath("mofjson.schema.json").read_text(encoding="utf-8")
    sys.stdout.write(schema)
    if not schema.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mofjson", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a MOFJSON document")
    validate_parser.add_argument("path", type=Path)
    validate_parser.set_defaults(handler=_validate_command)

    canonical_parser = subparsers.add_parser(
        "canonicalize", help="write a deterministically ordered document"
    )
    canonical_parser.add_argument("path", type=Path)
    canonical_parser.add_argument("-o", "--output", type=Path)
    canonical_parser.add_argument(
        "--allow-invalid", action="store_true", help="canonicalize even if validation fails"
    )
    canonical_parser.set_defaults(handler=_canonicalize_command)

    stats_parser = subparsers.add_parser("stats", help="summarize representation size")
    stats_parser.add_argument("path", type=Path)
    stats_parser.set_defaults(handler=_stats_command)

    schema_parser = subparsers.add_parser("schema", help="print the bundled JSON Schema")
    schema_parser.set_defaults(handler=_schema_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(args.handler(args))
