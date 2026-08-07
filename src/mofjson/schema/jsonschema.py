"""JSON Schema emission: layers, profiles, and provider-specific dialects.

Phase 03.6 ships a minimal metal_node layer emitter. Phase 06 adds metadata layers
and the document shell. Profiles / dialects arrive fully in Phase 10.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

from mofjson.schema.crystal import CrystalStructure
from mofjson.schema.document import MOFJSONDocument
from mofjson.schema.metadata import (
    DocumentProvenance,
    Identifiers,
    Property,
    PropertyProvenance,
)
from mofjson.schema.metal_node import MetalNode
from mofjson.schema.reticular import ReticularStructure

_LAYER_MODELS: dict[str, type[BaseModel]] = {
    "metal_node": MetalNode,
    "crystal": CrystalStructure,
    "reticular": ReticularStructure,
    "identifiers": Identifiers,
    "property": Property,
    "provenance": DocumentProvenance,
    "property_provenance": PropertyProvenance,
    "document": MOFJSONDocument,
    "metadata": Identifiers,  # alias: identifiers block as the metadata entry point
}

# Anthropic rejects minimum/maximum on integers. Convert small closed ranges to enum;
# drop bounds on large ranges (e.g. space_group_number 1–230) and rely on validators.
_MAX_ENUM_SPAN = 32

_DROP_KEYWORDS = frozenset(
    {
        "title",
        "examples",
        "default",
        "exclusiveMinimum",
        "exclusiveMaximum",
    }
)


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline $defs / $ref for providers with limited $ref support."""
    defs = schema.pop("$defs", None) or schema.pop("definitions", None) or {}
    if not defs:
        return schema

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                name = ref.rsplit("/", 1)[-1]
                if name not in defs:
                    return node
                return resolve(copy.deepcopy(defs[name]))
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    resolved = resolve(schema)
    return resolved if isinstance(resolved, dict) else schema


def _is_integer_type(node: dict[str, Any]) -> bool:
    t = node.get("type")
    if t == "integer":
        return True
    if isinstance(t, list) and "integer" in t and "null" in t and len(t) == 2:
        return True
    return False


def _maybe_int_enum(
    node: dict[str, Any], *, convert_bounds: bool
) -> dict[str, Any]:
    """Replace small integer min/max with enum when convert_bounds is True."""
    if not convert_bounds or not _is_integer_type(node):
        node.pop("minimum", None)
        node.pop("maximum", None)
        return node
    lo = node.get("minimum")
    hi = node.get("maximum")
    if (
        isinstance(lo, (int, float))
        and isinstance(hi, (int, float))
        and float(lo).is_integer()
        and float(hi).is_integer()
    ):
        ilo, ihi = int(lo), int(hi)
        if 0 <= ihi - ilo <= _MAX_ENUM_SPAN:
            node["enum"] = list(range(ilo, ihi + 1))
            node.pop("minimum", None)
            node.pop("maximum", None)
            return node
    node.pop("minimum", None)
    node.pop("maximum", None)
    return node


def _strictify_object(node: dict[str, Any]) -> None:
    """OpenAI strict: additionalProperties false; every property required."""
    if node.get("type") == "object" or "properties" in node:
        node.setdefault("additionalProperties", False)
        props = node.get("properties")
        if isinstance(props, dict) and props:
            node["required"] = sorted(props.keys())


def _transform_node(node: Any, *, convert_int_bounds: bool) -> Any:
    """Shared OpenAI/Anthropic transform over a schema tree."""
    if isinstance(node, list):
        return [_transform_node(x, convert_int_bounds=convert_int_bounds) for x in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for k, v in node.items():
        if k in _DROP_KEYWORDS:
            continue
        # Keep descriptions — they help models; titles from Pydantic are noise.
        if k == "description":
            out[k] = v
            continue
        if k in ("minimum", "maximum"):
            # Handled after the loop once type is known.
            continue
        out[k] = _transform_node(v, convert_int_bounds=convert_int_bounds)

    if "minimum" in node or "maximum" in node:
        if "minimum" in node:
            out["minimum"] = node["minimum"]
        if "maximum" in node:
            out["maximum"] = node["maximum"]
        out = _maybe_int_enum(out, convert_bounds=convert_int_bounds)
        # Float bounds: drop for providers (unsupported / noisy); validators remain.
        if out.get("type") == "number" or (
            isinstance(out.get("type"), list) and "number" in (out.get("type") or [])
        ):
            out.pop("minimum", None)
            out.pop("maximum", None)

    _strictify_object(out)
    return out


def to_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """Provider dialect: OpenAI structured-output / strict JSON Schema subset.

    Restrictions handled here (commented because undocumented and easily re-broken):
    - additionalProperties must be false on every object
    - every property must appear in required (use nullable types, not optional keys)
    - drop title/default/examples; keep description
    - drop minimum/maximum (OpenAI accepts them but Anthropic parity + smaller schemas)
    - $defs inlined by caller before this transform
    """
    out = _transform_node(copy.deepcopy(schema), convert_int_bounds=False)
    if not isinstance(out, dict):
        raise TypeError("to_openai expected a dict schema root")
    return out


def to_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """Provider dialect: Anthropic tool-use JSON Schema subset.

    Restrictions handled here:
    - No minimum/maximum on integers — convert small closed ranges to enum
      (MolJSON charge [-5, 5] pattern). Large ranges (space_group_number 1–230)
      drop bounds instead of exploding the enum.
    - Same object strictness as OpenAI (additionalProperties false, all required).
    """
    out = _transform_node(copy.deepcopy(schema), convert_int_bounds=True)
    if not isinstance(out, dict):
        raise TypeError("to_anthropic expected a dict schema root")
    return out


def _apply_dialect(schema: dict[str, Any], dialect: str) -> dict[str, Any]:
    if dialect == "openai":
        return to_openai(schema)
    if dialect == "anthropic":
        return to_anthropic(schema)
    if dialect == "jsonschema":
        return schema
    if dialect == "paper":
        return schema
    raise ValueError(f"Unsupported dialect: {dialect!r}")


def _emit_model_schema(model: type[BaseModel], title: str, dialect: str) -> dict[str, Any]:
    raw = model.model_json_schema(mode="validation")
    inlined = _inline_defs(copy.deepcopy(raw))
    if dialect in ("openai", "anthropic"):
        cleaned = _apply_dialect(inlined, dialect)
    else:
        # paper / jsonschema: unconstrained reference — keep min/max, no strict rewrite
        cleaned = inlined if isinstance(inlined, dict) else {}
    if not isinstance(cleaned, dict):
        cleaned = {}
    cleaned = dict(cleaned)
    cleaned["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    cleaned["title"] = title
    return cleaned


def MetalNodeJsonSchema() -> dict[str, Any]:
    """Return an OpenAI-strict-ish JSON Schema for MetalNode."""
    return _emit_model_schema(MetalNode, "MetalNode", "openai")


def CrystalStructureJsonSchema() -> dict[str, Any]:
    """Return an OpenAI-strict-ish JSON Schema for CrystalStructure."""
    return _emit_model_schema(CrystalStructure, "CrystalStructure", "openai")


def ReticularStructureJsonSchema() -> dict[str, Any]:
    """Return an OpenAI-strict-ish JSON Schema for ReticularStructure (Phase 05)."""
    return _emit_model_schema(ReticularStructure, "ReticularStructure", "openai")


def GetPaperSchema(layer: str | None = None) -> dict[str, Any]:
    """Unconstrained reference schema (keeps min/max; no strict required-all-keys).

    Frozen under schemas/paper/ for published benchmark reproducibility.
    """
    if layer is None:
        layer = "document"
    if layer == "metadata":
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "MOFJSONMetadata",
            "type": "object",
            "properties": {
                "identifiers": _emit_model_schema(Identifiers, "Identifiers", "paper"),
                "properties": {
                    "type": "array",
                    "items": _emit_model_schema(Property, "Property", "paper"),
                },
                "provenance": _emit_model_schema(
                    DocumentProvenance, "DocumentProvenance", "paper"
                ),
            },
        }
    model = _LAYER_MODELS.get(layer)
    if model is None:
        raise ValueError(f"Unknown schema layer: {layer!r}")
    return _emit_model_schema(model, model.__name__, "paper")


def GetSchema(
    layer: str | None = None,
    profile: str | None = None,
    dialect: str = "jsonschema",
) -> dict[str, Any]:
    """Public entry point for JSON Schema emission."""
    if dialect not in ("jsonschema", "openai", "anthropic", "paper"):
        raise ValueError(f"Unsupported dialect: {dialect!r}")

    if profile is not None:
        # Lazy import avoids a circular dependency with mofjson.llm.profiles.
        from mofjson.llm.profiles import GetProfileSchema

        return GetProfileSchema(profile, dialect=dialect)

    if layer is None:
        layer = "document"

    if layer == "metal_node":
        # Phase 03 golden pins the OpenAI-strict form as the default layer schema.
        if dialect == "paper":
            return GetPaperSchema(layer="metal_node")
        if dialect == "anthropic":
            return _emit_model_schema(MetalNode, "MetalNode", "anthropic")
        return MetalNodeJsonSchema()

    if layer == "crystal":
        if dialect == "paper":
            return GetPaperSchema(layer="crystal")
        if dialect == "anthropic":
            return _emit_model_schema(CrystalStructure, "CrystalStructure", "anthropic")
        return CrystalStructureJsonSchema()

    if layer == "reticular":
        if dialect == "paper":
            return GetPaperSchema(layer="reticular")
        if dialect == "anthropic":
            return _emit_model_schema(ReticularStructure, "ReticularStructure", "anthropic")
        return ReticularStructureJsonSchema()

    if layer == "metadata":
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "MOFJSONMetadata",
            "type": "object",
            "properties": {
                "identifiers": _emit_model_schema(Identifiers, "Identifiers", dialect),
                "properties": {
                    "type": "array",
                    "items": _emit_model_schema(Property, "Property", dialect),
                },
                "provenance": _emit_model_schema(
                    DocumentProvenance, "DocumentProvenance", dialect
                ),
            },
            "additionalProperties": False,
        }

    model = _LAYER_MODELS.get(layer)
    if model is None:
        raise NotImplementedError(f"phase-06.7: schema layer {layer!r} not implemented yet")
    return _emit_model_schema(model, model.__name__, dialect)


__all__ = [
    "CrystalStructureJsonSchema",
    "GetPaperSchema",
    "GetSchema",
    "MetalNodeJsonSchema",
    "ReticularStructureJsonSchema",
    "to_anthropic",
    "to_openai",
]
