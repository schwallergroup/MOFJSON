# Model-training view

A dataset record and a model target should not be identical.

## Complete dataset record

The storage/exchange record may contain:

```json
{
  "schema_version": "0.1",
  "name": "MOF-5",
  "identifiers": {"mofid": "...", "database_id": "..."},
  "building_blocks": [],
  "net": {},
  "realization": {},
  "properties": {},
  "provenance": {}
}
```

## Generative target

For unconditional structure generation, train only on:

```json
{
  "building_blocks": [],
  "net": {}
}
```

The data loader can add the fixed `schema_version` before validation. Exclude global names, MOFid/MOFkey/database identifiers, provenance, citations, split labels, and measured properties from the output target.

Do not remove local graph ids. Atom ids (`C1`), building-block ids (`bdc_linker`), connection-site ids (`a`), vertex ids (`linker_x`), and edge endpoint references are the pointers that define the graph. They should be generated according to a deterministic local naming convention.

For conditional generation, properties belong in the input prompt or a separate `conditions` object, not in the target design:

```json
{
  "conditions": {
    "topology": "pcu",
    "metal": "Zn",
    "target_property": {"name": "band_gap_eV", "maximum": 2.0}
  },
  "target": {
    "building_blocks": [],
    "net": {}
  }
}
```

This prevents accidental leakage of the answer through database ids or memorized MOF names. A separate evaluation table can join generated designs to computed or predicted properties after generation.

The complete generation-only MOF-5-like payload is [`../examples/generation/mof5-like.design.json`](../examples/generation/mof5-like.design.json).
