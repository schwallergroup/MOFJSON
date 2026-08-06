# Four representations of the same MOF-5-like design

These examples use a Zn4O node, a benzene-1,4-dicarboxylate (BDC) linker, the `pcu` topology, and no interpenetration. They are intended to make the benchmark controls concrete, not to replace a crystallographic MOF-5 record.

## 1. MOFid

```text
[O-]C(=O)c1ccc(cc1)C(=O)[O-].[Zn][O]([Zn])([Zn])[Zn] MOFid-v1.pcu.cat0;MOF-5
```

The dot-separated prefix contains canonicalized building-block SMILES. `MOFid-v1` is the format marker, `pcu` is the topology, `cat0` denotes no additional interpenetrating net, and text following the semicolon is a comment.

MOFGPT uses a normalized training spelling of the same basic information:

```text
[O-]C(=O)c1ccc(cc1)C(=O)[O-].[Zn][O]([Zn])([Zn])[Zn]&&pcu.cat0
```

Full example: [`mofid.txt`](mofid.txt).

## 2. MOFid-in-JSON

```json
{
  "representation": "MOFid-in-JSON",
  "version": "1",
  "components": [
    "[O-]C(=O)c1ccc(cc1)C(=O)[O-]",
    "[Zn][O]([Zn])([Zn])[Zn]"
  ],
  "topology": "pcu",
  "catenation": 0,
  "comment": "MOF-5"
}
```

This is an experimental control, not an official format. It preserves the same SMILES and metadata as MOFid but puts them in named JSON fields. Comparing it with MOFid tests the effect of JSON syntax without replacing SMILES or adding explicit connectivity.

Full example: [`mofid-in-json.json`](mofid-in-json.json).

## 3. MOFJSON-Chem

```json
{
  "representation": "MOFJSON-Chem",
  "building_blocks": [
    {
      "id": "zn4o_node",
      "role": "node",
      "atom_graph": {
        "atoms": [{"id": "Zn1", "element": "Zn"}],
        "bonds": []
      }
    }
  ],
  "topology": {"name": "pcu", "dimension": 3, "catenation": 0}
}
```

The shortened snippet omits most atoms and bonds; the linked file is complete. MOFJSON-Chem replaces each building-block SMILES with explicit atom and bond lists but still represents framework connectivity only by the topology label. It isolates the MolJSON-like benefit of explicit chemical graphs.

Full example: [`mofjson-chem.json`](mofjson-chem.json).

## 4. MOFJSON-Core

```json
{
  "schema_version": "0.1",
  "building_blocks": [
    {
      "id": "bdc_linker",
      "role": "linker",
      "atom_graph": {"atoms": [], "bonds": []},
      "connection_sites": [
        {"id": "a", "anchor_atoms": ["O1", "O2"]},
        {"id": "b", "anchor_atoms": ["O3", "O4"]}
      ]
    }
  ],
  "net": {
    "name": "pcu",
    "dimension": 3,
    "vertices": [],
    "edges": [
      {
        "id": "e2",
        "source": {"vertex": "linker_x", "connection_site": "b", "image": [0, 0, 0]},
        "target": {"vertex": "node_0", "connection_site": "nx", "image": [1, 0, 0]}
      }
    ]
  }
}
```

Again, this snippet is abbreviated and is not independently valid. The full Core example adds named connection sites, building-block instances, explicit edges, and periodic image shifts. The `[1, 0, 0]` target image states that the connection crosses the unit-cell boundary in the positive `a` direction.

Full valid example: [`../pcu-design.json`](../pcu-design.json).

## What changes across the four conditions?

| Feature | MOFid | MOFid-in-JSON | MOFJSON-Chem | MOFJSON-Core |
|---|---:|---:|---:|---:|
| Building-block SMILES | Yes | Yes | No | No |
| Explicit building-block atom/bond graph | No | No | Yes | Yes |
| Named topology and catenation | Yes | Yes | Yes | Yes |
| Explicit connection sites | No | No | No | Yes |
| Explicit periodic quotient-graph edges | No | No | No | Yes |
| Unique 3D coordinates | No | No | No | No |

MOFJSON-Core still does not determine a unique 3D crystal. Restricted 3D assembly additionally needs fixed geometry/topology templates; a lossless crystal requires the optional MOFJSON-Realization layer.
