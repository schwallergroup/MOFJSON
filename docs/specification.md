# MOFJSON v0.1: design notes

Status: experimental draft for a two-week representation study. This is not a proposed community standard yet.

## Design goal

MOFJSON represents a MOF as a hierarchy of explicit graphs that a language model can read and generate without learning SMILES traversal rules or inferring periodic connectivity from an RCSR label:

1. Each unique building block has an explicit atom/bond graph.
2. Each building block exposes named connection sites anchored to atoms.
3. A periodic quotient graph instantiates the building blocks and connects their sites.
4. Integer image shifts on edge endpoints make periodic connections explicit.
5. An optional realization stores lattice vectors and fractional atomic coordinates when lossless crystal exchange is required.

The first four items form `MOFJSON-Core`, the LLM-facing reticular design. The optional fifth item forms `MOFJSON-Realization`. Core can drive an assembler only when the required building-block geometries and topology embedding are supplied by a fixed external template library; it does not determine unique 3D coordinates by itself.

## Why this is not just MOFid in braces

MOFid is deliberately an identifier: dot-separated building-block SMILES followed by topology and catenation metadata. It does not preserve a topology embedding, connection-site mapping, cell metrics, fractional coordinates, disorder, or defects. Consequently, a MOFid does not generally select one unique CIF.

MOFJSON-Core adds explicit building-block atom graphs, ports, and periodic connectivity. `MOFJSON-Realization` can be a lossless carrier for an idealized crystal. A benchmark must keep these claims separate:

- Compare MOFid with information-matched JSON ablations to measure serialization effects.
- Compare MOFid with MOFJSON-Core to measure the benefit of explicit graph information.
- Test CIF round trips only for MOFJSON-Realization.

## Core objects

### `building_blocks`

Each unique node or linker contains:

- a stable local `id`;
- a `role` of `node` or `linker`;
- an `atom_graph` with order-independent atoms and bonds;
- named `connection_sites` anchored to one or more atoms;
- optionally, local connection vectors in a later schema revision.

Bond order `"coordination"` is distinct from conventional covalent orders. This avoids inventing metal oxidation states or applying organic valence assumptions to coordination bonds.

### `net`

`vertices` instantiate unique building blocks. `edges` reference a vertex, a connection-site id, and an integer periodic image `[i, j, k]` at each endpoint. The quotient graph therefore states which port connects to which port across a unit-cell boundary.

The RCSR `name` is useful metadata, not the connectivity itself. Unknown or non-RCSR nets can still be represented.

### `realization`

The optional realization begins with lattice vectors and fractional atom sites. The draft intentionally keeps this layer small. Symmetry, disorder, occupancies, defects, and a mapping from design atoms to realized sites need to be specified before claiming general CIF fidelity.

## Identity and canonical form

The order of building blocks, atoms, bonds, connection sites, vertices, and edges is semantically irrelevant. The package supplies deterministic ordering for exact-match evaluation. This is a serialization canonical form, not yet a graph-isomorphism-based chemical canonicalization.

## Validation levels

The prototype validator currently checks:

- required fields and primitive types;
- unique identifiers;
- atom and building-block references;
- vertex and connection-site references;
- three-integer periodic image shifts;
- one use of each instantiated connection site;
- basic realization vector shapes.

Research-grade validation should add element symbols, valence policies separated by bond type, charge balance, net connectedness and dimensionality, coordination compatibility, graph isomorphism, and optional 3D assembly checks.

## Non-goals for v0.1

- A universal representation of defects, disorder, guests, composites, or framework dynamics.
- A unique prediction of 3D geometry from chemistry and an RCSR name alone.
- A replacement for CIF, MOFid, or MOFkey in their established exchange and identification roles.
- A claim of experimental synthesizability from schema validity.
