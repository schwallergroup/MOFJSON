# MOFJSON

MOFJSON is a research prototype for testing whether explicit hierarchical graphs are a better language-model interface for metal-organic frameworks than MOFid strings.

The project starts from a deliberately narrow observation: MOFid is a useful identifier made from building-block SMILES, topology, and catenation, but it was not designed to be a lossless 3D crystal representation. MOFJSON v0.1 therefore separates:

- **Core:** explicit building-block atom graphs, connection sites, and periodic quotient-net edges for language-model reasoning, generation, and restricted template-based assembly.
- **Realization (optional):** lattice vectors and fractional atomic sites for lossless-exchange experiments.

This repository is a two-week workshop-paper scaffold, not evidence yet that MOFJSON outperforms MOFid. The benchmark in [`docs/two-week-plan.md`](docs/two-week-plan.md) uses both broad frozen-API evaluation and matched GPU fine-tuning, with controls that avoid conflating JSON syntax, extra information, data exposure, and compute budget.

## Quick start

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run mofjson validate examples/pcu-design.json
uv run mofjson stats examples/pcu-design.json
uv run python -m unittest discover -s tests -v
```

Print the bundled draft schema or create a deterministic serialization:

```bash
uv run mofjson schema
uv run mofjson canonicalize examples/pcu-design.json
```

## Repository map

- [`HANDOFF.md`](HANDOFF.md): complete student/agent initialization, current state, priorities, controls, and takeover prompt.
- [`docs/specification.md`](docs/specification.md): v0.1 semantics, invariants, and non-goals.
- [`docs/model-training-view.md`](docs/model-training-view.md): separates full dataset records, model targets, and conditioning information.
- [`docs/two-week-plan.md`](docs/two-week-plan.md): hypotheses, controls, experiments, statistics, risks, and daily schedule.
- [`src/mofjson/mofjson.schema.json`](src/mofjson/mofjson.schema.json): JSON Schema draft.
- [`src/mofjson/validation.py`](src/mofjson/validation.py): dependency-free referential validator.
- [`examples/pcu-design.json`](examples/pcu-design.json): illustrative Core document.
- [`examples/generation/mof5-like.design.json`](examples/generation/mof5-like.design.json): complete generation-only payload with full linker information.
- [`examples/representations/README.md`](examples/representations/README.md): the same MOF-5-like design as MOFid, MOFid-in-JSON, MOFJSON-Chem, and MOFJSON-Core.
- [`paper/outline.md`](paper/outline.md): claim ladder and manuscript skeleton.
- [`references.md`](references.md): how the three inspiration papers shape the project.

## Scientific boundaries

Schema-valid, graph-consistent, assembly-ready, chemically plausible, and synthesizable are different claims. The v0.1 validator checks the first two only. The planned paper must report them separately and should not imply experimental feasibility from a valid JSON document.
