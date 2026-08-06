# MOFJSON project hand-off and initialization

Audience: a student researcher or another coding/research agent taking over the project.

Last updated: 2026-08-06.

## 1. Mission

Develop and evaluate MOFJSON, an explicit hierarchical representation of metal-organic frameworks intended for language-model reasoning and generation.

The intended two-week workshop-paper claim is deliberately narrow:

> On matched MOF graph tasks, an explicit hierarchical JSON representation improves language-model reliability and restricted template-based assembly relative to MOFid, at the cost of longer sequences.

This is a hypothesis, not a current result. The project must produce controlled experimental evidence before making it.

Suggested paper title:

> **MOFJSON: Explicit Reticular Graphs as a Language-Model Interface for Metal-Organic Frameworks**

## 2. Current state

### Already implemented

- A `uv`-based Python 3.11 package and lockfile.
- A draft MOFJSON v0.1 JSON Schema.
- Dependency-free structural and referential validation.
- Deterministic serialization for order-insensitive arrays.
- CLI commands for validation, statistics, canonicalization, and printing the schema.
- A complete MOF-5-like Core example.
- A generation-only MOF-5-like design with full BDC linker information.
- Examples of MOFid, MOFid-in-JSON, MOFJSON-Chem, and MOFJSON-Core.
- Seven passing unit tests at hand-off.
- A two-week experimental plan and workshop-paper outline.

### Not implemented

- Dataset download or license-resolution code.
- CIF/MOFid to a shared semantic-record converter.
- Robust MOF decomposition, bond perception, or topology assignment.
- MOFid/MOFid-JSON/MOFJSON-Chem batch serializers.
- Graph-isomorphism-based semantic comparison.
- Deterministic benchmark task generation and scoring.
- API-model evaluation or result caching.
- GPU training code.
- CIF round-trip or template-assembly code.
- Statistical analysis, figures, tables, or paper results.

No experiment has yet shown that MOFJSON is better than MOFid.

### Repository status

This directory was not a Git repository at the time of hand-off. Do not assume commits or history exist. Ask the project owner before initializing or publishing a repository.

## 3. Read these files first

Read in this order:

1. [`README.md`](README.md) - short project overview and commands.
2. [`docs/specification.md`](docs/specification.md) - representation semantics and non-goals.
3. [`docs/model-training-view.md`](docs/model-training-view.md) - separation of stored records, model conditions, and generated targets.
4. [`examples/generation/mof5-like.design.json`](examples/generation/mof5-like.design.json) - complete generative payload.
5. [`examples/representations/README.md`](examples/representations/README.md) - four controlled representation conditions.
6. [`docs/two-week-plan.md`](docs/two-week-plan.md) - hypotheses, experiments, statistics, timeline, and risks.
7. [`paper/outline.md`](paper/outline.md) - manuscript claim ladder.
8. [`references.md`](references.md) and the three PDFs under `Inspiration/`.

The inspiration papers serve different roles:

- Bucior et al. define MOFid/MOFkey and explain their identifier purpose and limitations.
- Badrinarayanan et al. demonstrate MOFid-based language-model generation in MOFGPT.
- Runcie et al. provide the MolJSON representation-comparison methodology to adapt.

## 4. Initialize the environment

From the repository root:

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run mofjson validate examples/pcu-design.json
uv run mofjson stats examples/pcu-design.json
```

Expected state at hand-off:

```text
Ran 7 tests
OK
VALID examples/pcu-design.json
```

Inspect the examples:

```bash
uv run python -m json.tool examples/generation/mof5-like.design.json
uv run mofjson canonicalize examples/pcu-design.json
```

The generation-only payload intentionally omits `schema_version`. Add the fixed wrapper before validating it:

```python
import json
from pathlib import Path

from mofjson import validate

payload = json.loads(
    Path("examples/generation/mof5-like.design.json").read_text()
)
issues = validate({"schema_version": "0.1", **payload})
assert not issues, issues
```

Do not put API keys, tokens, credentials, or private dataset URLs in the repository. Use workstation secret management or environment variables outside committed configuration.

## 5. Conceptual model

### Full stored record

A full dataset or exchange record may contain:

```text
schema version
metadata and external identifiers
conditioning properties
building-block graphs
periodic net
optional atomic realization
provenance
```

### Generative target

The model should normally generate only:

```json
{
  "building_blocks": [],
  "net": {}
}
```

External names, MOFid/MOFkey values, database ids, citations, provenance, split labels, and measured properties are excluded from the output target.

Local graph ids remain essential:

- `C1` identifies an atom within a building block.
- `bdc_linker` identifies a unique building-block graph.
- `a` identifies a connection site on that linker.
- `linker_x` identifies an instance in the quotient graph.
- Edge endpoints refer to those local ids.

These are graph pointers, not metadata leakage.

### Conditional generation

Requested topology, chemistry, or property targets belong in the input condition. The generated structural design remains the output:

```json
{
  "conditions": {
    "topology": "pcu",
    "metal": "Zn",
    "target_property": {
      "name": "band_gap_eV",
      "maximum": 2.0
    }
  },
  "target": {
    "building_blocks": [],
    "net": {}
  }
}
```

## 6. Representation conditions

Every semantic MOF record must be serializable in four ways:

1. **MOFid** - building-block SMILES plus topology and catenation.
2. **MOFid-in-JSON** - exactly the same information in named JSON fields. This controls for JSON syntax.
3. **MOFJSON-Chem** - explicit building-block atom/bond graphs plus the same topology metadata. This controls for explicit chemistry graphs.
4. **MOFJSON-Core** - explicit chemistry, connection sites, building-block instances, and periodic quotient-graph edges.

Do not interpret Core versus MOFid as a pure syntax comparison: Core contains more information. Use the intermediate conditions to attribute gains.

`MOFJSON-Realization` is separate. It can hold lattice and fractional-coordinate information for CIF round-trip tests, but it is not information-matched to MOFid.

## 7. Scientific definitions that must not be collapsed

Use these terms precisely:

- **Parseable:** the output can be parsed as the declared format.
- **Schema-valid:** required fields and primitive constraints pass.
- **Referentially valid:** all atom, building-block, vertex, and connection-site references resolve.
- **Graph-consistent:** degrees, site use, connectivity, and periodic edges satisfy deterministic invariants.
- **Chemically plausible:** passes a documented chemistry policy; this is stronger than schema validity.
- **Assembly-successful:** a declared template assembler creates a 3D structure for an eligible design.
- **Structure-matched:** an assembled or round-tripped structure matches a reference within declared tolerances.
- **Synthesizable:** supported by experimental or serious synthetic-feasibility evidence. This project does not establish synthesizability.

Never call every schema-valid output a “valid MOF.” State the exact validation level.

## 8. Non-negotiable experimental controls

- Derive all four representations from the same underlying semantic record.
- Use paired train/dev/test ids across representations.
- Split by building-block combination or scaffold, not by random database row.
- Deduplicate before splitting.
- Keep model, prompt, examples, decoding, output budget, and retry policy matched.
- Treat truncated and unparsable output as failure in primary metrics.
- Do not silently repair primary outputs. Report deterministic repair separately if useful.
- Do not use an LLM judge for primary structural correctness.
- Do not use exact JSON string match as semantic equivalence.
- Normalize local ids, order-independent arrays, and periodic-image gauge before graph comparison.
- Report representation length, tokenizer tokens, inference tokens, cost, latency, and truncation.
- Compare matched training under both equal-record and equal-token budgets.
- Freeze schema, prompts, metrics, and inference settings after the pilot.
- Preserve raw prompts, raw responses, evaluator version, model id, and API date.
- Report negative results and all applicable denominators.

## 9. Critical path

Do not begin large model runs first. The critical path is a trusted paired semantic corpus and exact evaluators.

### Milestone A - freeze the semantic data contract

Deliver:

- A short decision record describing every field and invariant.
- A canonical internal semantic record independent of all four serializations.
- Explicit policies for bond types, charges, implicit hydrogens, connection sites, periodic shifts, duplicate components, and catenation.
- A decision on how local ids are canonicalized.
- A decision on how periodic graph equivalence is evaluated.

Acceptance criterion:

> Every generated field has a deterministic evaluator or is removed from v0.1.

### Milestone B - obtain a trusted paired corpus

Target 10,000-50,000 clean structures, with a manually reviewed test set of at least 500 structures across at least eight topologies. The fallback is 1,000 training and 200 verified test designs in a restricted library setting.

For each source, record:

- stable URL or DOI;
- version/download date;
- license and redistribution status;
- file checksum;
- exact preparation command;
- inclusion/exclusion counts and reasons.

Primary subset restrictions may include:

- resolvable single RCSR topology;
- no interpenetration for the first benchmark;
- no disorder or partial occupancy;
- no disconnected guests;
- successful node/linker decomposition;
- explicit connection mapping.

Acceptance criterion:

> At least 90% deterministic conversion success on the declared restricted subset, with a reviewed error taxonomy for the rest.

### Milestone C - implement paired serializers and validators

Deliver one serializer per condition plus batch validation. Unit-test each serializer against the same semantic fixtures.

Required tests include:

- representation round trips where meaningful;
- invariance to array order and local id renaming;
- dangling references;
- duplicate ids;
- site reuse and degree mismatch;
- invalid periodic shifts;
- disconnected nets;
- unknown topology handling;
- deterministic hashes and split stability.

Acceptance criterion:

> The same fixture produces all four conditions deterministically, and all intended information differences are documented.

### Milestone D - implement exact benchmark tasks

Start with tasks that are cheap to verify:

- building-block and element counts;
- node/linker role recognition;
- topology and catenation;
- vertex degree and adjacency;
- connection-site use;
- finite periodic-graph shortest paths;
- library-constrained design generation.

Separate tasks whose answers exist in all conditions from Core-only capability tasks.

Acceptance criterion:

> One hundred generated questions and answers are independently hand-checked before model evaluation.

### Milestone E - run a small pilot

Pilot all representation conditions on a small paired sample using one API model and one small GPU-trained model.

The pilot must detect:

- representation-specific prompt advantages;
- parser/evaluator bugs;
- context-window truncation;
- incorrect output-token matching;
- schema-constrained decoding advantages;
- excessive MOFJSON length;
- train/test leakage.

Freeze the benchmark after corrections. Do not repeatedly change it in response to main results.

## 10. Main experiments after the pilot

### Frozen API models

- Use at least three models across two provider families and two capability tiers.
- Pin dated model ids.
- Use temperature zero for deterministic reasoning tasks.
- Use one preregistered sampling setup for generation.
- Cache every response and usage record.
- Apply structured output either equally to all appropriate JSON conditions or as a separate ablation.

### GPU training

Use the same open-weight base model and tokenizer for every representation.

Run:

1. equal-record training;
2. equal-token or equal-update training;
3. data-scale curves near 1%, 10%, and 100%;
4. at least 1,000 generated samples per condition, preferably 10,000.

Keep trainable parameter count, optimizer family, precision, context length, decoding grid, and early-stopping policy matched. Use one shared pilot hyperparameter grid rather than tuning every representation independently.

Before selecting a model, inventory:

- GPU type and count;
- memory per GPU;
- scheduler and job limits;
- local model cache;
- permitted model licenses;
- expected training time at the longest representation length.

Record GPU-hours, peak memory, package revisions, model revision, seed, and energy if available.

### Small 3D experiment

Keep two claims separate:

- `MOFJSON-Core + fixed block/topology templates -> assembler -> CIF` for eligible structures.
- `CIF -> MOFJSON-Realization -> CIF` for 25-50 clean round-trip structures.

Core does not uniquely determine coordinates without external geometry/template information.

## 11. Metrics and statistics

Primary comparison:

> MOFJSON-Core versus MOFid on the fraction of constrained generations satisfying all applicable constraints.

Report:

- parse success;
- schema and reference validity;
- graph consistency;
- all-constraints-satisfied;
- uniqueness and novelty;
- assembly success on the eligible denominator;
- exact reasoning accuracy;
- input/output/reasoning tokens;
- cost and latency;
- truncation rate;
- GPU compute.

Use paired bootstrap 95% confidence intervals and McNemar's test for paired binary endpoints, with Holm correction for multiple primary comparisons. Always report effect sizes and raw denominators.

## 12. Expected project layout

Add new components without placing generated data in the package source:

```text
data/
  raw/                    # immutable downloads; ignored
  processed/
    records.jsonl         # canonical semantic records
    splits.json
    benchmark.jsonl
  checksums.txt
experiments/
  benchmark.toml
  prompts/
  training/
results/
  raw/                    # cached model outputs; ignored until release decision
  tables/
  figures/
src/mofjson/
  io/                     # source readers and semantic record
  representations/        # four serializers/parsers
  evaluation/             # deterministic metrics
  tasks/                  # benchmark generation
tests/
```

Prefer `uv add --group <group> ...` when research dependencies are selected. Keep the base validator lightweight. Do not add a large training stack until the workstation and model choice are known.

## 13. Two-week execution order

| Day | Main outcome |
|---|---|
| 1 | Freeze the v0.1 semantic contract, fields, and evaluation policy |
| 2 | Build the paired corpus and manually review initial conversions |
| 3 | Finish serializers, validators, hashes, and leakage-aware splits |
| 4 | Generate and hand-check deterministic benchmark tasks |
| 5 | Run API/GPU pilot, fix protocol bugs, then freeze the benchmark |
| 6 | Run main API evaluation and equal-record GPU training |
| 7 | Run equal-token/data-scale training and final generation |
| 8 | Complete statistics, ablations, error analysis, and small 3D tests |
| 9 | Write the complete workshop manuscript and figures |
| 10 | Reproduce from a clean environment, polish, and package artifacts |

See [`docs/two-week-plan.md`](docs/two-week-plan.md) for detailed go/no-go criteria.

## 14. Decisions requiring owner input

Do not guess silently if any of these materially change the study:

- Which MOF dataset may be downloaded and redistributed?
- Is there an existing internal CIF/MOFid corpus or converter?
- Which topology and building-block template libraries are available?
- What GPU hardware and scheduler are available?
- Which API providers and models are approved?
- What approximate API/compute budget should cap the main runs?
- Which workshop, page limit, template, and submission deadline apply?
- Should the repository be initialized and published, and under what license?

Continue with read-only investigation and small local prototypes while waiting, but do not start expensive or externally visible work without the relevant answer.

## 15. Stop and reassess conditions

Escalate to the owner if:

- fewer than 200 reliable paired structures can be produced;
- conversion errors cannot be separated from representation errors;
- information content differs without a matching ablation;
- an evaluator depends on subjective LLM judgment;
- MOFJSON routinely exceeds the selected context window;
- the 3D claim requires unavailable proprietary templates;
- dataset or model licensing prevents release;
- the pilot has no interpretable endpoint or reveals leakage;
- the schedule would require claiming synthesizability from structural validity.

## 16. Definition of done

The workshop project is complete only when it has:

- a frozen versioned schema and semantic contract;
- paired representation serializers;
- a redistributable corpus or deterministic build instructions;
- a manually reviewed benchmark subset;
- exact evaluators with tests;
- raw API/GPU outputs and pinned configurations;
- paired confidence intervals and error analysis;
- reproducible tables and figures;
- a complete 4-6 page manuscript;
- limitations that distinguish schema validity, chemistry, assembly, and synthesis.

## 17. End-of-session update template

At the end of every work session, append or deliver:

```text
Date:
Objective:
Completed:
Files changed:
Commands/tests run:
Results and evidence:
Decisions made:
Open risks/blockers:
Next exact task:
Owner input needed:
```

Do not report “done” without the relevant test, artifact, or result path.

## 18. Initialization prompt for another LLM agent

The following can be pasted into a new agent session:

```text
You are taking over the MOFJSON research prototype in this workspace.

First read HANDOFF.md completely, followed by README.md, docs/specification.md,
docs/model-training-view.md, docs/two-week-plan.md, and the representation examples.
Run the existing test suite before changing files.

The goal is a two-week workshop paper testing whether explicit hierarchical JSON
improves LLM reasoning and generation over MOFid. This is an unproven hypothesis.
Maintain four paired conditions: MOFid, MOFid-in-JSON, MOFJSON-Chem, and
MOFJSON-Core. Never conflate JSON syntax with additional graph information.

Treat the trusted semantic corpus and exact deterministic evaluators as the
critical path. Do not begin expensive model runs until paired serialization,
splits, tasks, metrics, and a small pilot are verified. Do not use an LLM judge
for primary correctness. Do not claim unique 3D reconstruction from Core or
synthesizability from schema validity.

Preserve user files and secrets. Use uv for dependencies and commands. Add tests
for every invariant. Record data licenses, checksums, model revisions, prompts,
token use, cost, and compute. Keep the owner informed of decisions that change
scope, information content, licensing, compute budget, or the paper claim.

Start by reporting: (1) test status, (2) repository state, (3) the first missing
critical-path artifact, and (4) a concrete next action. Then proceed with the
highest-priority safe task.
```
