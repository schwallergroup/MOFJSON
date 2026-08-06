# Two-week workshop-paper plan

## Decision

The project is feasible in two weeks if the paper makes a narrow, representation-level claim and uses the available GPU/API budget for matched experiments rather than a large property-optimization pipeline:

> On matched MOF graph tasks, an explicit hierarchical JSON representation improves language-model reliability and assembly readiness relative to MOFid, at the cost of longer sequences.

Do not promise general de novo discovery, synthesizability, or unique 3D reconstruction from a core design. Those require substantially more than two weeks.

## Research questions and preregistered hypotheses

**RQ1 - Generation reliability.** Can an LLM emit syntactically and referentially valid MOF designs more often in MOFJSON than in MOFid?

- H1: MOFJSON improves parse rate, reference integrity, topology/degree consistency, and total constraint satisfaction.

**RQ2 - Graph use.** Does explicit periodic connectivity improve reasoning about building blocks and nets?

- H2: MOFJSON improves exact accuracy on coordination, adjacency, degree, component, and periodic shortest-path questions.

**RQ3 - Reconstructability.** Does the new representation make the path to 3D less ambiguous?

- H3a: Given fixed building-block geometries and topology templates, MOFJSON-Core yields a higher assembly success rate than MOFid on a restricted corpus.
- H3b: MOFJSON-Realization round-trips clean CIFs within numerical tolerances. This is a separate lossless-exchange result.

**RQ4 - Efficiency trade-off.** Is any gain worth the additional tokens?

- H4: MOFJSON has a better accuracy-per-dollar or accuracy-per-reasoning-token frontier for graph tasks, despite longer input/output sequences.

Freeze the schema, tasks, prompts, model versions, decoding settings, sample sizes, and metrics after the Day 5 pilot. Report negative results.

## Representation conditions

Use four conditions so that extra information is not mistaken for better syntax:

1. **MOFid:** canonical building-block SMILES plus topology and catenation.
2. **MOFid-JSON:** the same SMILES and metadata placed in named JSON fields. This controls for JSON structure alone.
3. **MOFJSON-Chem:** explicit atom/bond graphs for building blocks plus the same topology metadata. This isolates explicit chemistry graphs.
4. **MOFJSON-Core:** MOFJSON-Chem plus connection sites and periodic quotient-graph edges. This is the proposed format.

Use `MOFJSON-Realization` only for the CIF round-trip experiment. It should not be compared with MOFid as though they contain equal information.

## Data

### Primary corpus

Select 10,000-50,000 clean paired structures from the same hMOF/QMOF-style sources used by MOFGPT, subject to what can be legally and reproducibly downloaded on Days 1-2. Require:

- an available CIF and MOFid;
- a resolvable single, non-interpenetrated RCSR topology for the primary analysis;
- no disorder, partial occupancy, disconnected guests, or unresolved bonds in the minimum viable subset;
- node/linker decomposition and explicit connection mapping;
- a deterministic train/development/test split by building-block combination, not random rows.

Deduplicate by a canonical design-graph hash. Split by linker scaffold and node family where possible to prevent near-duplicate leakage. Stratify the test set by topology, number of unique building blocks, atom count, metal family, and quotient-graph size.

### Minimum viable benchmark

Keep a manually verified test set of at least 500 MOFs across at least 8 topologies. If conversion is the bottleneck, a fallback experiment may use 1,000 training and 200 verified test designs in a library-constrained setting. A small trusted benchmark is preferable to thousands of uncertain decompositions.

## Experiments

### E1: representation statistics

For every condition report characters, tokenizer-specific tokens, gzip size, parsing time, canonicalization stability, and information fields present. This exposes the cost of explicitness.

### E2: read/translate

Ask frozen LLMs to answer deterministic questions from each representation: unique node/linker counts, element counts, building-block degrees, named topology, catenation, adjacency, and shortest paths on a finite quotient-graph expansion. Use only questions whose answers are represented in all compared conditions; put Core-only questions in a separately labeled capability experiment.

Primary metric: exact answer accuracy. Secondary metrics: refusal/parse failure and reasoning/output tokens.

### E3: constrained generation

Provide a library of allowed nodes, linkers, and compatible nets, then request designs satisfying 3-5 constraints. Evaluate without an LLM judge:

- JSON or MOFid parse success;
- unique-id and reference integrity;
- atom-graph validity under an explicitly documented coordination-bond policy;
- connection-site use and degree compatibility;
- known-net consistency;
- constraint satisfaction;
- uniqueness, novelty against train data, and diversity;
- template-assembler success for the restricted corpus.

The primary endpoint is the fraction satisfying all applicable constraints. Schema-valid is not synonymous with chemically valid or synthesizable.

Do not use exact string match as the semantic generation metric. Local ids and order-independent arrays may differ without changing a design. Normalize identifiers and periodic-image gauges, then compare building-block and quotient graphs by isomorphism. Exact canonical serialization is a useful secondary diagnostic only. If robust periodic graph isomorphism is not ready by Day 5, constrain generation to canonical library templates and state that limitation.

### E4: matched open-model training (primary GPU experiment)

Fine-tune the same open-weight decoder-only base model separately on every representation condition. Use two training objectives if the data pipeline is ready:

1. next-token modeling of serialized MOF designs, matching the basic MOFGPT generation setting;
2. instruction fine-tuning from algorithmically generated design constraints to the target representation.

Run two fairness regimes because MOFJSON sequences are longer:

- **Equal records:** every model sees the same semantic structures for the same number of epochs. This measures data efficiency per MOF.
- **Equal tokens/updates:** every model gets the same training-token or optimizer-step budget. This measures compute efficiency.

Use the same base weights, tokenizer, train/dev/test ids, trainable parameter count, optimizer family, precision, context length, decoding grid, and early-stopping rule. Record GPU model, GPU-hours, peak memory, energy if available, and package/model revisions. Plot learning curves at approximately 1%, 10%, and 100% of the paired training set. Generate at least 1,000 samples per condition for final validity/uniqueness estimates; 10,000 is preferred if inference is cheap.

If workstation capacity permits, repeat the frozen protocol at a second model scale. Do not tune hyperparameters independently per representation beyond one shared pilot grid, as this would undermine the comparison.

### E5: 3D round trip (small, separate)

For 25-50 clean structures, test `CIF -> MOFJSON-Realization -> CIF`. Compare composition, lattice, fractional coordinates modulo periodicity/symmetry, and structure matching tolerance. Separately attempt `Core + fixed building-block/topology template libraries -> assembler -> CIF` and report the eligible denominator and failure categories. Core alone does not determine unique coordinates.

### E6: ablations and error analysis

- Remove explicit connection sites.
- Replace atom graphs with SMILES.
- Remove JSON Schema constrained decoding.
- Shuffle order-independent arrays.
- Bin failures by atom count, number of building blocks, topology, interpenetration, and token length.

At minimum run the first three. They identify whether gains come from JSON, explicit graphs, extra information, or constrained decoding.

## Models and inference protocol

Use at least three frozen API models spanning two provider families and two capability tiers; the available credits make model diversity more valuable than very large prompt counts on one model. Pin dated model identifiers. Run temperature 0 for reasoning/translation and a preregistered sampling configuration for generation. Use the same system prompt, chemical context, exemplars, output-token budget, and retry policy in all representation conditions. Use native structured outputs only in a separate ablation or apply equally to all JSON conditions.

Cache raw requests, responses, token counts, latency, model id, seed where supported, and evaluator version. Never silently repair model outputs for the primary metric; report an optional deterministic-repair metric separately.

## Statistics

- Use paired items across representations.
- Report bootstrap 95% confidence intervals for rates and paired differences.
- Use McNemar's test for paired binary endpoints, with Holm correction across primary comparisons.
- Report effect sizes and raw denominators, not only p-values.
- Predeclare one primary comparison: MOFJSON-Core vs MOFid on all-constraints-satisfied in E3.
- Treat invalid or truncated output as failure; report truncation separately.

## Ten working days

| Day | Deliverable | Go/no-go criterion |
|---|---|---|
| 1 | Freeze claim, schema semantics, benchmark contract, data license/source | Every field has an evaluator or is removed |
| 2 | CIF/MOFid ingestion, large paired corpus, and 200+ manually verified examples | At least 90% deterministic conversion on the restricted subset |
| 3 | Canonicalizers, validators, graph invariants, design hashes | Round-trip/canonicalization tests pass |
| 4 | Deterministic task generator and evaluators | 100 hand-checked tasks have exact answers |
| 5 | API and GPU pilot across all conditions; freeze protocol | No representation-specific prompt/evaluator bug; shared training settings selected |
| 6 | Main frozen-API runs plus matched open-model training | At least 95% of API calls completed; equal-record training complete |
| 7 | Equal-token and data-scale training runs; generation | At least one model scale complete; drop only the second scale if needed |
| 8 | Statistics, ablations, error taxonomy, 3D subset | Primary table and two figures final |
| 9 | Write complete 4-6 page workshop draft | Full story, limitations, and reproducibility checklist present |
| 10 | Independent rerun from clean environment, polish, release artifact | One-command tests and benchmark subset reproduce reported numbers |

## Paper shape (4-6 pages plus references)

1. **Introduction:** representation is an experimental variable; MOFid was designed for identification, not LLM graph manipulation or unique 3D reconstruction.
2. **MOFJSON:** hierarchical schema, periodic quotient graph, canonicalization, validation, Core versus Realization.
3. **Benchmark:** paired information controls, frozen models, constrained generation, deterministic evaluation.
4. **Results:** reliability/cost Pareto, graph reasoning, assembly success, ablations, error modes.
5. **Limitations:** restricted clean subset, bond perception, topology assignment, no synthesizability claim, API-model reproducibility.
6. **Conclusion:** explicit reticular graphs are a promising interface; community schema and larger benchmarks are future work.

Suggested title: **MOFJSON: Explicit Reticular Graphs as a Language-Model Interface for Metal-Organic Frameworks**.

## Required figures and tables

- Figure 1: one MOF shown as MOFid, MOFid-JSON, MOFJSON-Core, and optional realization.
- Figure 2: benchmark flow from paired semantic object to model output to deterministic validators/assembler.
- Figure 3: all-constraints-satisfied and graph-reasoning accuracy versus tokens/cost.
- Table 1: schema capability and information-content comparison.
- Table 2: primary results with paired confidence intervals.
- Small error table: syntax, dangling reference, degree mismatch, invalid building block, unknown topology, assembly failure.

## Main risks and mitigations

| Risk | Mitigation |
|---|---|
| CIF decomposition and bond perception consume the schedule | Restrict to a trusted subset; do not solve universal MOF chemistry in this paper |
| MOFJSON wins only because it contains more information | Include MOFid-JSON and MOFJSON-Chem ablations; label capability experiments |
| JSON is much longer and truncates | Match output budgets, report truncation, and plot accuracy/cost Pareto |
| Structured decoding guarantees syntax | Separate schema-constrained and unconstrained settings; semantic checks remain primary |
| “Valid” outputs are not synthesizable | Use precise names: schema-valid, graph-consistent, assembly-successful; avoid synthetic claims |
| Small sample gives unstable conclusions | Pair examples, use confidence intervals, publish all raw outputs, narrow the claim |
| No improvement appears by Day 5 | Publish a representation audit/negative result only if the benchmark reveals a useful failure taxonomy; otherwise stop rather than overclaim |

## Definition of done

- Versioned schema and specification.
- At least 10,000 paired training designs plus 500 verified test designs, or a clearly labeled fallback corpus with deterministic download/build instructions.
- Exact evaluators with unit tests and no LLM judge in primary metrics.
- Raw prompts/responses plus pinned model metadata.
- One command to reproduce tables and figures from cached outputs.
- A complete workshop manuscript with the claim limited to measured tasks.
