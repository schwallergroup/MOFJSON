# Workshop-paper outline

Working title: **MOFJSON: Explicit Reticular Graphs as a Language-Model Interface for Metal-Organic Frameworks**

## Abstract skeleton

1. MOF language models commonly use MOFid, an identifier that combines building-block SMILES with topology metadata.
2. MOFid is compact and chemically informed, but is not an explicit periodic graph or unique 3D crystal representation.
3. Introduce MOFJSON-Core and the optional MOFJSON-Realization layer.
4. Compare MOFid, information-matched JSON controls, and MOFJSON using frozen API models and matched open-model training on paired reasoning and constrained-generation tasks.
5. Report validity, constraint satisfaction, assembly success, token cost, confidence intervals, and main error modes.
6. Conclude only what the completed experiments support.

## Claim ladder

- Safe now: MOFJSON is machine-parseable and can express explicit building-block and periodic quotient graphs.
- Requires benchmark evidence: LLMs generate and reason over it more reliably than MOFid.
- Requires restricted assembler evidence: MOFJSON-Core more often produces assembly-ready designs.
- Requires realization round trips: MOFJSON-Realization can preserve selected clean 3D crystals.
- Out of scope: generated structures are synthesizable or experimentally superior.

## Tables/figures placeholders

- `fig1-representations`: visual comparison and information content.
- `fig2-benchmark`: paired task/evaluator pipeline.
- `fig3-pareto`: frozen-model accuracy or constraint satisfaction versus tokens/API cost.
- `fig4-learning-curves`: matched-model validity versus training records and tokens.
- `table1-capabilities`: compactness, explicit graph, periodic edges, 3D, canonicalization.
- `table2-results`: preregistered primary/secondary endpoints.
- `table3-errors`: deterministic failure taxonomy.

## Reproducibility appendix

Dataset versions and exclusions; split construction; schema version; prompt templates; dated model ids; decoding parameters; API dates; evaluator commit; statistical tests; compute/API budget; licenses.
