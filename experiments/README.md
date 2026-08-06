# Experiments

The primary experiment matrix is specified in [`../docs/two-week-plan.md`](../docs/two-week-plan.md).

Keep immutable configuration files here and write raw model responses under `results/raw/` (ignored by Git until the release policy is decided). Processed, non-sensitive aggregate tables should go under `results/tables/`. Prompt templates must be versioned and shared across representation conditions except for the requested output format.

Suggested first commands once the benchmark runner exists:

```bash
uv run mofjson validate examples/pcu-design.json
uv run python -m unittest discover -s tests -v
```
