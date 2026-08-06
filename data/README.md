# Data layout

Raw and processed datasets are intentionally not committed at project creation.

Expected layout after the data source and license are frozen:

```text
data/
  raw/                 # immutable downloaded CIF/MOFid records
  processed/
    records.jsonl      # canonical semantic records
    splits.json        # leakage-aware train/dev/test ids
    benchmark.jsonl    # generated tasks and exact answers
  checksums.txt
```

Every raw source must have a version or download date, URL/DOI, license, checksum, and deterministic preparation command. The primary benchmark must remain reproducible from cached inputs if an upstream database changes.
