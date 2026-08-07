"""Ingestion from the MOFology KG, the CIF corpus, and ODAC25.

Phase 09.
"""

from mofjson.ingest.corpus import BuildCorpus, CorpusReport
from mofjson.ingest.crossvalidate import (
    BuildCrossValidationReport,
    CrossValidate,
    CrossValidationReport,
    Disagreement,
)
from mofjson.ingest.identifiers import (
    ClassifyLocalName,
    IdentifierFamily,
    ParseMofid,
    ResolveIdentifiers,
    UriToLocalName,
)
from mofjson.ingest.kg_csv import KgRecord, ReadKgCsv, ReadKgCsvWithCoverage
from mofjson.ingest.kg_ttl import CountSubjectBlocks, IterSubjectBlocks, ReadKgTtl
from mofjson.ingest.manifest import BuildManifest, LoadKgStructureManifest, Manifest
from mofjson.ingest.merge import GroupDuplicates, MergeGroup, MergeNormalizedWithStructure
from mofjson.ingest.normalize import QUIRK_HANDLERS, NormalizedRecord, NormalizeRecord
from mofjson.ingest.odac import (
    IndexAseLmdb,
    IterOdacStructures,
    MatchStabNamesToOdac,
    OpenAseLmdb,
)

__all__ = [
    "BuildCorpus",
    "BuildCrossValidationReport",
    "BuildManifest",
    "ClassifyLocalName",
    "CorpusReport",
    "CountSubjectBlocks",
    "CrossValidate",
    "CrossValidationReport",
    "Disagreement",
    "GroupDuplicates",
    "IdentifierFamily",
    "IndexAseLmdb",
    "IterOdacStructures",
    "IterSubjectBlocks",
    "KgRecord",
    "LoadKgStructureManifest",
    "Manifest",
    "MatchStabNamesToOdac",
    "MergeGroup",
    "MergeNormalizedWithStructure",
    "NormalizedRecord",
    "NormalizeRecord",
    "OpenAseLmdb",
    "ParseMofid",
    "QUIRK_HANDLERS",
    "ReadKgCsv",
    "ReadKgCsvWithCoverage",
    "ReadKgTtl",
    "ResolveIdentifiers",
    "UriToLocalName",
]
