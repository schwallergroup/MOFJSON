from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from mofjson import canonical_dumps, validate


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "pcu-design.json"
GENERATION_EXAMPLE = ROOT / "examples" / "generation" / "mof5-like.design.json"


class MOFJSONTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_example_is_valid(self) -> None:
        self.assertEqual(validate(self.document), [])

    def test_unknown_connection_site_is_reported(self) -> None:
        invalid = copy.deepcopy(self.document)
        invalid["net"]["edges"][0]["source"]["connection_site"] = "missing"
        messages = [str(issue) for issue in validate(invalid)]
        self.assertTrue(any("unknown site 'missing'" in message for message in messages))

    def test_reused_connection_site_is_reported(self) -> None:
        invalid = copy.deepcopy(self.document)
        invalid["net"]["edges"][1]["target"]["connection_site"] = "px"
        messages = [str(issue) for issue in validate(invalid)]
        self.assertTrue(any("is reused" in message for message in messages))

    def test_canonicalization_is_invariant_to_array_order(self) -> None:
        reordered = copy.deepcopy(self.document)
        reordered["building_blocks"].reverse()
        reordered["net"]["vertices"].reverse()
        reordered["net"]["edges"].reverse()
        self.assertEqual(canonical_dumps(self.document), canonical_dumps(reordered))

    def test_non_numeric_connection_vector_is_reported(self) -> None:
        invalid = copy.deepcopy(self.document)
        invalid["building_blocks"][0]["connection_sites"][0]["vector"] = ["x", 0, 0]
        messages = [str(issue) for issue in validate(invalid)]
        self.assertTrue(any("must contain only numbers" in message for message in messages))

    def test_malformed_identifier_does_not_crash_validator(self) -> None:
        invalid = copy.deepcopy(self.document)
        invalid["net"]["vertices"][0]["id"] = ["not", "hashable"]
        messages = [str(issue) for issue in validate(invalid)]
        self.assertTrue(any("vertex ids must be present and unique" in message for message in messages))

    def test_generation_payload_is_valid_after_version_is_added(self) -> None:
        payload = json.loads(GENERATION_EXAMPLE.read_text(encoding="utf-8"))
        record = {"schema_version": "0.1", **payload}
        self.assertEqual(validate(record), [])


if __name__ == "__main__":
    unittest.main()
