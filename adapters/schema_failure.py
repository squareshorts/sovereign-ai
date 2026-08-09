"""
Schema-failure fixture -- deterministic test backend that returns
output missing required fields.

Used to verify that the institutional workflow layer correctly
detects and rejects structurally invalid adapter outputs.
"""

import json
from .base import BaseAdapter


class SchemaFailureFixture(BaseAdapter):
    FIXTURE_ID = "schema_failure_fixture"
    MODEL_ID = "deterministic-schema-fail-v1"
    ADAPTER_VERSION = "1.0.0"

    def infer(self, prompt: str) -> str:
        """Return output missing required extraction fields.

        This is intentionally malformed: it omits both 'request_meds'
        and 'statement_meds', and also does not match the pre-reconciled
        output format.
        """
        output = {
            "some_unexpected_field": [],
        }
        return json.dumps(output, sort_keys=True)
