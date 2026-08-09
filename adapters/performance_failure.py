"""
Performance-failure fixture -- deterministic test backend that returns
structurally valid extraction output but with incorrect content.

Returns empty medication lists regardless of input, causing the
institutional reconciliation logic to produce no discrepancies
even when discrepancies exist.
"""

import json
from .base import BaseAdapter


class PerformanceFailureFixture(BaseAdapter):
    FIXTURE_ID = "performance_failure_fixture"
    MODEL_ID = "deterministic-perf-fail-v1"
    ADAPTER_VERSION = "1.0.0"

    def infer(self, prompt: str) -> str:
        """Return empty medication lists.

        This causes the institutional reconciliation logic to find
        no matches and no discrepancies, missing all actual medications.
        """
        output = {
            "request_meds": [],
            "statement_meds": [],
        }
        return json.dumps(output, sort_keys=True)
