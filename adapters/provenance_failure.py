"""
Provenance-failure fixture -- deterministic test backend that returns
valid extraction output but reports incomplete adapter metadata.

Used to verify that the provenance validation machinery detects
missing audit fields.
"""

import json
from .base import BaseAdapter


class ProvenanceFailureFixture(BaseAdapter):
    FIXTURE_ID = "provenance_failure_fixture"
    MODEL_ID = ""  # Deliberately empty -- provenance failure
    ADAPTER_VERSION = ""  # Deliberately empty -- provenance failure

    def infer(self, prompt: str) -> str:
        """Return conformant extraction output (same as conformant fixture)."""
        data = json.loads(prompt)

        request_meds = []
        for r in data.get("requests", []):
            med = r.get("medication", "").strip()
            dose = r.get("dose", "").strip()
            if med:
                request_meds.append({"medication": med, "dose": dose})

        statement_meds = []
        for s in data.get("statements", []):
            med = s.get("medication", "").strip()
            dose = s.get("dose", "").strip()
            if med:
                statement_meds.append({"medication": med, "dose": dose})

        output = {
            "request_meds": request_meds,
            "statement_meds": statement_meds,
        }
        return json.dumps(output, sort_keys=True)
