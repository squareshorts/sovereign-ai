"""
Conformant fixture -- deterministic test backend that extracts
medication data from the input and returns it as raw structured lists.

This fixture does NOT perform reconciliation. It returns extracted
medication lists so the institutional workflow layer can apply its
own reconciliation logic via reconcile_medications().
"""

import json
from .base import BaseAdapter


class ConformantFixture(BaseAdapter):
    FIXTURE_ID = "conformant_fixture"
    MODEL_ID = "deterministic-conformant-v1"
    ADAPTER_VERSION = "1.0.0"

    def infer(self, prompt: str) -> str:
        """Extract medication lists from input and return as JSON.

        Returns a dict with 'request_meds' and 'statement_meds' lists.
        The institutional workflow layer applies reconciliation logic.
        """
        data = json.loads(prompt)
        
        # If history contains embedded JSON, use it (for testing)
        if "history" in data and len(data["history"]) > 0:
            try:
                hist_data = json.loads(data["history"][0])
                if "requests" in hist_data or "statements" in hist_data:
                    data = hist_data
            except:
                pass

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
