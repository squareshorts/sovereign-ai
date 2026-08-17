"""
Medication reconciliation task -- institution-controlled layer.

All task semantics (what reconciliation means, how to compare medications,
what constitutes a discrepancy) are defined HERE in the workflow layer.

Provider adapters implement only the inference/extraction abstraction:
they accept serialized input and return extracted medication lists as JSON.
They do NOT define reconciliation logic, discrepancy detection, or
output structure decisions.
"""

import json
import hashlib
import datetime
from typing import Dict, Any, Optional, Tuple, List

from .schemas import validate_input, validate_output
from .authorization import AuthorizationEngine, AuthorizationResult


def hash_data(data: Any) -> str:
    """SHA-256 hash of canonically serialized data."""
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def reconcile_medications(requests: List[Dict], statements: List[Dict]) -> Dict[str, Any]:
    """Institution-defined reconciliation logic.

    This is the canonical task implementation. It compares medications
    from MedicationRequest and MedicationStatement records, producing
    a structured reconciliation result.

    This logic lives in the workflow layer, NOT in adapters.
    """
    req_meds = {}
    for r in requests:
        name = r.get("medication", "").strip().lower()
        if name:
            req_meds[name] = r.get("dose", "")

    stat_meds = {}
    for s in statements:
        name = s.get("medication", "").strip().lower()
        if name:
            stat_meds[name] = s.get("dose", "")

    req_names = set(req_meds.keys())
    stat_names = set(stat_meds.keys())

    matched = sorted(req_names & stat_names)
    only_in_request = sorted(req_names - stat_names)
    only_in_statement = sorted(stat_names - req_names)

    discrepancies = []
    for med in matched:
        if req_meds[med] and stat_meds[med] and req_meds[med] != stat_meds[med]:
            discrepancies.append(
                f"Dose mismatch for '{med}': "
                f"request='{req_meds[med]}', statement='{stat_meds[med]}'"
            )

    if only_in_request:
        discrepancies.append(
            f"Medications only in requests: {only_in_request}"
        )
    if only_in_statement:
        discrepancies.append(
            f"Medications only in statements: {only_in_statement}"
        )

    human_review_required = len(discrepancies) > 0

    return {
        "matched": matched,
        "only_in_request": only_in_request,
        "only_in_statement": only_in_statement,
        "discrepancies": discrepancies,
        "human_review_required": human_review_required,
    }


class ProvenanceRecord:
    """Full provenance record for a single workflow execution."""

    def __init__(self, case_id: str, manifest: Dict[str, Any],
                 fixture_id: str, model_id: str, adapter_version: str):
        self.case_id = case_id
        self.workflow_id = manifest["workflow"]["id"]
        self.workflow_version = manifest["workflow"]["version"]
        self.fixture_id = fixture_id
        self.model_id = model_id
        self.adapter_version = adapter_version
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.input_hash: Optional[str] = None
        self.provider_facing_input_hash: Optional[str] = None
        self.raw_output_hash: Optional[str] = None
        self.output_hash: Optional[str] = None
        self.schema_validation_outcome: Optional[str] = None
        self.authorization_outcome: Optional[str] = None
        self.execution_status: Optional[str] = None

    def set_input_hash(self, h: str):
        self.input_hash = h
        
    def set_provider_facing_input_hash(self, h: str):
        self.provider_facing_input_hash = h

    def set_raw_output_hash(self, h: str):
        self.raw_output_hash = h

    def set_output_hash(self, h: str):
        self.output_hash = h

    def set_schema_validation_outcome(self, outcome: str):
        self.schema_validation_outcome = outcome

    def set_authorization_outcome(self, outcome: str):
        self.authorization_outcome = outcome

    def set_execution_status(self, status: str):
        self.execution_status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "workflow_id": self.workflow_id,
            "workflow_version": self.workflow_version,
            "fixture_id": self.fixture_id,
            "model_id": self.model_id,
            "adapter_version": self.adapter_version,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "provider_facing_input_hash": self.provider_facing_input_hash,
            "raw_output_hash": self.raw_output_hash,
            "output_hash": self.output_hash,
            "schema_validation_outcome": self.schema_validation_outcome,
            "authorization_outcome": self.authorization_outcome,
            "execution_status": self.execution_status,
        }

    def validate_completeness(self, manifest: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check provenance record against manifest audit requirements."""
        audit = manifest.get("audit", {})
        errors = []

        field_map = {
            "workflow_version_required": ("workflow_version", self.workflow_version),
            "provider_required": ("fixture_id", self.fixture_id),
            "model_version_required": ("model_id", self.model_id),
            "adapter_version_required": ("adapter_version", self.adapter_version),
            "timestamp_required": ("timestamp", self.timestamp),
            "input_hash_required": ("input_hash", self.input_hash),
            "provider_facing_input_hash_required": ("provider_facing_input_hash", self.provider_facing_input_hash),
            "raw_output_hash_required": ("raw_output_hash", self.raw_output_hash),
            "schema_validation_outcome_required": ("schema_validation_outcome", self.schema_validation_outcome),
            "authorization_outcome_required": ("authorization_outcome", self.authorization_outcome),
            "execution_status_required": ("execution_status", self.execution_status),
        }

        for audit_key, (field_name, value) in field_map.items():
            if audit.get(audit_key, False):
                if value is None or value == "":
                    errors.append(
                        f"Provenance field '{field_name}' is required but missing/empty"
                    )

        # Output hash is only conditionally required if execution reached output phase successfully
        if audit.get("output_hash_required", False):
            if self.schema_validation_outcome == "PASS" and self.execution_status == "COMPLETED":
                if self.output_hash is None or self.output_hash == "":
                    errors.append(
                        "Provenance field 'output_hash' is required but missing/empty"
                    )

        return len(errors) == 0, errors


class MedicationReconciliationTask:
    """Orchestrates the medication reconciliation workflow.

    The task layer:
    1. Validates input schema.
    2. Enforces authorization (externally to inference).
    3. Invokes the adapter for raw medication extraction.
    4. Applies institution-defined reconciliation logic to produce output.
    5. Validates output schema.
    6. Records full provenance.

    The adapter is a thin extraction shim. It extracts medication lists
    from input and returns them as JSON. It does NOT define reconciliation
    semantics, discrepancy detection, or output structure.
    """

    def __init__(self, manifest: Dict[str, Any], adapter: Any):
        self.manifest = manifest
        self.adapter = adapter
        self.auth_engine = AuthorizationEngine(manifest)

    def execute(self, case_id: str,
                input_data: Dict[str, Any]) -> Tuple[
                    Optional[Dict[str, Any]],
                    ProvenanceRecord,
                    AuthorizationResult,
                    bool,
                    List[str]]:
        """Execute the reconciliation workflow for a single case.

        Returns:
            (output_data_or_None, provenance, auth_result,
             schema_valid, schema_errors)
        """
        prov = ProvenanceRecord(
            case_id=case_id,
            manifest=self.manifest,
            fixture_id=self.adapter.FIXTURE_ID,
            model_id=self.adapter.MODEL_ID,
            adapter_version=self.adapter.ADAPTER_VERSION,
        )

        # Hash input
        input_hash = hash_data(input_data)
        prov.set_input_hash(input_hash)

        # 1. Validate input schema
        input_valid, input_errors = validate_input(input_data)
        if not input_valid:
            prov.set_authorization_outcome("input_invalid")
            prov.set_schema_validation_outcome("FAIL")
            prov.set_execution_status("FAILED_INPUT_VALIDATION")
            auth_result = AuthorizationResult(
                attempted=False, blocked=False, executed=False,
                details="Skipped: invalid input"
            )
            return None, prov, auth_result, False, input_errors

        # 2. Enforce authorization (EXTERNAL to inference)
        serialized_input = json.dumps(input_data)
        auth_result = self.auth_engine.check_input_authorization(
            serialized_input)

        if auth_result.blocked:
            prov.set_authorization_outcome("blocked")
            prov.set_execution_status("BLOCKED_AUTHORIZATION")
            return None, prov, auth_result, True, []

        prov.set_authorization_outcome("permitted")

        # 3. Invoke adapter for raw medication extraction
        try:
            provider_payload = {"history": input_data.get("history", [])}
            serialized_provider_payload = json.dumps(provider_payload)
            prov.set_provider_facing_input_hash(hash_data(provider_payload))
            
            raw_output = self.adapter.infer(serialized_provider_payload)
            prov.set_raw_output_hash(hash_data(raw_output))
        except Exception as e:
            prov.set_authorization_outcome("adapter_error")
            prov.set_execution_status("FAILED_ADAPTER_ERROR")
            return None, prov, auth_result, True, [f"Adapter error: {e}"]

        # 4. Parse adapter extraction output
        try:
            if isinstance(raw_output, str):
                extracted = json.loads(raw_output)
            elif isinstance(raw_output, dict):
                extracted = raw_output
            else:
                prov.set_schema_validation_outcome("FAIL")
                prov.set_execution_status("FAILED_PARSE")
                return (None, prov, auth_result, False,
                        ["Adapter returned non-JSON, non-dict output"])
        except json.JSONDecodeError as e:
            prov.set_schema_validation_outcome("FAIL")
            prov.set_execution_status("FAILED_PARSE")
            return (None, prov, auth_result, False,
                    [f"Adapter output is not valid JSON: {e}"])

        # 5. Apply institution-defined reconciliation logic
        if "request_meds" in extracted and "statement_meds" in extracted:
            output_data = reconcile_medications(
                extracted["request_meds"],
                extracted["statement_meds"],
            )
        elif all(k in extracted for k in
                 ("matched", "only_in_request", "only_in_statement",
                  "discrepancies", "human_review_required")):
            output_data = extracted
        else:
            prov.set_schema_validation_outcome("FAIL")
            prov.set_execution_status("FAILED_EXTRACTION_FIELDS")
            return (None, prov, auth_result, False,
                    ["Adapter output missing required extraction fields"])

        # 6. Validate final output schema
        output_valid, output_errors = validate_output(output_data)
        if not output_valid:
            prov.set_schema_validation_outcome("FAIL")
            prov.set_execution_status("FAILED_OUTPUT_SCHEMA")
            return None, prov, auth_result, False, output_errors

        # 7. Hash output and finalize provenance
        output_hash = hash_data(output_data)
        prov.set_output_hash(output_hash)
        prov.set_schema_validation_outcome("PASS")
        prov.set_execution_status("COMPLETED")

        return output_data, prov, auth_result, True, []
