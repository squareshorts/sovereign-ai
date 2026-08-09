"""
Schema validation for SPST workflow inputs and outputs.

This module enforces structural conformance of data flowing through
the institution-controlled workflow layer. It validates against the
field requirements specified in workflow_manifest.json.
"""

from typing import Dict, Any, List, Tuple


REQUIRED_INPUT_KEYS = {"history", "requests", "statements"}
REQUIRED_OUTPUT_KEYS = {
    "matched", "only_in_request", "only_in_statement",
    "discrepancies", "human_review_required"
}


def validate_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate input data against the required schema.

    Returns (is_valid, list_of_errors).
    """
    errors = []
    if not isinstance(data, dict):
        return False, ["Input is not a dictionary"]

    missing = REQUIRED_INPUT_KEYS - set(data.keys())
    if missing:
        errors.append(f"Missing required input keys: {sorted(missing)}")

    if not isinstance(data.get("history"), list):
        errors.append("'history' must be a list")
    if not isinstance(data.get("requests"), list):
        errors.append("'requests' must be a list")
    if not isinstance(data.get("statements"), list):
        errors.append("'statements' must be a list")

    return len(errors) == 0, errors


def validate_output(data: Any) -> Tuple[bool, List[str]]:
    """Validate output data against the required schema.

    Returns (is_valid, list_of_errors).
    """
    errors = []
    if not isinstance(data, dict):
        return False, ["Output is not a dictionary"]

    missing = REQUIRED_OUTPUT_KEYS - set(data.keys())
    if missing:
        errors.append(f"Missing required output keys: {sorted(missing)}")

    if "matched" in data and not isinstance(data["matched"], list):
        errors.append("'matched' must be a list")
    if "only_in_request" in data and not isinstance(data["only_in_request"], list):
        errors.append("'only_in_request' must be a list")
    if "only_in_statement" in data and not isinstance(data["only_in_statement"], list):
        errors.append("'only_in_statement' must be a list")
    if "discrepancies" in data and not isinstance(data["discrepancies"], list):
        errors.append("'discrepancies' must be a list")
    if "human_review_required" in data and not isinstance(data["human_review_required"], bool):
        errors.append("'human_review_required' must be a boolean")

    return len(errors) == 0, errors
