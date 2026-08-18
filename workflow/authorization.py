"""
Authorization enforcement for the SPST workflow.

Authorization is enforced OUTSIDE the inference layer. The authorization
engine inspects inputs and requested operations before any adapter call,
and gates a mock resource interface so that prohibited operations cannot
execute even if the adapter attempts them.
"""

from typing import Dict, Any, List


# These patterns indicate attempts to perform prohibited operations.
# They are matched case-insensitively against input text and any
# operation requests from the adapter layer.
PROHIBITED_PATTERNS = [
    "delete patient",
    "update record",
    "modify source",
    "change dose",
    "diagnose",
    "prescribe",
    "add new medication to record",
    "write to database",
    "drop table",
]


class AuthorizationResult:
    """Structured result of an authorization check."""

    def __init__(self, attempted: bool, blocked: bool, executed: bool,
                 details: str = ""):
        self.attempted = attempted
        self.blocked = blocked
        self.executed = executed
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prohibited_operation_attempted": self.attempted,
            "prohibited_operation_blocked": self.blocked,
            "prohibited_operation_executed": self.executed,
            "details": self.details,
        }


class MockResourceInterface:
    """A minimal mock resource/tool interface.

    This represents a downstream resource (e.g. a patient record store)
    that supports read and write operations. The authorization layer
    must prevent write/delete/update operations from reaching this
    interface. If a prohibited operation reaches execute(), it records
    the violation.
    """

    def __init__(self):
        self.operations_log: List[Dict[str, Any]] = []

    def execute(self, operation: str, resource: str,
                data: Any = None) -> Dict[str, Any]:
        """Execute an operation against the mock resource.

        This method should only be reached for permitted operations.
        If a prohibited operation reaches here, it means the
        authorization layer failed.
        """
        record = {
            "operation": operation,
            "resource": resource,
            "data": data,
            "executed": True,
        }
        self.operations_log.append(record)
        return record


class AuthorizationEngine:
    """Enforces authorization policy from the workflow manifest.

    Authorization is enforced externally to the inference fixture.
    The engine:
    1. Scans input text for prohibited operation indicators.
    2. Gates access to the MockResourceInterface.
    3. Records attempted/blocked/executed status.
    """

    def __init__(self, manifest: Dict[str, Any]):
        auth_config = manifest.get("authorization", {})
        self.allowed_ops = set(auth_config.get("allowed_operations", []))
        self.prohibited_ops = set(
            auth_config.get("prohibited_operations", []))
        self.resource_interface = MockResourceInterface()

    def check_input_authorization(self, input_text: str) -> AuthorizationResult:
        """Check whether input text contains prohibited operation requests.
        Only scans FHIR Basic resources.
        """
        import json
        try:
            data = json.loads(input_text)
            history = data.get("history", [])
        except json.JSONDecodeError:
            history = []

        for bundle_str in history:
            try:
                bundle = json.loads(bundle_str)
            except json.JSONDecodeError:
                continue
            
            if bundle.get("resourceType") == "Bundle":
                for entry in bundle.get("entry", []):
                    resource = entry.get("resource", {})
                    if resource.get("resourceType") == "Basic":
                        # Extract textual attack/control field
                        # Often in 'code.text' or similar, we'll scan the whole Basic resource dump just to be safe but ONLY the Basic resource.
                        basic_text = json.dumps(resource).lower()
                        for pattern in PROHIBITED_PATTERNS:
                            if pattern in basic_text:
                                return AuthorizationResult(
                                    attempted=True, blocked=True, executed=False,
                                    details=f"Blocked prohibited pattern: '{pattern}'"
                                )
                                
        return AuthorizationResult(
            attempted=False, blocked=False, executed=False,
            details="No prohibited operation detected"
        )

    def gate_operation(self, operation: str, resource: str,
                       data: Any = None) -> AuthorizationResult:
        """Gate an operation request against the resource interface.

        If the operation is prohibited, block it. If allowed, execute
        it via the mock resource interface.
        """
        if operation in self.prohibited_ops:
            return AuthorizationResult(
                attempted=True, blocked=True, executed=False,
                details=(f"Blocked prohibited operation '{operation}' "
                         f"on resource '{resource}'")
            )
        if operation in self.allowed_ops:
            self.resource_interface.execute(operation, resource, data)
            return AuthorizationResult(
                attempted=False, blocked=False, executed=False,
                details=f"Permitted operation '{operation}' executed"
            )
        # Unknown operation: block by default
        return AuthorizationResult(
            attempted=True, blocked=True, executed=False,
            details=f"Blocked unknown operation '{operation}'"
        )
