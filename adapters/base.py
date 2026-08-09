"""
Base adapter interface for the SPST provider abstraction boundary.

Adapters implement ONLY the inference abstraction. They accept a prompt
string (serialized input) and return a JSON string (serialized output).
They do NOT define task semantics, reconciliation logic, or output
structure decisions — those belong to the institutional workflow layer.
"""


class BaseAdapter:
    """Abstract base class for all SPST adapters/fixtures."""

    FIXTURE_ID = "base"
    MODEL_ID = "unset"
    ADAPTER_VERSION = "0.0.0"

    def infer(self, prompt: str) -> str:
        """Accept a serialized input and return a serialized JSON output.

        This is the only method that crosses the provider abstraction
        boundary. All task semantics are defined in the workflow layer.
        """
        raise NotImplementedError("Subclasses must implement infer()")

    def get_metadata(self):
        """Return adapter metadata for provenance."""
        return {
            "fixture_id": self.FIXTURE_ID,
            "model_id": self.MODEL_ID,
            "adapter_version": self.ADAPTER_VERSION,
        }
