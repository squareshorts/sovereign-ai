import os
import json
import urllib.request
import urllib.error
from adapters.base import BaseAdapter

class AnthropicAdapter(BaseAdapter):
    FIXTURE_ID = "anthropic_real"
    ADAPTER_VERSION = "1.0.0"

    def __init__(self, model_id: str, temperature: float = 0.0):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.model_id = model_id
        self.MODEL_ID = model_id
        self.temperature = temperature
        
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "..", "workflow", "provider_extraction_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    def infer(self, prompt: str) -> str:
        # We use Claude's tool use to force structured JSON output.
        tool_schema = {
            "name": "extract_medications",
            "description": "Extract request and statement medications into structured JSON.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "request_meds": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "medication": {"type": "string"},
                                "dose": {"type": "string"}
                            },
                            "required": ["medication", "dose"]
                        }
                    },
                    "statement_meds": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "medication": {"type": "string"},
                                "dose": {"type": "string"}
                            },
                            "required": ["medication", "dose"]
                        }
                    }
                },
                "required": ["request_meds", "statement_meds"]
            }
        }

        payload = {
            "model": self.model_id,
            "temperature": self.temperature,
            "system": self.system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "tools": [tool_schema],
            "tool_choice": {"type": "tool", "name": "extract_medications"},
            "max_tokens": 4096
        }

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_body = response.read().decode("utf-8")
                resp_json = json.loads(resp_body)
                
                # We expect the model to invoke the tool. We extract the tool input as the JSON response.
                for content_block in resp_json.get("content", []):
                    if content_block.get("type") == "tool_use" and content_block.get("name") == "extract_medications":
                        return json.dumps(content_block.get("input", {}))
                
                # Fallback if no tool use found
                raise RuntimeError("Anthropic model did not use the forced structured extraction tool.")

        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Anthropic API Error: {e.code} {e.read().decode('utf-8')}")
        except Exception as e:
            raise RuntimeError(f"Anthropic Transport Error: {str(e)}")
