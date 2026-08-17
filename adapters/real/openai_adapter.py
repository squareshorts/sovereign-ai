import os
import json
import urllib.request
import urllib.error
from adapters.base import BaseAdapter

class OpenAIAdapter(BaseAdapter):
    FIXTURE_ID = "openai_real"
    ADAPTER_VERSION = "1.0.0"

    def __init__(self, model_id: str, temperature: float = 0.0):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.model_id = model_id
        # We store the requested alias/model, and update if the API returns a different one
        self.MODEL_ID = model_id 
        self.temperature = temperature
        
        # Load canonical prompt
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "..", "workflow", "provider_extraction_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    def infer(self, prompt: str) -> str:
        # Schema definition for strict structured output
        extraction_schema = {
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
                        "required": ["medication", "dose"],
                        "additionalProperties": False
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
                        "required": ["medication", "dose"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["request_meds", "statement_meds"],
            "additionalProperties": False
        }

        payload = {
            "model": self.model_id,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "medication_extraction",
                    "strict": True,
                    "schema": extraction_schema
                }
            }
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_body = response.read().decode("utf-8")
                resp_json = json.loads(resp_body)
                # Update actual model returned by API
                if "model" in resp_json:
                    self.MODEL_ID = resp_json["model"]
                return resp_json["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            # Re-raise to trigger exponential backoff in the orchestrator
            raise RuntimeError(f"OpenAI API Error: {e.code} {e.read().decode('utf-8')}")
        except Exception as e:
            raise RuntimeError(f"OpenAI Transport Error: {str(e)}")
