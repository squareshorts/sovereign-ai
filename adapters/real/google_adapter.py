import os
import json
import urllib.request
import urllib.error
from adapters.base import BaseAdapter

class GoogleAdapter(BaseAdapter):
    FIXTURE_ID = "google_real"
    ADAPTER_VERSION = "1.0.0"

    def __init__(self, model_id: str, temperature: float = 0.0):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.model_id = model_id
        self.MODEL_ID = model_id
        self.temperature = temperature
        
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "..", "workflow", "provider_extraction_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    def infer(self, prompt: str) -> str:
        # Schema definition for Google Gemini generateContent
        extraction_schema = {
            "type": "OBJECT",
            "properties": {
                "request_meds": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "medication": {"type": "STRING"},
                            "dose": {"type": "STRING"}
                        },
                        "required": ["medication", "dose"]
                    }
                },
                "statement_meds": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "medication": {"type": "STRING"},
                            "dose": {"type": "STRING"}
                        },
                        "required": ["medication", "dose"]
                    }
                }
            },
            "required": ["request_meds", "statement_meds"]
        }

        payload = {
            "system_instruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "responseSchema": extraction_schema
            }
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"

        import time
        max_retries = 3
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json"
                },
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    resp_body = response.read().decode("utf-8")
                    resp_json = json.loads(resp_body)
                    
                    if "modelVersion" in resp_json:
                        self.MODEL_ID = resp_json["modelVersion"]

                    try:
                        text_content = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                        return text_content
                    except (KeyError, IndexError) as e:
                        raise RuntimeError(f"Malformed response from Gemini: {resp_json}")
                    
            except urllib.error.HTTPError as e:
                if attempt < max_retries and e.code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini API Error: {e.code} {e.read().decode('utf-8')}")
            except urllib.error.URLError as e:
                if attempt < max_retries and ("timed out" in str(e.reason).lower() or isinstance(e.reason, (TimeoutError, OSError))):
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini Transport Error: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Gemini Transport Error: {str(e)}")
