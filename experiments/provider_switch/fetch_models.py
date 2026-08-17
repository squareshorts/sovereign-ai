import os
import json
import urllib.request
import urllib.error

def fetch_openai_models():
    api_key = os.environ.get("OPENAI_API_KEY")
    req = urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["id"] for m in data["data"]]
    except Exception as e:
        print("OpenAI models fetch error:", e)
        return []

def fetch_anthropic_models():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    req = urllib.request.Request("https://api.anthropic.com/v1/models", headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["id"] for m in data["data"]]
    except Exception as e:
        print("Anthropic models fetch error:", e)
        return []

def fetch_gemini_models():
    api_key = os.environ.get("GEMINI_API_KEY")
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"].replace("models/", "") for m in data["models"]]
    except Exception as e:
        print("Gemini models fetch error:", e)
        return []

print("OpenAI:", fetch_openai_models()[:10])
print("Anthropic:", fetch_anthropic_models())
print("Gemini:", fetch_gemini_models()[:10])
