import os
import json
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from adapters.real.openai_adapter import OpenAIAdapter
from adapters.real.anthropic_adapter import AnthropicAdapter
from adapters.real.google_adapter import GoogleAdapter
from workflow.task import MedicationReconciliationTask

def main():
    manifest_path = os.path.join(PROJECT_ROOT, "workflow_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    calibration_path = os.path.join(PROJECT_ROOT, "experiments", "provider_switch", "calibration_inputs.jsonl")
    cases = []
    with open(calibration_path, "r") as f:
        for line in f:
            cases.append(json.loads(line))
            
    # Selected models based on predefined rule
    models = {
        "OpenAI": "gpt-5.4-mini-2026-03-17",
        "Anthropic": "claude-haiku-4-5-20251001",
        "Google": "gemini-3.6-flash"
    }
            
    print(f"Loaded {len(cases)} calibration cases.")
    
    adapters = [
        ("OpenAI", OpenAIAdapter(models["OpenAI"])),
        ("Anthropic", AnthropicAdapter(models["Anthropic"])),
        ("Gemini", GoogleAdapter(models["Google"]))
    ]
    
    for name, adapter in adapters:

        print(f"\n--- Testing {name} Adapter ---")
        task = MedicationReconciliationTask(manifest, adapter)
        for i, case in enumerate(cases):
            print(f"  Case {i+1} ({case['case_id']}):")
            try:
                res = task.execute(case['case_id'], case['input'])
                output = res.extracted_output
                schema_ok = res.schema_valid
                errs = res.schema_errors
                if output is not None:
                    print(f"    Success! Schema OK: {schema_ok}")
                else:
                    print(f"    Failed! Errors: {errs}")
            except Exception as e:
                print(f"    Error: {e}")

if __name__ == "__main__":
    main()
